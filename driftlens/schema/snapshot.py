"""Snapshot assembly from identity-layer output (Step 20)."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import driftlens
from driftlens.config.schema import Config, SideOptions
from driftlens.errors import DiagnosticSink, InvalidInputError, UnsupportedFormatError
from driftlens.identity.chunks import (
    all_chunk_identity_keys,
    compute_embedding_digest_field,
)
from driftlens.identity.documents import (
    Granularity,
    GranularitySource,
    assign_identity,
    disambiguate_duplicate_ids,
)
from driftlens.identity.hashing import DTYPE_TAGS, canonical_json, sha256_hex
from driftlens.identity.snapshot_id import (
    EmbeddingDescriptor,
    build_descriptor,
    compute_snapshot_id,
)
from driftlens.schema.manifest import (
    SNAPSHOT_SCHEMA_VERSION,
    FieldMappingSummary,
    Manifest,
)
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.security.limits import RuntimeLimits, loads_limited
from driftlens.text.normalize import normalization_version
from driftlens.text.tokenizer import tokenizer_id

_EMBEDDING_CONFIDENCE_RANK = {
    "UNKNOWN": 0,
    "INFERRED_FROM_DATA": 1,
    "USER_SUPPLIED": 2,
    "ADAPTER_METADATA": 3,
    "DECLARED": 4,
}
_IDENTITY_CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_CONFIDENCE_DESCRIPTORS = ("model", "provider", "similarity_metric", "dtype", "normalized")


@dataclass
class StatisticsAccumulator:
    """Mutable statistics container; later steps overwrite the None sentinels."""

    document_count: int | None = None
    chunk_count: int | None = None
    embedding_model_counts: dict[str, int] | None = None
    identity_summary: dict[str, int] | None = None
    embedding_validity: dict[str, int] | None = None
    chunk_profile_histograms: dict[str, Any] | None = None
    metadata_schema: dict[str, Any] | None = None
    anomaly_counts: dict[str, int] | None = None


@dataclass
class DocumentRow:
    """Per-document identity layer (text hashes deferred to Step 25/26)."""

    document_id: str
    chunk_count: int
    identity_confidence: str
    document_raw_hash: str | None = None
    document_content_hash: str | None = None
    total_tokens: int = 0
    reconstruction_confidence: str | None = None
    first_created_at: str | None = None
    last_updated_at: str | None = None


@dataclass
class DocumentTable:
    """Document rows keyed by final document_id."""

    rows: dict[str, DocumentRow] = field(default_factory=dict)


@dataclass
class Snapshot:
    """Manifest, canonical records, document table, and statistics."""

    manifest: Manifest
    records: list[Record]
    documents: DocumentTable
    statistics: StatisticsAccumulator


def _distinct_or_mixed(values: list[str]) -> str | None:
    distinct = sorted(set(values))
    if not distinct:
        return None
    if len(distinct) == 1:
        return distinct[0]
    return "MIXED"


def _embedding_bearing(records: list[Record]) -> list[Record]:
    return [record for record in records if record.embedding.vector is not None]


def _check_dimension_uniformity(records: list[Record]) -> int | None:
    dimensions = [
        record.embedding.dimension
        for record in records
        if record.embedding.vector is not None
    ]
    if len(set(dimensions)) > 1:
        modal = Counter(dimensions).most_common(1)[0][0]
        offending = min(
            (
                record
                for record in records
                if record.embedding.vector is not None
                and record.embedding.dimension != modal
            ),
            key=lambda record: record.row_number,
        )
        raise InvalidInputError(
            location=f"row {offending.row_number}",
            message=(
                f"embedding dimension {offending.embedding.dimension}, "
                f"expected {modal} (modal dimension)"
            ),
        )
    return dimensions[0] if dimensions else None


def _embedding_metadata_confidence(records: list[Record]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in _CONFIDENCE_DESCRIPTORS:
        worst_rank: int | None = None
        worst_level = "UNKNOWN"
        for record in records:
            mapping = getattr(record, "_embedding_metadata_confidence", None) or {}
            level = mapping.get(name, "UNKNOWN")
            rank = _EMBEDDING_CONFIDENCE_RANK.get(level, 0)
            if worst_rank is None or rank < worst_rank:
                worst_rank = rank
                worst_level = level
        result[name] = worst_level
    return result


def _document_table(records: list[Record]) -> DocumentTable:
    groups: dict[str, list[Record]] = {}
    order: list[str] = []
    for record in records:
        key = record.document_id if record.document_id is not None else ""
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)

    rows: dict[str, DocumentRow] = {}
    for key in order:
        group = groups[key]
        worst = min(
            group,
            key=lambda record: _IDENTITY_CONFIDENCE_RANK.get(
                record.identity_confidence or "LOW", 0
            ),
        )
        created = [record.created_at for record in group if record.created_at is not None]
        updated = [record.updated_at for record in group if record.updated_at is not None]
        rows[key] = DocumentRow(
            document_id=key,
            chunk_count=len(group),
            identity_confidence=worst.identity_confidence or "LOW",
            total_tokens=sum(record.token_count or 0 for record in group),
            first_created_at=min(created) if created else None,
            last_updated_at=max(updated) if updated else None,
        )
    return DocumentTable(rows=rows)


def build_snapshot(
    records: list[Record],
    *,
    side: SideOptions,
    cfg: Config,
    source_type: str,
    sink: DiagnosticSink,
    limits: RuntimeLimits,
    granularity_override: Granularity | None,
    field_mapping_summary: FieldMappingSummary,
    unsafe_pickle: bool,
) -> Snapshot:
    """Assign identity, aggregate descriptors, and assemble a snapshot."""

    records, granularity, granularity_source = assign_identity(
        records,
        side=side,
        sink=sink,
        granularity_override=granularity_override,
    )

    embedding_dimension = _check_dimension_uniformity(records)
    embedding_records = _embedding_bearing(records)

    embedding_model = _distinct_or_mixed(
        [r.embedding.model for r in embedding_records if r.embedding.model is not None]
    )
    embedding_provider = _distinct_or_mixed(
        [r.embedding.provider for r in embedding_records if r.embedding.provider is not None]
    )
    embedding_dtype = _distinct_or_mixed(
        [r.embedding.dtype for r in embedding_records if r.embedding.dtype is not None]
    )
    similarity_metric = _distinct_or_mixed(
        [
            r.embedding.similarity_metric
            for r in embedding_records
            if r.embedding.similarity_metric is not None
        ]
    )
    model_counts = Counter(
        r.embedding.model if r.embedding.model is not None else "UNKNOWN"
        for r in embedding_records
    )

    normalized_values = [r.embedding.normalized for r in embedding_records]
    if normalized_values and all(value is True for value in normalized_values):
        embedding_normalized: bool | None = True
    elif normalized_values and all(value is False for value in normalized_values):
        embedding_normalized = False
    else:
        embedding_normalized = None

    metadata_confidence = _embedding_metadata_confidence(embedding_records)

    chunk_identity_levels: dict[str, list[int]] = {}
    for record in records:
        keys = all_chunk_identity_keys(
            record,
            document_identity_confidence=None,
            duplicate_rank_within_document=0,
        )
        chunk_identity_levels[record.record_id] = sorted(keys)

    confidence_counter = Counter(record.identity_confidence for record in records)
    identity_summary = {
        level: confidence_counter.get(level, 0) for level in ("HIGH", "MEDIUM", "LOW")
    }

    descriptor = EmbeddingDescriptor(
        model=embedding_model,
        provider=embedding_provider,
        dtype=embedding_dtype,
        dimension=embedding_dimension,
        similarity_metric=similarity_metric,
        normalized=embedding_normalized,
    )
    descriptor_bytes = build_descriptor(
        SNAPSHOT_SCHEMA_VERSION,
        normalization_version(),
        tokenizer_id(),
        granularity,
        descriptor,
    )

    disambiguate_duplicate_ids(
        records,
        on_duplicate_id=cfg.identity.on_duplicate_id,
        sink=sink,
    )

    snapshot_key, content = compute_snapshot_id(records, descriptor_bytes)

    documents = _document_table(records)
    statistics = StatisticsAccumulator(
        document_count=len(documents.rows),
        chunk_count=len(records),
        embedding_model_counts=dict(model_counts),
        identity_summary=dict(identity_summary),
    )

    records.sort(key=lambda record: record.record_id)

    manifest = Manifest(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        snapshot_id=snapshot_key,
        content_digest=content.hex(),
        normalization_version=normalization_version(),
        tokenizer_id=tokenizer_id(),
        text_params={
            "min_overlap_chars": cfg.text.min_overlap_chars,
            "overlap_window": cfg.text.overlap_window,
            "casefold": cfg.text.casefold,
        },
        driftlens_version=driftlens.__version__,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_type=source_type,
        granularity=granularity,
        granularity_source=granularity_source,
        document_count=len(documents.rows),
        chunk_count=len(records),
        embedding_dimension=embedding_dimension,
        embedding_dtype=embedding_dtype,
        embedding_model=embedding_model,
        embedding_model_counts=dict(model_counts),
        embedding_provider=embedding_provider,
        embedding_metadata_confidence=metadata_confidence,
        similarity_metric=similarity_metric,
        embedding_normalized=embedding_normalized,
        chunk_identity_levels=chunk_identity_levels,
        identity_summary=dict(identity_summary),
        field_mapping=field_mapping_summary,
        unsafe_pickle=unsafe_pickle,
        limits_applied=limits.applied(),
    )

    return Snapshot(
        manifest=manifest,
        records=records,
        documents=documents,
        statistics=statistics,
    )

# --- Step 36: on-disk snapshot format (write/read) -------------------------

_DESCRIPTOR_FIELDS = ("model", "provider", "dtype", "similarity_metric", "normalized")
_MANIFEST_DESCRIPTOR_KEYS = {
    "model": "embedding_model",
    "provider": "embedding_provider",
    "dtype": "embedding_dtype",
    "similarity_metric": "similarity_metric",
    "normalized": "embedding_normalized",
}
_ARROW_DTYPES = {
    "float16": pa.float16(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "int8": pa.int8(),
}
_NUMPY_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
}
_RECORDS_FILE = "records.parquet"
_DOCUMENTS_FILE = "documents.parquet"
_STATISTICS_FILE = "statistics.json"
_MANIFEST_FILE = "manifest.json"
_EMBEDDINGS_DIR = "embeddings"


def sparse_embedding_field(record: Record, field_name: str, manifest_value: Any) -> Any:
    """Return the per-record override, or None when it matches the manifest."""
    record_value = getattr(record.embedding, field_name, None)
    return None if record_value == manifest_value else record_value


def _concrete_dtype_and_dimension(record: Record) -> tuple[str, int]:
    vector = record.embedding.vector
    dtype = record.embedding.dtype or str(vector.dtype)
    dimension = record.embedding.dimension or int(vector.shape[-1])
    return dtype, dimension


def _embedding_filename(dtype: str, dimension: int) -> str:
    return f"vectors_{DTYPE_TAGS.get(dtype, dtype)}_{dimension}.parquet"


def _numpy_dtype_for_arrow(arrow_type: pa.DataType) -> np.dtype:
    for name, candidate in _ARROW_DTYPES.items():
        if candidate.equals(arrow_type):
            return np.dtype(_NUMPY_DTYPES[name])
    raise InvalidInputError(message=f"unsupported embedding dtype {arrow_type}")


def _string_column(values: list[Any]) -> pa.Array:
    return pa.array(values, type=pa.string())


def _int32_column(values: list[Any]) -> pa.Array:
    return pa.array(values, type=pa.int32())


def _bool_column(values: list[Any]) -> pa.Array:
    return pa.array(values, type=pa.bool_())


def _records_table(records: list[Record], *, no_text: bool, manifest: Manifest) -> pa.Table:
    manifest_values = {
        field_name: getattr(manifest, key)
        for field_name, key in _MANIFEST_DESCRIPTOR_KEYS.items()
    }
    columns: dict[str, pa.Array] = {
        "record_id": _string_column([record.record_id for record in records]),
        "document_id": _string_column([record.document_id for record in records]),
        "identity_source": _string_column([record.identity_source for record in records]),
        "identity_confidence": _string_column(
            [record.identity_confidence for record in records]
        ),
        "token_count": _int32_column([record.token_count for record in records]),
        "char_count": _int32_column([record.char_count for record in records]),
        "sentence_count": _int32_column([record.sentence_count for record in records]),
        "encoding_damage": _bool_column([record.encoding_damage for record in records]),
        "raw_text_hash": _string_column([record.raw_text_hash for record in records]),
        "normalized_text_hash": _string_column(
            [record.normalized_text_hash for record in records]
        ),
    }
    if not no_text:
        columns["text"] = _string_column([record.text for record in records])
        columns["normalized_text"] = _string_column(
            [record.normalized_text for record in records]
        )
    columns.update(
        {
            "chunk_index": _int32_column([record.chunk_index for record in records]),
            "source": _string_column([record.source for record in records]),
            "created_at": _string_column([record.created_at for record in records]),
            "updated_at": _string_column([record.updated_at for record in records]),
            "metadata": _string_column(
                [canonical_json(record.metadata).decode("utf-8") for record in records]
            ),
            "embedding_ref": _string_column(
                [
                    None
                    if record.embedding.vector is None
                    else _embedding_filename(*_concrete_dtype_and_dimension(record))
                    for record in records
                ]
            ),
        }
    )
    for field_name in _DESCRIPTOR_FIELDS:
        if field_name == "normalized":
            columns["embedding_normalized_override"] = _bool_column(
                [
                    sparse_embedding_field(record, field_name, manifest_values[field_name])
                    for record in records
                ]
            )
        else:
            columns[f"embedding_{field_name}_override"] = _string_column(
                [
                    sparse_embedding_field(record, field_name, manifest_values[field_name])
                    for record in records
                ]
            )
    return pa.table(columns)


def _documents_table(documents: DocumentTable) -> pa.Table:
    names = [f.name for f in fields(DocumentRow)]
    rows = list(documents.rows.values())
    return pa.table(
        {name: [getattr(row, name) for row in rows] for name in names}
    )


def _write_embeddings(
    records: list[Record], embeddings_dir: Path
) -> list[str]:
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        if record.embedding.vector is None:
            continue
        dtype, dimension = _concrete_dtype_and_dimension(record)
        filename = _embedding_filename(dtype, dimension)
        if filename not in groups:
            groups[filename] = {"dtype": dtype, "dimension": dimension, "ids": [], "vectors": []}
            order.append(filename)
        groups[filename]["ids"].append(record.record_id)
        groups[filename]["vectors"].append(record.embedding.vector)
    for filename in order:
        group = groups[filename]
        arrow_type = pa.list_(_ARROW_DTYPES[group["dtype"]], group["dimension"])
        column = pa.array(
            [np.asarray(vector).tolist() for vector in group["vectors"]],
            type=arrow_type,
        )
        table = pa.table({"record_id": _string_column(group["ids"]), "vector": column})
        pq.write_table(table, embeddings_dir / filename)
    return order


def _is_snapshot_dir(path: Path) -> bool:
    return (path / _MANIFEST_FILE).is_file()


def write_snapshot(snapshot: Snapshot, out_dir: Path, *, no_text: bool) -> None:
    """Write a snapshot directory: manifest, records, documents, embeddings, stats."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        if out_dir.is_dir() and not any(out_dir.iterdir()):
            pass
        elif out_dir.is_dir() and _is_snapshot_dir(out_dir):
            shutil.rmtree(out_dir)
        else:
            raise InvalidInputError(
                location=str(out_dir),
                message="output directory is not empty and is not a snapshot directory",
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = out_dir / _EMBEDDINGS_DIR
    embeddings_dir.mkdir(exist_ok=True)

    records_table = _records_table(snapshot.records, no_text=no_text, manifest=snapshot.manifest)
    pq.write_table(records_table, out_dir / _RECORDS_FILE)
    pq.write_table(_documents_table(snapshot.documents), out_dir / _DOCUMENTS_FILE)
    embedding_files = _write_embeddings(snapshot.records, embeddings_dir)
    (out_dir / _STATISTICS_FILE).write_text(
        json.dumps(asdict(snapshot.statistics)), encoding="utf-8"
    )

    manifest_data = asdict(snapshot.manifest)
    manifest_data["files"] = {}
    (out_dir / _MANIFEST_FILE).write_text(
        json.dumps(manifest_data), encoding="utf-8"
    )

    checksums: dict[str, str] = {}
    checksums[_RECORDS_FILE] = sha256_hex((out_dir / _RECORDS_FILE).read_bytes())
    checksums[_DOCUMENTS_FILE] = sha256_hex((out_dir / _DOCUMENTS_FILE).read_bytes())
    for filename in embedding_files:
        relative = f"{_EMBEDDINGS_DIR}/{filename}"
        checksums[relative] = sha256_hex((out_dir / relative).read_bytes())
    checksums[_STATISTICS_FILE] = sha256_hex((out_dir / _STATISTICS_FILE).read_bytes())
    manifest_data["files"] = checksums
    (out_dir / _MANIFEST_FILE).write_text(
        json.dumps(manifest_data), encoding="utf-8"
    )


def _read_records(snapshot_dir: Path, manifest_data: dict[str, Any]) -> list[Record]:
    table = pq.read_table(snapshot_dir / _RECORDS_FILE)
    has_text = "text" in table.column_names
    manifest_values = {
        field_name: manifest_data.get(key)
        for field_name, key in _MANIFEST_DESCRIPTOR_KEYS.items()
    }
    cache: dict[str, dict[str, np.ndarray]] = {}
    records: list[Record] = []
    dimension = manifest_data.get("embedding_dimension")
    for row in table.to_pylist():
        embedding_ref = row.get("embedding_ref")
        vector = None
        if embedding_ref is not None:
            if embedding_ref not in cache:
                cache[embedding_ref] = _read_embedding_file(
                    snapshot_dir / _EMBEDDINGS_DIR / embedding_ref
                )
            vector = cache[embedding_ref].get(row["record_id"])
        resolved: dict[str, Any] = {}
        for field_name in _DESCRIPTOR_FIELDS:
            override = row.get(f"embedding_{field_name}_override")
            manifest_value = manifest_values[field_name]
            if override is not None:
                resolved[field_name] = override
            elif manifest_value == "MIXED":
                resolved[field_name] = None
            else:
                resolved[field_name] = manifest_value
        if dimension is None and vector is not None:
            dimension = int(vector.shape[-1])
        dtype = resolved["dtype"]
        if dtype is None and vector is not None:
            dtype = str(vector.dtype)
        embedding = EmbeddingInfo(
            vector=vector,
            dtype=dtype,
            dimension=dimension if vector is not None else None,
            norm=None,
            validity=None,
            model=resolved["model"],
            provider=resolved["provider"],
            similarity_metric=resolved["similarity_metric"],
            normalized=resolved["normalized"],
        )
        record = Record(
            record_id=row["record_id"],
            document_id=row.get("document_id"),
            identity_source=row.get("identity_source"),
            identity_confidence=row.get("identity_confidence"),
            text=row.get("text") if has_text else None,
            normalized_text=row.get("normalized_text") if has_text else None,
            token_count=row.get("token_count"),
            char_count=row.get("char_count"),
            sentence_count=row.get("sentence_count"),
            encoding_damage=row.get("encoding_damage"),
            raw_text_hash=row.get("raw_text_hash"),
            normalized_text_hash=row.get("normalized_text_hash"),
            embedding=embedding,
            metadata=json.loads(row["metadata"]),
            chunk_index=row.get("chunk_index"),
            source=row.get("source"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
        compute_embedding_digest_field(record)
        records.append(record)
    return records


def _read_embedding_file(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    numpy_dtype = _numpy_dtype_for_arrow(table.schema.field("vector").type.value_type)
    result: dict[str, np.ndarray] = {}
    for row in table.to_pylist():
        result[row["record_id"]] = np.asarray(row["vector"], dtype=numpy_dtype)
    return result


def _read_documents(snapshot_dir: Path) -> DocumentTable:
    table = pq.read_table(snapshot_dir / _DOCUMENTS_FILE)
    rows: dict[str, DocumentRow] = {}
    for row in table.to_pylist():
        document = DocumentRow(**row)
        rows[document.document_id] = document
    return DocumentTable(rows=rows)


def read_snapshot(snapshot_dir: Path, *, limits: RuntimeLimits, verify_full: bool) -> Snapshot:
    """Read a snapshot directory, verifying checksums before any parsing."""
    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / _MANIFEST_FILE
    manifest_data = loads_limited(
        manifest_path.read_text(encoding="utf-8"), location=str(manifest_path)
    )

    schema_version = str(manifest_data.get("schema_version", ""))
    if schema_version.split(".")[0] != SNAPSHOT_SCHEMA_VERSION.split(".")[0]:
        raise UnsupportedFormatError(
            message=f"snapshot schema {schema_version} is not supported by this version"
        )

    for name, expected in manifest_data.get("files", {}).items():
        path = snapshot_dir / name
        limits.add_input_bytes(path.stat().st_size, location=str(path))
        if sha256_hex(path.read_bytes()) != expected:
            raise InvalidInputError(location=str(path), message="checksum mismatch")

    records = _read_records(snapshot_dir, manifest_data)
    documents = _read_documents(snapshot_dir)
    statistics_path = snapshot_dir / _STATISTICS_FILE
    statistics = StatisticsAccumulator(
        **json.loads(statistics_path.read_text(encoding="utf-8"))
    )
    manifest = Manifest(**manifest_data)

    if verify_full:
        descriptor_bytes = build_descriptor(
            manifest.schema_version,
            manifest.normalization_version,
            manifest.tokenizer_id,
            manifest.granularity,
            EmbeddingDescriptor(
                model=manifest.embedding_model,
                provider=manifest.embedding_provider,
                dtype=manifest.embedding_dtype,
                dimension=manifest.embedding_dimension,
                similarity_metric=manifest.similarity_metric,
                normalized=manifest.embedding_normalized,
            ),
        )
        recomputed, _content = compute_snapshot_id(records, descriptor_bytes)
        if recomputed != manifest.snapshot_id:
            raise InvalidInputError(
                location=str(manifest_path), message="snapshot id mismatch"
            )

    return Snapshot(
        manifest=manifest,
        records=records,
        documents=documents,
        statistics=statistics,
    )

