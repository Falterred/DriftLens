"""Snapshot assembly from identity-layer output (Step 20)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import driftlens
from driftlens.config.schema import Config, SideOptions
from driftlens.errors import DiagnosticSink, InvalidInputError
from driftlens.identity.chunks import all_chunk_identity_keys
from driftlens.identity.documents import (
    Granularity,
    GranularitySource,
    assign_identity,
    disambiguate_duplicate_ids,
)
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
from driftlens.schema.record import Record
from driftlens.security.limits import RuntimeLimits
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
