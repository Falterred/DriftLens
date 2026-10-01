"""Parquet input adapter (Step 16)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from driftlens.adapters._embedding_codec import NUMPY_DTYPE
from driftlens.adapters.detect import FieldMapping, resolve_fields
from driftlens.adapters.jsonl import LoadedInput, row_is_empty
from driftlens.config.schema import SideOptions
from driftlens.errors import InvalidInputError
from driftlens.schema.record import (
    EmbeddingInfo,
    MetadataScalar,
    Record,
    canonicalize_timestamp,
)
from driftlens.security.limits import RuntimeLimits
from driftlens.text.normalize import check_utf8_clean, normalize_text
from driftlens.text.sentences import sentence_count
from driftlens.text.tokenizer import token_count

PARQUET_EMBEDDING_ARROW_TYPES = {
    pa.float16(): "float16",
    pa.float32(): "float32",
    pa.float64(): "float64",
}
DESCRIPTOR_KEYS = (
    "embedding_model",
    "embedding_provider",
    "embedding_dtype",
    "similarity_metric",
)
TIMESTAMP_ROLES = ("created_at", "updated_at", "modified_at", "timestamp")
EMPTY_EMBEDDING = EmbeddingInfo(None, None, None, None, None, None, None, None, None)


def arrow_value_to_metadata(
    value: Any, *, location: str = "parquet metadata"
) -> MetadataScalar:
    """Coerce one PyArrow-derived Python value into a metadata scalar."""

    if isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return canonicalize_timestamp(
            value.isoformat(), unit="iso", location=location
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise InvalidInputError(
            message="binary values have no metadata representation",
            location=location,
        )
    if isinstance(value, (list, dict)):
        return value
    return value


def _encoding_damage(text: str) -> bool:
    if "\ufffd" in text:
        return True
    return any(ord(ch) < 0x20 and ch not in "\t\n" for ch in text)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _is_embedding_list(arrow_type: Any) -> bool:
    return (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    )


def _build_record(
    row: dict[str, Any],
    mapping: FieldMapping,
    *,
    forced_dtypes: dict[str, str],
    row_number: int,
    location: str,
) -> Record:
    """Build one canonical Record from a Parquet row dict."""

    text: str | None = None
    normalized: str | None = None
    tokens: int | None = None
    chars: int | None = None
    sentences: int | None = None
    damage: bool | None = None

    if mapping.text:
        value = row.get(mapping.text)
        if value is not None:
            if not isinstance(value, str):
                raise InvalidInputError(
                    message="text field is not a string", location=location
                )
            text = check_utf8_clean(value, location=location)
            normalized = normalize_text(text)
            tokens = token_count(text)
            chars = len(text)
            sentences = sentence_count(text)
            damage = _encoding_damage(text)

    embedding = EMPTY_EMBEDDING
    if mapping.embedding:
        value = row.get(mapping.embedding)
        if value is not None:
            dtype_used = forced_dtypes.get(mapping.embedding)
            if dtype_used is None:
                raise InvalidInputError(
                    message="embedding column must be a list of floats",
                    location=location,
                )
            vector = np.asarray(value, dtype=NUMPY_DTYPE[dtype_used])
            embedding = EmbeddingInfo(
                vector=vector,
                dtype=dtype_used,
                dimension=int(vector.shape[0]),
                norm=float(np.linalg.norm(np.asarray(vector, dtype=np.float64))),
                validity=None,
                model=None,
                provider=None,
                similarity_metric=None,
                normalized=None,
            )

    consumed = mapping.consumed_keys
    metadata: dict[str, MetadataScalar] = {}
    for name, value in row.items():
        if name in consumed:
            continue
        if value is None:
            metadata[name] = None
        else:
            metadata[name] = arrow_value_to_metadata(value, location=location)

    chunk_index: int | None = None
    if mapping.chunk_index:
        value = row.get(mapping.chunk_index)
        if value is not None:
            chunk_index = int(value)

    created_at: str | None = None
    updated_at: str | None = None
    for role in TIMESTAMP_ROLES:
        key = mapping.timestamp_fields.get(role)
        if key is None:
            continue
        value = row.get(key)
        if value is None:
            continue
        rendered = arrow_value_to_metadata(value, location=location)
        if role == "created_at" and created_at is None:
            created_at = rendered if isinstance(rendered, str) else str(rendered)
        elif role in ("updated_at", "modified_at") and updated_at is None:
            updated_at = rendered if isinstance(rendered, str) else str(rendered)
        elif role == "timestamp" and created_at is None:
            created_at = rendered if isinstance(rendered, str) else str(rendered)

    return Record(
        record_id=_optional_text(row.get(mapping.chunk_id))
        if mapping.chunk_id
        else None,
        document_id=_optional_text(row.get(mapping.document_id))
        if mapping.document_id
        else None,
        text=text,
        normalized_text=normalized,
        token_count=tokens,
        char_count=chars,
        sentence_count=sentences,
        encoding_damage=damage,
        embedding=embedding,
        metadata=metadata,
        chunk_index=chunk_index,
        created_at=created_at,
        updated_at=updated_at,
        row_number=row_number,
    )


def _file_descriptors(schema: pa.Schema) -> dict[str, str]:
    descriptors: dict[str, str] = {}
    raw_metadata = schema.metadata or {}
    for key, value in raw_metadata.items():
        try:
            name = key.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if name == "pandas":
            continue
        if name not in DESCRIPTOR_KEYS:
            continue
        try:
            descriptors[name] = value.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return descriptors


def load_parquet(
    path: Path, *, side: SideOptions, limits: RuntimeLimits
) -> LoadedInput:
    """Load a Parquet file into canonical records."""

    path = Path(path)
    parquet_file = pq.ParquetFile(path)
    limits.add_input_bytes(path.stat().st_size, location=str(path))

    schema = parquet_file.schema_arrow
    forced_dtypes: dict[str, str] = {}
    available_keys: list[str] = []
    for name in schema.names:
        arrow_type = schema.field(name).type
        if _is_embedding_list(arrow_type):
            dtype_name = PARQUET_EMBEDDING_ARROW_TYPES.get(arrow_type.value_type)
            if dtype_name is not None:
                forced_dtypes[name] = dtype_name
                available_keys.append(name)
            continue
        available_keys.append(name)

    mapping = resolve_fields(
        available_keys,
        granularity_hint="AUTO",
        overrides=side,
        location=str(path),
    )

    if mapping.chunk_index:
        chunk_type = schema.field(mapping.chunk_index).type
        if not pa.types.is_integer(chunk_type):
            raise InvalidInputError(
                message="chunk_index must be an integer-typed column",
                location=str(path),
            )

    for name in schema.names:
        arrow_type = schema.field(name).type
        if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            if name not in mapping.consumed_keys:
                raise InvalidInputError(
                    message=f"binary column {name} has no metadata representation",
                    location=str(path),
                )

    file_descriptors = _file_descriptors(schema)

    records: list[Record] = []
    row_number = 0
    for batch in parquet_file.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            row_number += 1
            location = f"{path} row {row_number}"
            record = _build_record(
                row,
                mapping,
                forced_dtypes=forced_dtypes,
                row_number=row_number,
                location=location,
            )
            if row_is_empty(row, record, mapping):
                raise InvalidInputError(
                    message="row has no id, text or embedding", location=location
                )
            records.append(record)

    return LoadedInput(
        records=records,
        field_mapping=mapping,
        file_descriptors=file_descriptors,
        source_type="parquet",
    )
