"""JSONL input adapter (Step 14)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from driftlens.adapters._embedding_codec import (
    decode_embedding_array,
    decode_embedding_b64,
)
from driftlens.adapters.detect import FieldMapping, resolve_fields
from driftlens.config.schema import SideOptions
from driftlens.errors import InvalidInputError
from driftlens.schema.record import EmbeddingInfo, Record, canonicalize_timestamp
from driftlens.security.limits import RuntimeLimits, loads_limited
from driftlens.text.normalize import check_utf8_clean, normalize_text
from driftlens.text.sentences import sentence_count
from driftlens.text.tokenizer import token_count

SourceType = Literal["jsonl", "json", "csv", "parquet", "chroma", "faiss"]


@dataclass
class LoadedInput:
    """Adapter output: canonical records plus the resolved schema of the input."""

    records: list[Record]
    field_mapping: FieldMapping
    file_descriptors: dict[str, str]
    source_type: SourceType


def _opt_text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _encoding_damage(text: str) -> bool:
    if "\ufffd" in text:
        return True
    return any(ord(ch) < 0x20 and ch not in "\t\n" for ch in text)


def _timestamp_value(
    obj: dict[str, Any],
    mapping: FieldMapping,
    role: str,
    *,
    unit: str,
    location: str,
) -> str | None:
    key = mapping.timestamp_fields.get(role)
    if key is None:
        return None
    value = obj.get(key)
    if value is None:
        return None
    return canonicalize_timestamp(value, unit=unit, location=location)


def build_record(
    obj: dict[str, Any],
    mapping: FieldMapping | None,
    *,
    file_descriptors: dict[str, str],
    row_number: int,
    location: str,
    timestamp_unit: str,
) -> Record:
    """Build one canonical Record from a parsed JSON object row."""

    text: str | None = None
    normalized: str | None = None
    tokens: int | None = None
    chars: int | None = None
    sentences: int | None = None
    damage: bool | None = None

    raw_text = obj.get(mapping.text) if mapping is not None and mapping.text else None
    if raw_text is not None:
        if not isinstance(raw_text, str):
            raise InvalidInputError(
                message="text field is not a string", location=location
            )
        text = check_utf8_clean(raw_text, location=location)
        normalized = normalize_text(text)
        tokens = token_count(text)
        chars = len(text)
        sentences = sentence_count(text)
        damage = _encoding_damage(text)

    embedding = EmbeddingInfo(None, None, None, None, None, None, None, None, None)
    raw_embedding = (
        obj.get(mapping.embedding)
        if mapping is not None and mapping.embedding
        else None
    )
    if raw_embedding is not None:
        row_dtype = obj.get(mapping.dtype) if mapping.dtype else None
        declared_dtype = _opt_text(row_dtype)
        if declared_dtype is None:
            declared_dtype = file_descriptors.get("embedding_dtype")
        if mapping.embedding_is_b64:
            vector, dtype_used = decode_embedding_b64(
                raw_embedding, declared_dtype=declared_dtype, location=location
            )
        else:
            vector, dtype_used = decode_embedding_array(
                raw_embedding, declared_dtype=declared_dtype, location=location
            )
        model = _opt_text(obj.get(mapping.model) if mapping.model else None)
        provider = _opt_text(obj.get(mapping.provider) if mapping.provider else None)
        metric = _opt_text(obj.get(mapping.metric) if mapping.metric else None)
        if model is None:
            model = file_descriptors.get("embedding_model")
        if provider is None:
            provider = file_descriptors.get("embedding_provider")
        if metric is None:
            metric = file_descriptors.get("similarity_metric")
        embedding = EmbeddingInfo(
            vector=vector,
            dtype=dtype_used,
            dimension=int(vector.shape[0]),
            norm=float(np.linalg.norm(np.asarray(vector, dtype=np.float64))),
            validity=None,
            model=model,
            provider=provider,
            similarity_metric=metric,
            normalized=None,
        )

    consumed = mapping.consumed_keys if mapping is not None else frozenset()
    metadata = {key: value for key, value in obj.items() if key not in consumed}

    raw_record_id = (
        obj.get(mapping.chunk_id) if mapping is not None and mapping.chunk_id else None
    )
    raw_document_id = (
        obj.get(mapping.document_id)
        if mapping is not None and mapping.document_id
        else None
    )
    raw_chunk_index = (
        obj.get(mapping.chunk_index)
        if mapping is not None and mapping.chunk_index
        else None
    )
    chunk_index = None
    if raw_chunk_index is not None:
        if isinstance(raw_chunk_index, bool) or not isinstance(raw_chunk_index, int):
            raise InvalidInputError(
                message="chunk_index must be an integer", location=location
            )
        chunk_index = int(raw_chunk_index)

    created_at = None
    updated_at = None
    if mapping is not None:
        created_at = _timestamp_value(
            obj, mapping, "created_at", unit=timestamp_unit, location=location
        )
        updated_at = _timestamp_value(
            obj, mapping, "updated_at", unit=timestamp_unit, location=location
        )
        if updated_at is None:
            updated_at = _timestamp_value(
                obj, mapping, "modified_at", unit=timestamp_unit, location=location
            )
        if created_at is None:
            created_at = _timestamp_value(
                obj, mapping, "timestamp", unit=timestamp_unit, location=location
            )

    return Record(
        record_id=_opt_text(raw_record_id),
        document_id=_opt_text(raw_document_id),
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


def row_is_empty(
    obj: dict[str, Any],
    record: Record,
    mapping: FieldMapping | None,
) -> bool:
    """True when a row carries no id value, no text and no embedding."""

    if record.text is not None or record.embedding.vector is not None:
        return False
    chunk_key = mapping.chunk_id if mapping is not None else None
    document_key = mapping.document_id if mapping is not None else None
    if chunk_key is not None and obj.get(chunk_key) is not None:
        return False
    if document_key is not None and obj.get(document_key) is not None:
        return False
    return True


def load_jsonl(path: Path, *, side: SideOptions, limits: RuntimeLimits) -> LoadedInput:
    """Load a JSON Lines file into canonical records."""

    path = Path(path)
    records: list[Record] = []
    mapping: FieldMapping | None = None
    resolved = False

    with path.open("rb") as handle:
        for row_number, line_bytes in enumerate(handle, start=1):
            if not line_bytes.strip():
                continue
            location = f"{path} line {row_number}"
            limits.add_input_bytes(len(line_bytes), location=location)
            try:
                line = line_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raise InvalidInputError(
                    message="invalid UTF-8", location=location
                ) from None
            obj = loads_limited(line, location=location)
            if not isinstance(obj, dict):
                raise InvalidInputError(
                    message="row is not a JSON object", location=location
                )
            if not resolved and obj:
                mapping = resolve_fields(
                    obj.keys(),
                    granularity_hint="AUTO",
                    overrides=side,
                    location=str(path),
                )
                resolved = True
            record = build_record(
                obj,
                mapping,
                file_descriptors={},
                row_number=row_number,
                location=location,
                timestamp_unit=side.timestamp_unit,
            )
            if row_is_empty(obj, record, mapping):
                raise InvalidInputError(
                    message="row has no id, text or embedding", location=location
                )
            records.append(record)

    if mapping is None:
        mapping = FieldMapping()
    return LoadedInput(
        records=records,
        field_mapping=mapping,
        file_descriptors={},
        source_type="jsonl",
    )
