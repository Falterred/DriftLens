"""CSV / TSV input adapter (Step 15)."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from driftlens.adapters._embedding_codec import decode_embedding_array
from driftlens.adapters.detect import FieldMapping, resolve_fields
from driftlens.adapters.jsonl import LoadedInput, row_is_empty
from driftlens.config.schema import SideOptions
from driftlens.errors import InvalidInputError
from driftlens.schema.record import (
    EmbeddingInfo,
    Record,
    canonicalize_timestamp,
    type_csv_cell,
)
from driftlens.security.limits import RuntimeLimits, loads_limited
from driftlens.text.normalize import check_utf8_clean, normalize_text
from driftlens.text.sentences import sentence_count
from driftlens.text.tokenizer import token_count

TIMESTAMP_ROLES = ("created_at", "updated_at", "modified_at", "timestamp")
EMPTY_EMBEDDING = EmbeddingInfo(None, None, None, None, None, None, None, None, None)


def _encoding_damage(text: str) -> bool:
    if "\ufffd" in text:
        return True
    return any(ord(ch) < 0x20 and ch not in "\t\n" for ch in text)


def _numeric_tokens(cell: str, *, location: str) -> list[float]:
    numbers: list[float] = []
    for index, token in enumerate(cell.split(",")):
        try:
            numbers.append(float(token.strip()))
        except ValueError:
            raise InvalidInputError(
                message=f"embedding element {index} is not numeric", location=location
            ) from None
    return numbers


def parse_embedding_cell_json(
    cell: str, *, declared_dtype: str | None, location: str
) -> tuple[np.ndarray, str]:
    """Parse a cell holding a JSON array literal of numbers."""

    parsed = loads_limited(cell, location=location)
    return decode_embedding_array(
        parsed, declared_dtype=declared_dtype, location=location
    )


def parse_embedding_cell_delimited(
    cell: str, *, declared_dtype: str | None, location: str
) -> tuple[np.ndarray, str]:
    """Parse a cell holding comma-separated numbers (independent of the CSV dialect)."""

    numbers = _numeric_tokens(cell, location=location)
    return decode_embedding_array(
        numbers, declared_dtype=declared_dtype, location=location
    )


def resolve_wide_embedding_columns(header: list[str], prefix: str) -> list[str] | None:
    """Return the widest contiguous zero-based run of prefixed columns."""

    if prefix is None or f"{prefix}0" not in header:
        return None
    columns: list[str] = []
    index = 0
    while f"{prefix}{index}" in header:
        columns.append(f"{prefix}{index}")
        index += 1
    return columns


def _embedding_info(
    vector: np.ndarray, dtype_used: str
) -> EmbeddingInfo:
    return EmbeddingInfo(
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


def _declared_dtype(raw_obj: dict[str, str], mapping: FieldMapping) -> str | None:
    if not mapping.dtype:
        return None
    cell = raw_obj.get(mapping.dtype)
    if cell is None or cell == "":
        return None
    return cell


def _build_record(
    raw_obj: dict[str, str],
    mapping: FieldMapping,
    *,
    wide_columns: list[str],
    row_number: int,
    location: str,
    timestamp_unit: str,
    embedding_format: str,
) -> Record:
    """Build one canonical Record from a CSV row of raw string cells."""

    text: str | None = None
    normalized: str | None = None
    tokens: int | None = None
    chars: int | None = None
    sentences: int | None = None
    damage: bool | None = None

    if mapping.text:
        cell = raw_obj.get(mapping.text)
        if cell is not None and cell != "":
            text = check_utf8_clean(cell, location=location)
            normalized = normalize_text(text)
            tokens = token_count(text)
            chars = len(text)
            sentences = sentence_count(text)
            damage = _encoding_damage(text)

    consumed = mapping.consumed_keys
    copied = mapping.copied_keys
    wide_set = set(wide_columns)
    metadata: dict[str, Any] = {}
    for key, cell in raw_obj.items():
        if key in consumed or key in copied or key in wide_set:
            continue
        _, typed = type_csv_cell(cell)
        metadata[key] = typed

    declared_dtype = _declared_dtype(raw_obj, mapping)

    embedding = EMPTY_EMBEDDING
    if wide_columns:
        numbers: list[float] = []
        for index, column in enumerate(wide_columns):
            try:
                numbers.append(float(raw_obj.get(column, "")))
            except ValueError:
                raise InvalidInputError(
                    message=f"embedding element {index} is not numeric",
                    location=location,
                ) from None
        vector, dtype_used = decode_embedding_array(
            numbers, declared_dtype=declared_dtype, location=location
        )
        embedding = _embedding_info(vector, dtype_used)
    elif mapping.embedding:
        cell = raw_obj.get(mapping.embedding)
        if cell is not None and cell != "":
            if embedding_format == "delimited":
                vector, dtype_used = parse_embedding_cell_delimited(
                    cell, declared_dtype=declared_dtype, location=location
                )
            else:
                vector, dtype_used = parse_embedding_cell_json(
                    cell, declared_dtype=declared_dtype, location=location
                )
            embedding = _embedding_info(vector, dtype_used)

    chunk_index: int | None = None
    if mapping.chunk_index:
        cell = raw_obj.get(mapping.chunk_index)
        if cell is not None and cell != "":
            type_name, typed = type_csv_cell(cell)
            if type_name != "int":
                raise InvalidInputError(
                    message="chunk_index must be an integer", location=location
                )
            chunk_index = int(typed)

    created_at: str | None = None
    updated_at: str | None = None
    for role in TIMESTAMP_ROLES:
        key = mapping.timestamp_fields.get(role)
        if key is None:
            continue
        cell = raw_obj.get(key)
        if cell is None or cell == "":
            continue
        field_location = f"{location} field {key}"
        if timestamp_unit == "iso":
            value: Any = cell
        else:
            try:
                value = float(cell)
            except ValueError:
                raise InvalidInputError(
                    message="timestamp value is not numeric", location=field_location
                ) from None
        rendered = canonicalize_timestamp(
            value, unit=timestamp_unit, location=field_location
        )
        if role == "created_at" and created_at is None:
            created_at = rendered
        elif role in ("updated_at", "modified_at") and updated_at is None:
            updated_at = rendered
        elif role == "timestamp" and created_at is None:
            created_at = rendered

    record_id = raw_obj.get(mapping.chunk_id) if mapping.chunk_id else None
    document_id = raw_obj.get(mapping.document_id) if mapping.document_id else None

    return Record(
        record_id=record_id if record_id else None,
        document_id=document_id if document_id else None,
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


def load_csv(
    path: Path, *, side: SideOptions, limits: RuntimeLimits
) -> LoadedInput:
    """Load a CSV or TSV file into canonical records."""

    path = Path(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    limits.add_input_bytes(os.stat(path).st_size, location=str(path))

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(
            handle, delimiter=delimiter, quotechar='"', doublequote=True
        )
        header = next(reader, None)
        if header is None:
            raise InvalidInputError(
                message="CSV file has no header row", location=str(path)
            )

        prefix = getattr(side, "embedding_prefix", None)
        wide_columns = (
            resolve_wide_embedding_columns(header, prefix) if prefix else None
        )
        if prefix and wide_columns is None:
            raise InvalidInputError(
                message=f"no column {prefix}0 found", location=str(path)
            )
        wide_set = set(wide_columns) if wide_columns else set()
        candidate_keys = [key for key in header if key not in wide_set]
        mapping = resolve_fields(
            candidate_keys,
            granularity_hint="AUTO",
            overrides=side,
            location=str(path),
        )

        embedding_format = getattr(side, "embedding_format", None) or "json"
        if embedding_format not in ("json", "delimited"):
            raise InvalidInputError(
                message=f"unknown embedding format {embedding_format!r}",
                location=str(path),
            )

        records: list[Record] = []
        for row_number, cells in enumerate(reader, start=1):
            location = f"{path} row {row_number}"
            if len(cells) != len(header):
                raise InvalidInputError(
                    message=(
                        f"expected {len(header)} columns, found {len(cells)}"
                    ),
                    location=location,
                )
            raw_obj = {
                header[index]: cells[index] for index in range(len(header))
            }
            record = _build_record(
                raw_obj,
                mapping,
                wide_columns=wide_columns or [],
                row_number=row_number,
                location=location,
                timestamp_unit=side.timestamp_unit,
                embedding_format=embedding_format,
            )
            if row_is_empty(raw_obj, record, mapping):
                raise InvalidInputError(
                    message="row has no id, text or embedding", location=location
                )
            records.append(record)

    return LoadedInput(
        records=records,
        field_mapping=mapping,
        file_descriptors={},
        source_type="csv",
    )
