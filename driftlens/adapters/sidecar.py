"""Sidecar table adapter for FAISS inputs (Step 17)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from driftlens.adapters.detect import FieldMapping, resolve_fields
from driftlens.adapters.jsonl import build_record
from driftlens.adapters.parquet import (
    PARQUET_EMBEDDING_ARROW_TYPES,
    _build_record as _build_parquet_record,
    _is_embedding_list,
)
from driftlens.config.schema import SideOptions
from driftlens.errors import InvalidInputError
from driftlens.schema.record import Record
from driftlens.security.limits import RuntimeLimits, loads_limited


@dataclass
class SidecarRow:
    """One sidecar row, keyed by its FAISS id, with its canonical record."""

    faiss_id: int
    record: Record


class SidecarTable(dict):
    """Sidecar rows keyed by faiss_id, carrying the resolved field mapping."""

    def __init__(self) -> None:
        super().__init__()
        self.field_mapping = FieldMapping()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_numeric_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    )


def _carry_model(
    record: Record, mapping: FieldMapping | None, obj: dict[str, Any]
) -> None:
    if mapping is None:
        return
    if mapping.model:
        value = obj.get(mapping.model)
        if value is not None:
            record.embedding.model = value if isinstance(value, str) else str(value)
    if mapping.provider:
        value = obj.get(mapping.provider)
        if value is not None:
            record.embedding.provider = value if isinstance(value, str) else str(value)


def _observe(observed: dict[str, list[Any]], obj: dict[str, Any]) -> None:
    for key, value in obj.items():
        if key == "faiss_id" or value is None:
            continue
        observed.setdefault(key, []).append(value)


def _validate(
    path: Path, ids: list[int], missing: int, observed: dict[str, list[Any]]
) -> None:
    if missing:
        raise InvalidInputError(
            message=f"{missing} missing faiss_id values", location=str(path)
        )
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise InvalidInputError(
            message=f"{duplicates} duplicate faiss_id values", location=str(path)
        )
    for column, values in observed.items():
        if values and all(_is_numeric_list(value) for value in values):
            raise InvalidInputError(
                message=f"sidecar column {column!r} holds embeddings",
                location=str(path),
            )


def _load_jsonl(path: Path, *, side: SideOptions, limits: RuntimeLimits) -> SidecarTable:
    table = SidecarTable()
    mapping: FieldMapping | None = None
    resolved = False
    ids: list[int] = []
    missing = 0
    observed: dict[str, list[Any]] = {}

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
                candidates = [key for key in obj if key != "faiss_id"]
                mapping = resolve_fields(
                    candidates,
                    granularity_hint="AUTO",
                    overrides=side,
                    location=str(path),
                )
                resolved = True
            faiss_id = obj.get("faiss_id")
            if faiss_id is None:
                missing += 1
            elif not _is_int(faiss_id):
                raise InvalidInputError(
                    message="faiss_id must be an integer", location=location
                )
            else:
                ids.append(faiss_id)
            # Join key, not metadata (see _load_parquet): consume before build.
            obj.pop("faiss_id", None)
            record = build_record(
                obj,
                mapping,
                file_descriptors={},
                row_number=row_number,
                location=location,
                timestamp_unit=side.timestamp_unit,
            )
            _carry_model(record, mapping, obj)
            _observe(observed, obj)
            if faiss_id is not None:
                table[faiss_id] = SidecarRow(faiss_id=faiss_id, record=record)

    table.field_mapping = mapping if mapping is not None else FieldMapping()
    _validate(path, ids, missing, observed)
    return table


def _load_parquet(
    path: Path, *, side: SideOptions, limits: RuntimeLimits
) -> SidecarTable:
    parquet_file = pq.ParquetFile(path)
    limits.add_input_bytes(path.stat().st_size, location=str(path))
    schema = parquet_file.schema_arrow
    if "faiss_id" not in schema.names:
        raise InvalidInputError(
            message="sidecar is missing the faiss_id column", location=str(path)
        )
    if not pa.types.is_int64(schema.field("faiss_id").type):
        raise InvalidInputError(
            message="faiss_id must be an int64 column", location=str(path)
        )

    forced_dtypes: dict[str, str] = {}
    available_keys: list[str] = []
    for name in schema.names:
        if name == "faiss_id":
            continue
        arrow_type = schema.field(name).type
        if _is_embedding_list(arrow_type):
            dtype_name = PARQUET_EMBEDDING_ARROW_TYPES.get(arrow_type.value_type)
            if dtype_name is not None:
                forced_dtypes[name] = dtype_name
        available_keys.append(name)

    mapping = resolve_fields(
        available_keys,
        granularity_hint="AUTO",
        overrides=side,
        location=str(path),
    )
    table = SidecarTable()
    table.field_mapping = mapping
    ids: list[int] = []
    missing = 0
    observed: dict[str, list[Any]] = {}
    row_number = 0

    for batch in parquet_file.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            row_number += 1
            location = f"{path} row {row_number}"
            faiss_id = row.get("faiss_id")
            if faiss_id is None:
                missing += 1
            elif not _is_int(faiss_id):
                raise InvalidInputError(
                    message="faiss_id must be an integer", location=location
                )
            else:
                ids.append(faiss_id)
            # The join key is positional bookkeeping, not user metadata:
            # consuming it here keeps faiss_id out of record.metadata so a
            # re-keyed successor is not misreported as a metadata change.
            row.pop("faiss_id", None)
            record = _build_parquet_record(
                row,
                mapping,
                forced_dtypes=forced_dtypes,
                row_number=row_number,
                location=location,
            )
            _carry_model(record, mapping, row)
            _observe(observed, row)
            if faiss_id is not None:
                table[faiss_id] = SidecarRow(faiss_id=faiss_id, record=record)

    _validate(path, ids, missing, observed)
    return table


def load_sidecar(
    path: Path, *, side: SideOptions, limits: RuntimeLimits
) -> SidecarTable:
    """Load a JSONL or Parquet sidecar table keyed by faiss_id."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return _load_jsonl(path, side=side, limits=limits)
    if suffix == ".parquet":
        return _load_parquet(path, side=side, limits=limits)
    raise InvalidInputError(
        message=f"sidecar must be JSONL or Parquet, got {path.suffix!r}",
        location=str(path),
    )
