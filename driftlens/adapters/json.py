"""Whole-file JSON input adapter (Step 14)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from driftlens.adapters.detect import FieldMapping, resolve_fields
from driftlens.adapters.jsonl import LoadedInput, build_record, row_is_empty
from driftlens.config.schema import SideOptions
from driftlens.errors import InvalidInputError
from driftlens.security.limits import RuntimeLimits, loads_limited

_DESCRIPTOR_KEYS = (
    "embedding_model",
    "embedding_provider",
    "embedding_dtype",
    "similarity_metric",
)


def load_json(path: Path, *, side: SideOptions, limits: RuntimeLimits) -> LoadedInput:
    """Load a whole-file JSON array or records-wrapper object."""

    path = Path(path)
    limits.add_input_bytes(path.stat().st_size, location=str(path))
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise InvalidInputError(message="invalid UTF-8", location=str(path)) from None

    top = loads_limited(text, location=str(path))
    file_descriptors: dict[str, str] = {}

    if isinstance(top, list):
        rows: list[Any] = top
    elif isinstance(top, dict):
        rows = []
        found = False
        for key in ("records", "data", "items"):
            value = top.get(key)
            if isinstance(value, list):
                rows = value
                found = True
                break
        if not found:
            raise InvalidInputError(
                message="JSON object has no records array", location=str(path)
            )
        file_descriptors = {k: top[k] for k in _DESCRIPTOR_KEYS if k in top}
    else:
        raise InvalidInputError(
            message="JSON input is not a records array", location=str(path)
        )

    records = []
    mapping: FieldMapping | None = None
    resolved = False
    for position, obj in enumerate(rows, start=1):
        location = f"{path} row {position}"
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
            file_descriptors=file_descriptors,
            row_number=position,
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
        file_descriptors=file_descriptors,
        source_type="json",
    )
