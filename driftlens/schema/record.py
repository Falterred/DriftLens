"""Canonical record shape and value-typing helpers (Step 12)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Literal

import numpy as np

from driftlens.errors import InvalidInputError
from driftlens.security.limits import loads_limited

IdentityConfidence = Literal["HIGH", "MEDIUM", "LOW"]
EmbeddingValidity = Literal["ok", "nan_inf", "zero"]
MetadataScalar = None | bool | int | float | str | list | dict
EmbeddingDtype = Literal["float16", "float32", "float64", "int8"]
TimeUnit = Literal["iso", "seconds", "milliseconds"]


@dataclass
class EmbeddingInfo:
    """Embedding payload plus the declared-or-inferred attributes of its space."""

    vector: np.ndarray | None
    dtype: EmbeddingDtype | None
    dimension: int | None
    norm: float | None
    validity: EmbeddingValidity | None
    model: str | None
    provider: str | None
    similarity_metric: Literal["cosine", "dot", "l2"] | None
    normalized: bool | None


@dataclass
class Record:
    """Canonical record table row shared by every adapter and identity stage."""

    record_id: str
    document_id: str | None = None
    identity_source: str | None = None
    identity_confidence: IdentityConfidence | None = None
    text: str | None = None
    normalized_text: str | None = None
    token_count: int | None = None
    char_count: int | None = None
    sentence_count: int | None = None
    encoding_damage: bool | None = None
    raw_text_hash: str | None = None
    normalized_text_hash: str | None = None
    embedding: EmbeddingInfo = field(
        default_factory=lambda: EmbeddingInfo(
            None, None, None, None, None, None, None, None, None
        )
    )
    metadata: dict[str, MetadataScalar] = field(default_factory=dict)
    chunk_index: int | None = None
    source: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    row_number: int = 0
    identity_source_hint: str | None = None


_INT_RE = re.compile(r"^-?(0|[1-9]\d*)$")
_FLOAT_RE = re.compile(
    r"^-?(?:(?:0|[1-9]\d*)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
)


def is_iso8601_datetime_string(s: str) -> bool:
    """True for a complete RFC 3339 date or date-time string."""

    if _DATE_RE.match(s):
        try:
            date.fromisoformat(s)
        except ValueError:
            return False
        return True
    if _DATETIME_RE.match(s):
        text = s[:-1] + "+00:00" if s.endswith("Z") else s
        try:
            datetime.fromisoformat(text)
        except ValueError:
            return False
        return True
    return False


def type_csv_cell(raw: str) -> tuple[str, MetadataScalar]:
    """Type a raw CSV cell in the fixed §2.7 priority order."""

    if raw == "":
        return ("null", None)
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return ("bool", lowered == "true")
    if _INT_RE.match(raw):
        return ("int", int(raw))
    if _FLOAT_RE.match(raw):
        parsed = float(raw)
        if math.isfinite(parsed):
            return ("float", parsed)
    if is_iso8601_datetime_string(raw):
        return ("datetime", raw)
    try:
        decoded = loads_limited(raw, location="csv cell")
    except InvalidInputError:
        decoded = _UNSET
    if isinstance(decoded, list):
        return ("list", decoded)
    if isinstance(decoded, dict):
        return ("json", decoded)
    return ("string", raw)


class _Unset:
    __slots__ = ()


_UNSET = _Unset()


def canonicalize_timestamp(value: str | int | float, *, unit: TimeUnit, location: str) -> str:
    """Render a timestamp role value as canonical UTC ISO 8601."""

    if unit == "iso":
        if not isinstance(value, str) or not is_iso8601_datetime_string(value):
            raise InvalidInputError(
                message="expected ISO 8601 timestamp", location=location
            )
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        moment = parsed.astimezone(timezone.utc)
    elif unit in ("seconds", "milliseconds"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidInputError(
                message="expected numeric timestamp", location=location
            )
        seconds = float(value)
        if unit == "milliseconds":
            seconds /= 1000.0
        try:
            moment = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            raise InvalidInputError(
                message="timestamp out of range", location=location
            ) from None
    else:
        raise ValueError(f"unknown timestamp unit: {unit}")
    base = moment.strftime("%Y-%m-%dT%H:%M:%S")
    if moment.microsecond:
        return f"{base}.{moment.microsecond:06d}".rstrip("0") + "Z"
    return base + "Z"
