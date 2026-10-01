"""Canonical serialization and digests for record identity (Step 11)."""

from __future__ import annotations

import hashlib
import math
from typing import Any

ABSENT_LENGTH = (1 << 64) - 1
DTYPE_TAGS = {"float16": "f16", "float32": "f32", "float64": "f64", "int8": "i8"}

_HEX_ESCAPES = {chr(i): "\\u%04x" % i for i in range(0x20)}
_HEX_ESCAPES['"'] = '\\"'
_HEX_ESCAPES["\\"] = "\\\\"


class NonFiniteError(ValueError):
    """Raised by canonical_json when a float is NaN or infinite."""


def sha256_hex(data: bytes) -> str:
    """Lowercase hexadecimal SHA-256 digest of data."""

    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: bytes) -> bytes:
    """Raw 32-byte SHA-256 digest of data."""

    return hashlib.sha256(data).digest()


def _encode_json_string(value: str) -> bytes:
    out = ['"']
    for ch in value:
        replacement = _HEX_ESCAPES.get(ch)
        out.append(replacement if replacement is not None else ch)
    out.append('"')
    return "".join(out).encode("utf-8")


def canonical_json(value: Any) -> bytes:
    """dl-json-1 bytes: RFC 8785 (JCS) plus the int/float typed deviations."""

    if value is None:
        return b"null"
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if isinstance(value, int):
        return str(value).encode("ascii")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise NonFiniteError("non-finite float is not canonicalizable")
        text = repr(value)
        if "." not in text and "e" not in text and "E" not in text:
            text += ".0"
        return text.encode("ascii")
    if isinstance(value, str):
        return _encode_json_string(value)
    if isinstance(value, list):
        return b"[" + b",".join(canonical_json(item) for item in value) + b"]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError("canonical_json object keys must be str")
        items = sorted(value.items())
        return (
            b"{"
            + b",".join(
                _encode_json_string(key) + b":" + canonical_json(item)
                for key, item in items
            )
            + b"}"
        )
    raise TypeError(f"canonical_json cannot encode {type(value).__name__}")


def encode_field(value: Any) -> bytes:
    """8-byte big-endian length prefix followed by the field payload."""

    if value is None:
        return ABSENT_LENGTH.to_bytes(8, "big")
    if type(value) is bytes:
        payload = value
    elif type(value) is str:
        payload = value.encode("utf-8")
    elif type(value) is int:
        payload = str(value).encode("ascii")
    else:
        raise TypeError(f"encode_field cannot encode {type(value).__name__}")
    return len(payload).to_bytes(8, "big") + payload


def record_digest(
    *,
    record_id: str,
    document_id: str | None,
    raw_text_hash: str | None,
    normalized_text_hash: str | None,
    chunk_index: int | None,
    created_at: str | None,
    updated_at: str | None,
    metadata_canonical: bytes | None,
    embedding_model: str | None,
    embedding_digest_hex: str | None,
) -> bytes:
    """Raw 32-byte record digest over the ten §3.5 fields, in order."""

    hasher = hashlib.sha256()
    for field_value in (
        record_id,
        document_id,
        raw_text_hash,
        normalized_text_hash,
        chunk_index,
        created_at,
        updated_at,
        metadata_canonical,
        embedding_model,
        embedding_digest_hex,
    ):
        hasher.update(encode_field(field_value))
    return hasher.digest()


def content_digest(record_digests_by_id: dict[str, bytes]) -> bytes:
    """Raw 32-byte digest over record digests concatenated in record_id order."""

    hasher = hashlib.sha256()
    for record_id in sorted(record_digests_by_id):
        hasher.update(record_digests_by_id[record_id])
    return hasher.digest()


def snapshot_id(content_digest_bytes: bytes, descriptor_canonical: bytes) -> str:
    """Prefixed snapshot identifier: sha256 over content digest and descriptor."""

    return "sha256:" + sha256_hex(content_digest_bytes + descriptor_canonical)


def sample_hash(seed: int, record_id: str) -> int:
    """256-bit big-endian sampling hash for a record under a numeric seed."""

    material = f"{seed}".encode("ascii") + b"\x1f" + record_id.encode("utf-8")
    return int.from_bytes(sha256_bytes(material), "big")


def embedding_digest(dtype: str, dimension: int, raw_le_bytes: bytes) -> str:
    """Lowercase hex digest binding dtype, dimension, and raw little-endian bytes."""

    tag = DTYPE_TAGS.get(dtype)
    if tag is None:
        raise ValueError(f"unknown embedding dtype: {dtype}")
    material = (
        tag.encode("ascii")
        + b"\x1f"
        + str(dimension).encode("ascii")
        + b"\x1f"
        + raw_le_bytes
    )
    return sha256_hex(material)
