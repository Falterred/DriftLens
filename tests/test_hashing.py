"""Canonical serialization and digests (Step 11)."""

from __future__ import annotations

import re

import pytest

from driftlens.identity.hashing import (
    ABSENT_LENGTH,
    NonFiniteError,
    canonical_json,
    content_digest,
    encode_field,
    embedding_digest,
    record_digest,
    sample_hash,
    sha256_bytes,
    snapshot_id,
)


def test_canonical_json_key_sort() -> None:
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_canonical_json_int_vs_float() -> None:
    assert canonical_json({"x": 1}) == b'{"x":1}'
    assert canonical_json({"x": 1.0}) == b'{"x":1.0}'
    assert canonical_json({"x": 1}) != canonical_json({"x": 1.0})


def test_canonical_json_big_int() -> None:
    assert canonical_json(2**60 + 7) == str(2**60 + 7).encode("ascii")


def test_canonical_json_nan_rejected() -> None:
    with pytest.raises(NonFiniteError):
        canonical_json({"x": float("nan")})


def test_canonical_json_bool_not_int() -> None:
    assert canonical_json(True) == b"true"
    assert canonical_json(False) == b"false"


def test_canonical_json_no_slash_escape() -> None:
    assert canonical_json("a/b") == b'"a/b"'


def test_encode_field_absent_vs_empty() -> None:
    absent = encode_field(None)
    empty = encode_field("")
    assert absent == ABSENT_LENGTH.to_bytes(8, "big")
    assert len(absent) == 8
    assert empty == b"\x00" * 8
    assert len(empty) == 8
    assert absent != empty
    assert int.from_bytes(absent[:8], "big") == ABSENT_LENGTH
    assert int.from_bytes(empty[:8], "big") == 0
    # Neither is a prefix of the other's continuation.
    assert not absent.startswith(empty)
    assert not empty.startswith(absent)


def test_record_digest_field_order_matters() -> None:
    common = dict(
        record_id="r1",
        document_id="d1",
        raw_text_hash=None,
        normalized_text_hash=None,
        chunk_index=None,
        metadata_canonical=None,
        embedding_model=None,
        embedding_digest_hex=None,
    )
    first = record_digest(created_at="t1", updated_at="t2", **common)
    second = record_digest(created_at="t2", updated_at="t1", **common)
    assert first != second


def test_content_digest_order_independent_of_dict_insertion() -> None:
    d_a = sha256_bytes(b"a")
    d_b = sha256_bytes(b"b")
    assert content_digest({"b": d_b, "a": d_a}) == content_digest({"a": d_a, "b": d_b})


def test_snapshot_id_format() -> None:
    assert re.match(r"^sha256:[0-9a-f]{64}$", snapshot_id(b"\x00" * 32, b"{}"))


def test_sample_hash_deterministic_and_seed_sensitive() -> None:
    assert sample_hash(42, "r1") == sample_hash(42, "r1")
    assert sample_hash(42, "r1") != sample_hash(43, "r1")


def test_embedding_digest_dtype_sensitive() -> None:
    assert embedding_digest("float32", 3, b"\x00" * 12) != embedding_digest(
        "float64", 3, b"\x00" * 12
    )
