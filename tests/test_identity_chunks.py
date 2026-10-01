"""Chunk identity tests (Step 19)."""

from __future__ import annotations

from driftlens.identity.chunks import (
    all_chunk_identity_keys,
    compute_chunk_identity,
    compute_embedding_digest_field,
    compute_text_hashes,
)
from driftlens.schema.record import Record


def _record(record_id=None, *, document_id=None, chunk_index=None, text=None):
    return Record(
        record_id=record_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
    )


def _prepared(record):
    compute_text_hashes(record)
    compute_embedding_digest_field(record)
    return record


def test_chunk_identity_level1_explicit() -> None:
    record = _prepared(_record(record_id="c99", text="hello"))
    result = compute_chunk_identity(
        record, document_identity_confidence="HIGH", duplicate_rank_within_document=0
    )
    assert result.level == 1
    assert result.confidence == "HIGH"
    assert result.key == "c1:c99"


def test_chunk_identity_level1_indexidmap_medium() -> None:
    record = _prepared(_record(record_id="c99", text="hello"))
    record._identity_source_hint = "indexidmap"
    result = compute_chunk_identity(
        record, document_identity_confidence="HIGH", duplicate_rank_within_document=0
    )
    assert result.level == 1
    assert result.confidence == "MEDIUM"
    assert result.key == "c1:c99"


def test_chunk_identity_level2_needs_nonlow_document() -> None:
    record = _prepared(
        _record(record_id=None, document_id="d1", chunk_index=3, text="hello")
    )
    result = compute_chunk_identity(
        record, document_identity_confidence="LOW", duplicate_rank_within_document=0
    )
    assert result.level == 4
    assert result.confidence == "LOW"


def test_chunk_identity_level3_ordinal_no_collision() -> None:
    records = [
        _prepared(_record(record_id=None, document_id="d1", text="same"))
        for _ in range(3)
    ]
    keys = [
        all_chunk_identity_keys(
            record,
            document_identity_confidence="HIGH",
            duplicate_rank_within_document=index,
        )[3]
        for index, record in enumerate(records)
    ]
    assert len(set(keys)) == 3
    assert keys[0].endswith("#0")
    assert keys[1].endswith("#1")
    assert keys[2].endswith("#2")
    result = compute_chunk_identity(
        records[0], document_identity_confidence="HIGH", duplicate_rank_within_document=0
    )
    assert result.level == 3
