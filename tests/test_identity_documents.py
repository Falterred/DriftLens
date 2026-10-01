"""Document identity and granularity tests (Step 19)."""

from __future__ import annotations

import pytest

from driftlens.config.schema import SideOptions
from driftlens.errors import Code, DiagnosticSink, InvalidInputError
from driftlens.identity.chunks import compute_embedding_digest_field, compute_text_hashes
from driftlens.identity.documents import (
    assign_identity,
    compute_document_identity,
    detect_granularity,
    disambiguate_duplicate_ids,
)
from driftlens.identity.hashing import sha256_hex
from driftlens.schema.record import Record
from driftlens.text.normalize import normalize_text


def _record(record_id=None, *, document_id=None, chunk_index=None, text=None, metadata=None):
    return Record(
        record_id=record_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        metadata=dict(metadata or {}),
    )


def _prepared(record):
    compute_text_hashes(record)
    compute_embedding_digest_field(record)
    return record


def _identity(record, *, value, source_identifier=False, granularity="CHUNK", chunk1=None):
    return compute_document_identity(
        record,
        adapter_hint=None,
        explicit_column_value=value,
        explicit_column_is_source_identifier=source_identifier,
        granularity=granularity,
        chunk_level1_id=chunk1,
    )


def test_document_identity_level1_vs_level2() -> None:
    record = _prepared(_record(text="hello", metadata={"source": "meta-src"}))
    primary = _identity(record, value="primary")
    source_tier = _identity(record, value="primary", source_identifier=True)
    assert primary.level == 1
    assert primary.confidence == "HIGH"
    assert primary.document_id == "primary"
    assert source_tier.level == 2
    assert source_tier.confidence == "HIGH"
    assert primary.document_id != "meta-src"


def test_document_identity_level3_metadata_fallback() -> None:
    record = _prepared(_record(text="hello", metadata={"source_id": "abc"}))
    result = _identity(record, value=None)
    assert result.level == 3
    assert result.confidence == "MEDIUM"
    assert result.document_id == "abc"


def test_document_identity_level5_content_derived() -> None:
    record = _prepared(_record(text="hello"))
    result = _identity(record, value=None)
    assert result.level == 5
    assert result.confidence == "LOW"
    assert result.document_id == "cid:" + sha256_hex(normalize_text("hello").encode("utf-8"))


def test_granularity_assumed_when_no_ids_anywhere() -> None:
    records = [_prepared(_record(text="a")), _prepared(_record(text="b"))]
    assert detect_granularity(records, forced=None) == ("DOCUMENT", "assumed")


def test_granularity_detected_chunk_via_repeated_doc_id() -> None:
    records = [
        _prepared(_record(document_id="d1", text="a")),
        _prepared(_record(document_id="d1", text="b")),
    ]
    assert detect_granularity(records, forced=None) == ("CHUNK", "detected")


def test_granularity_detected_chunk_via_chunk_index() -> None:
    records = [_prepared(_record(document_id="d1", chunk_index=2, text="a"))]
    assert detect_granularity(records, forced=None) == ("CHUNK", "detected")


def test_duplicate_id_disambiguation_suffix() -> None:
    records = [
        _prepared(_record(record_id="dup", text="alpha")),
        _prepared(_record(record_id="dup", text="beta")),
    ]
    sink = DiagnosticSink()
    out = disambiguate_duplicate_ids(records, on_duplicate_id="suffix", sink=sink)
    assert sorted(record.record_id for record in out) == ["dup#0", "dup#1"]
    assert sink.has(Code.IDENTITY_WARNING)


def test_duplicate_id_error_mode() -> None:
    records = [
        _prepared(_record(record_id="dup", text="alpha")),
        _prepared(_record(record_id="dup", text="beta")),
    ]
    sink = DiagnosticSink()
    with pytest.raises(InvalidInputError):
        disambiguate_duplicate_ids(records, on_duplicate_id="error", sink=sink)
    assert sink.has(Code.IDENTITY_WARNING)


def test_assign_identity_end_to_end_chunk_granularity() -> None:
    records = []
    for document in ("d1", "d2"):
        for index in range(3):
            records.append(
                _record(
                    record_id=None,
                    document_id=document,
                    chunk_index=index,
                    text=f"{document}-{index}",
                )
            )
    sink = DiagnosticSink()
    out, granularity, source = assign_identity(
        records, side=SideOptions(), sink=sink, granularity_override=None
    )
    assert granularity == "CHUNK"
    assert source == "detected"
    assert all(record.identity_confidence is not None for record in out)
    for document in ("d1", "d2"):
        ids = [record.record_id for record in out if record.document_id == document]
        assert len(ids) == len(set(ids))
