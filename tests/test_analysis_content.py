"""Document reconstruction and content diffing (Step 25)."""

from __future__ import annotations

import pytest

from driftlens.analysis.content import (
    ComparisonLevelResult,
    ReconstructedDocument,
    classify_chunks,
    compare_content,
    determine_comparison_level,
    diff_documents,
    reconstruct_document_text,
)
from driftlens.config.schema import DocumentsCfg, TextCfg, default_config
from driftlens.errors import Code, DiagnosticSink, IncompatibleError
from driftlens.identity.chunks import compute_text_hashes
from driftlens.identity.hashing import sha256_hex
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import Record
from driftlens.schema.snapshot import (
    DocumentRow,
    DocumentTable,
    Snapshot,
    StatisticsAccumulator,
)


def _chunk(record_id, text, *, document_id="d1", chunk_index=None):
    record = Record(
        record_id=record_id,
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
    )
    compute_text_hashes(record)
    return record


def _manifest(levels: dict[str, list[int]], *, granularity: str = "CHUNK") -> Manifest:
    return Manifest(
        schema_version="1.0",
        snapshot_id="snap",
        content_digest="digest",
        normalization_version="nfc-ws-1",
        tokenizer_id="simple-v1",
        text_params={
            "min_overlap_chars": 16,
            "overlap_window": 2000,
            "casefold": True,
        },
        driftlens_version="0.1.0",
        created_at="2026-01-01T00:00:00+00:00",
        source_type="jsonl",
        granularity=granularity,
        granularity_source="detected" if granularity == "CHUNK" else "assumed",
        document_count=len(levels),
        chunk_count=len(levels),
        embedding_dimension=None,
        embedding_dtype=None,
        embedding_model=None,
        embedding_model_counts={},
        embedding_provider=None,
        embedding_metadata_confidence={},
        similarity_metric=None,
        embedding_normalized=None,
        chunk_identity_levels=levels,
        identity_summary={"HIGH": len(levels), "MEDIUM": 0, "LOW": 0},
        field_mapping=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _table(document_ids):
    return DocumentTable(
        rows={
            document_id: DocumentRow(
                document_id=document_id,
                chunk_count=1,
                identity_confidence="HIGH",
            )
            for document_id in document_ids
        }
    )


def _recon(document_id, *, text=None, raw=None, content=None):
    raw_hash = raw if raw is not None else sha256_hex((text or "").encode("utf-8"))
    content_hash = content if content is not None else raw_hash
    return ReconstructedDocument(
        document_id=document_id,
        text=text,
        document_raw_hash=raw_hash,
        document_content_hash=content_hash,
        reconstruction_confidence="HIGH",
        order_unknown=False,
    )


def _snapshot(texts: dict[str, str]) -> Snapshot:
    records = []
    rows = {}
    for index, (document_id, text) in enumerate(texts.items()):
        record = Record(
            record_id=f"{document_id}::0",
            document_id=document_id,
            text=text,
            row_number=index,
        )
        compute_text_hashes(record)
        records.append(record)
        rows[document_id] = DocumentRow(
            document_id=document_id, chunk_count=1, identity_confidence="HIGH"
        )
    manifest = _manifest(
        {record.record_id: [4] for record in records}, granularity="DOCUMENT"
    )
    return Snapshot(
        manifest=manifest,
        records=records,
        documents=DocumentTable(rows=rows),
        statistics=StatisticsAccumulator(),
    )


def test_reconstruct_document_overlap_removed() -> None:
    first_text = "one two three four five six seven"
    second_text = "four five six seven eight nine ten"
    first = _chunk("c1", first_text, chunk_index=0)
    second = _chunk("c2", second_text, chunk_index=1)
    recon = reconstruct_document_text(
        "d1", [first, second], granularity="CHUNK", cfg=TextCfg()
    )
    assert recon.text == first_text + " eight nine ten"
    assert recon.reconstruction_confidence == "HIGH"
    assert recon.order_unknown is False


def test_reconstruct_document_no_chunk_index_is_low_confidence() -> None:
    first = _chunk("c1", "alpha", chunk_index=None)
    second = _chunk("c2", "beta", chunk_index=None)
    recon = reconstruct_document_text(
        "d1", [first, second], granularity="CHUNK", cfg=TextCfg()
    )
    assert recon.reconstruction_confidence == "LOW"
    assert recon.order_unknown is True
    assert recon.text == "alpha\nbeta"


def test_reconstruct_document_level_hashes_differ_by_whitespace() -> None:
    tight = _chunk("a", "hello world")
    loose = _chunk("b", "hello   world")
    tight_recon = reconstruct_document_text(
        "a", [tight], granularity="DOCUMENT", cfg=TextCfg()
    )
    loose_recon = reconstruct_document_text(
        "b", [loose], granularity="DOCUMENT", cfg=TextCfg()
    )
    assert tight_recon.document_raw_hash != loose_recon.document_raw_hash
    assert tight_recon.document_content_hash == loose_recon.document_content_hash


def test_comparison_level_picks_best_common() -> None:
    v1 = _manifest({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    v2 = _manifest({"a": [2, 3, 4], "b": [2, 3, 4]})
    result = determine_comparison_level(v1, v2, [], [])
    assert result.level == 2


def test_comparison_level_raises_identity_unavailable() -> None:
    v1 = _manifest({"a": [1, 2], "b": [1, 3]})
    v2 = _manifest({"x": [2], "y": [3]})
    with pytest.raises(IncompatibleError) as info:
        determine_comparison_level(v1, v2, [], [])
    assert info.value.code == Code.IDENTITY_UNAVAILABLE
    assert info.value.exit_code == 4


def test_diff_documents_added_removed() -> None:
    rows = diff_documents(
        _table({"a"}),
        _table({"b"}),
        {"a": _recon("a", text="text a")},
        {"b": _recon("b", text="text b")},
        cfg=DocumentsCfg(),
    )
    statuses = {row.document_id: row.status for row in rows}
    assert statuses == {"a": "REMOVED", "b": "ADDED"}


def test_diff_documents_unchanged() -> None:
    rows = diff_documents(
        _table({"a"}),
        _table({"a"}),
        {"a": _recon("a", text="same text")},
        {"a": _recon("a", text="same text")},
        cfg=DocumentsCfg(),
    )
    assert rows[0].status == "UNCHANGED"


def test_diff_documents_format_only() -> None:
    rows = diff_documents(
        _table({"a"}),
        _table({"a"}),
        {"a": _recon("a", text="hello world", raw="raw1", content="content1")},
        {"a": _recon("a", text="hello   world", raw="raw2", content="content1")},
        cfg=DocumentsCfg(),
    )
    assert rows[0].status == "FORMAT_ONLY_CHANGE"


def test_diff_documents_replaced_vs_modified() -> None:
    low1 = " ".join(f"a{index}" for index in range(20))
    low2 = " ".join(f"b{index}" for index in range(20))
    high1 = " ".join(f"c{index}" for index in range(20))
    high2 = " ".join([f"c{index}" for index in range(19)] + ["z"])

    rows = diff_documents(
        _table({"low", "high"}),
        _table({"low", "high"}),
        {
            "low": _recon("low", text=low1, raw="l1", content="l1c"),
            "high": _recon("high", text=high1, raw="h1", content="h1c"),
        },
        {
            "low": _recon("low", text=low2, raw="l2", content="l2c"),
            "high": _recon("high", text=high2, raw="h2", content="h2c"),
        },
        cfg=DocumentsCfg(),
    )
    statuses = {row.document_id: row.status for row in rows}
    assert statuses["low"] == "REPLACED"
    assert statuses["high"] == "MODIFIED"


def test_diff_documents_embedding_only_defaults_to_modified() -> None:
    rows = diff_documents(
        _table({"a"}),
        _table({"a"}),
        {"a": _recon("a", text=None, raw="r1", content="c1")},
        {"a": _recon("a", text=None, raw="r2", content="c2")},
        cfg=DocumentsCfg(),
    )
    assert rows[0].status == "MODIFIED"
    assert rows[0].similarity is None


def test_classify_chunks_unchanged() -> None:
    first = _chunk("c1", "hello world", chunk_index=0)
    second = _chunk("c1", "hello world", chunk_index=0)
    result = classify_chunks(
        [first],
        [second],
        comparison_level=ComparisonLevelResult(1, "text"),
        shared_documents={"d1"},
    )
    assert [pair[2] for pair in result.pairs] == ["UNCHANGED"]


def test_classify_chunks_format_only() -> None:
    first = _chunk("c1", "hello world", chunk_index=0)
    second = _chunk("c1", "hello  world", chunk_index=0)
    result = classify_chunks(
        [first],
        [second],
        comparison_level=ComparisonLevelResult(1, "text"),
        shared_documents={"d1"},
    )
    assert [pair[2] for pair in result.pairs] == ["FORMAT_ONLY_CHANGE"]


def test_classify_chunks_moved() -> None:
    first = _chunk("c1", "hello world", chunk_index=0)
    second = _chunk("c1", "hello world", chunk_index=1)
    result = classify_chunks(
        [first],
        [second],
        comparison_level=ComparisonLevelResult(1, "text"),
        shared_documents={"d1"},
    )
    assert [pair[2] for pair in result.pairs] == ["MOVED"]


def test_classify_chunks_modified_identity() -> None:
    first = _chunk("c1", "hello world", chunk_index=0)
    second = _chunk("c1", "goodbye moon", chunk_index=0)
    result = classify_chunks(
        [first],
        [second],
        comparison_level=ComparisonLevelResult(1, "text"),
        shared_documents={"d1"},
    )
    assert [pair[2] for pair in result.pairs] == ["MODIFIED"]


def test_classify_chunks_added_removed() -> None:
    first = _chunk("c1", "hello world", chunk_index=0)
    second = _chunk("c2", "hello world", chunk_index=0)
    result = classify_chunks(
        [first],
        [second],
        comparison_level=ComparisonLevelResult(1, "text"),
        shared_documents={"d1"},
    )
    statuses = sorted(pair[2] for pair in result.pairs)
    assert statuses == ["ADDED", "REMOVED"]
    assert result.unmatched_v1 == [first]
    assert result.unmatched_v2 == [second]


def test_classify_chunks_moved_na_without_chunk_index() -> None:
    first = _chunk("c1", "hello world", chunk_index=None)
    second = _chunk("c1", "hello world", chunk_index=None)
    result = classify_chunks(
        [first],
        [second],
        comparison_level=ComparisonLevelResult(1, "text"),
        shared_documents={"d1"},
    )
    statuses = [pair[2] for pair in result.pairs]
    assert "MOVED" not in statuses
    assert statuses == ["UNCHANGED"]


def test_content_change_rate_excludes_format_only() -> None:
    unchanged = {f"u{index}": f"unchanged text {index}" for index in range(3)}
    format_v1 = {f"f{index}": f"format text {index}" for index in range(3)}
    format_v2 = {f"f{index}": f"format  text {index}" for index in range(3)}
    modified_v1 = " ".join(f"w{index}" for index in range(50))
    modified_v2 = " ".join([f"w{index}" for index in range(49)] + ["zz"])

    v1_texts = dict(unchanged)
    v1_texts.update(format_v1)
    v1_texts["m"] = modified_v1
    v1_texts["rem"] = "removed text"

    v2_texts = dict(unchanged)
    v2_texts.update(format_v2)
    v2_texts["m"] = modified_v2
    v2_texts["add1"] = "added one"
    v2_texts["add2"] = "added two"

    _rows, _chunks, metrics = compare_content(
        _snapshot(v1_texts),
        _snapshot(v2_texts),
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    metric = next(item for item in metrics if item.name == "content_change_rate")
    assert metric.change == pytest.approx(4 / 10)
