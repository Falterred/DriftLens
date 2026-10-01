"""Chunk profile, overlap and anomaly analysis (Step 26)."""

from __future__ import annotations

from driftlens.analysis.chunking import (
    classify_chunk_anomaly,
    compare_chunking,
    compute_anomaly_counts,
    compute_chunk_profile,
    compute_overlap_estimate,
)
from driftlens.config.schema import ChunkingCfg, Threshold, default_config
from driftlens.identity.chunks import compute_text_hashes
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import Record
from driftlens.schema.snapshot import (
    DocumentRow,
    DocumentTable,
    Snapshot,
    StatisticsAccumulator,
)


def _record(
    record_id,
    *,
    text="x",
    token_count=None,
    char_count=None,
    sentence_count=None,
    encoding_damage=None,
    chunk_index=None,
    document_id=None,
    normalized_text=None,
):
    return Record(
        record_id=record_id,
        document_id=document_id,
        text=text,
        normalized_text=normalized_text,
        token_count=token_count,
        char_count=char_count,
        sentence_count=sentence_count,
        encoding_damage=encoding_damage,
        chunk_index=chunk_index,
    )


def _manifest(levels):
    return Manifest(
        schema_version="1.0",
        snapshot_id="snap",
        content_digest="digest",
        normalization_version="nfc-ws-1",
        tokenizer_id="simple-v1",
        text_params={"min_overlap_chars": 16, "overlap_window": 2000, "casefold": True},
        driftlens_version="0.1.0",
        created_at="2026-01-01T00:00:00+00:00",
        source_type="jsonl",
        granularity="CHUNK",
        granularity_source="detected",
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


def _snapshot(records):
    return Snapshot(
        manifest=_manifest({record.record_id: [4] for record in records}),
        records=list(records),
        documents=DocumentTable(rows={}),
        statistics=StatisticsAccumulator(),
    )


def _table(counts):
    return DocumentTable(
        rows={
            document_id: DocumentRow(
                document_id=document_id,
                chunk_count=chunk_count,
                identity_confidence="HIGH",
            )
            for document_id, chunk_count in counts.items()
        }
    )


def test_anomaly_cascade_empty_wins() -> None:
    flags = classify_chunk_anomaly(_record("r", token_count=0), cfg=ChunkingCfg())
    assert flags.size_category == "empty"


def test_anomaly_cascade_very_short() -> None:
    flags = classify_chunk_anomaly(_record("r", token_count=2), cfg=ChunkingCfg())
    assert flags.size_category == "very_short"


def test_anomaly_cascade_suspicious_short() -> None:
    flags = classify_chunk_anomaly(_record("r", token_count=15), cfg=ChunkingCfg())
    assert flags.size_category == "suspicious_short"


def test_anomaly_oversized_plain() -> None:
    flags = classify_chunk_anomaly(_record("r", token_count=3000), cfg=ChunkingCfg())
    assert flags.size_category == "oversized"


def test_anomaly_over_embedder_limit_with_safety_factor() -> None:
    cfg = ChunkingCfg(embedding_max_tokens=1000, token_safety_factor=1.3)
    flags = classify_chunk_anomaly(_record("r", token_count=800), cfg=cfg)
    assert flags.size_category == "over_embedder_limit"


def test_anomaly_under_embedder_limit_with_safety_factor_not_flagged() -> None:
    cfg = ChunkingCfg(embedding_max_tokens=1000, token_safety_factor=1.3)
    flags = classify_chunk_anomaly(_record("r", token_count=700), cfg=cfg)
    assert flags.size_category is None


def test_anomalous_rate_no_double_count() -> None:
    record = _record("r", token_count=0, encoding_damage=True)
    counts = compute_anomaly_counts([record], cfg=ChunkingCfg())
    assert counts.anomalous_rate == 1.0


def test_anomalous_rate_excludes_suspicious_short() -> None:
    records = [_record(f"r{index}", token_count=15) for index in range(4)]
    counts = compute_anomaly_counts(records, cfg=ChunkingCfg())
    assert counts.anomalous_rate == 0.0


def test_overlap_estimate_none_without_chunk_index() -> None:
    records = [
        _record("a", text="alpha beta gamma", chunk_index=None),
        _record("b", text="beta gamma delta", chunk_index=None),
    ]
    assert compute_overlap_estimate(records, seed=42) is None


def test_overlap_estimate_basic() -> None:
    first = _record(
        "a",
        text="t0 t1 t2 t3 t4 t5 t6",
        document_id="d1",
        chunk_index=0,
    )
    second = _record(
        "b",
        text="t2 t3 t4 t5 t6 t7 t8 t9",
        document_id="d1",
        chunk_index=1,
    )
    compute_text_hashes(first)
    compute_text_hashes(second)
    estimate = compute_overlap_estimate([first, second], seed=42)
    assert estimate is not None
    assert estimate.share_with_overlap == 1.0
    assert estimate.median_l == 5


def test_chunks_per_document_matches_document_table() -> None:
    documents = _table({"d1": 2, "d2": 5})
    profile = compute_chunk_profile([], documents)
    assert profile.chunks_per_document == {"d1": 2, "d2": 5}


def test_compare_chunking_anomalous_rate_tests_v2_only() -> None:
    v1_records = [
        _record("a1", token_count=0),
        _record("a2", token_count=10),
    ]
    v2_records = [
        _record(f"b{index}", token_count=0 if index == 0 else 10)
        for index in range(100)
    ]
    cfg = default_config()
    cfg.drift["anomalous_chunk_rate"] = Threshold(warning=0.02)

    metrics = compare_chunking(_snapshot(v1_records), _snapshot(v2_records), cfg=cfg)
    metric = next(item for item in metrics if item.name == "anomalous_chunk_rate")
    assert metric.status == "PASS"


def test_compare_chunking_produces_report_only_char_sentence_metrics() -> None:
    v1_records = [
        _record("a1", token_count=10, char_count=50, sentence_count=2),
        _record("a2", token_count=14, char_count=70, sentence_count=4),
    ]
    v2_records = [
        _record("b1", token_count=12, char_count=60, sentence_count=3),
        _record("b2", token_count=16, char_count=80, sentence_count=5),
    ]

    metrics = compare_chunking(
        _snapshot(v1_records), _snapshot(v2_records), cfg=default_config()
    )
    by_name = {item.name: item for item in metrics}
    assert "char_ks" in by_name
    assert "sentence_ks" in by_name
    assert by_name["char_ks"].is_threshold_key is False
    assert by_name["sentence_ks"].is_threshold_key is False
