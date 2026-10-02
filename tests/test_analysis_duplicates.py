"""Exact-duplicate grouping (Step 29)."""

from __future__ import annotations

import pytest

from driftlens.analysis.duplicates import compare_duplicates, compute_exact_groups
from driftlens.analysis.sampling import hash_threshold_sample
from driftlens.config.schema import DuplicatesCfg, Threshold, default_config
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import Record
from driftlens.schema.snapshot import DocumentTable, Snapshot, StatisticsAccumulator


def _record(record_id, *, text_hash=None, embedding_digest=None):
    text = f"text {text_hash}" if text_hash is not None else None
    record = Record(record_id=record_id, text=text, normalized_text=text)
    record.normalized_text_hash = text_hash
    if embedding_digest is not None:
        record._embedding_digest_hex = embedding_digest
    return record


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


def test_exact_group_by_text_hash() -> None:
    records = [
        _record("a", text_hash="h"),
        _record("b", text_hash="h"),
        _record("c", text_hash="h"),
    ]
    result = compute_exact_groups(records, cfg=DuplicatesCfg(), seed=42)
    assert len(result.groups) == 1
    assert sorted(result.groups[0].member_ids) == ["a", "b", "c"]
    assert result.redundant_count == 2


def test_exact_group_embedding_only() -> None:
    records = [
        _record("a", embedding_digest="d"),
        _record("b", embedding_digest="d"),
    ]
    result = compute_exact_groups(records, cfg=DuplicatesCfg(), seed=42)
    assert len(result.groups) == 1
    assert result.redundant_count == 1


def test_representative_is_smallest_id() -> None:
    records = [
        _record("z1", text_hash="h"),
        _record("a1", text_hash="h"),
        _record("m1", text_hash="h"),
    ]
    result = compute_exact_groups(records, cfg=DuplicatesCfg(), seed=42)
    assert result.groups[0].representative_id == "a1"


def test_no_duplicates_zero_rate() -> None:
    records = [_record(f"r{index}", text_hash=f"h{index}") for index in range(3)]
    result = compute_exact_groups(records, cfg=DuplicatesCfg(), seed=42)
    assert result.duplicate_rate == 0.0
    assert result.groups == []


def test_duplicate_rate_denominator_is_full_snapshot() -> None:
    ids = [f"r{index}" for index in range(10)]
    sampled = hash_threshold_sample(42, 6, ids)
    considered = [record_id for record_id in ids if record_id in sampled.v1_ids]
    pair = considered[:2]

    records = [
        _record(
            record_id,
            text_hash="dup" if record_id in pair else f"u{record_id}",
        )
        for record_id in ids
    ]
    result = compute_exact_groups(
        records, cfg=DuplicatesCfg(max_exact_vectors=6), seed=42
    )
    assert result.redundant_count == 1
    assert result.duplicate_rate == 1 / 10


def test_covered_fraction_full_when_under_cap() -> None:
    records = [_record(f"r{index}", text_hash=f"h{index}") for index in range(5)]
    result = compute_exact_groups(records, cfg=DuplicatesCfg(), seed=42)
    assert result.covered_fraction == 1.0
    assert result.sample is None


def test_covered_fraction_partial_when_sampled() -> None:
    records = [_record(f"r{index}", text_hash=f"h{index}") for index in range(1000)]
    result = compute_exact_groups(
        records, cfg=DuplicatesCfg(max_exact_vectors=100), seed=42
    )
    assert result.covered_fraction == pytest.approx(0.1)
    assert result.sample is not None


def test_units_include_singletons() -> None:
    records = [_record(f"r{index}", text_hash=f"h{index}") for index in range(5)]
    result = compute_exact_groups(records, cfg=DuplicatesCfg(), seed=42)
    assert len(result.units) == len(records)


def _rate_fixture():
    v1_records = [
        _record("v1a", text_hash="d1"),
        _record("v1b", text_hash="d1"),
        _record("v1c", text_hash="d1"),
        _record("v1d", text_hash="solo"),
    ]
    v2_records = [_record(f"v2{index}", text_hash=f"s{index}") for index in range(99)]
    v2_records.append(_record("v2dup1", text_hash="shared"))
    v2_records.append(_record("v2dup2", text_hash="shared"))
    return v1_records, v2_records


def test_compare_duplicates_tests_v2_level_not_change() -> None:
    v1_records, v2_records = _rate_fixture()
    drift_cfg = default_config().drift
    drift_cfg["duplicate_rate"] = Threshold(warning=0.02)

    _v1, _v2, metrics = compare_duplicates(
        _snapshot(v1_records),
        _snapshot(v2_records),
        cfg=DuplicatesCfg(),
        drift_cfg=drift_cfg,
    )
    metric = next(item for item in metrics if item.name == "duplicate_rate")
    assert metric.status == "PASS"


def test_duplicate_rate_near_semantic_unbaselined_not_skipped() -> None:
    v1_records, v2_records = _rate_fixture()
    _v1, _v2, metrics = compare_duplicates(
        _snapshot(v1_records),
        _snapshot(v2_records),
        cfg=DuplicatesCfg(),
        drift_cfg=default_config().drift,
    )
    by_name = {item.name: item for item in metrics}
    for name in ("duplicate_rate_near", "duplicate_rate_semantic"):
        # Step 37: the stages now run, so the metrics carry real (here zero)
        # values and are merely unbaselined, no longer N/A.
        assert by_name[name].status == "UNBASELINED"
        assert by_name[name].reason is None
        assert by_name[name].v2 == 0.0


def test_duplicate_rate_exact_equals_duplicate_rate_at_v01() -> None:
    v1_records, v2_records = _rate_fixture()
    _v1, _v2, metrics = compare_duplicates(
        _snapshot(v1_records),
        _snapshot(v2_records),
        cfg=DuplicatesCfg(),
        drift_cfg=default_config().drift,
    )
    by_name = {item.name: item for item in metrics}
    assert by_name["duplicate_rate"].v2 == by_name["duplicate_rate_exact"].v2
