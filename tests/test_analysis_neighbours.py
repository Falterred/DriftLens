"""Neighbour-overlap analysis (Step 46, section 8.9)."""

from __future__ import annotations

import numpy as np
import pytest

from driftlens.analysis.compatibility import CompatibilityResult
from driftlens.analysis.content import ComparisonLevelResult
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.analysis.neighbours import (
    NeighbourSearchIndex,
    build_pool,
    compare_neighbours,
    score_pair,
    select_queries,
)
from driftlens.config.schema import default_config
from driftlens.evaluation.retrievers import DenseProvidedIndex
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import DocumentTable, Snapshot, StatisticsAccumulator


def _record(record_id: str, vector, *, validity: str = "ok") -> Record:
    value = np.asarray(vector, dtype=np.float64)
    return Record(
        record_id=record_id,
        embedding=EmbeddingInfo(
            vector=value,
            dtype="float64",
            dimension=len(value),
            norm=float(np.linalg.norm(value)),
            validity=validity,  # type: ignore[arg-type]
            model=None,
            provider=None,
            similarity_metric=None,
            normalized=None,
        ),
    )


def _manifest(records: list[Record]) -> Manifest:
    return Manifest(
        schema_version="1.0",
        snapshot_id="snap",
        content_digest="digest",
        normalization_version="nfc-ws-1",
        tokenizer_id="simple-v1",
        text_params={"min_overlap_chars": 16, "overlap_window": 2000, "casefold": True},
        driftlens_version="0.2.0",
        created_at="2026-01-01T00:00:00+00:00",
        source_type="jsonl",
        granularity="CHUNK",
        granularity_source="detected",
        document_count=len(records),
        chunk_count=len(records),
        embedding_dimension=None,
        embedding_dtype=None,
        embedding_model=None,
        embedding_model_counts={},
        embedding_provider=None,
        embedding_metadata_confidence={},
        similarity_metric=None,
        embedding_normalized=None,
        chunk_identity_levels={record.record_id: [4] for record in records},
        identity_summary={"HIGH": len(records), "MEDIUM": 0, "LOW": 0},
        field_mapping=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _snapshot(records: list[Record]) -> Snapshot:
    return Snapshot(
        manifest=_manifest(records),
        records=list(records),
        documents=DocumentTable(rows={}),
        statistics=StatisticsAccumulator(),
    )


def _summary(metric: str | None) -> EmbeddingSnapshotSummary:
    return EmbeddingSnapshotSummary(
        model="E1",
        provider=None,
        dtype="float64",
        similarity_metric=metric,
        similarity_metric_derived=False,
        normalized=None,
        confidence={},
        invalid_count=0,
        zero_count=0,
        coverage=1.0,
        norm_summary=None,
        embeddings_unavailable=False,
    )


def _compat(state: str = "COMPATIBLE") -> CompatibilityResult:
    return CompatibilityResult(
        state=state,  # type: ignore[arg-type]
        space_test=None,
        label_conflict=False,
        dimension_v1=None,
        dimension_v2=None,
        metric_v1=None,
        metric_v2=None,
        model_v1=None,
        model_v2=None,
    )


def _level() -> ComparisonLevelResult:
    return ComparisonLevelResult(level=1, hash_kind="text")


def _shared_records(count: int) -> list[Record]:
    return [
        _record("r%04d" % index, [float(index + 1), 1.0]) for index in range(count)
    ]


def test_score_pair_matches_step39_formulas() -> None:
    a = np.array([0.2, 0.5, -0.3])
    b = np.array([-0.1, 0.4, 0.9])
    for metric in ("cosine", "dot", "l2"):
        index = DenseProvidedIndex(
            [_record("r1", a), _record("r2", b)], metric=metric  # type: ignore[arg-type]
        )
        by_id = {result.record_id: result.score for result in index.search(a, top_k=2)}
        assert by_id["r2"] == pytest.approx(score_pair(a, b, metric))  # type: ignore[arg-type]


def test_build_pool_excludes_invalid_and_zero() -> None:
    records = [
        _record("ok", [1.0, 0.0]),
        _record("nan", [1.0, 2.0], validity="nan_inf"),
        _record("zero_tag", [1.0, 2.0], validity="zero"),
        _record("zero_vector", [0.0, 0.0]),
    ]
    pool = build_pool(records, seed=1, sample_size=100)
    assert pool.record_ids == ["ok"]


def test_build_pool_samples_when_over_size() -> None:
    records = [_record("r%03d" % index, [float(index + 1), 1.0]) for index in range(100)]
    pool = build_pool(records, seed=7, sample_size=10)
    assert len(pool.record_ids) <= 10
    assert pool.vectors.shape[0] == len(pool.record_ids)


def test_neighbour_index_excludes_query_itself() -> None:
    records = [_record("r%02d" % index, [float(index + 1), 1.0]) for index in range(5)]
    pool = build_pool(records, seed=1, sample_size=100)
    index = NeighbourSearchIndex(pool, metric="cosine")
    neighbours = index.top_k_other_than("r00", k=5)
    assert "r00" not in neighbours
    assert neighbours


def test_neighbour_index_ties_broken_by_record_id() -> None:
    records = [
        _record("a", [1.0, 0.0]),
        _record("b", [2.0, 0.0]),
        _record("c", [3.0, 0.0]),
    ]
    pool = build_pool(records, seed=1, sample_size=100)
    index = NeighbourSearchIndex(pool, metric="cosine")
    assert index.top_k_other_than("a", k=1) == {"b"}


def test_select_queries_uses_pool_intersection_not_full_snapshot() -> None:
    v1_records = [
        _record("a", [1.0, 0.0]),
        _record("b", [0.0, 1.0]),
        _record("c", [1.0, 1.0], validity="nan_inf"),
    ]
    v2_records = [
        _record("a", [1.0, 0.0]),
        _record("b", [0.0, 1.0]),
        _record("c", [1.0, 1.0]),
        _record("d", [1.0, 1.0]),
    ]
    v1_pool = build_pool(v1_records, seed=1, sample_size=100)
    v2_pool = build_pool(v2_records, seed=1, sample_size=100)
    queries, _sample = select_queries(
        v1_pool,
        v2_pool,
        comparison_level=_level(),
        v1_records_by_id={record.record_id: record for record in v1_records},
        v2_records_by_id={record.record_id: record for record in v2_records},
        max_queries=100,
        seed=1,
    )
    assert sorted(queries) == [("a", "a"), ("b", "b")]


def test_select_queries_caps_at_max_queries() -> None:
    count = 5000
    records = [_record("r%04d" % index, [1.0]) for index in range(count)]
    by_id = {record.record_id: record for record in records}
    v1_pool = build_pool(records, seed=1, sample_size=count)
    v2_pool = build_pool(records, seed=1, sample_size=count)
    queries, _sample = select_queries(
        v1_pool,
        v2_pool,
        comparison_level=_level(),
        v1_records_by_id=by_id,
        v2_records_by_id=by_id,
        max_queries=2000,
        seed=1,
    )
    assert len(queries) <= 2000


def test_compare_neighbours_metric_unknown_checked_before_compat_state() -> None:
    records = _shared_records(40)
    v1 = _snapshot(records)
    v2 = _snapshot(records)
    unknown = _summary(None)
    known = _summary("cosine")

    compatible = compare_neighbours(
        v1,
        v2,
        compat=_compat("COMPATIBLE"),
        v1_summary=unknown,
        v2_summary=known,
        comparison_level=_level(),
        cfg=default_config(),
    )
    assert compatible.reason == "METRIC_UNKNOWN"

    # MIXED_MODELS *does* suspend neighbours; METRIC_UNKNOWN still wins, proving
    # the metric check happens before the compatibility-state table is consulted.
    suspended = compare_neighbours(
        v1,
        v2,
        compat=_compat("MIXED_MODELS"),
        v1_summary=unknown,
        v2_summary=known,
        comparison_level=_level(),
        cfg=default_config(),
    )
    assert suspended.reason == "METRIC_UNKNOWN"


def test_compare_neighbours_runs_under_model_changed() -> None:
    records = _shared_records(40)
    result = compare_neighbours(
        _snapshot(records),
        _snapshot(records),
        compat=_compat("MODEL_CHANGED"),
        v1_summary=_summary("cosine"),
        v2_summary=_summary("cosine"),
        comparison_level=_level(),
        cfg=default_config(),
    )
    assert result.reason is None
    assert result.change is not None


def test_compare_neighbours_suspended_under_mixed_models() -> None:
    records = _shared_records(40)
    result = compare_neighbours(
        _snapshot(records),
        _snapshot(records),
        compat=_compat("MIXED_MODELS"),
        v1_summary=_summary("cosine"),
        v2_summary=_summary("cosine"),
        comparison_level=_level(),
        cfg=default_config(),
    )
    assert result.reason == "SKIPPED_MIXED_MODELS"
    assert result.change is None


def test_compare_neighbours_insufficient_data_below_30_queries() -> None:
    records = _shared_records(10)
    result = compare_neighbours(
        _snapshot(records),
        _snapshot(records),
        compat=_compat("COMPATIBLE"),
        v1_summary=_summary("cosine"),
        v2_summary=_summary("cosine"),
        comparison_level=_level(),
        cfg=default_config(),
    )
    assert result.reason == "INSUFFICIENT_DATA"
    assert result.change is None


def test_compare_neighbours_tolerates_differing_metrics() -> None:
    records = _shared_records(40)
    result = compare_neighbours(
        _snapshot(records),
        _snapshot(records),
        compat=_compat("METRIC_MISMATCH"),
        v1_summary=_summary("cosine"),
        v2_summary=_summary("l2"),
        comparison_level=_level(),
        cfg=default_config(),
    )
    assert result.reason is None
    assert result.change is not None


def test_compare_neighbours_identical_snapshots_perfect_overlap() -> None:
    records = _shared_records(40)
    result = compare_neighbours(
        _snapshot(records),
        _snapshot(records),
        compat=_compat("COMPATIBLE"),
        v1_summary=_summary("cosine"),
        v2_summary=_summary("cosine"),
        comparison_level=_level(),
        cfg=default_config(),
    )
    assert result.change == pytest.approx(1.0)
    assert result.details is not None
    assert result.details["num_queries"] == 40
