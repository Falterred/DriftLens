"""Maximum mean discrepancy point estimate (Step 30)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from driftlens.analysis.compatibility import CompatibilityResult
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.analysis.mmd import (
    compare_mmd,
    compute_mmd,
    median_heuristic_bandwidth,
    rbf_kernel_matrix,
)
from driftlens.config.schema import AnalysisCfg, default_config
from driftlens.errors import DiagnosticSink
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import DocumentTable, Snapshot, StatisticsAccumulator


def _record(record_id, vector):
    value = np.asarray(vector, dtype=np.float64)
    info = EmbeddingInfo(
        vector=value,
        dtype="float64",
        dimension=len(value),
        norm=float(np.linalg.norm(value)),
        validity="ok",
        model=None,
        provider=None,
        similarity_metric=None,
        normalized=None,
    )
    return Record(record_id=record_id, embedding=info)


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


def _summary(metric):
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


def _compat(state="COMPATIBLE"):
    return CompatibilityResult(
        state=state,
        space_test=None,
        label_conflict=False,
        dimension_v1=None,
        dimension_v2=None,
        metric_v1=None,
        metric_v2=None,
        model_v1=None,
        model_v2=None,
    )


def test_rbf_kernel_identity_pairs_are_one() -> None:
    points = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -1.0]])
    kernel = rbf_kernel_matrix(points, points, 1.0)
    assert np.allclose(np.diag(kernel), 1.0)


def test_rbf_kernel_symmetric() -> None:
    rng = np.random.default_rng(0)
    first = rng.normal(size=(4, 3))
    second = rng.normal(size=(5, 3))
    assert np.allclose(
        rbf_kernel_matrix(first, second, 1.3),
        rbf_kernel_matrix(second, first, 1.3).T,
    )


def test_median_heuristic_matches_manual_computation() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [2.0, 5.0],
            [5.0, 5.0],
        ]
    )
    expected = float(np.median(pdist(points, metric="euclidean")))
    assert median_heuristic_bandwidth(points, seed=42) == expected


def test_insufficient_data_fewer_than_2() -> None:
    result = compute_mmd(
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        ["a"],
        ["b", "c"],
        cfg=AnalysisCfg(),
        fixed_bandwidth=None,
        seed=42,
    )
    assert result.reason == "INSUFFICIENT_DATA"


def test_degenerate_kernel_identical_samples() -> None:
    vectors = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    result = compute_mmd(
        vectors,
        vectors,
        ["a", "b", "c"],
        ["d", "e", "f"],
        cfg=AnalysisCfg(),
        fixed_bandwidth=None,
        seed=42,
    )
    assert result.mmd == 0.0
    assert result.reason is None


def test_degenerate_kernel_zero_sigma_different_samples() -> None:
    result = compute_mmd(
        np.zeros((3, 2)),
        np.ones((3, 2)),
        ["a", "b", "c"],
        ["d", "e", "f"],
        cfg=AnalysisCfg(),
        fixed_bandwidth=0.0,
        seed=42,
    )
    assert result.reason == "DEGENERATE_KERNEL"


def test_mmd_clamped_nonnegative() -> None:
    rng = np.random.default_rng(7)
    result = compute_mmd(
        rng.normal(size=(20, 2)),
        rng.normal(size=(20, 2)),
        [f"a{index}" for index in range(20)],
        [f"b{index}" for index in range(20)],
        cfg=AnalysisCfg(),
        fixed_bandwidth=None,
        seed=42,
    )
    assert result.mmd_squared is not None
    assert result.mmd_squared >= 0.0
    assert result.mmd is not None
    assert result.mmd >= 0.0


def test_mmd_known_separation() -> None:
    rng = np.random.default_rng(1)
    first = rng.normal(loc=0.0, scale=0.1, size=(20, 2))
    second = rng.normal(loc=10.0, scale=0.1, size=(20, 2))
    control = rng.normal(loc=0.0, scale=0.1, size=(20, 2))
    ids_a = [f"a{index}" for index in range(20)]
    ids_b = [f"b{index}" for index in range(20)]

    separated = compute_mmd(
        first, second, ids_a, ids_b, cfg=AnalysisCfg(), fixed_bandwidth=None, seed=42
    )
    baseline = compute_mmd(
        first, control, ids_a, ids_b, cfg=AnalysisCfg(), fixed_bandwidth=None, seed=42
    )
    assert separated.mmd is not None
    assert baseline.mmd is not None
    assert separated.mmd > baseline.mmd


def test_compare_mmd_suspended_by_model_changed() -> None:
    metric = compare_mmd(
        _snapshot([]),
        _snapshot([]),
        compat=_compat("MODEL_CHANGED"),
        v1_summary=_summary(None),
        v2_summary=_summary(None),
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    assert metric.status == "SKIPPED"
    assert metric.reason == "SKIPPED_MODEL_CHANGED"


def test_compare_mmd_cosine_normalization_applied() -> None:
    first = _snapshot([_record("a0", [3.0, 0.0]), _record("a1", [0.0, 4.0])])
    second = _snapshot([_record("b0", [6.0, 0.0]), _record("b1", [0.0, 8.0])])
    cfg = default_config()

    metric = compare_mmd(
        first,
        second,
        compat=_compat(),
        v1_summary=_summary("cosine"),
        v2_summary=_summary("cosine"),
        cfg=cfg,
        sink=DiagnosticSink(),
    )
    normalized = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected = compute_mmd(
        normalized,
        normalized,
        ["a0", "a1"],
        ["b0", "b1"],
        cfg=cfg.analysis,
        fixed_bandwidth=None,
        seed=cfg.analysis.seed,
    )
    assert metric.change == pytest.approx(expected.mmd)


def test_compare_mmd_not_normalized_for_l2_metric() -> None:
    first = _snapshot([_record("a0", [3.0, 0.0]), _record("a1", [0.0, 4.0])])
    second = _snapshot([_record("b0", [6.0, 0.0]), _record("b1", [0.0, 8.0])])
    cfg = default_config()

    metric = compare_mmd(
        first,
        second,
        compat=_compat(),
        v1_summary=_summary("l2"),
        v2_summary=_summary("l2"),
        cfg=cfg,
        sink=DiagnosticSink(),
    )
    expected = compute_mmd(
        np.array([[3.0, 0.0], [0.0, 4.0]]),
        np.array([[6.0, 0.0], [0.0, 8.0]]),
        ["a0", "a1"],
        ["b0", "b1"],
        cfg=cfg.analysis,
        fixed_bandwidth=None,
        seed=cfg.analysis.seed,
    )
    assert metric.change == pytest.approx(expected.mmd)


def test_compare_mmd_approximate_flag_propagates() -> None:
    first = _snapshot([_record("a0", [1.0, 0.0]), _record("a1", [0.0, 1.0])])
    second = _snapshot([_record("b0", [1.0, 0.0]), _record("b1", [0.0, 1.0])])
    first.vectors_approximate = True

    metric = compare_mmd(
        first,
        second,
        compat=_compat(),
        v1_summary=_summary(None),
        v2_summary=_summary(None),
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    assert metric.approximate is True
