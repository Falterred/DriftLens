"""HDBSCAN cluster analysis (Step 47, section 9)."""

from __future__ import annotations

import sys

import hdbscan
import numpy as np
import pytest

from driftlens.analysis import clusters
from driftlens.analysis.clusters import (
    ClusterAssignment,
    ClusteringResult,
    classify_cluster_transitions,
    cluster_centroid,
    compare_clusters,
    fit_pca,
    run_hdbscan,
)
from driftlens.analysis.compatibility import CompatibilityResult
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.config.schema import Config, default_config
from driftlens.errors import Code, DiagnosticSink
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import DocumentTable, Snapshot, StatisticsAccumulator


def _record(record_id, vector, *, validity="ok"):
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


def _manifest(records):
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


def _snapshot(records):
    return Snapshot(
        manifest=_manifest(records),
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


def _cluster_cfg(**overrides):
    cfg: Config = default_config()
    for key, value in overrides.items():
        setattr(cfg.cluster, key, value)
    return cfg


def _result(pairs):
    return ClusteringResult(
        assignments=[
            ClusterAssignment(record_id, label) for record_id, label in pairs
        ],
        noise_fraction=0.0,
        min_cluster_size=2,
    )


def _fake_hdbscan(label_fn):
    class _FakeHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_predict(self, X):
            return np.asarray([label_fn(row) for row in np.asarray(X)], dtype=int)

    return _FakeHDBSCAN


class _IdentityPCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)


def _identity_fit(pooled, *, n_components):
    return _IdentityPCA(n_components)


def test_mode_off_returns_empty():
    cfg = _cluster_cfg(mode="off")
    sink = DiagnosticSink()
    result = compare_clusters(
        _snapshot([]),
        _snapshot([]),
        compat=_compat(),
        v1_summary=_summary(None),
        v2_summary=_summary(None),
        cfg=cfg,
        sink=sink,
    )
    assert result == []
    assert sink.items() == []


def test_mode_auto_missing_extra_skipped(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "hdbscan", None)
    sink = DiagnosticSink()
    result = compare_clusters(
        _snapshot([]),
        _snapshot([]),
        compat=_compat(),
        v1_summary=_summary(None),
        v2_summary=_summary(None),
        cfg=_cluster_cfg(),
        sink=sink,
    )
    assert len(result) == 1
    assert result[0].status == "SKIPPED"
    assert result[0].reason == "CLUSTER_EXTRA_MISSING"
    assert sink.has(Code.CLUSTER_EXTRA_MISSING)


def test_suspended_by_model_changed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        clusters, "run_hdbscan", lambda *args, **kwargs: calls.append(1) or _result([])
    )
    result = compare_clusters(
        _snapshot([]),
        _snapshot([]),
        compat=_compat("MODEL_CHANGED"),
        v1_summary=_summary(None),
        v2_summary=_summary(None),
        cfg=_cluster_cfg(mode="on"),
        sink=DiagnosticSink(),
    )
    assert len(result) == 1
    assert result[0].status == "SKIPPED"
    assert result[0].reason == "SKIPPED_MODEL_CHANGED"
    assert calls == []


def test_pca_shared_fit_independent_transform(monkeypatch):
    pooled = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    pca = fit_pca(pooled, n_components=50)
    assert pca.components_.shape == (2, 5)
    v1 = pooled[:2]
    v2 = pooled[2:]
    assert np.allclose(pca.transform(v1), (v1 - pca.mean_) @ pca.components_.T)
    assert np.allclose(pca.transform(v2), (v2 - pca.mean_) @ pca.components_.T)

    spy = {"fit": 0, "transform": 0}

    class _SpyPCA(_IdentityPCA):
        def transform(self, X):
            spy["transform"] += 1
            return super().transform(X)

    def _fit(pooled_vectors, *, n_components):
        spy["fit"] += 1
        return _SpyPCA(n_components)

    monkeypatch.setattr(clusters, "fit_pca", _fit)
    monkeypatch.setattr(hdbscan, "HDBSCAN", _fake_hdbscan(lambda row: 0))
    compare_clusters(
        _snapshot([_record("v1_%d" % index, [1.0, 0.0]) for index in range(6)]),
        _snapshot([_record("v2_%d" % index, [1.0, 0.0]) for index in range(6)]),
        compat=_compat(),
        v1_summary=_summary(None),
        v2_summary=_summary(None),
        cfg=_cluster_cfg(min_cluster_size=2),
        sink=DiagnosticSink(),
    )
    assert spy == {"fit": 1, "transform": 2}


def test_hdbscan_deterministic_label_canonicalization(monkeypatch):
    monkeypatch.setattr(
        hdbscan,
        "HDBSCAN",
        _fake_hdbscan(lambda row: 5 if row[0] < 3 else (3 if row[0] < 6 else -1)),
    )
    vectors = np.array([[float(index), 1.0] for index in range(1, 7)])
    ids = ["r%d" % index for index in range(1, 7)]
    cfg = _cluster_cfg(min_cluster_size=2)
    first = run_hdbscan(vectors, ids, cfg=cfg.cluster)
    second = run_hdbscan(vectors, ids, cfg=cfg.cluster)
    assert first.assignments == second.assignments
    labels = {item.record_id: item.label for item in first.assignments}
    assert labels == {"r1": 1, "r2": 1, "r3": 0, "r4": 0, "r5": 0, "r6": -1}


def test_hdbscan_noise_excluded_from_fraction_denominator(monkeypatch):
    captured = {}

    def factory(label_fn):
        class _Capturing:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def fit_predict(self, X):
                return np.asarray([label_fn(row) for row in np.asarray(X)], dtype=int)

        return _Capturing

    monkeypatch.setattr(hdbscan, "HDBSCAN", factory(lambda row: -1 if row[0] <= 3 else 1))
    vectors = np.array([[float(index), 1.0] for index in range(1, 11)])
    ids = ["r%02d" % index for index in range(1, 11)]
    cfg = _cluster_cfg(min_cluster_size=3)
    result = run_hdbscan(vectors, ids, cfg=cfg.cluster)
    assert result.noise_fraction == pytest.approx(3 / 10)
    assert result.noise_fraction != pytest.approx(3 / 7)
    assert result.min_cluster_size == 3
    assert captured == {
        "min_cluster_size": 3,
        "min_samples": 3,
        "cluster_selection_method": "eom",
        "allow_single_cluster": False,
    }


def test_cluster_centroid_normalizes_when_cosine_space():
    assignment = _result([("a", 0), ("b", 0)])
    vectors = np.array([[3.0, 0.0], [0.0, 4.0]])
    plain = cluster_centroid(vectors, assignment, 0, l2_normalize_first=False)
    normalized = cluster_centroid(vectors, assignment, 0, l2_normalize_first=True)
    assert np.allclose(plain, [1.5, 2.0])
    assert np.allclose(normalized, [0.5, 0.5])
    assert not np.allclose(plain, normalized)


def test_transition_stable():
    cfg = _cluster_cfg()
    v1 = _result([("a", 0), ("b", 0), ("c", 0)])
    v2 = _result([("a", 0), ("b", 0), ("c", 0)])
    vectors = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0]), "c": np.array([1.0, 0.0])}
    transitions = classify_cluster_transitions(
        v1, v2, vectors, vectors, cfg=cfg.cluster, cosine_space=False
    )
    assert len(transitions) == 1
    assert transitions[0].kind == "STABLE"
    assert transitions[0].similarity_kind == "jaccard"
    assert transitions[0].similarity == pytest.approx(1.0)


def test_transition_shifted_by_size_ratio():
    cfg = _cluster_cfg()
    v1 = _result([("a", 0), ("b", 0)])
    v2 = _result([(name, 0) for name in ("c", "d", "e", "f", "g", "h")])
    v1_vectors = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}
    v2_vectors = {name: np.array([1.0, 0.0]) for name in ("c", "d", "e", "f", "g", "h")}
    transitions = classify_cluster_transitions(
        v1, v2, v1_vectors, v2_vectors, cfg=cfg.cluster, cosine_space=False
    )
    assert len(transitions) == 1
    assert transitions[0].kind == "SHIFTED"
    assert transitions[0].similarity_kind == "centroid_cosine"
    assert transitions[0].similarity == pytest.approx(1.0)


def test_transition_falls_back_to_centroid_cosine():
    cfg = _cluster_cfg()
    v1 = _result([("a", 0), ("b", 0)])
    v2 = _result([("c", 0), ("d", 0)])
    v1_vectors = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}
    v2_vectors = {"c": np.array([0.9, 0.1]), "d": np.array([0.9, 0.1])}
    transitions = classify_cluster_transitions(
        v1, v2, v1_vectors, v2_vectors, cfg=cfg.cluster, cosine_space=False
    )
    assert len(transitions) == 1
    assert transitions[0].similarity_kind == "centroid_cosine"
    assert transitions[0].kind == "STABLE"


def test_transition_split():
    cfg = _cluster_cfg(min_common=0.6)
    v1 = _result([("s%d" % index, 0) for index in range(10)])
    v2 = _result(
        [("s0", 0), ("s1", 0), ("u0", 0), ("u1", 0)]
        + [("s2", 1), ("s3", 1), ("u2", 1), ("u3", 1)]
    )
    v1_vectors = {"s%d" % index: np.array([1.0, 0.0]) for index in range(2)}
    v1_vectors.update({"s%d" % index: np.array([0.0, 1.0]) for index in range(2, 10)})
    v2_vectors = {
        "s0": np.array([1.0, 0.0]),
        "s1": np.array([1.0, 0.0]),
        "u0": np.array([-1.0, 0.0]),
        "u1": np.array([-1.0, 0.0]),
        "s2": np.array([0.0, 1.0]),
        "s3": np.array([0.0, 1.0]),
        "u2": np.array([0.0, -1.0]),
        "u3": np.array([0.0, -1.0]),
    }
    transitions = classify_cluster_transitions(
        v1, v2, v1_vectors, v2_vectors, cfg=cfg.cluster, cosine_space=False
    )
    assert len(transitions) == 1
    assert transitions[0].kind == "SPLIT"
    assert transitions[0].v1_labels == [0]
    assert transitions[0].v2_labels == [0, 1]


def test_transition_merged():
    cfg = _cluster_cfg(min_common=0.6)
    v1 = _result(
        [("a0", 0), ("a1", 0), ("p0", 0), ("p1", 0)]
        + [("b0", 1), ("b1", 1), ("q0", 1), ("q1", 1)]
    )
    v2 = _result(
        [("a0", 0), ("a1", 0), ("b0", 0), ("b1", 0)]
        + [("r%d" % index, 0) for index in range(6)]
    )
    v1_vectors = {
        "a0": np.array([1.0, 0.0]),
        "a1": np.array([1.0, 0.0]),
        "p0": np.array([-1.0, 0.0]),
        "p1": np.array([-1.0, 0.0]),
        "b0": np.array([0.0, 1.0]),
        "b1": np.array([0.0, 1.0]),
        "q0": np.array([0.0, -1.0]),
        "q1": np.array([0.0, -1.0]),
    }
    v2_vectors = {
        "a0": np.array([1.0, 0.0]),
        "a1": np.array([1.0, 0.0]),
        "b0": np.array([0.0, 1.0]),
        "b1": np.array([0.0, 1.0]),
    }
    v2_vectors.update({"r%d" % index: np.array([1.0, 1.0]) for index in range(6)})
    transitions = classify_cluster_transitions(
        v1, v2, v1_vectors, v2_vectors, cfg=cfg.cluster, cosine_space=False
    )
    assert len(transitions) == 1
    assert transitions[0].kind == "MERGED"
    assert transitions[0].v1_labels == [0, 1]
    assert transitions[0].v2_labels == [0]


def test_transition_new_disappeared():
    cfg = _cluster_cfg()
    v1 = _result([("a", 0), ("b", 0)])
    v2 = _result([("c", 0), ("d", 0)])
    v1_vectors = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}
    v2_vectors = {"c": np.array([0.0, 1.0]), "d": np.array([0.0, 1.0])}
    transitions = classify_cluster_transitions(
        v1, v2, v1_vectors, v2_vectors, cfg=cfg.cluster, cosine_space=False
    )
    by_kind = {transition.kind: transition for transition in transitions}
    assert set(by_kind) == {"DISAPPEARED", "NEW"}
    assert by_kind["DISAPPEARED"].v1_labels == [0]
    assert by_kind["DISAPPEARED"].v2_labels == []
    assert by_kind["NEW"].v1_labels == []
    assert by_kind["NEW"].v2_labels == [0]


def test_semantic_na_no_nonnoise_clusters_v1(monkeypatch):
    monkeypatch.setattr(clusters, "fit_pca", _identity_fit)
    monkeypatch.setattr(
        hdbscan, "HDBSCAN", _fake_hdbscan(lambda row: -1 if row[0] >= 0.5 else 0)
    )
    v1 = [_record("a%d" % index, [1.0, 0.0]) for index in range(6)]
    v2 = [_record("b%d" % index, [0.0, 1.0]) for index in range(6)]
    result = compare_clusters(
        _snapshot(v1),
        _snapshot(v2),
        compat=_compat(),
        v1_summary=_summary(None),
        v2_summary=_summary(None),
        cfg=_cluster_cfg(min_cluster_size=2),
        sink=DiagnosticSink(),
    )
    assert len(result) == 1
    assert result[0].reason == "N/A"


def test_semantic_insufficient_data_below_2x_min_cluster_size(monkeypatch):
    calls = []
    monkeypatch.setattr(
        clusters, "run_hdbscan", lambda *args, **kwargs: calls.append(1) or _result([])
    )
    records = [_record("r%d" % index, [1.0, float(index)]) for index in range(4)]
    result = compare_clusters(
        _snapshot(records),
        _snapshot(records),
        compat=_compat(),
        v1_summary=_summary(None),
        v2_summary=_summary(None),
        cfg=_cluster_cfg(min_cluster_size=5),
        sink=DiagnosticSink(),
    )
    assert len(result) == 1
    assert result[0].reason == "INSUFFICIENT_DATA"
    assert calls == []


def test_semantic_rate_counts_disappeared_as_not_stable(monkeypatch):
    monkeypatch.setattr(clusters, "fit_pca", _identity_fit)

    def label(row):
        if row[0] > 0.5:
            return 0
        if row[1] > 0.5:
            return 1
        return 2

    monkeypatch.setattr(hdbscan, "HDBSCAN", _fake_hdbscan(label))
    v1 = (
        [_record("a%d" % index, [1.0, 0.0, 0.0]) for index in range(4)]
        + [_record("b%d" % index, [0.0, 1.0, 0.0]) for index in range(4)]
        + [_record("c%d" % index, [0.0, 0.0, 1.0]) for index in range(4)]
    )
    v2 = [_record("a%d" % index, [1.0, 0.0, 0.0]) for index in range(4)] + [
        _record("b%d" % index, [0.0, 1.0, 0.0]) for index in range(10)
    ]
    result = compare_clusters(
        _snapshot(v1),
        _snapshot(v2),
        compat=_compat(),
        v1_summary=_summary(None),
        v2_summary=_summary(None),
        cfg=_cluster_cfg(min_cluster_size=2),
        sink=DiagnosticSink(),
    )
    assert len(result) == 1
    assert result[0].change == pytest.approx(2 / 3)


