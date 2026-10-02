"""Baseline construction and application (Step 48, section 12.4)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from driftlens.analysis import baselines
from driftlens.analysis.baselines import (
    BaselineFile,
    apply_baseline,
    build_baseline,
    compute_threshold_from_values,
    load_baseline,
    validate_baseline_snapshots,
    write_baseline,
)
from driftlens.analysis.health import make_metric, round9
from driftlens.config.schema import Threshold, default_config
from driftlens.errors import Code, DiagnosticSink, InvalidInputError
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import DocumentTable, Snapshot, StatisticsAccumulator


@pytest.fixture()
def workdir():
    base = Path(__file__).resolve().parent / "_analysis_baselines_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _record(record_id, vector, *, model=None):
    value = np.asarray(vector, dtype=np.float64)
    return Record(
        record_id=record_id,
        embedding=EmbeddingInfo(
            vector=value,
            dtype="float64",
            dimension=len(value),
            norm=float(np.linalg.norm(value)),
            validity="ok",
            model=model,
            provider=None,
            similarity_metric=None,
            normalized=None,
        ),
    )


def _manifest(
    records,
    *,
    normalization="nfc-ws-1",
    tokenizer="simple-v1",
    dimension=None,
    model=None,
    similarity_metric=None,
):
    return Manifest(
        schema_version="1.0",
        snapshot_id="snap",
        content_digest="digest",
        normalization_version=normalization,
        tokenizer_id=tokenizer,
        text_params={"min_overlap_chars": 16, "overlap_window": 2000, "casefold": True},
        driftlens_version="0.2.0",
        created_at="2026-01-01T00:00:00+00:00",
        source_type="jsonl",
        granularity="CHUNK",
        granularity_source="detected",
        document_count=len(records),
        chunk_count=len(records),
        embedding_dimension=dimension,
        embedding_dtype=None,
        embedding_model=model,
        embedding_model_counts={},
        embedding_provider=None,
        embedding_metadata_confidence={},
        similarity_metric=similarity_metric,
        embedding_normalized=None,
        chunk_identity_levels={record.record_id: [4] for record in records},
        identity_summary={"HIGH": len(records), "MEDIUM": 0, "LOW": 0},
        field_mapping=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _snapshot(records, **manifest_kwargs):
    return Snapshot(
        manifest=_manifest(records, **manifest_kwargs),
        records=list(records),
        documents=DocumentTable(rows={}),
        statistics=StatisticsAccumulator(),
    )


def _vectors(snapshot):
    return np.vstack(
        [np.asarray(record.embedding.vector, dtype=np.float64) for record in snapshot.records]
    )


def _baseline(**overrides):
    settings = {
        "retrieval.primary_metric": "recall@5",
        "retrieval.k": [1, 5, 10],
        "ci.retrieval_drop_mode": "absolute",
        "sample_size": 10000,
        "seed": 42,
        "bandwidth": None,
    }
    base = dict(
        format_version="1.0",
        thresholds={},
        embedding_model=None,
        embedding_dimension=None,
        similarity_metric=None,
        tokenizer_id="simple-v1",
        normalization_version="nfc-ws-1",
        settings=settings,
        pair_count=1,
        method="mean_std",
    )
    base.update(overrides)
    return BaselineFile(**base)


def _patch_pipeline(monkeypatch, *, mmd_skipped_pairs=frozenset()):
    calls = {"mmd": 0}
    monkeypatch.setattr(
        baselines,
        "compare_content",
        lambda v1, v2, **kw: (
            None,
            None,
            [make_metric("content_change_rate", value=0.1, threshold=None)],
        ),
    )
    monkeypatch.setattr(
        baselines,
        "compare_chunking",
        lambda v1, v2, **kw: [make_metric("chunk_ks", value=0.2, threshold=None)],
    )
    monkeypatch.setattr(
        baselines,
        "compare_metadata",
        lambda v1, v2, **kw: (
            None,
            [make_metric("metadata_js", value=0.3, threshold=None)],
        ),
    )
    monkeypatch.setattr(
        baselines,
        "compare_duplicates",
        lambda v1, v2, **kw: (
            None,
            None,
            [
                make_metric(
                    "duplicate_rate",
                    value=0.4,
                    threshold=None,
                    v1=0.3,
                    v2=0.4,
                    change=0.1,
                )
            ],
        ),
    )
    monkeypatch.setattr(
        baselines,
        "compare_clusters",
        lambda v1, v2, **kw: [make_metric("semantic", value=0.5, threshold=None)],
    )

    def fake_mmd(v1, v2, **kwargs):
        index = calls["mmd"]
        calls["mmd"] += 1
        if index in mmd_skipped_pairs:
            return make_metric("mmd", value=None, threshold=None, reason="MIXED_MODELS")
        return make_metric("mmd", value=0.05, threshold=None)

    monkeypatch.setattr(baselines, "compare_mmd", fake_mmd)
    return calls


def test_validate_baseline_rejects_differing_normalization():
    first = _snapshot([_record("c1", [1.0, 0.0])], normalization="nfc-ws-1")
    second = _snapshot([_record("c1", [1.0, 0.0])], normalization="nfd-ws-2")
    with pytest.raises(InvalidInputError):
        validate_baseline_snapshots([first, second])


def test_validate_baseline_allows_differing_models():
    first = _snapshot([_record("c1", [1.0, 0.0], model="E1")], dimension=2, model="E1")
    second = _snapshot([_record("c1", [1.0, 0.0], model="E2")], dimension=2, model="E2")
    validate_baseline_snapshots([first, second])


def test_fixed_bandwidth_from_newest_pair(monkeypatch):
    first = _snapshot(
        [
            _record("c1", [0.5, 0.0, 0.0, 0.0]),
            _record("c2", [0.0, 0.5, 0.0, 0.0]),
            _record("c3", [0.0, 0.0, 0.5, 0.0]),
        ],
        dimension=4,
    )
    second = _snapshot(
        [
            _record("c1", [2.0, 0.0, 0.0, 0.0]),
            _record("c2", [0.0, 2.0, 0.0, 0.0]),
            _record("c3", [0.0, 0.0, 2.0, 0.0]),
        ],
        dimension=4,
    )
    third = _snapshot(
        [
            _record("c1", [5.0, 0.0, 0.0, 0.0]),
            _record("c2", [0.0, 5.0, 0.0, 0.0]),
            _record("c3", [0.0, 0.0, 5.0, 0.0]),
        ],
        dimension=4,
    )
    cfg = default_config()
    cfg.significance.permutations = 5
    cfg.significance.bootstrap = 5
    captured = []
    real = baselines.compare_mmd
    _patch_pipeline(monkeypatch, mmd_skipped_pairs=frozenset())

    def spy(v1, v2, **kwargs):
        metric = real(v1, v2, **kwargs)
        captured.append(metric.details.get("sigma") if metric.details else None)
        return metric

    monkeypatch.setattr(baselines, "compare_mmd", spy)

    build_baseline([first, second, third], eval_set=None, cfg=cfg)

    newest = round9(
        float(
            baselines.median_heuristic_bandwidth(
                np.vstack([_vectors(second), _vectors(third)]),
                seed=cfg.analysis.seed,
            )
        )
    )
    older = round9(
        float(
            baselines.median_heuristic_bandwidth(
                np.vstack([_vectors(first), _vectors(second)]),
                seed=cfg.analysis.seed,
            )
        )
    )
    assert captured[0] == newest
    assert captured[0] != older


def test_per_key_independent_valid_counts(monkeypatch):
    _patch_pipeline(monkeypatch, mmd_skipped_pairs=frozenset({0, 2}))
    snapshots = [
        _snapshot([_record("c1", [1.0, 0.0]), _record("c2", [0.0, 1.0])], dimension=2)
        for _ in range(6)
    ]
    cfg = default_config()
    cfg.baseline.min_pairs = 5
    result = build_baseline(snapshots, eval_set=None, cfg=cfg)
    assert result.pair_count == 5
    assert "mmd" not in result.thresholds
    assert "chunk_ks" in result.thresholds


def test_min_pairs_not_met_key_absent(monkeypatch):
    _patch_pipeline(monkeypatch)
    snapshots = [
        _snapshot([_record("c1", [1.0, 0.0]), _record("c2", [0.0, 1.0])], dimension=2)
        for _ in range(4)
    ]
    cfg = default_config()
    cfg.baseline.min_pairs = 5
    result = build_baseline(snapshots, eval_set=None, cfg=cfg)
    assert result.pair_count == 3
    assert result.thresholds == {}


def test_mean_std_method():
    values = [1.0, 2.0, 3.0]
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    threshold = compute_threshold_from_values(values, method="mean_std", min_pairs=3)
    assert threshold == Threshold(
        warning=round9(mean + 2.0 * std), critical=round9(mean + 3.0 * std)
    )


def test_quantile_method():
    values = [1.0, 2.0, 3.0, 4.0]
    threshold = compute_threshold_from_values(values, method="quantile", min_pairs=4)
    assert threshold == Threshold(
        warning=round9(float(np.percentile(values, 90))),
        critical=round9(float(np.percentile(values, 99))),
    )


def test_threshold_clipped_at_zero():
    mean_std = compute_threshold_from_values(
        [-5.0, -4.0, -3.0], method="mean_std", min_pairs=3
    )
    assert mean_std == Threshold(warning=0.0, critical=0.0)
    quantile = compute_threshold_from_values(
        [-5.0, -4.0, -3.0], method="quantile", min_pairs=3
    )
    assert quantile == Threshold(warning=0.0, critical=0.0)


def test_apply_baseline_explicit_wins():
    cfg = default_config()
    cfg.drift["mmd"] = Threshold(warning=0.05, critical=0.10)
    baseline = _baseline(thresholds={"mmd": Threshold(warning=0.5, critical=0.9)})
    manifest = _manifest([_record("c1", [1.0, 0.0])], dimension=2)
    sink = DiagnosticSink()
    result = apply_baseline(
        cfg, baseline, v1_manifest=manifest, v2_manifest=manifest, sink=sink
    )
    assert result.drift["mmd"] == Threshold(warning=0.05, critical=0.10)


def test_apply_baseline_fills_unset():
    cfg = default_config()
    baseline = _baseline(thresholds={"chunk_ks": Threshold(warning=0.2, critical=0.3)})
    manifest = _manifest([_record("c1", [1.0, 0.0])], dimension=2)
    sink = DiagnosticSink()
    result = apply_baseline(
        cfg, baseline, v1_manifest=manifest, v2_manifest=manifest, sink=sink
    )
    assert result.drift["chunk_ks"] == Threshold(warning=0.2, critical=0.3)
    assert result.drift["mmd"] is None


def test_apply_baseline_incompatible_dimension():
    cfg = default_config()
    baseline = _baseline(embedding_dimension=768)
    manifest = _manifest([_record("c1", [1.0, 0.0, 0.0, 0.0])], dimension=1536)
    sink = DiagnosticSink()
    result = apply_baseline(
        cfg, baseline, v1_manifest=manifest, v2_manifest=manifest, sink=sink
    )
    assert result is cfg
    assert sink.has(Code.BASELINE_INCOMPATIBLE)


def test_apply_baseline_unknown_model_matches_only_unknown():
    cfg = default_config()
    baseline = _baseline()
    unknown = _manifest([_record("c1", [1.0, 0.0])], dimension=2, model=None)
    sink = DiagnosticSink()
    result = apply_baseline(
        cfg, baseline, v1_manifest=unknown, v2_manifest=unknown, sink=sink
    )
    assert not sink.has(Code.BASELINE_INCOMPATIBLE)
    assert result is not cfg

    known = _manifest([_record("c1", [1.0, 0.0], model="E1")], dimension=2, model="E1")
    known_sink = DiagnosticSink()
    known_result = apply_baseline(
        cfg, baseline, v1_manifest=known, v2_manifest=known, sink=known_sink
    )
    assert known_sink.has(Code.BASELINE_INCOMPATIBLE)
    assert known_result is cfg


def test_baseline_write_read_round_trip(workdir):
    baseline = _baseline(
        thresholds={
            "chunk_ks": Threshold(warning=0.2, critical=0.3),
            "mmd": Threshold(warning=None, critical=0.5),
        },
        embedding_model="E1",
        embedding_dimension=768,
        similarity_metric="cosine",
        pair_count=4,
        method="quantile",
    )
    path = workdir / "nested" / "baseline.json"
    write_baseline(baseline, path)
    loaded = load_baseline(path)
    assert loaded == baseline
