"""Permutation tests and bootstrap CIs (Step 45, sections 8.7/8.8)."""

from __future__ import annotations

import numpy as np
import pytest

from driftlens.analysis import mmd as mmd_module
from driftlens.analysis.compatibility import CompatibilityResult
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.analysis.mmd import compare_mmd
from driftlens.analysis.significance import (
    Partition,
    bootstrap_ci,
    choose_permutation_design,
    compute_significance,
    generate_permutation,
    permutation_test,
)
from driftlens.config.schema import SignificanceCfg, default_config
from driftlens.errors import DiagnosticSink
from driftlens.evaluation.retrieval import _ci_details
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import DocumentTable, Snapshot, StatisticsAccumulator


def _record(record_id: str, vector, *, confidence: str | None = None) -> Record:
    value = np.asarray(vector, dtype=np.float64)
    return Record(
        record_id=record_id,
        embedding=EmbeddingInfo(
            vector=value,
            dtype="float64",
            dimension=len(value),
            norm=float(np.linalg.norm(value)),
            validity="ok",
            model=None,
            provider=None,
            similarity_metric=None,
            normalized=None,
        ),
        identity_confidence=confidence,
    )


def _manifest(records: list[Record]) -> Manifest:
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


def test_choose_design_pairing_aware_with_trustworthy_shared_ids() -> None:
    design, paired = choose_permutation_design(
        ["a", "b", "c"], ["a", "b", "d"], {"a": "HIGH", "b": "MEDIUM"}
    )
    assert design == "PAIRING_AWARE"
    assert paired == {"a", "b"}


def test_choose_design_two_sample_when_all_shared_are_low() -> None:
    design, paired = choose_permutation_design(
        ["a", "b"], ["a", "b"], {"a": "LOW", "b": "LOW"}
    )
    assert design == "TWO_SAMPLE"
    assert paired == set()


def test_choose_design_two_sample_when_no_shared_ids() -> None:
    design, paired = choose_permutation_design(["a", "b"], ["c", "d"], {})
    assert design == "TWO_SAMPLE"
    assert paired == set()


def test_generate_permutation_pairing_preserves_group_sizes() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        partition = generate_permutation(
            "PAIRING_AWARE",
            n_v1=3,
            n_v2=3,
            paired_positions=[(0, 3), (1, 4)],
            unpaired_v1_positions=[2],
            unpaired_v2_positions=[5],
            rng=rng,
        )
        assert len(partition.v1_indices) == 3
        assert len(partition.v2_indices) == 3
        assert sorted(partition.v1_indices + partition.v2_indices) == list(range(6))


def test_generate_permutation_two_sample_preserves_group_sizes() -> None:
    rng = np.random.default_rng(1)
    for _ in range(100):
        partition = generate_permutation(
            "TWO_SAMPLE",
            n_v1=4,
            n_v2=2,
            paired_positions=[],
            unpaired_v1_positions=[],
            unpaired_v2_positions=[],
            rng=rng,
        )
        assert len(partition.v1_indices) == 4
        assert len(partition.v2_indices) == 2
        assert sorted(partition.v1_indices + partition.v2_indices) == list(range(6))


def test_permutation_test_formula() -> None:
    values = [1.0, 0.0, 0.5, 1.5]
    calls = {"n": 0}

    def statistic_fn(partition: Partition) -> float:
        value = values[calls["n"]]
        calls["n"] += 1
        return value

    p_value = permutation_test(
        n_v1=2,
        n_v2=2,
        statistic_fn=statistic_fn,
        observed_partition=Partition(v1_indices=[0, 1], v2_indices=[2, 3]),
        design="TWO_SAMPLE",
        paired_positions=[],
        unpaired_v1_positions=[],
        unpaired_v2_positions=[],
        cfg=SignificanceCfg(permutations=3),
        seed=0,
    )
    assert p_value == pytest.approx((1 + 1) / (1 + 3))
    assert calls["n"] == 4


def test_permutation_test_deterministic_with_seed() -> None:
    def statistic_fn(partition: Partition) -> float:
        return float(sum(partition.v1_indices))

    kwargs = dict(
        n_v1=3,
        n_v2=3,
        statistic_fn=statistic_fn,
        observed_partition=Partition(v1_indices=[0, 1, 2], v2_indices=[3, 4, 5]),
        design="TWO_SAMPLE",
        paired_positions=[],
        unpaired_v1_positions=[],
        unpaired_v2_positions=[],
        cfg=SignificanceCfg(permutations=20),
    )
    first = permutation_test(seed=7, **kwargs)
    second = permutation_test(seed=7, **kwargs)
    assert first == second
    assert first is not None


def test_bootstrap_ci_percentile_method() -> None:
    pool = [float(index) for index in range(200)]
    counter = {"n": 0}

    def statistic_fn(partition: Partition) -> float:
        value = pool[counter["n"]]
        counter["n"] += 1
        return value

    ci = bootstrap_ci(
        statistic_fn=statistic_fn,
        n_v1=4,
        n_v2=4,
        cfg=SignificanceCfg(bootstrap=200),
        seed=0,
    )
    assert ci is not None
    assert ci[0] == pytest.approx(float(np.percentile(pool, 2.5)))
    assert ci[1] == pytest.approx(float(np.percentile(pool, 97.5)))


def test_bootstrap_tolerates_repeated_indices() -> None:
    seen = {"duplicate": False}

    def statistic_fn(partition: Partition) -> float:
        if len(set(partition.v1_indices)) < len(partition.v1_indices):
            seen["duplicate"] = True
        if len(set(partition.v2_indices)) < len(partition.v2_indices):
            seen["duplicate"] = True
        return 0.0

    ci = bootstrap_ci(
        statistic_fn=statistic_fn,
        n_v1=5,
        n_v2=5,
        cfg=SignificanceCfg(bootstrap=200),
        seed=1,
    )
    assert ci is not None
    assert seen["duplicate"] is True


def test_compute_significance_below_20_skipped() -> None:
    def statistic_fn(partition: Partition) -> float:
        return 0.5

    result = compute_significance(
        v1_ids=[f"v{index}" for index in range(15)],
        v2_ids=[f"w{index}" for index in range(15)],
        id_confidence={},
        statistic_fn=statistic_fn,
        observed=0.5,
        cfg=default_config().significance,
        seed=0,
    )
    assert result.reason == "SIGNIFICANCE_SKIPPED"
    assert result.p_value is None
    assert result.b == 0
    assert result.ci_low is None and result.ci_high is None


def test_compute_significance_ci_needs_30() -> None:
    def statistic_fn(partition: Partition) -> float:
        return float(len(partition.v1_indices))

    result = compute_significance(
        v1_ids=[f"v{index}" for index in range(25)],
        v2_ids=[f"w{index}" for index in range(25)],
        id_confidence={},
        statistic_fn=statistic_fn,
        observed=25.0,
        cfg=SignificanceCfg(permutations=50, bootstrap=20, max_n=5000),
        seed=0,
    )
    assert result.p_value is not None
    assert result.ci_low is None and result.ci_high is None


def test_compute_significance_budget_reduces_large_populations() -> None:
    state = {"n": 0, "max_v1": 0, "max_v2": 0, "obs_v1": 0, "obs_v2": 0}

    def statistic_fn(partition: Partition) -> float:
        if state["n"] == 0:
            state["obs_v1"] = len(partition.v1_indices)
            state["obs_v2"] = len(partition.v2_indices)
        state["n"] += 1
        state["max_v1"] = max(state["max_v1"], len(partition.v1_indices))
        state["max_v2"] = max(state["max_v2"], len(partition.v2_indices))
        return 0.5

    result = compute_significance(
        v1_ids=[f"v{index}" for index in range(10000)],
        v2_ids=[f"w{index}" for index in range(10000)],
        id_confidence={},
        statistic_fn=statistic_fn,
        observed=0.5,
        cfg=SignificanceCfg(permutations=10, bootstrap=5, max_n=5000),
        seed=0,
    )
    assert result.p_value is not None
    assert state["obs_v1"] <= 5000 and state["obs_v2"] <= 5000
    assert state["max_v1"] <= 5000 and state["max_v2"] <= 5000


def test_mmd_kernel_matrix_computed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(3)
    first = _snapshot(
        [_record(f"a{index}", rng.normal(size=4), confidence="HIGH") for index in range(25)]
    )
    second = _snapshot(
        [_record(f"b{index}", rng.normal(size=4), confidence="HIGH") for index in range(25)]
    )
    cfg = default_config()
    cfg.significance = SignificanceCfg(permutations=50, bootstrap=20, max_n=5000)

    calls = {"n": 0}
    original = mmd_module.rbf_kernel_matrix

    def spy(x, y, sigma):
        calls["n"] += 1
        return original(x, y, sigma)

    monkeypatch.setattr(mmd_module, "rbf_kernel_matrix", spy)
    metric = compare_mmd(
        first,
        second,
        compat=_compat(),
        v1_summary=_summary("l2"),
        v2_summary=_summary("l2"),
        cfg=cfg,
        sink=DiagnosticSink(),
    )
    assert metric.details is not None
    assert metric.details["p_value"] is not None
    assert metric.details["test_design"] == "TWO_SAMPLE"
    assert calls["n"] == 4
    assert calls["n"] < cfg.significance.permutations


def test_retrieval_ci_paired_by_case() -> None:
    cfg = default_config()
    cfg.significance = SignificanceCfg(permutations=999, bootstrap=200, max_n=5000)
    n_cases = 40
    v1_values = [float(index % 2) for index in range(n_cases)]
    v2_values = [
        v1_values[index] + (1.0 if index < n_cases // 2 else -1.0)
        for index in range(n_cases)
    ]

    details = _ci_details(
        v1_values, v2_values, cfg=cfg, seed=cfg.analysis.seed
    )
    assert details is not None
    paired = details["ci_95"]
    assert isinstance(paired, list)

    def paired_fn(partition: Partition) -> float:
        total = 0.0
        for index in partition.v1_indices:
            total += v2_values[index] - v1_values[index]
        return total / len(partition.v1_indices)

    expected = bootstrap_ci(
        statistic_fn=paired_fn,
        n_v1=n_cases,
        n_v2=n_cases,
        cfg=cfg.significance,
        seed=cfg.analysis.seed,
    )
    assert expected is not None
    assert paired[0] == pytest.approx(expected[0])
    assert paired[1] == pytest.approx(expected[1])

    def unpaired_fn(partition: Partition) -> float:
        first_mean = sum(v1_values[i] for i in partition.v1_indices) / len(
            partition.v1_indices
        )
        second_mean = sum(v2_values[i] for i in partition.v2_indices) / len(
            partition.v2_indices
        )
        return second_mean - first_mean

    unpaired = bootstrap_ci(
        statistic_fn=unpaired_fn,
        n_v1=n_cases,
        n_v2=n_cases,
        cfg=cfg.significance,
        seed=cfg.analysis.seed,
    )
    assert unpaired is not None
    assert (
        abs(unpaired[0] - float(paired[0])) > 1e-9
        or abs(unpaired[1] - float(paired[1])) > 1e-9
    )
