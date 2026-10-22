"""Maximum mean discrepancy point estimate (Step 30, v0.1)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist

from driftlens.analysis.compatibility import (
    ANALYSES_SUSPENDED_BY_STATE,
    CompatibilityResult,
    suspension_reason_code,
)
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.analysis.health import Metric, SampleInfo, make_metric, round9
from driftlens.analysis.sampling import hash_threshold_sample, nested_subsample, to_sample_info
from driftlens.analysis.significance import SignificanceResult, compute_significance
from driftlens.config.schema import AnalysisCfg, Config
from driftlens.errors import DiagnosticSink
from driftlens.schema.snapshot import Snapshot

_DEFAULT_MAX_SUBSAMPLE = 2000


@dataclass
class MMDResult:
    mmd: float | None
    mmd_squared: float | None
    sigma: float | None
    estimator: str
    reason: str | None
    sample_v1: SampleInfo | None
    sample_v2: SampleInfo | None
    mmd_squared_raw: float | None = None


def rbf_kernel_matrix(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix via the expand-dot-product trick."""
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    x2 = np.sum(first ** 2, axis=1)[:, None]
    y2 = np.sum(second ** 2, axis=1)[None, :]
    squared = np.maximum(x2 + y2 - 2 * first @ second.T, 0.0)
    return np.exp(-squared / (2 * sigma ** 2))


def median_heuristic_bandwidth(
    pooled: np.ndarray, *, seed: int, max_subsample: int = _DEFAULT_MAX_SUBSAMPLE
) -> float:
    """Median pairwise Euclidean distance over a deterministic subsample."""
    points = np.asarray(pooled, dtype=np.float64)
    if len(points) < 2:
        return 0.0

    if len(points) > max_subsample:
        ids = [str(index) for index in range(len(points))]
        sample = hash_threshold_sample(seed, max_subsample, ids, v2_record_ids=None)
        sub = points[[int(record_id) for record_id in sample.v1_ids]]
    else:
        sub = points

    if len(sub) < 2:
        return 0.0
    return float(np.median(pdist(sub, metric="euclidean")))


def _vector_set(vectors: np.ndarray) -> set[tuple[float, ...]]:
    return {
        tuple(round(float(value), 9) for value in row)
        for row in np.asarray(vectors, dtype=np.float64)
    }


def _subset(
    vectors: np.ndarray, ids: Sequence[str], kept: frozenset[str]
) -> np.ndarray:
    rows = [vectors[index] for index, record_id in enumerate(ids) if record_id in kept]
    if not rows:
        return np.empty((0, vectors.shape[1]), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _mmd_from_kernel(
    kernel_matrix: np.ndarray, v1_indices: Sequence[int], v2_indices: Sequence[int]
) -> float:
    """Unbiased MMD statistic recomputed by reindexing a fixed kernel matrix.

    P2: gathers use flat fancy indexing on a cached raveled kernel instead of
    three np.ix_ blocks plus np.trace. The summed element sets are identical
    to the previous form; only floating-point summation order can differ.
    """

    kernel = np.asarray(kernel_matrix, dtype=np.float64)
    m = len(v1_indices)
    n = len(v2_indices)
    if m < 2 or n < 2:
        return 0.0
    rows = np.fromiter(v1_indices, dtype=np.intp, count=m)
    cols = np.fromiter(v2_indices, dtype=np.intp, count=n)
    size = kernel.shape[0]
    flat = kernel.reshape(-1)
    kxx = flat[rows[:, None] * size + rows[None, :]]
    kyy = flat[cols[:, None] * size + cols[None, :]]
    kxy = flat[rows[:, None] * size + cols[None, :]]
    sum_xx = (kxx.sum() - flat[rows * size + rows].sum()) / (m * (m - 1))
    sum_yy = (kyy.sum() - flat[cols * size + cols].sum()) / (n * (n - 1))
    sum_xy = kxy.sum() / (m * n)
    return math.sqrt(max(float(sum_xx + sum_yy - 2 * sum_xy), 0.0))


def mmd_permutation_statistic_fn(
    kernel_matrix: np.ndarray, partition
) -> float:
    """statistic_fn closure body reused by both permutation test and bootstrap."""

    return _mmd_from_kernel(
        kernel_matrix, partition.v1_indices, partition.v2_indices
    )


def compute_mmd(
    v1_vectors: np.ndarray,
    v2_vectors: np.ndarray,
    v1_ids: Sequence[str],
    v2_ids: Sequence[str],
    *,
    cfg: AnalysisCfg,
    fixed_bandwidth: float | None,
    seed: int,
) -> MMDResult:
    """Unbiased U-statistic MMD point estimate (diagonal excluded)."""
    first = np.asarray(v1_vectors, dtype=np.float64)
    second = np.asarray(v2_vectors, dtype=np.float64)

    sample = hash_threshold_sample(seed, cfg.sample_size, list(v1_ids), list(v2_ids))
    sample_v1 = to_sample_info(sample, side="v1")
    sample_v2 = to_sample_info(sample, side="v2")

    v1_sub = _subset(first, v1_ids, sample.v1_ids)
    v2_sub = _subset(second, v2_ids, sample.v2_ids)

    m = int(v1_sub.shape[0])
    n = int(v2_sub.shape[0])
    if m < 2 or n < 2:
        return MMDResult(
            mmd=None,
            mmd_squared=None,
            sigma=None,
            estimator="unbiased",
            reason="INSUFFICIENT_DATA",
            sample_v1=sample_v1,
            sample_v2=sample_v2,
        )

    if fixed_bandwidth is not None:
        sigma = float(fixed_bandwidth)
    else:
        pooled = np.vstack([v1_sub, v2_sub])
        sigma = median_heuristic_bandwidth(pooled, seed=seed)

    if sigma == 0.0:
        if _vector_set(v1_sub) == _vector_set(v2_sub):
            return MMDResult(
                mmd=0.0,
                mmd_squared=0.0,
                sigma=0.0,
                estimator="unbiased",
                reason=None,
                sample_v1=sample_v1,
                sample_v2=sample_v2,
            )
        return MMDResult(
            mmd=None,
            mmd_squared=None,
            sigma=0.0,
            estimator="unbiased",
            reason="DEGENERATE_KERNEL",
            sample_v1=sample_v1,
            sample_v2=sample_v2,
        )

    kxx = rbf_kernel_matrix(v1_sub, v1_sub, sigma)
    kyy = rbf_kernel_matrix(v2_sub, v2_sub, sigma)
    kxy = rbf_kernel_matrix(v1_sub, v2_sub, sigma)

    sum_xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    sum_yy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    sum_xy = kxy.sum() / (m * n)

    mmd_squared_raw = sum_xx + sum_yy - 2 * sum_xy
    mmd_squared = max(float(mmd_squared_raw), 0.0)
    mmd = math.sqrt(mmd_squared)

    return MMDResult(
        mmd=round9(mmd),
        mmd_squared=round9(mmd_squared),
        mmd_squared_raw=float(mmd_squared_raw),
        sigma=round9(sigma),
        estimator="unbiased",
        reason=None,
        sample_v1=sample_v1,
        sample_v2=sample_v2,
    )


def _analysis_vectors(
    snapshot: Snapshot, *, normalize: bool
) -> tuple[np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    for record in snapshot.records:
        if record.embedding.validity != "ok" or record.embedding.vector is None:
            continue
        vector = np.asarray(record.embedding.vector, dtype=np.float64)
        if normalize:
            vector = vector / float(np.linalg.norm(vector))
        vectors.append(vector)
        ids.append(record.record_id)
    if not vectors:
        return np.empty((0, 0), dtype=np.float64), []
    return np.vstack(vectors), ids


def _mmd_significance(
    v1: Snapshot,
    v2: Snapshot,
    v1_vectors: np.ndarray,
    v2_vectors: np.ndarray,
    v1_ids: Sequence[str],
    v2_ids: Sequence[str],
    *,
    cfg: Config,
    sigma: float,
    observed: float,
) -> SignificanceResult:
    """Build one combined kernel and run the seeded permutation/bootstrap test."""

    sample = hash_threshold_sample(
        cfg.analysis.seed, cfg.analysis.sample_size, list(v1_ids), list(v2_ids)
    )
    kept_v1 = [record_id for record_id in v1_ids if record_id in sample.v1_ids]
    kept_v2 = [record_id for record_id in v2_ids if record_id in sample.v2_ids]
    if len(kept_v1) > cfg.significance.max_n or len(kept_v2) > cfg.significance.max_n:
        deeper = nested_subsample(
            cfg.analysis.seed, cfg.significance.max_n, kept_v1, kept_v2
        )
        kept_v1 = [record_id for record_id in kept_v1 if record_id in deeper.v1_ids]
        kept_v2 = [record_id for record_id in kept_v2 if record_id in deeper.v2_ids]

    first = _subset(np.asarray(v1_vectors, dtype=np.float64), v1_ids, frozenset(kept_v1))
    second = _subset(np.asarray(v2_vectors, dtype=np.float64), v2_ids, frozenset(kept_v2))
    kernel = rbf_kernel_matrix(np.vstack([first, second]), np.vstack([first, second]), float(sigma))

    id_confidence: dict[str, str] = {}
    for record in list(v1.records) + list(v2.records):
        if record.identity_confidence is not None:
            id_confidence[record.record_id] = record.identity_confidence

    return compute_significance(
        v1_ids=kept_v1,
        v2_ids=kept_v2,
        id_confidence=id_confidence,
        statistic_fn=lambda partition: mmd_permutation_statistic_fn(kernel, partition),
        observed=observed,
        cfg=cfg.significance,
        seed=cfg.analysis.seed,
    )


def compare_mmd(
    v1: Snapshot,
    v2: Snapshot,
    *,
    compat: CompatibilityResult,
    v1_summary: EmbeddingSnapshotSummary,
    v2_summary: EmbeddingSnapshotSummary,
    cfg: Config,
    sink: DiagnosticSink,
) -> Metric:
    """MMD metric for the pair, honoring compatibility suspension."""
    if "mmd" in ANALYSES_SUSPENDED_BY_STATE[compat.state]:
        reason_code = suspension_reason_code(compat.state)
        reason = reason_code.value if reason_code is not None else compat.state
        return make_metric(
            "mmd", value=None, threshold=cfg.drift["mmd"], reason=reason
        )

    should_normalize = (
        v1_summary.similarity_metric == "cosine"
        and v2_summary.similarity_metric == "cosine"
    )
    v1_vectors, v1_ids = _analysis_vectors(v1, normalize=should_normalize)
    v2_vectors, v2_ids = _analysis_vectors(v2, normalize=should_normalize)

    result = compute_mmd(
        v1_vectors,
        v2_vectors,
        v1_ids,
        v2_ids,
        cfg=cfg.analysis,
        fixed_bandwidth=cfg.analysis.bandwidth,
        seed=cfg.analysis.seed,
    )

    approximate = bool(
        getattr(v1, "vectors_approximate", False)
        or getattr(v2, "vectors_approximate", False)
    )
    details: dict[str, object] = {
        "sigma": result.sigma,
        "estimator": result.estimator,
    }
    if (
        getattr(result, "mmd_squared_raw", None) is not None
        and result.mmd_squared == 0.0
    ):
        details["mmd_squared_raw"] = result.mmd_squared_raw
        details["clamped_to_zero"] = True
    if result.reason is None and result.mmd is not None and result.sigma:
        significance = _mmd_significance(
            v1,
            v2,
            v1_vectors,
            v2_vectors,
            v1_ids,
            v2_ids,
            cfg=cfg,
            sigma=float(result.sigma),
            observed=result.mmd,
        )
        details["p_value"] = significance.p_value
        details["test_design"] = significance.design
        details["ci_95"] = [significance.ci_low, significance.ci_high]
    return make_metric(
        "mmd",
        value=result.mmd,
        threshold=cfg.drift["mmd"],
        reason=result.reason,
        sample=result.sample_v1,
        details=details,
        approximate=approximate,
    )
