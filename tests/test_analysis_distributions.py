"""Distribution distances and binning (Step 24)."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from driftlens.analysis.distributions import (
    FIXED_BIN_EDGES,
    js_divergence_categorical,
    ks_statistic,
    mean_median_variance_quantiles,
    numeric_js_bin_edges,
    wasserstein1_normalized,
)
from driftlens.analysis.health import round9


def test_ks_empty_sample() -> None:
    assert ks_statistic([], [1, 2, 3]) == (None, "EMPTY_SAMPLE")


def test_ks_identical_distributions() -> None:
    assert ks_statistic([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == (0.0, None)


def test_ks_disjoint_distributions() -> None:
    assert ks_statistic([1, 1, 1], [100, 100, 100]) == (1.0, None)


def test_wasserstein_v1_zero_variance_uses_pooled() -> None:
    v1 = [5, 5, 5, 5]
    v2 = [1, 2, 3, 4, 5]
    value, reason = wasserstein1_normalized(v1, v2)
    assert value is not None
    assert reason is None
    pooled_std = math.sqrt(
        (len(v1) * float(np.var(v1, ddof=0)) + len(v2) * float(np.var(v2, ddof=0)))
        / (len(v1) + len(v2))
    )
    expected = round9(float(stats.wasserstein_distance(v1, v2)) / pooled_std)
    assert value == expected


def test_wasserstein_both_constant_equal() -> None:
    assert wasserstein1_normalized([7, 7, 7], [7, 7, 7]) == (0.0, None)


def test_wasserstein_both_constant_unequal() -> None:
    assert wasserstein1_normalized([7, 7, 7], [9, 9, 9]) == (None, "CONSTANT_SHIFT")


def test_js_categorical_identical() -> None:
    assert js_divergence_categorical({"a": 10, "b": 10}, {"a": 5, "b": 5}) == (
        0.0,
        None,
    )


def test_js_categorical_disjoint_single() -> None:
    assert js_divergence_categorical({"a": 10}, {"b": 10}) == (1.0, None)


def test_js_categorical_both_empty() -> None:
    assert js_divergence_categorical({}, {}) == (None, "EMPTY_SAMPLE")


def test_js_categorical_one_sided_empty_is_not_empty_sample() -> None:
    assert js_divergence_categorical({"a": 10}, {}) != (None, "EMPTY_SAMPLE")
    assert js_divergence_categorical({"a": 10}, {}) == (1.0, None)


def test_numeric_js_bin_edges_constant_data() -> None:
    assert numeric_js_bin_edges([5, 5, 5, 5]) == [5, 5 + 1e-9]


def test_numeric_js_bin_edges_fd_normal() -> None:
    pooled = list(range(1, 101))
    edges = numeric_js_bin_edges(pooled, max_bins=100)
    iqr = float(np.percentile(pooled, 75) - np.percentile(pooled, 25))
    fd_width = 2 * iqr / (len(pooled) ** (1 / 3))
    expected_bins = min(
        100, max(1, math.ceil((max(pooled) - min(pooled)) / fd_width))
    )
    assert len(edges) == expected_bins + 1

    capped = numeric_js_bin_edges(pooled, max_bins=3)
    assert len(capped) <= 4


def test_fixed_bin_edges_shape() -> None:
    assert FIXED_BIN_EDGES[0] == 0
    assert FIXED_BIN_EDGES[-1] == math.inf
    inner = FIXED_BIN_EDGES[1:-1]
    assert inner[0] == 1
    assert inner[-1] == 1048576
    assert all(value == 2 ** index for index, value in enumerate(inner))


def test_mean_median_variance_quantiles_empty() -> None:
    assert mean_median_variance_quantiles([]) is None


def test_round9_applied_everywhere() -> None:
    value, _reason = ks_statistic([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert value is not None
    assert value == float(f"{value:.9g}")
