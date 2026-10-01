"""Item 6: section 8.8 degenerate-data table coverage (Step 34)."""

from __future__ import annotations

import numpy as np

from driftlens.analysis.distributions import (
    js_divergence_categorical,
    ks_statistic,
    wasserstein1_normalized,
)
from driftlens.analysis.mmd import compute_mmd
from driftlens.config.schema import AnalysisCfg

EXPECTED_ROWS = [
    "zero_variance_v1_uses_pooled",
    "both_constant_equal",
    "both_constant_unequal",
    "one_category_equal",
    "one_category_different",
    "empty_sample",
    "fewer_than_2_vectors",
    "zero_sigma_identical",
    "zero_sigma_different",
]

DEGENERATE_ROWS = {
    "zero_variance_v1_uses_pooled": "test_degenerate_zero_variance_v1_uses_pooled",
    "both_constant_equal": "test_degenerate_both_constant_equal",
    "both_constant_unequal": "test_degenerate_both_constant_unequal",
    "one_category_equal": "test_degenerate_one_category_equal",
    "one_category_different": "test_degenerate_one_category_different",
    "empty_sample": "test_degenerate_empty_sample",
    "fewer_than_2_vectors": "test_degenerate_fewer_than_2_vectors",
    "zero_sigma_identical": "test_degenerate_zero_sigma_identical",
    "zero_sigma_different": "test_degenerate_zero_sigma_different",
}


def test_degenerate_zero_variance_v1_uses_pooled() -> None:
    value, reason = wasserstein1_normalized([5, 5, 5, 5], [1, 2, 3, 4, 5])
    assert value is not None
    assert reason is None


def test_degenerate_both_constant_equal() -> None:
    assert wasserstein1_normalized([7, 7, 7], [7, 7, 7]) == (0.0, None)


def test_degenerate_both_constant_unequal() -> None:
    assert wasserstein1_normalized([7, 7, 7], [9, 9, 9]) == (None, "CONSTANT_SHIFT")


def test_degenerate_one_category_equal() -> None:
    assert js_divergence_categorical({"a": 10}, {"a": 5}) == (0.0, None)


def test_degenerate_one_category_different() -> None:
    assert js_divergence_categorical({"a": 10}, {"b": 10}) == (1.0, None)


def test_degenerate_empty_sample() -> None:
    assert ks_statistic([], [1, 2, 3]) == (None, "EMPTY_SAMPLE")


def test_degenerate_fewer_than_2_vectors() -> None:
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


def test_degenerate_zero_sigma_identical() -> None:
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


def test_degenerate_zero_sigma_different() -> None:
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


def test_degenerate_suite_covers_every_row() -> None:
    assert set(DEGENERATE_ROWS) == set(EXPECTED_ROWS)
    for test_name in DEGENERATE_ROWS.values():
        assert test_name in globals()
