"""Deterministic hash-threshold sampling (Step 23)."""

from __future__ import annotations

import random

from driftlens.analysis.health import SampleInfo
from driftlens.analysis.sampling import (
    hash_threshold_sample,
    nested_subsample,
    to_sample_info,
)


def _ids(count: int) -> list[str]:
    return [f"id{index}" for index in range(count)]


def test_deterministic_repeatable() -> None:
    ids = _ids(1000)
    first = hash_threshold_sample(42, 100, ids)
    second = hash_threshold_sample(42, 100, ids)
    assert first.v1_ids == second.v1_ids


def test_order_independent() -> None:
    ids = _ids(1000)
    shuffled = list(ids)
    random.Random(7).shuffle(shuffled)
    assert hash_threshold_sample(42, 100, ids).v1_ids == hash_threshold_sample(
        42, 100, shuffled
    ).v1_ids


def test_seed_sensitivity() -> None:
    ids = _ids(1000)
    assert hash_threshold_sample(42, 50, ids).v1_ids != hash_threshold_sample(
        43, 50, ids
    ).v1_ids


def test_small_population_keeps_everything() -> None:
    ids = _ids(10)
    result = hash_threshold_sample(42, 100, ids)
    assert result.v1_ids == frozenset(ids)
    assert result.tau == 2 ** 256


def test_at_most_k_records() -> None:
    ids = _ids(10000)
    result = hash_threshold_sample(42, 100, ids)
    assert len(result.v1_ids) <= 100
    assert len(result.v1_ids) == 100


def test_pairing_survives_both_snapshots() -> None:
    v1_ids = ["a", "b", "c", "d", "e"]
    v2_ids = ["a", "b", "x", "y"]
    shared = set(v1_ids) & set(v2_ids)

    result = None
    for seed in range(1000):
        candidate = hash_threshold_sample(seed, 3, v1_ids, v2_ids)
        if shared & candidate.v1_ids:
            result = candidate
            break
    assert result is not None
    assert shared & result.v1_ids

    for record_id in shared:
        assert (record_id in result.v1_ids) == (record_id in result.v2_ids)


def test_nesting_property() -> None:
    ids = _ids(1000)
    larger = hash_threshold_sample(42, 500, ids)
    smaller = nested_subsample(42, 50, ids)
    assert smaller.v1_ids <= larger.v1_ids


def test_single_snapshot_mode() -> None:
    ids = _ids(1000)
    result = hash_threshold_sample(42, 100, ids, v2_record_ids=None)
    assert result.v2_ids == frozenset()
    assert result.v2_total == 0


def test_to_sample_info_shape() -> None:
    result = hash_threshold_sample(42, 100, _ids(1000), _ids(800))
    info = to_sample_info(result)
    assert isinstance(info, SampleInfo)
    assert info.v1_n == len(result.v1_ids)
    assert info.v1_of == result.v1_total
    assert info.v2_n == len(result.v2_ids)
    assert info.v2_of == result.v2_total
    assert info.seed == result.seed
