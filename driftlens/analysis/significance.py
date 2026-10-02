"""Permutation tests and bootstrap confidence intervals (Step 45, sections 8.7/8.8)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from driftlens.analysis.sampling import nested_subsample
from driftlens.config.schema import SignificanceCfg

PermutationDesign = Literal["PAIRING_AWARE", "TWO_SAMPLE"]

_MIN_PERMUTATION_VECTORS = 20
_MIN_BOOTSTRAP_VECTORS = 30
_SKIPPED_REASON = "SIGNIFICANCE_SKIPPED"


@dataclass
class Partition:
    """Positions into a pooled combined array (v1 first, then v2)."""

    v1_indices: list[int]
    v2_indices: list[int]


@dataclass
class SignificanceResult:
    """Significance outcome attached to a drift metric (section 8.7)."""

    p_value: float | None
    design: PermutationDesign
    b: int
    ci_low: float | None
    ci_high: float | None
    reason: str | None


def choose_permutation_design(
    v1_ids: Sequence[str],
    v2_ids: Sequence[str],
    id_confidence: dict[str, str],
) -> tuple[PermutationDesign, set[str]]:
    """Pairing-aware only when at least one shared id is non-LOW confidence."""

    shared = set(v1_ids) & set(v2_ids)
    trustworthy = {
        record_id for record_id in shared if id_confidence.get(record_id) != "LOW"
    }
    if trustworthy:
        return ("PAIRING_AWARE", trustworthy)
    return ("TWO_SAMPLE", set())


def generate_permutation(
    design: PermutationDesign,
    *,
    n_v1: int,
    n_v2: int,
    paired_positions: list[tuple[int, int]],
    unpaired_v1_positions: list[int],
    unpaired_v2_positions: list[int],
    rng: np.random.Generator,
) -> Partition:
    """Draw one random relabeling that preserves the original group sizes."""

    if design == "PAIRING_AWARE":
        v1_indices: list[int] = []
        v2_indices: list[int] = []
        for pos_v1, pos_v2 in paired_positions:
            if rng.random() < 0.5:
                v1_indices.append(pos_v2)
                v2_indices.append(pos_v1)
            else:
                v1_indices.append(pos_v1)
                v2_indices.append(pos_v2)
        pooled = list(unpaired_v1_positions) + list(unpaired_v2_positions)
        rng.shuffle(pooled)
        split = len(unpaired_v1_positions)
        v1_indices.extend(pooled[:split])
        v2_indices.extend(pooled[split:])
        return Partition(v1_indices=v1_indices, v2_indices=v2_indices)

    all_positions = list(range(n_v1 + n_v2))
    rng.shuffle(all_positions)
    return Partition(
        v1_indices=all_positions[:n_v1], v2_indices=all_positions[n_v1:]
    )


def permutation_test(
    *,
    n_v1: int,
    n_v2: int,
    statistic_fn: Callable[[Partition], float],
    observed_partition: Partition,
    design: PermutationDesign,
    paired_positions: list[tuple[int, int]],
    unpaired_v1_positions: list[int],
    unpaired_v2_positions: list[int],
    cfg: SignificanceCfg,
    seed: int,
) -> float | None:
    """One-sided permutation p-value p = (1 + count_ge) / (1 + B)."""

    rng = np.random.default_rng(seed)
    observed_stat = float(statistic_fn(observed_partition))
    if not math.isfinite(observed_stat):
        return None

    count_ge = 0
    for _ in range(cfg.permutations):
        permuted = generate_permutation(
            design,
            n_v1=n_v1,
            n_v2=n_v2,
            paired_positions=paired_positions,
            unpaired_v1_positions=unpaired_v1_positions,
            unpaired_v2_positions=unpaired_v2_positions,
            rng=rng,
        )
        if float(statistic_fn(permuted)) >= observed_stat:
            count_ge += 1
    return (1 + count_ge) / (1 + cfg.permutations)


def bootstrap_ci(
    *,
    statistic_fn: Callable[[Partition], float],
    n_v1: int,
    n_v2: int,
    cfg: SignificanceCfg,
    seed: int,
) -> tuple[float, float] | None:
    """Percentile bootstrap 95% CI, resampling each side independently."""

    if n_v1 < 1 or n_v2 < 1:
        return None
    rng = np.random.default_rng(seed + 1)
    values: list[float] = []
    for _ in range(cfg.bootstrap):
        v1_indices = [int(index) for index in rng.integers(0, n_v1, size=n_v1)]
        v2_indices = [int(index) for index in rng.integers(0, n_v2, size=n_v2)]
        values.append(
            float(
                statistic_fn(
                    Partition(v1_indices=v1_indices, v2_indices=v2_indices)
                )
            )
        )
    if not values:
        return None
    low = float(np.percentile(values, 2.5))
    high = float(np.percentile(values, 97.5))
    return (low, high)


def compute_significance(
    *,
    v1_ids: Sequence[str],
    v2_ids: Sequence[str],
    id_confidence: dict[str, str],
    statistic_fn: Callable[[Partition], float],
    observed: float,
    cfg: SignificanceCfg,
    seed: int,
) -> SignificanceResult:
    """Budget check, design choice, permutation test, and bootstrap CI (8.7)."""

    del observed  # the observed statistic is recomputed by permutation_test
    first_ids = list(v1_ids)
    second_ids = list(v2_ids)
    if len(first_ids) > cfg.max_n or len(second_ids) > cfg.max_n:
        sample = nested_subsample(seed, cfg.max_n, first_ids, second_ids)
        first_ids = [record_id for record_id in first_ids if record_id in sample.v1_ids]
        second_ids = [record_id for record_id in second_ids if record_id in sample.v2_ids]
        id_confidence = {
            record_id: confidence
            for record_id, confidence in id_confidence.items()
            if record_id in sample.v1_ids or record_id in sample.v2_ids
        }

    design, paired_ids = choose_permutation_design(
        first_ids, second_ids, id_confidence
    )
    n_v1 = len(first_ids)
    n_v2 = len(second_ids)
    if n_v1 < _MIN_PERMUTATION_VECTORS or n_v2 < _MIN_PERMUTATION_VECTORS:
        return SignificanceResult(
            p_value=None,
            design=design,
            b=0,
            ci_low=None,
            ci_high=None,
            reason=_SKIPPED_REASON,
        )

    v2_positions = {
        record_id: index for index, record_id in enumerate(second_ids)
    }
    paired_positions: list[tuple[int, int]] = []
    unpaired_v1_positions: list[int] = []
    for index, record_id in enumerate(first_ids):
        if record_id in paired_ids:
            paired_positions.append((index, v2_positions[record_id]))
        else:
            unpaired_v1_positions.append(index)
    unpaired_v2_positions = [
        index
        for index, record_id in enumerate(second_ids)
        if record_id not in paired_ids
    ]

    observed_partition = Partition(
        v1_indices=list(range(n_v1)),
        v2_indices=list(range(n_v1, n_v1 + n_v2)),
    )
    p_value = permutation_test(
        n_v1=n_v1,
        n_v2=n_v2,
        statistic_fn=statistic_fn,
        observed_partition=observed_partition,
        design=design,
        paired_positions=paired_positions,
        unpaired_v1_positions=unpaired_v1_positions,
        unpaired_v2_positions=unpaired_v2_positions,
        cfg=cfg,
        seed=seed,
    )

    ci: tuple[float, float] | None = None
    if n_v1 >= _MIN_BOOTSTRAP_VECTORS and n_v2 >= _MIN_BOOTSTRAP_VECTORS:
        ci = bootstrap_ci(
            statistic_fn=statistic_fn, n_v1=n_v1, n_v2=n_v2, cfg=cfg, seed=seed
        )

    return SignificanceResult(
        p_value=p_value,
        design=design,
        b=cfg.permutations,
        ci_low=ci[0] if ci is not None else None,
        ci_high=ci[1] if ci is not None else None,
        reason=None if p_value is not None else _SKIPPED_REASON,
    )
