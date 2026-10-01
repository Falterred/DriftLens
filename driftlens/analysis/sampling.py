"""Deterministic hash-threshold sampling (Step 23).

One generic primitive is shared by every sampled metric (MMD, duplicate
overflow, significance budgets) so the sample accounting comes from a
single code path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from driftlens.analysis.health import SampleInfo
from driftlens.identity.hashing import sample_hash

# One past the maximum possible 256-bit sample hash, so h <= tau is true for
# every real hash ("no cutoff").
_NO_CUTOFF = 2 ** 256


@dataclass
class SampleResult:
    v1_ids: frozenset[str]
    v2_ids: frozenset[str]
    v1_total: int
    v2_total: int
    tau: int
    seed: int
    k: int


def _sample_one(
    seed: int, k: int, record_ids: Sequence[str]
) -> tuple[dict[str, int], int, int, frozenset[str]]:
    hashes = {record_id: sample_hash(seed, record_id) for record_id in record_ids}
    total = len(record_ids)
    if total <= k:
        tau = _NO_CUTOFF
    else:
        tau = sorted(hashes.values())[k - 1]
    kept = frozenset(record_id for record_id, h in hashes.items() if h <= tau)
    return hashes, total, tau, kept


def hash_threshold_sample(
    seed: int,
    k: int,
    v1_record_ids: Sequence[str],
    v2_record_ids: Sequence[str] | None = None,
) -> SampleResult:
    """Sample at most k records per side with one shared cutoff tau."""
    h1, v1_total, tau1, _keep1 = _sample_one(seed, k, v1_record_ids)

    if v2_record_ids is None:
        keep1 = frozenset(record_id for record_id, h in h1.items() if h <= tau1)
        return SampleResult(
            v1_ids=keep1,
            v2_ids=frozenset(),
            v1_total=v1_total,
            v2_total=0,
            tau=tau1,
            seed=seed,
            k=k,
        )

    h2, v2_total, tau2, _keep2 = _sample_one(seed, k, v2_record_ids)
    tau = min(tau1, tau2)
    keep1 = frozenset(record_id for record_id, h in h1.items() if h <= tau)
    keep2 = frozenset(record_id for record_id, h in h2.items() if h <= tau)
    return SampleResult(
        v1_ids=keep1,
        v2_ids=keep2,
        v1_total=v1_total,
        v2_total=v2_total,
        tau=tau,
        seed=seed,
        k=k,
    )


def nested_subsample(
    seed: int,
    smaller_k: int,
    v1_record_ids: Sequence[str],
    v2_record_ids: Sequence[str] | None = None,
) -> SampleResult:
    """Intent-documenting alias for a smaller-k sample under the same seed."""
    return hash_threshold_sample(seed, smaller_k, v1_record_ids, v2_record_ids)


def to_sample_info(
    result: SampleResult, *, side: Literal["v1", "v2", "both"] = "both"
) -> SampleInfo:
    """Convert a SampleResult into the health.SampleInfo shape."""
    return SampleInfo(
        v1_n=len(result.v1_ids),
        v1_of=result.v1_total,
        v2_n=len(result.v2_ids),
        v2_of=result.v2_total,
        seed=result.seed,
    )
