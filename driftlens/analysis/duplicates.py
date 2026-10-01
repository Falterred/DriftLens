"""Exact-duplicate grouping (Step 29, v0.1 exact stage only)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from driftlens.analysis.health import Metric, SampleInfo, make_metric
from driftlens.analysis.sampling import hash_threshold_sample, to_sample_info
from driftlens.config.schema import DuplicatesCfg, Threshold
from driftlens.schema.record import Record
from driftlens.schema.snapshot import Snapshot


@dataclass
class DuplicateUnit:
    unit_id: str
    representative: Record
    members: list[Record]


@dataclass
class DuplicateGroup:
    representative_id: str
    member_ids: list[str]
    class_: Literal["exact", "near", "semantic"]


@dataclass
class ExactDuplicateResult:
    groups: list[DuplicateGroup]
    units: list[DuplicateUnit]
    redundant_count: int
    total_chunks: int
    duplicate_rate: float
    covered_fraction: float
    sample: SampleInfo | None


def _grouping_key(record: Record) -> str | None:
    if record.text is not None:
        return record.normalized_text_hash
    digest = getattr(record, "_embedding_digest_hex", None)
    if digest is not None:
        return digest
    return None


def compute_exact_groups(
    records: Sequence[Record], *, cfg: DuplicatesCfg, seed: int
) -> ExactDuplicateResult:
    """Group records by their exact text/embedding hash (stage 1 only)."""
    total_chunks = len(records)

    sample: SampleInfo | None = None
    if total_chunks > cfg.max_exact_vectors:
        sampled = hash_threshold_sample(
            seed,
            cfg.max_exact_vectors,
            [record.record_id for record in records],
            v2_record_ids=None,
        )
        considered = [
            record for record in records if record.record_id in sampled.v1_ids
        ]
        covered_fraction = len(considered) / total_chunks
        sample = to_sample_info(sampled)
    else:
        considered = list(records)
        covered_fraction = 1.0

    buckets: dict[str, list[Record]] = defaultdict(list)
    for record in considered:
        key = _grouping_key(record)
        if key is not None:
            buckets[key].append(record)

    groups: list[DuplicateGroup] = []
    units: list[DuplicateUnit] = []
    for members in buckets.values():
        ordered = sorted(members, key=lambda record: record.record_id)
        representative = ordered[0]
        units.append(
            DuplicateUnit(
                unit_id=representative.record_id,
                representative=representative,
                members=ordered,
            )
        )
        if len(ordered) >= 2:
            groups.append(
                DuplicateGroup(
                    representative_id=representative.record_id,
                    member_ids=[record.record_id for record in ordered],
                    class_="exact",
                )
            )

    units.sort(key=lambda unit: unit.unit_id)
    groups.sort(key=lambda group: group.representative_id)

    redundant_count = sum(len(group.member_ids) - 1 for group in groups)
    duplicate_rate = redundant_count / total_chunks if total_chunks else 0.0

    return ExactDuplicateResult(
        groups=groups,
        units=units,
        redundant_count=redundant_count,
        total_chunks=total_chunks,
        duplicate_rate=duplicate_rate,
        covered_fraction=covered_fraction,
        sample=sample,
    )


def _count_metric(name: str, v1_value, v2_value):
    return make_metric(
        name,
        value=v2_value,
        threshold=None,
        v1=v1_value,
        v2=v2_value,
        change=v2_value - v1_value,
    )


def compare_duplicates(
    v1: Snapshot,
    v2: Snapshot,
    *,
    cfg: DuplicatesCfg,
    drift_cfg: dict[str, Threshold | None],
    seed: int = 42,
) -> tuple[ExactDuplicateResult, ExactDuplicateResult, list[Metric]]:
    """Exact-duplicate rates for both snapshots plus the DUPLICATION metrics."""
    v1_result = compute_exact_groups(v1.records, cfg=cfg, seed=seed)
    v2_result = compute_exact_groups(v2.records, cfg=cfg, seed=seed)

    rate_change = v2_result.duplicate_rate - v1_result.duplicate_rate
    metrics: list[Metric] = [
        make_metric(
            "duplicate_rate",
            value=v2_result.duplicate_rate,
            threshold=drift_cfg.get("duplicate_rate"),
            v1=v1_result.duplicate_rate,
            v2=v2_result.duplicate_rate,
            change=rate_change,
        ),
        make_metric(
            "duplicate_rate_exact",
            value=v2_result.duplicate_rate,
            threshold=drift_cfg.get("duplicate_rate_exact"),
            v1=v1_result.duplicate_rate,
            v2=v2_result.duplicate_rate,
            change=rate_change,
        ),
        make_metric(
            "duplicate_rate_near",
            value=None,
            threshold=drift_cfg.get("duplicate_rate_near"),
        ),
        make_metric(
            "duplicate_rate_semantic",
            value=None,
            threshold=drift_cfg.get("duplicate_rate_semantic"),
        ),
        _count_metric(
            "duplicate_groups",
            len(v1_result.groups),
            len(v2_result.groups),
        ),
        _count_metric(
            "redundant_chunks",
            v1_result.redundant_count,
            v2_result.redundant_count,
        ),
        make_metric(
            "duplicate_covered_fraction",
            value=v2_result.covered_fraction,
            threshold=None,
            v1=v1_result.covered_fraction,
            v2=v2_result.covered_fraction,
            change=v2_result.covered_fraction - v1_result.covered_fraction,
        ),
    ]
    return v1_result, v2_result, metrics
