"""Temporal and freshness-shift analysis (Step 44, section 5.4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from driftlens.analysis.content import DocumentDiffRow
from driftlens.analysis.distributions import ks_statistic
from driftlens.analysis.health import Metric, make_metric
from driftlens.errors import Code, DiagnosticSink
from driftlens.schema.record import Record, canonicalize_timestamp
from driftlens.schema.snapshot import Snapshot

FreshnessField = Literal["created_at", "updated_at"]

_TIMESTAMP_STATUSES: frozenset[str] = frozenset(
    {"UNCHANGED", "FORMAT_ONLY_CHANGE", "MODIFIED", "REPLACED"}
)
_CONTENT_CHANGED_STATUSES: frozenset[str] = frozenset({"MODIFIED", "REPLACED"})


@dataclass
class FreshnessShiftResult:
    """Freshness-shift KS outcome for one timestamp field (section 5.4)."""

    ks_d: float | None
    reason: str | None
    field: Literal["created_at", "updated_at"]


def _to_epoch_seconds(value: str) -> float:
    """Canonical ISO timestamp to POSIX seconds (Step 12 Z-normalization)."""

    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def compute_as_of(records: Sequence[Record]) -> str | None:
    """Pooled max of every non-None created_at/updated_at; None if none exist.

    Both fields are stored as fixed-width canonical UTC ISO strings (Step 12),
    so lexicographic max is chronological max.
    """

    values = [record.created_at for record in records if record.created_at]
    values += [record.updated_at for record in records if record.updated_at]
    return max(values) if values else None


def compute_freshness_shift(
    v1_records: Sequence[Record],
    v2_records: Sequence[Record],
    *,
    field: Literal["created_at", "updated_at"],
    as_of_override: str | None,
) -> FreshnessShiftResult:
    """KS D between v1 and v2 age distributions for one timestamp field.

    Ages are as_of_seconds minus record timestamp seconds.  Without an
    override each snapshot gets its own independently computed as-of, so this
    is a snapshot-relative comparison unless --as-of anchors both sides.
    """

    as_of_v1 = as_of_override or compute_as_of(v1_records)
    as_of_v2 = as_of_override or compute_as_of(v2_records)
    if as_of_v1 is None or as_of_v2 is None:
        return FreshnessShiftResult(None, "EMPTY_SAMPLE", field)

    as_of_v1_sec = _to_epoch_seconds(as_of_v1)
    as_of_v2_sec = _to_epoch_seconds(as_of_v2)
    ages_v1 = [
        as_of_v1_sec - _to_epoch_seconds(getattr(record, field))
        for record in v1_records
        if getattr(record, field)
    ]
    ages_v2 = [
        as_of_v2_sec - _to_epoch_seconds(getattr(record, field))
        for record in v2_records
        if getattr(record, field)
    ]
    d, reason = ks_statistic(ages_v1, ages_v2)
    return FreshnessShiftResult(d, reason, field)


def detect_future_timestamps(
    records: Sequence[Record],
    *,
    as_of_override: str | None,
) -> list[str]:
    """Record ids whose created_at or updated_at exceeds an explicit --as-of.

    With no override this is a documented no-op: the default as-of is the
    pooled max observed timestamp, so nothing can exceed it.
    """

    if as_of_override is None:
        return []
    cutoff = _to_epoch_seconds(
        canonicalize_timestamp(as_of_override, unit="iso", location="--as-of")
    )
    flagged: list[str] = []
    for record in records:
        if record.created_at and _to_epoch_seconds(record.created_at) > cutoff:
            flagged.append(record.record_id)
            continue
        if record.updated_at and _to_epoch_seconds(record.updated_at) > cutoff:
            flagged.append(record.record_id)
    return flagged


def _document_updated_at(records: Sequence[Record]) -> str | None:
    """Document-level updated_at as the max across its chunks (Step 20 convention)."""

    return max((record.updated_at for record in records if record.updated_at), default=None)


def compute_timestamp_inconsistent_flags(
    document_diff: list[DocumentDiffRow],
    chunk_match: "FullChunkMatchResult",
    v1_records_by_doc: dict[str, list[Record]],
    v2_records_by_doc: dict[str, list[Record]],
) -> set[str]:
    """Document ids whose timestamp_inconsistent flag should be True (5.4).

    Flagged when content changed but updated_at did not, or updated_at changed
    while content is unchanged.  ADDED/REMOVED documents are excluded.
    """

    _ = chunk_match
    flagged: set[str] = set()
    for row in document_diff:
        if row.status not in _TIMESTAMP_STATUSES:
            continue
        updated_at_v1 = _document_updated_at(v1_records_by_doc.get(row.document_id, []))
        updated_at_v2 = _document_updated_at(v2_records_by_doc.get(row.document_id, []))
        content_changed = row.status in _CONTENT_CHANGED_STATUSES
        updated_at_changed = updated_at_v1 != updated_at_v2
        if (content_changed and not updated_at_changed) or (
            not content_changed and updated_at_changed
        ):
            flagged.add(row.document_id)
    return flagged


def compare_temporal(
    v1: Snapshot,
    v2: Snapshot,
    *,
    as_of_override: str | None,
    sink: DiagnosticSink,
) -> list[Metric]:
    """Report-only freshness KS metrics and FUTURE_TIMESTAMP warnings (5.4)."""

    created_shift = compute_freshness_shift(
        v1.records, v2.records, field="created_at", as_of_override=as_of_override
    )
    updated_shift = compute_freshness_shift(
        v1.records, v2.records, field="updated_at", as_of_override=as_of_override
    )
    metrics = [
        make_metric(
            "freshness_ks_created_at",
            value=created_shift.ks_d,
            threshold=None,
            reason=created_shift.reason,
        ),
        make_metric(
            "freshness_ks_updated_at",
            value=updated_shift.ks_d,
            threshold=None,
            reason=updated_shift.reason,
        ),
    ]

    future_v1 = detect_future_timestamps(v1.records, as_of_override=as_of_override)
    future_v2 = detect_future_timestamps(v2.records, as_of_override=as_of_override)
    if future_v1:
        sink.add(
            Code.FUTURE_TIMESTAMP,
            f"{len(future_v1)} record(s) in v1",
            count=len(future_v1),
        )
    if future_v2:
        sink.add(
            Code.FUTURE_TIMESTAMP,
            f"{len(future_v2)} record(s) in v2",
            count=len(future_v2),
        )
    return metrics
