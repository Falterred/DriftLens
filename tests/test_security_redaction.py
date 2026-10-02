"""Snapshot-scoped identifier redaction (Step 41)."""

from __future__ import annotations

from driftlens.analysis.health import Metric
from driftlens.security.redaction import (
    apply_redaction_to_metric,
    redact_id,
    redact_label,
)


def test_redact_deterministic_within_snapshot() -> None:
    first = redact_id("doc-1", snapshot_id="snap-a")
    second = redact_id("doc-1", snapshot_id="snap-a")
    assert first == second
    assert first != "doc-1"
    assert len(first) == 12
    # redact_label shares the construction so id/label pairs stay joinable.
    assert redact_label("doc-1", snapshot_id="snap-a") == first


def test_redact_differs_across_snapshots() -> None:
    assert redact_id("doc-1", snapshot_id="snap-a") != redact_id(
        "doc-1", snapshot_id="snap-b"
    )


def test_apply_redaction_noop_when_not_requested() -> None:
    metric = Metric(
        name="duplication:largest_group",
        v1="doc-1",
        v2="doc-2",
        is_identifier_valued=True,
        identifier_kind="id",
    )
    result = apply_redaction_to_metric(metric, snapshot_id="snap-a", redact=False)
    assert result is metric


def test_apply_redaction_skips_non_identifier_metrics() -> None:
    metric = Metric(name="content:changed_count", v1=3.0, v2=4.0)
    result = apply_redaction_to_metric(metric, snapshot_id="snap-a", redact=True)
    assert result is metric
    assert result.v1 == 3.0
    assert result.v2 == 4.0
