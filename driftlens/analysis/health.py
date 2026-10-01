"""Analysis health: metric, dimension and overall status evaluation (Step 21).

This module owns the single shared status computation used by every metric in
the report.  It is pure: it never reads files, snapshots or configuration
beyond the dataclasses it is handed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from driftlens.config.schema import DIMENSIONS, THRESHOLD_KEYS, Config, Threshold

MetricStatus = Literal[
    "PASS",
    "WARN",
    "FAIL",
    "UNBASELINED",
    "SKIPPED",
    "N/A",
    "REPORT_ONLY",
]
DimensionStatus = Literal["PASS", "WARN", "FAIL", "UNBASELINED", "SKIPPED", "N/A"]
OverallStatus = Literal["PASS", "WARN", "FAIL", "UNBASELINED"]

# Severity ordering used when a dimension aggregates its metrics: higher is worse.
# UNBASELINED ranks below PASS, so unbaselined metrics never drag a passing
# dimension down (Inconsistency patch for step 21).
_DIMENSION_RANK = {"FAIL": 3, "WARN": 2, "PASS": 1, "UNBASELINED": 0}


def round9(value: float) -> float:
    """Round to 9 significant digits (PROJECT_ARCHITECTURE.md §5).

    Used for every float-derived discrete decision so a value within float noise
    of exactly a threshold does not flap between PASS/WARN across platforms.
    """
    return float(f"{value:.9g}")


@dataclass
class SampleInfo:
    v1_n: int
    v1_of: int
    v2_n: int
    v2_of: int
    seed: int


@dataclass
class Metric:
    name: str
    v1: float | str | None = None
    v2: float | str | None = None
    change: float | str | None = None
    status: MetricStatus = "N/A"
    reason: str | None = None
    sample: SampleInfo | None = None
    details: dict[str, Any] | None = None
    is_threshold_key: bool = False
    approximate: bool = False


@dataclass
class Dimension:
    name: str
    metrics: list[Metric] = field(default_factory=list)
    status: DimensionStatus = "N/A"


@dataclass
class HealthReport:
    dimensions: dict[str, Dimension]
    overall: OverallStatus
    exceeded: list[str]


def evaluate_status(
    *,
    is_threshold_key: bool,
    value: float | None,
    threshold: Threshold | None,
    reason: str | None,
    borderline: Any | None = None,
) -> MetricStatus:
    """Compute the status tag for a single metric (step 21 implementation logic 1)."""
    if not is_threshold_key:
        return "REPORT_ONLY"
    if reason is not None:
        return "SKIPPED"
    if value is None:
        return "N/A"
    if threshold is None:
        return "UNBASELINED"
    if borderline is not None:
        if threshold.critical is not None:
            borderline.note(value, threshold.critical)
        if threshold.warning is not None:
            borderline.note(value, threshold.warning)
    if threshold.critical is not None and round9(value) > round9(threshold.critical):
        return "FAIL"
    if threshold.warning is not None and round9(value) > round9(threshold.warning):
        return "WARN"
    return "PASS"


def make_metric(
    name: str,
    *,
    value: float | None,
    threshold: Threshold | None,
    reason: str | None = None,
    v1: float | str | None = None,
    v2: float | str | None = None,
    change: float | str | None = None,
    sample: SampleInfo | None = None,
    details: dict[str, Any] | None = None,
    approximate: bool = False,
    is_threshold_key: bool | None = None,
    borderline: Any | None = None,
) -> Metric:
    """Thin constructor deriving is_threshold_key and the status tag."""
    threshold_key = name in THRESHOLD_KEYS if is_threshold_key is None else is_threshold_key
    # A measurement that exists only for the pair (cross-snapshot distances such
    # as MMD) shows its value in the change column with v1/v2 blank (§15.2).
    if change is None and v1 is None and v2 is None:
        change = value
    return Metric(
        name=name,
        v1=v1,
        v2=v2,
        change=change,
        status=evaluate_status(
            is_threshold_key=threshold_key,
            value=value,
            threshold=threshold,
            reason=reason,
            borderline=borderline,
        ),
        reason=reason,
        sample=sample,
        details=details,
        is_threshold_key=threshold_key,
        approximate=approximate,
    )


def dimension_status(metrics: Sequence[Metric]) -> DimensionStatus:
    """Aggregate a dimension's metrics into one status (implementation logic 3)."""
    relevant = [m for m in metrics if m.status not in ("N/A", "REPORT_ONLY")]
    if not relevant:
        return "N/A"
    if all(m.status == "SKIPPED" for m in relevant):
        return "SKIPPED"
    non_skipped = [m for m in relevant if m.status != "SKIPPED"]
    return max(non_skipped, key=lambda m: _DIMENSION_RANK[m.status]).status


def build_health_report(dimensions: dict[str, list[Metric]], *, cfg: Config) -> HealthReport:
    """Build the full report, always in the fixed §12.1 dimension order."""
    ordered = {
        name: Dimension(
            name=name,
            metrics=dimensions.get(name, []),
            status=dimension_status(dimensions.get(name, [])),
        )
        for name in DIMENSIONS
    }
    overall, exceeded = overall_status(ordered)
    return HealthReport(dimensions=ordered, overall=overall, exceeded=exceeded)


def overall_status(
    dimensions: dict[str, Dimension],
) -> tuple[OverallStatus, list[str]]:
    """Compute the overall status and the list of exceeded dimensions (logic 5)."""
    thresholded = [d for d in dimensions.values() if d.status != "N/A"]
    if not thresholded:
        return ("UNBASELINED", [])
    exceeded = [d.name for d in thresholded if d.status in ("WARN", "FAIL")]
    if any(d.status == "FAIL" for d in thresholded):
        overall: OverallStatus = "FAIL"
    elif any(d.status == "WARN" for d in thresholded):
        overall = "WARN"
    elif any(d.status == "PASS" for d in thresholded):
        overall = "PASS"
    else:
        overall = "UNBASELINED"
    return (overall, exceeded)
