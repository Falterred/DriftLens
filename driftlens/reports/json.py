"""Machine-readable JSON report rendering (Step 31)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from driftlens.analysis.health import Dimension, HealthReport, Metric
from driftlens.config.schema import PRIVACY_MAX_LABEL_CATEGORIES, Config
from driftlens.errors import DiagnosticSink
from driftlens.schema.manifest import Manifest

# Frozen for DriftLens 1.0 (Step 51): additive fields only may bump the
# minor; any other change requires a new major schema (out of 1.x).
# See docs/schema-versioning.md.
# 1.1: additive - mmd details.mmd_squared_raw/clamped_to_zero, duplicates
# sample annotation + SAMPLED_DUPLICATES reason, optional
# chunks_below_target/chunks_above_target chunking metrics.
REPORT_SCHEMA_VERSION = "1.1"


class BorderlineCounter:
    """Counts threshold decisions within a relative 1e-6 of the bound."""

    def __init__(self) -> None:
        self.count = 0

    def note(self, value: float, bound: float) -> None:
        if bound == 0:
            return
        if abs(value - bound) / abs(bound) <= 1e-6:
            self.count += 1


def _sanitize(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def cap_category_labels(labels: dict[str, int]) -> tuple[dict[str, int], bool]:
    """Keep at most PRIVACY_MAX_LABEL_CATEGORIES labels, highest count first.

    Returns the retained mapping and whether anything was dropped.  This is the
    display-time cap; it is deliberately independent of the analysis-time
    metadata max_categories setting, which only decides whether a field is
    collected in the first place.
    """

    if len(labels) <= PRIVACY_MAX_LABEL_CATEGORIES:
        return dict(labels), False
    ordered = sorted(labels.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered[:PRIVACY_MAX_LABEL_CATEGORIES]), True


def sample_to_json(sample) -> dict | None:
    """Render a SampleInfo as its nested n/of shape, or None."""
    if sample is None:
        return None
    return {
        "v1": {"n": sample.v1_n, "of": sample.v1_of},
        "v2": {"n": sample.v2_n, "of": sample.v2_of},
        "seed": sample.seed,
    }


def metric_to_json(metric: Metric) -> dict:
    """Render one metric object in section 15.7's shape."""
    reason = metric.reason
    for value in (metric.v1, metric.v2, metric.change):
        if value is not None and _sanitize(value) is None and reason is None:
            reason = "NON_FINITE_VALUE"
    details = metric.details
    if isinstance(details, dict):
        raw_labels = details.get("category_labels")
        if isinstance(raw_labels, dict):
            capped, truncated = cap_category_labels(raw_labels)
            details = dict(details)
            details["category_labels"] = capped
            if truncated:
                details["category_labels_truncated"] = True
    return {
        "v1": _sanitize(metric.v1),
        "v2": _sanitize(metric.v2),
        "change": _sanitize(metric.change),
        "status": metric.status,
        "reason": reason,
        "sample": sample_to_json(metric.sample),
        "details": details,
    }


def dimension_to_json(dimension: Dimension) -> dict:
    """Render one dimension with its metrics keyed by metric name."""
    return {
        "status": dimension.status,
        "metrics": {
            metric.name: metric_to_json(metric) for metric in dimension.metrics
        },
    }


def _snapshot_summary(manifest: Manifest) -> dict:
    return {
        "snapshot_id": manifest.snapshot_id,
        "documents": manifest.document_count,
        "chunks": manifest.chunk_count,
        "embedding_model": manifest.embedding_model,
        "embedding_dimension": manifest.embedding_dimension,
        "granularity": manifest.granularity,
        "granularity_source": manifest.granularity_source,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


def _determinism_level(health: HealthReport) -> str:
    for dimension in health.dimensions.values():
        for metric in dimension.metrics:
            for value in (metric.v1, metric.v2, metric.change):
                if isinstance(value, float) and math.isfinite(value):
                    return "TOLERANCE"
    return "EXACT"


def _skipped_entries(health: HealthReport) -> list[dict]:
    entries: list[dict] = []
    for name, dimension in health.dimensions.items():
        if dimension.status != "SKIPPED":
            continue
        reason = next(
            metric.reason
            for metric in dimension.metrics
            if metric.status == "SKIPPED"
        )
        entries.append({"dimension": name, "reason": reason})
    return entries


def build_report_json(
    *,
    health: HealthReport,
    v1_manifest: Manifest,
    v2_manifest: Manifest,
    sink: DiagnosticSink,
    cfg: Config,
    borderline: BorderlineCounter,
    diff_artifact_paths: dict[str, str],
    driftlens_version: str,
    breakdowns: dict[str, list[dict]] | None = None,
    no_timestamp: bool = False,
) -> dict:
    """Build the full section 15.7 report object."""
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "driftlens_version": driftlens_version,
        "created_at": (
            "1970-01-01T00:00:00+00:00"
            if no_timestamp
            else datetime.now(timezone.utc).isoformat()
        ),
        "snapshots": {
            "v1": _snapshot_summary(v1_manifest),
            "v2": _snapshot_summary(v2_manifest),
        },
        "status": {"overall": health.overall, "exceeded": health.exceeded},
        "dimensions": {
            name: dimension_to_json(dimension)
            for name, dimension in health.dimensions.items()
        },
        "warnings": [
            {
                "code": diagnostic.code.value,
                "count": diagnostic.count,
                "message": diagnostic.message,
                "dimension": diagnostic.dimension,
            }
            for diagnostic in sink.items()
        ],
        "skipped": _skipped_entries(health),
        "config": {"effective": _jsonable(asdict(cfg))},
        "reproducibility": {
            "determinism_level": _determinism_level(health),
            "borderline_decisions": borderline.count,
        },
        "diff_artifacts": diff_artifact_paths,
    }
    if breakdowns is not None:
        report["breakdowns"] = breakdowns
    return report


def write_report_json(report: dict, path: Path) -> None:
    """Write the report as UTF-8 JSON, rejecting any stray NaN/Infinity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            sort_keys=False,
            allow_nan=False,
            ensure_ascii=False,
        )
