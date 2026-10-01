"""Human-readable terminal report rendering (Step 32)."""

from __future__ import annotations

from driftlens.analysis.health import (
    Dimension,
    HealthReport,
    Metric,
    MetricStatus,
)
from driftlens.errors import DiagnosticSink
from driftlens.schema.manifest import Manifest

NAME_WIDTH = 28
VALUE_WIDTH = 14
ANSI = {
    "WARN": "\033[33m",
    "FAIL": "\033[31m",
    "UNBASELINED": "\033[2m",
    "RESET": "\033[0m",
}
_SEPARATOR = "=" * 79


def format_value(value: float | str | None, *, unit: str | None = None) -> str:
    """Render a single value cell (no sign forcing, no unit guessing)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if abs(value) < 1000:
            text = f"{value:.4g}"
        else:
            text = f"{value:,.0f}"
    else:
        text = str(value)
    return f"{text} {unit}" if unit else text


def format_tag(status: MetricStatus, *, isatty: bool) -> str:
    """Status tag, ANSI-wrapped only when writing to a terminal."""
    base = {
        "WARN": "[WARN]",
        "FAIL": "[FAIL]",
        "UNBASELINED": "[UNBASELINED]",
    }.get(status, "")
    if base and isatty:
        return ANSI[status] + base + ANSI["RESET"]
    return base


def format_metric_row(metric: Metric, *, isatty: bool) -> str:
    """One metric line: name, v1, v2, change, tag."""
    if metric.status == "SKIPPED":
        return f"  {metric.name} SKIPPED ({metric.reason})"
    if metric.status == "N/A":
        suffix = f": {metric.reason}" if metric.reason else ""
        return f"  {metric.name} n/a{suffix}"

    name_col = metric.name.ljust(NAME_WIDTH)[:NAME_WIDTH]
    v1_col = format_value(metric.v1).rjust(VALUE_WIDTH)
    v2_col = format_value(metric.v2).rjust(VALUE_WIDTH)
    change_col = format_value(metric.change).rjust(VALUE_WIDTH)
    tag = format_tag(metric.status, isatty=isatty)
    line = f"  {name_col}{v1_col}{v2_col}{change_col} {tag}".rstrip()
    if metric.approximate:
        line += " (approximate)"
    return line


def _dimension_header(dimension: Dimension, *, isatty: bool) -> str:
    name = dimension.name.upper()
    if dimension.status == "SKIPPED":
        reason = next(
            (
                metric.reason
                for metric in dimension.metrics
                if metric.status == "SKIPPED"
            ),
            None,
        )
        return f"{name} SKIPPED ({reason})"
    if dimension.status == "N/A":
        return name
    tag = format_tag(dimension.status, isatty=isatty)
    return f"{name} {tag}" if tag else name


def format_dimension_block(dimension: Dimension, *, isatty: bool) -> str:
    """The dimension header plus its printable metric rows."""
    lines = [_dimension_header(dimension, isatty=isatty)]
    for metric in dimension.metrics:
        if (
            metric.status == "N/A"
            and metric.v1 is None
            and metric.v2 is None
            and metric.change is None
        ):
            continue
        lines.append(format_metric_row(metric, isatty=isatty))
    return "\n".join(lines)


def format_status_line(health: HealthReport) -> str:
    """The overall STATUS line."""
    if health.overall == "UNBASELINED":
        return "STATUS: UNBASELINED"
    if health.overall == "PASS":
        return "STATUS: PASS"
    return "STATUS: " + health.overall + " " + ", ".join(
        dimension.upper() for dimension in health.exceeded
    )


def format_warnings_block(sink: DiagnosticSink, *, max_lines: int = 10) -> str:
    """The WARNINGS block, truncated with a pointer to --json-out."""
    items = sink.items()
    selected = items[:max_lines]
    lines = ["WARNINGS"]
    lines.extend(
        f"  {diagnostic.code.value} {diagnostic.message} ({diagnostic.count})"
        for diagnostic in selected
    )
    remainder = len(items) - len(selected)
    if remainder > 0:
        lines.append(f"  +{remainder} more, see --json-out")
    return "\n".join(lines)


def format_footer(
    v1_manifest: Manifest,
    v2_manifest: Manifest,
    *,
    seed: int,
    determinism_level: str,
    report_out: str | None,
    diff_out: str | None,
) -> str:
    """The run footer with seed, determinism level and artifact paths."""
    text = f"seed {seed} determinism {determinism_level}"
    if report_out:
        text += f" details: {report_out}"
    if diff_out:
        text += f" records: {diff_out}"
    return text


def render_terminal_report(
    *,
    health: HealthReport,
    v1_manifest: Manifest,
    v2_manifest: Manifest,
    sink: DiagnosticSink,
    determinism_level: str,
    isatty: bool,
    seed: int,
    report_out: str | None = None,
    diff_out: str | None = None,
) -> str:
    """Full section 15.2 report, in the fixed order."""
    lines = [
        f"DRIFTLENS v1 {v1_manifest.snapshot_id} -> v2 {v2_manifest.snapshot_id}",
        format_status_line(health),
        _SEPARATOR,
    ]
    for dimension in health.dimensions.values():
        if dimension.status == "N/A" and len(dimension.metrics) == 0:
            continue
        lines.append(format_dimension_block(dimension, isatty=isatty))
        lines.append("")
    if sink.items():
        lines.append(format_warnings_block(sink))
    lines.append(_SEPARATOR)
    lines.append(
        format_footer(
            v1_manifest,
            v2_manifest,
            seed=seed,
            determinism_level=determinism_level,
            report_out=report_out,
            diff_out=diff_out,
        )
    )
    return "\n".join(lines)


def _key_change_suffix(dimension: Dimension) -> str:
    metric = next(
        (
            item
            for item in dimension.metrics
            if item.status == dimension.status and item.is_threshold_key
        ),
        None,
    )
    if metric is None:
        return ""
    return (
        f" {metric.name} {format_value(metric.v1)} -> {format_value(metric.v2)}"
    )


def render_quiet_report(health: HealthReport) -> str:
    """Status line plus one key-change line per non-PASS dimension."""
    lines = [format_status_line(health)]
    for dimension in health.dimensions.values():
        if dimension.status in ("PASS", "N/A"):
            continue
        if dimension.status == "SKIPPED":
            reason = next(
                (
                    metric.reason
                    for metric in dimension.metrics
                    if metric.status == "SKIPPED"
                ),
                None,
            )
            lines.append(f"{dimension.name.upper()} SKIPPED {reason}")
            continue
        lines.append(
            f"{dimension.name.upper()} {dimension.status}"
            f"{_key_change_suffix(dimension)}"
        )
    return "\n".join(lines)


def render_ci_report(health: HealthReport) -> tuple[str, int]:
    """One line per fixed dimension plus the v0.1 exit-code stub."""
    lines: list[str] = []
    for dimension in health.dimensions.values():
        line = f"{dimension.name.upper()} {dimension.status}"
        if dimension.status in ("WARN", "FAIL"):
            line += _key_change_suffix(dimension)
        lines.append(line)
    return "\n".join(lines), 0
