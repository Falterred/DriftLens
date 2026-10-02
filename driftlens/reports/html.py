"""DriftLens self-contained HTML report generation (Step 42).

The renderer is intentionally dependency-free: it emits one UTF-8 document
with inline CSS and inline SVG only. There are no scripts and no external
references. Every data-derived value passes through esc() before it reaches
the markup.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any

from driftlens.analysis.chunking import AnomalyCounts, ChunkProfile
from driftlens.analysis.compatibility import CompatibilityResult
from driftlens.analysis.content import ChunkDiffResult, DocumentDiffRow
from driftlens.analysis.distributions import FIXED_BIN_EDGES
from driftlens.analysis.duplicates import ExactDuplicateResult
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.analysis.health import HealthReport, Metric
from driftlens.analysis.metadata import FieldDrift
from driftlens.config.schema import Config
from driftlens.errors import CODE_CLASS, DiagnosticSink
from driftlens.reports.terminal import format_value
from driftlens.schema.manifest import Manifest

_SECTION_TITLES = (
    "Executive Summary",
    "Snapshot Information",
    "Document Changes",
    "Chunk Changes",
    "Metadata Drift",
    "Chunking Distributions",
    "Embedding Compatibility",
    "Embedding Drift",
    "Duplicates",
    "Clusters",
    "Retrieval",
    "Statistical Details",
    "Configuration",
    "Reproducibility",
    "Warnings",
    "Breakdowns",
)

_DOCUMENT_STATUSES = (
    "ADDED",
    "REMOVED",
    "MODIFIED",
    "REPLACED",
    "FORMAT_ONLY_CHANGE",
    "UNCHANGED",
)

_CHUNK_STATUSES = ("UNCHANGED", "MODIFIED", "MOVED", "ADDED", "REMOVED")

_ANOMALY_CATEGORIES = (
    "empty",
    "very_short",
    "suspicious_short",
    "oversized",
    "over_embedder_limit",
    "encoding_damage",
)

_STATUS_CLASS_MAP: dict[str, str] = {
    "PASS": "dl-status-pass",
    "WARN": "dl-status-warn",
    "FAIL": "dl-status-fail",
    "UNBASELINED": "dl-status-unbaselined",
    "SKIPPED": "dl-status-skipped",
    "N/A": "dl-status-na",
    "REPORT_ONLY": "dl-status-report-only",
    "COMPATIBLE": "dl-status-pass",
    "SAME_SPACE": "dl-status-pass",
    "DIFFERENT_SPACE": "dl-status-fail",
    "EMBEDDING_SPACE_INCOMPATIBLE": "dl-status-fail",
    "MODEL_CHANGED": "dl-status-warn",
    "SPACE_CHANGED": "dl-status-warn",
    "PARTIAL_MIGRATION": "dl-status-warn",
    "MIXED_MODELS": "dl-status-warn",
    "MODEL_UNKNOWN": "dl-status-warn",
    "METRIC_MISMATCH": "dl-status-fail",
}
_DEFAULT_STATUS_CLASS = "dl-status-neutral"

_DIMENSION_SECTION = {
    "content": 3,
    "chunking": 4,
    "metadata": 5,
    "embedding": 8,
    "duplication": 9,
    "retrieval": 11,
}


def esc(s: Any) -> str:
    """Single escaping chokepoint for all data-derived values."""
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


def _status_class(status: Any) -> str:
    if status is None:
        return _DEFAULT_STATUS_CLASS
    return _STATUS_CLASS_MAP.get(str(status), _DEFAULT_STATUS_CLASS)


def _num(value: Any) -> str:
    """Compact numeric rendering for SVG geometry."""
    try:
        return "%g" % float(value)
    except (TypeError, ValueError):
        return "0"


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _metric_change(metric: Metric) -> float | str | None:
    if metric.change is not None:
        return metric.change
    return metric.v2


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def _cell(value: Any) -> str:
    return format_value(value)


def _describe_change(v1: Any, v2: Any) -> str:
    return _cell(v1) + " to " + _cell(v2)


# --- Static design tokens (inline CSS, no external references) ---

_REPORT_CSS = """
:root {
  --dl-bg: #f6f7f9;
  --dl-surface: #ffffff;
  --dl-border: #dfe3e8;
  --dl-text: #1f2933;
  --dl-muted: #5c6b7a;
  --dl-pass: #1f7a45;
  --dl-warn: #9a6700;
  --dl-fail: #b3261e;
  --dl-neutral: #52606d;
  --dl-v1: #2f6fed;
  --dl-v2: #d97706;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--dl-bg);
  color: var(--dl-text);
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
}
.dl-layout { display: grid; grid-template-columns: 250px minmax(0, 1fr); }
.dl-sidebar {
  position: sticky;
  top: 0;
  align-self: start;
  height: 100vh;
  overflow-y: auto;
  background: var(--dl-surface);
  border-right: 1px solid var(--dl-border);
  padding: 16px 12px;
}
.dl-nav-title { font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--dl-muted); margin: 0 0 10px; }
.dl-nav-list { list-style: none; margin: 0; padding: 0; }
.dl-nav-list li { margin: 2px 0; }
.dl-nav-link { display: block; padding: 5px 8px; border-radius: 6px; color: var(--dl-text); text-decoration: none; }
.dl-nav-link:hover { background: var(--dl-bg); color: var(--dl-v1); }
.dl-main { min-width: 0; padding: 24px; }
.dl-section { background: var(--dl-surface); border: 1px solid var(--dl-border); border-radius: 10px; padding: 20px; margin-bottom: 20px; }
.dl-section-title { margin: 0 0 14px; font-size: 20px; }
.dl-subtitle { margin: 18px 0 8px; font-size: 15px; color: var(--dl-muted); }
.dl-badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 600; color: #fff; }
.dl-status-pass { background: var(--dl-pass); }
.dl-status-warn { background: var(--dl-warn); }
.dl-status-fail { background: var(--dl-fail); }
.dl-status-neutral { background: var(--dl-neutral); }
.dl-status-unbaselined { background: #6f42c1; }
.dl-status-skipped { background: #6c757d; }
.dl-status-na { background: #adb5bd; }
.dl-status-report-only { background: #0b7285; }
.dl-kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.dl-kpi { border: 1px solid var(--dl-border); border-radius: 8px; padding: 12px; background: var(--dl-bg); }
.dl-kpi-label { font-size: 12px; color: var(--dl-muted); text-transform: uppercase; letter-spacing: 0.04em; }
.dl-kpi-value { font-size: 20px; font-weight: 600; margin: 6px 0; word-break: break-word; }
.dl-kpi-detail { font-size: 12px; color: var(--dl-muted); }
.dl-jumps { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.dl-jump { display: inline-block; padding: 3px 10px; border: 1px solid var(--dl-fail); color: var(--dl-fail); border-radius: 999px; text-decoration: none; font-size: 12px; }
.dl-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.dl-card { border: 1px solid var(--dl-border); border-radius: 8px; padding: 14px; }
.dl-table-wrap { overflow-x: auto; margin: 10px 0; }
.dl-table { border-collapse: collapse; width: 100%; font-size: 13px; }
.dl-table th, .dl-table td { border: 1px solid var(--dl-border); padding: 6px 9px; text-align: left; vertical-align: top; }
.dl-table th { background: var(--dl-bg); position: sticky; top: 0; }
.dl-kv { display: grid; grid-template-columns: minmax(140px, max-content) 1fr; gap: 4px 12px; margin: 0; }
.dl-kv dt { color: var(--dl-muted); font-weight: 500; word-break: break-word; }
.dl-kv dd { margin: 0; word-break: break-word; }
.dl-empty { padding: 12px 14px; border: 1px dashed var(--dl-border); border-radius: 8px; color: var(--dl-muted); background: var(--dl-bg); }
.dl-details { margin: 10px 0; }
.dl-details summary { cursor: pointer; font-weight: 600; padding: 6px 0; }
.dl-accordion { border: 1px solid var(--dl-border); border-radius: 8px; padding: 10px 12px; margin: 8px 0; }
.dl-accordion summary { cursor: pointer; font-weight: 600; }
.dl-pre { white-space: pre-wrap; word-break: break-word; background: var(--dl-bg); border: 1px solid var(--dl-border); border-radius: 6px; padding: 8px 10px; font-size: 12px; }
.dl-alert { display: flex; align-items: center; gap: 10px; padding: 12px 14px; border: 1px solid var(--dl-border); border-radius: 8px; background: var(--dl-bg); margin-bottom: 8px; }
.dl-alert-text { font-weight: 600; }
.dl-chart { display: block; width: 100%; height: auto; margin: 8px 0; }
.dl-chart-title { fill: var(--dl-text); font-size: 13px; font-weight: 600; }
.dl-axis { fill: var(--dl-muted); font-size: 10px; }
.dl-grid { stroke: var(--dl-border); stroke-width: 1; }
.dl-bar { fill: var(--dl-v1); }
.dl-bar-v1 { fill: var(--dl-v1); }
.dl-bar-v2 { fill: var(--dl-v2); }
.dl-bar-current { fill: var(--dl-neutral); }
.dl-threshold-warning { stroke: var(--dl-warn); stroke-width: 2; stroke-dasharray: 5 4; }
.dl-threshold-critical { stroke: var(--dl-fail); stroke-width: 2; stroke-dasharray: 5 4; }
.dl-threshold-axis { stroke: var(--dl-text); stroke-width: 1; }
.dl-marker { fill: var(--dl-text); }
.dl-legend { font-size: 12px; color: var(--dl-muted); margin: 4px 0 0; }
.dl-legend-key { display: inline-block; margin-right: 14px; }
.dl-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
.dl-swatch-v1 { background: var(--dl-v1); }
.dl-swatch-v2 { background: var(--dl-v2); }
@media (max-width: 900px) {
  .dl-layout { grid-template-columns: 1fr; }
  .dl-sidebar { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--dl-border); }
  .dl-nav-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .dl-main { padding: 12px; }
}
"""


# --- SVG rendering engine ---

@dataclass(frozen=True)
class _SVGContext:
    width: float
    height: float
    pad_left: float = 52.0
    pad_right: float = 16.0
    pad_top: float = 30.0
    pad_bottom: float = 46.0

    @property
    def plot_width(self) -> float:
        return max(self.width - self.pad_left - self.pad_right, 1.0)

    @property
    def plot_height(self) -> float:
        return max(self.height - self.pad_top - self.pad_bottom, 1.0)


def _svg_open(ctx: _SVGContext, title: str) -> str:
    return (
        '<svg class="dl-chart" role="img" viewBox="0 0 '
        + _num(ctx.width)
        + " "
        + _num(ctx.height)
        + '" aria-label="'
        + esc(title)
        + '">'
    )


def _svg_close() -> str:
    return "</svg>"


def _svg_title(ctx: _SVGContext, title: str) -> str:
    if not title:
        return ""
    return (
        '<text class="dl-chart-title" x="'
        + _num(ctx.width / 2.0)
        + '" y="'
        + _num(ctx.pad_top / 1.7)
        + '" text-anchor="middle">'
        + esc(title)
        + "</text>"
    )


def _y_gridlines(ctx: _SVGContext, vmax: float) -> str:
    parts: list[str] = []
    for step in range(5):
        ratio = step / 4.0
        y = ctx.pad_top + ctx.plot_height * ratio
        value = vmax * (1.0 - ratio)
        parts.append(
            '<line class="dl-grid" x1="'
            + _num(ctx.pad_left)
            + '" y1="'
            + _num(y)
            + '" x2="'
            + _num(ctx.pad_left + ctx.plot_width)
            + '" y2="'
            + _num(y)
            + '"/>'
        )
        parts.append(
            '<text class="dl-axis" x="'
            + _num(ctx.pad_left - 6)
            + '" y="'
            + _num(y + 3)
            + '" text-anchor="end">'
            + esc(_num(value))
            + "</text>"
        )
    return "".join(parts)


def svg_bar_chart(
    labels: list[str],
    values: list[float],
    *,
    title: str = "",
    width: int = 600,
    height: int = 240,
) -> str:
    """Single-series vertical bar chart."""
    ctx = _SVGContext(float(width), float(height))
    count = len(values)
    vmax = max([float(v) for v in values] + [0.0])
    if vmax <= 0.0:
        vmax = 1.0
    parts = [_svg_open(ctx, title), _svg_title(ctx, title), _y_gridlines(ctx, vmax)]
    if count == 0:
        parts.append(_empty_svg_note(ctx, "no data"))
    else:
        slot = ctx.plot_width / count
        bar_width = slot * 0.62
        baseline = ctx.pad_top + ctx.plot_height
        for index, value in enumerate(values):
            magnitude = _as_float(value)
            bar_height = ctx.plot_height * (magnitude / vmax)
            x = ctx.pad_left + slot * index + (slot - bar_width) / 2.0
            y = baseline - bar_height
            parts.append(
                f'<rect class="dl-bar" x="{esc(_num(x))}" y="{esc(_num(y))}" '
                f'width="{esc(_num(bar_width))}" height="{esc(_num(bar_height))}">'
                f'<title>{esc(labels[index] if index < len(labels) else "")}: '
                f'{esc(_num(magnitude))}</title></rect>'
            )
            label = labels[index] if index < len(labels) else ""
            parts.append(
                f'<text class="dl-axis" x="{esc(_num(x + bar_width / 2.0))}" '
                f'y="{esc(_num(baseline + 14))}" text-anchor="end" '
                f'transform="rotate(-35 {esc(_num(x + bar_width / 2.0))} '
                f'{esc(_num(baseline + 14))})">{esc(label)}</text>'
            )
    parts.append(_svg_close())
    return "".join(parts)


def svg_grouped_bar_chart(
    categories: list[str],
    series_v1: list[float],
    series_v2: list[float],
    *,
    v1_label: str = "V1",
    v2_label: str = "V2",
    title: str = "",
    width: int = 640,
    height: int = 260,
) -> str:
    """Two-series vertical bar chart; emits two bars per category."""
    ctx = _SVGContext(float(width), float(height))
    count = len(categories)
    vmax = max([_as_float(v) for v in list(series_v1) + list(series_v2)] + [0.0])
    if vmax <= 0.0:
        vmax = 1.0
    parts = [_svg_open(ctx, title), _svg_title(ctx, title), _y_gridlines(ctx, vmax)]
    if count == 0:
        parts.append(_empty_svg_note(ctx, "no data"))
    else:
        slot = ctx.plot_width / count
        bar_width = slot * 0.3
        baseline = ctx.pad_top + ctx.plot_height
        for index, category in enumerate(categories):
            left = ctx.pad_left + slot * index + slot * 0.12
            value_v1 = _as_float(series_v1[index]) if index < len(series_v1) else 0.0
            value_v2 = _as_float(series_v2[index]) if index < len(series_v2) else 0.0
            for offset, value, css_class, label in (
                (0.0, value_v1, "dl-bar-v1", v1_label),
                (bar_width * 1.05, value_v2, "dl-bar-v2", v2_label),
            ):
                bar_height = ctx.plot_height * (value / vmax)
                x = left + offset
                y = baseline - bar_height
                parts.append(
                    f'<rect class="{esc(css_class)}" x="{esc(_num(x))}" '
                    f'y="{esc(_num(y))}" width="{esc(_num(bar_width))}" '
                    f'height="{esc(_num(bar_height))}">'
                    f'<title>{esc(category)} {esc(label)}: {esc(_num(value))}</title>'
                    f'</rect>'
                )
            parts.append(
                f'<text class="dl-axis" x="{esc(_num(left + bar_width))}" '
                f'y="{esc(_num(baseline + 14))}" text-anchor="end" '
                f'transform="rotate(-35 {esc(_num(left + bar_width))} '
                f'{esc(_num(baseline + 14))})">{esc(category)}</text>'
            )
    parts.append(
        '<p class="dl-legend"><span class="dl-legend-key">'
        '<span class="dl-swatch dl-swatch-v1"></span>'
        + esc(v1_label)
        + '</span><span class="dl-legend-key">'
        '<span class="dl-swatch dl-swatch-v2"></span>'
        + esc(v2_label)
        + "</span></p>"
    )
    parts.append(_svg_close())
    return "".join(parts)


def svg_histogram(
    bin_edges: list[float],
    counts_v1: list[int],
    counts_v2: list[int],
    *,
    title: str = "",
    width: int = 640,
    height: int = 260,
) -> str:
    """Dual-series histogram over explicit bin edges."""
    ctx = _SVGContext(float(width), float(height))
    bin_count = max(len(bin_edges) - 1, 0)
    vmax = max([float(c) for c in list(counts_v1) + list(counts_v2)] + [0.0])
    if vmax <= 0.0:
        vmax = 1.0
    parts = [_svg_open(ctx, title), _svg_title(ctx, title), _y_gridlines(ctx, vmax)]
    if bin_count == 0:
        parts.append(_empty_svg_note(ctx, "no bins"))
    else:
        slot = ctx.plot_width / bin_count
        bar_width = slot * 0.3
        baseline = ctx.pad_top + ctx.plot_height
        for index in range(bin_count):
            low = bin_edges[index]
            high = bin_edges[index + 1]
            label = _num(low) + "-" + _num(high)
            left = ctx.pad_left + slot * index + slot * 0.12
            first = float(counts_v1[index]) if index < len(counts_v1) else 0.0
            second = float(counts_v2[index]) if index < len(counts_v2) else 0.0
            for offset, value, css_class in (
                (0.0, first, "dl-bar-v1"),
                (bar_width * 1.05, second, "dl-bar-v2"),
            ):
                bar_height = ctx.plot_height * (value / vmax)
                x = left + offset
                y = baseline - bar_height
                parts.append(
                    f'<rect class="{esc(css_class)}" x="{esc(_num(x))}" '
                    f'y="{esc(_num(y))}" width="{esc(_num(bar_width))}" '
                    f'height="{esc(_num(bar_height))}">'
                    f'<title>{esc(label)}: {esc(_num(value))}</title></rect>'
                )
            parts.append(
                f'<text class="dl-axis" x="{esc(_num(left + bar_width))}" '
                f'y="{esc(_num(baseline + 14))}" text-anchor="end" '
                f'transform="rotate(-35 {esc(_num(left + bar_width))} '
                f'{esc(_num(baseline + 14))})">{esc(label)}</text>'
            )
    parts.append(_svg_close())
    return "".join(parts)


def svg_ranked_horizontal_bars(
    labels: list[str],
    values: list[float],
    *,
    title: str = "",
    width: int = 600,
    height: int = 260,
    threshold: float | None = None,
) -> str:
    """Horizontal ranked bars for distance-style metrics."""
    ctx = _SVGContext(float(width), float(height), pad_left=150.0)
    count = len(values)
    vmax = max([_as_float(v) for v in values] + [0.0])
    if vmax <= 0.0:
        vmax = 1.0
    if threshold is not None and threshold > vmax:
        vmax = float(threshold)
    parts = [_svg_open(ctx, title), _svg_title(ctx, title)]
    row_height = ctx.plot_height / max(count, 1)
    bar_height = max(row_height * 0.6, 2.0)
    if count == 0:
        parts.append(_empty_svg_note(ctx, "no data"))
    else:
        for index, label in enumerate(labels):
            value = _as_float(values[index]) if index < len(values) else 0.0
            bar_width = ctx.plot_width * (value / vmax)
            y = ctx.pad_top + row_height * index + (row_height - bar_height) / 2.0
            parts.append(
                f'<rect class="dl-bar" x="{esc(_num(ctx.pad_left))}" '
                f'y="{esc(_num(y))}" width="{esc(_num(bar_width))}" '
                f'height="{esc(_num(bar_height))}">'
                f'<title>{esc(label)}: {esc(_num(value))}</title></rect>'
            )
            parts.append(
                f'<text class="dl-axis" x="{esc(_num(ctx.pad_left - 8))}" '
                f'y="{esc(_num(y + bar_height))}" text-anchor="end">'
                f'{esc(label)}</text>'
            )
            parts.append(
                f'<text class="dl-axis" x="{esc(_num(ctx.pad_left + bar_width + 6))}" '
                f'y="{esc(_num(y + bar_height))}">{esc(_num(value))}</text>'
            )
        if threshold is not None:
            x = ctx.pad_left + ctx.plot_width * (float(threshold) / vmax)
            parts.append(
                f'<line class="dl-threshold-warning" x1="{esc(_num(x))}" '
                f'y1="{esc(_num(ctx.pad_top))}" x2="{esc(_num(x))}" '
                f'y2="{esc(_num(ctx.pad_top + ctx.plot_height))}"/></line>'
            )
    parts.append(_svg_close())
    return "".join(parts)


def _empty_svg_note(ctx: _SVGContext, text: str) -> str:
    return (
        '<text class="dl-axis" x="'
        + _num(ctx.width / 2.0)
        + '" y="'
        + _num(ctx.height / 2.0)
        + '" text-anchor="middle">'
        + esc(text)
        + "</text>"
    )


def _threshold_axis_x(ctx: _SVGContext, value: float, maximum: float) -> float:
    if maximum <= 0.0:
        maximum = 1.0
    ratio = value / maximum
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0
    return ctx.pad_left + ctx.plot_width * ratio


def _svg_threshold_linear_bar(
    *,
    value: float,
    warning: float | None = None,
    critical: float | None = None,
    maximum: float | None = None,
    title: str = "",
    width: int = 600,
    height: int = 104,
) -> str:
    """Honest linear threshold bar: 0, measured marker, warning and critical."""
    ctx = _SVGContext(float(width), float(height), pad_top=24.0, pad_bottom=40.0)
    candidates = [value]
    if warning is not None:
        candidates.append(float(warning))
    if critical is not None:
        candidates.append(float(critical))
    resolved_max = float(maximum) if maximum is not None else max(candidates + [1.0])
    if resolved_max <= 0.0:
        resolved_max = 1.0
    axis_y = ctx.pad_top + ctx.plot_height
    x_start = ctx.pad_left
    x_end = ctx.pad_left + ctx.plot_width
    parts = [_svg_open(ctx, title), _svg_title(ctx, title)]
    parts.append(
        f'<line class="dl-threshold-axis" x1="{esc(_num(x_start))}" '
        f'y1="{esc(_num(axis_y))}" x2="{esc(_num(x_end))}" '
        f'y2="{esc(_num(axis_y))}"/>'
    )
    parts.append(
        f'<text class="dl-axis" x="{esc(_num(x_start))}" '
        f'y="{esc(_num(axis_y + 16))}" text-anchor="middle">0</text>'
    )
    parts.append(
        f'<text class="dl-axis" x="{esc(_num(x_end))}" '
        f'y="{esc(_num(axis_y + 16))}" text-anchor="middle">'
        f'{esc(_num(resolved_max))}</text>'
    )
    if warning is not None:
        x_warning = _threshold_axis_x(ctx, float(warning), resolved_max)
        parts.append(
            f'<line class="dl-threshold-warning" x1="{esc(_num(x_warning))}" '
            f'y1="{esc(_num(ctx.pad_top))}" x2="{esc(_num(x_warning))}" '
            f'y2="{esc(_num(axis_y))}"/>'
        )
        parts.append(
            f'<text class="dl-axis" x="{esc(_num(x_warning))}" '
            f'y="{esc(_num(ctx.pad_top + 10))}" text-anchor="middle">warn '
            f'{esc(_num(warning))}</text>'
        )
    if critical is not None:
        x_critical = _threshold_axis_x(ctx, float(critical), resolved_max)
        parts.append(
            f'<line class="dl-threshold-critical" x1="{esc(_num(x_critical))}" '
            f'y1="{esc(_num(ctx.pad_top))}" x2="{esc(_num(x_critical))}" '
            f'y2="{esc(_num(axis_y))}"/>'
        )
        parts.append(
            f'<text class="dl-axis" x="{esc(_num(x_critical))}" '
            f'y="{esc(_num(ctx.pad_top + 22))}" text-anchor="middle">crit '
            f'{esc(_num(critical))}</text>'
        )
    x_value = _threshold_axis_x(ctx, value, resolved_max)
    parts.append(
        f'<circle class="dl-marker" cx="{esc(_num(x_value))}" '
        f'cy="{esc(_num(axis_y - 10))}" r="5">'
        f'<title>measured {esc(_num(value))}</title></circle>'
    )
    parts.append(
        f'<text class="dl-axis" x="{esc(_num(x_value))}" '
        f'y="{esc(_num(axis_y - 22))}" text-anchor="middle">'
        f'{esc(_num(value))}</text>'
    )
    parts.append(_svg_close())
    return "".join(parts)


# --- Reusable HTML components ---

def _status_badge(status: Any) -> str:
    text = "" if status is None else str(status)
    return '<span class="dl-badge ' + _status_class(status) + '">' + esc(text) + "</span>"


def _kpi_card(label: Any, value: Any, *, status: Any = None, detail: Any = None) -> str:
    parts = ['<div class="dl-kpi">']
    parts.append('<div class="dl-kpi-label">' + esc(label) + "</div>")
    parts.append('<div class="dl-kpi-value">' + esc(value) + "</div>")
    if status is not None:
        parts.append(_status_badge(status))
    if detail is not None:
        parts.append('<div class="dl-kpi-detail">' + esc(detail) + "</div>")
    parts.append("</div>")
    return "".join(parts)


def _section_container(number: int, title: str, body: str) -> str:
    return (
        '<section id="sec-'
        + str(number)
        + '" class="dl-section">'
        + '<h2 class="dl-section-title">'
        + esc(title)
        + "</h2>"
        + body
        + "</section>"
    )


def _empty_state(message: str) -> str:
    return '<div class="dl-empty" role="status">' + esc(message) + "</div>"


def _data_table(headers: list[Any], rows: list[list[Any]]) -> str:
    parts = ['<div class="dl-table-wrap"><table class="dl-table"><thead><tr>']
    for header in headers:
        parts.append("<th>" + esc(header) + "</th>")
    parts.append("</tr></thead><tbody>")
    if not rows:
        parts.append(
            '<tr><td colspan="'
            + str(max(len(headers), 1))
            + '" class="dl-empty">'
            + esc("no rows")
            + "</td></tr>"
        )
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            parts.append("<td>" + esc(cell) + "</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _key_value_grid(pairs: list[tuple[Any, Any]]) -> str:
    parts = ['<dl class="dl-kv">']
    for key, value in pairs:
        parts.append("<dt>" + esc(key) + "</dt><dd>" + esc(value) + "</dd>")
    parts.append("</dl>")
    return "".join(parts)


# --- Section context ---

@dataclass(frozen=True)
class _ReportContext:
    health: HealthReport
    v1_manifest: Manifest
    v2_manifest: Manifest
    document_diff: list[DocumentDiffRow]
    chunk_diff: ChunkDiffResult
    field_drifts: list[FieldDrift]
    profile_v1: ChunkProfile | None
    profile_v2: ChunkProfile | None
    anomaly_v1: AnomalyCounts | None
    anomaly_v2: AnomalyCounts | None
    compat: CompatibilityResult
    v1_summary: EmbeddingSnapshotSummary
    v2_summary: EmbeddingSnapshotSummary
    mmd_metric: Metric
    dup_v1: ExactDuplicateResult
    dup_v2: ExactDuplicateResult
    cfg: Config
    sink: DiagnosticSink
    determinism_level: str
    borderline: int
    limits_applied: dict[str, Any]
    unsafe_pickle: bool
    top_n_ids: list[str] | None
    diagnostic: bool
    breakdowns: dict[str, list[Any]] | None


def _mmd_threshold(ctx: _ReportContext) -> Any:
    return ctx.cfg.drift.get("mmd")


def _threshold_max(value: float, warning: float | None, critical: float | None) -> float:
    candidates = [value]
    if warning is not None:
        candidates.append(float(warning))
    if critical is not None:
        candidates.append(float(critical))
    return max(candidates + [1.0]) * 1.15


def _ranked_document_rows(ctx: _ReportContext) -> list[DocumentDiffRow]:
    tier = {
        "REPLACED": 4,
        "MODIFIED": 3,
        "ADDED": 2,
        "REMOVED": 2,
        "FORMAT_ONLY_CHANGE": 1,
        "UNCHANGED": 0,
    }
    return sorted(
        ctx.document_diff,
        key=lambda row: (-tier.get(row.status, 0), row.document_id),
    )


def _binned_counts(values: list[int], edges: list[float]) -> list[int]:
    if len(edges) < 2:
        return []
    counts = [0] * (len(edges) - 1)
    for value in values:
        index = len(edges) - 2
        for position in range(len(edges) - 1):
            if value < edges[position + 1]:
                index = position
                break
        counts[index] += 1
    return counts


# --- Section renderers ---

def _render_section_1(ctx: _ReportContext) -> str:
    v1 = ctx.v1_manifest
    v2 = ctx.v2_manifest
    mmd_value = _metric_change(ctx.mmd_metric)
    threshold = _mmd_threshold(ctx)
    warning = getattr(threshold, "warning", None) if threshold is not None else None
    parts = ['<div class="dl-kpi-grid">']
    parts.append(_kpi_card("Overall status", ctx.health.overall, status=ctx.health.overall))
    parts.append(
        _kpi_card("Documents", str(v1.document_count) + " to " + str(v2.document_count))
    )
    parts.append(_kpi_card("Chunks", str(v1.chunk_count) + " to " + str(v2.chunk_count)))
    parts.append(_kpi_card("Embedding model", _describe_change(v1.embedding_model, v2.embedding_model)))
    parts.append(
        _kpi_card(
            "Embedding dimension",
            _describe_change(v1.embedding_dimension, v2.embedding_dimension),
        )
    )
    parts.append(
        _kpi_card(
            "MMD",
            _cell(mmd_value),
            status=ctx.mmd_metric.status,
            detail="warning at " + _cell(warning),
        )
    )
    parts.append(_kpi_card("Duplicate rate (V2)", _cell(ctx.dup_v2.duplicate_rate)))
    parts.append(
        _kpi_card(
            "Breached dimensions",
            str(len(ctx.health.exceeded)),
            status="WARN" if ctx.health.exceeded else "PASS",
        )
    )
    parts.append("</div>")
    if ctx.health.exceeded:
        links = ['<div class="dl-jumps">']
        for name in ctx.health.exceeded:
            target = _DIMENSION_SECTION.get(name, 1)
            links.append(
                '<a class="dl-jump" href="#sec-'
                + str(target)
                + '">'
                + esc(name)
                + "</a>"
            )
        links.append("</div>")
        parts.append("".join(links))
    return _section_container(1, _SECTION_TITLES[0], "".join(parts))


def _manifest_card(label: str, manifest: Manifest) -> str:
    pairs = [
        ("snapshot_id", manifest.snapshot_id),
        ("schema_version", manifest.schema_version),
        ("content_digest", manifest.content_digest),
        ("driftlens_version", manifest.driftlens_version),
        ("created_at", manifest.created_at),
        ("source_type", manifest.source_type),
        ("granularity", manifest.granularity),
        ("normalization_version", manifest.normalization_version),
        ("tokenizer_id", manifest.tokenizer_id),
        ("document_count", manifest.document_count),
        ("chunk_count", manifest.chunk_count),
        ("embedding_model", manifest.embedding_model),
        ("embedding_provider", manifest.embedding_provider),
        ("embedding_dimension", manifest.embedding_dimension),
        ("embedding_dtype", manifest.embedding_dtype),
        ("similarity_metric", manifest.similarity_metric),
        ("embedding_normalized", manifest.embedding_normalized),
    ]
    return (
        '<div class="dl-card"><h3 class="dl-subtitle">'
        + esc(label)
        + "</h3>"
        + _key_value_grid(pairs)
        + "</div>"
    )


def _render_section_2(ctx: _ReportContext) -> str:
    body = (
        '<div class="dl-card-grid">'
        + _manifest_card("V1 snapshot", ctx.v1_manifest)
        + _manifest_card("V2 snapshot", ctx.v2_manifest)
        + "</div>"
    )
    return _section_container(2, _SECTION_TITLES[1], body)


def _render_section_3(ctx: _ReportContext) -> str:
    counts = {status: 0 for status in _DOCUMENT_STATUSES}
    for row in ctx.document_diff:
        counts[row.status] = counts.get(row.status, 0) + 1
    both = {"MODIFIED", "REPLACED", "FORMAT_ONLY_CHANGE", "UNCHANGED"}
    series_v1 = [
        float(counts[status]) if status in both or status == "REMOVED" else 0.0
        for status in _DOCUMENT_STATUSES
    ]
    series_v2 = [
        float(counts[status]) if status in both or status == "ADDED" else 0.0
        for status in _DOCUMENT_STATUSES
    ]
    chart = svg_grouped_bar_chart(
        list(_DOCUMENT_STATUSES), series_v1, series_v2, title="Documents by status"
    )
    ranked = _ranked_document_rows(ctx)
    if ctx.top_n_ids is None:
        top_rows = ranked[:10]
        rest = ranked[10:]
    else:
        wanted = set(ctx.top_n_ids)
        top_rows = [row for row in ranked if row.document_id in wanted]
        rest = [row for row in ranked if row.document_id not in wanted]
    headers = ["document_id", "status", "similarity", "reconstruction V1/V2"]
    table = _data_table(headers, [_document_row(row) for row in top_rows])
    body = chart + table
    if rest:
        body += (
            '<details class="dl-details"><summary>Remaining documents ('
            + str(len(rest))
            + ")</summary>"
            + _data_table(headers, [_document_row(row) for row in rest])
            + "</details>"
        )
    return _section_container(3, _SECTION_TITLES[2], body)


def _document_row(row: DocumentDiffRow) -> list[Any]:
    return [
        row.document_id,
        row.status,
        format_value(row.similarity),
        _cell(row.reconstruction_confidence_v1)
        + " / "
        + _cell(row.reconstruction_confidence_v2),
    ]


def _render_section_4(ctx: _ReportContext) -> str:
    counts = {status: 0 for status in _CHUNK_STATUSES}
    for _first, _second, status in ctx.chunk_diff.pairs:
        counts[status] = counts.get(status, 0) + 1
    counts["REMOVED"] += len(ctx.chunk_diff.unmatched_v1)
    counts["ADDED"] += len(ctx.chunk_diff.unmatched_v2)
    both = {"UNCHANGED", "MODIFIED", "MOVED"}
    series_v1 = [
        float(counts[status]) if status in both or status == "REMOVED" else 0.0
        for status in _CHUNK_STATUSES
    ]
    series_v2 = [
        float(counts[status]) if status in both or status == "ADDED" else 0.0
        for status in _CHUNK_STATUSES
    ]
    chart = svg_grouped_bar_chart(
        list(_CHUNK_STATUSES), series_v1, series_v2, title="Chunks by status"
    )
    table = _data_table(
        ["measure", "V1", "V2"],
        [
            ["manifest chunk_count", str(ctx.v1_manifest.chunk_count), str(ctx.v2_manifest.chunk_count)],
            ["matched pairs", str(len(ctx.chunk_diff.pairs)), str(len(ctx.chunk_diff.pairs))],
            ["unmatched", str(len(ctx.chunk_diff.unmatched_v1)), str(len(ctx.chunk_diff.unmatched_v2))],
        ],
    )
    return _section_container(4, _SECTION_TITLES[3], chart + table)


def _render_section_5(ctx: _ReportContext) -> str:
    if not ctx.field_drifts:
        return _section_container(5, _SECTION_TITLES[4], _empty_state("no metadata fields"))
    drifts = sorted(
        ctx.field_drifts,
        key=lambda drift: (drift.distance is None, -(drift.distance or 0.0), drift.name),
    )
    labels = [drift.name for drift in drifts]
    values = [float(drift.distance or 0.0) for drift in drifts]
    chart = svg_ranked_horizontal_bars(labels, values, title="Metadata drift by distance")
    rows = [
        [
            drift.name,
            format_value(drift.type_v1),
            format_value(drift.type_v2),
            format_value(drift.null_rate_v1),
            format_value(drift.null_rate_v2),
            format_value(drift.distance_kind),
            format_value(drift.distance),
        ]
        for drift in drifts
    ]
    table = _data_table(
        ["field", "type V1", "type V2", "null rate V1", "null rate V2", "distance kind", "distance"],
        rows,
    )
    return _section_container(5, _SECTION_TITLES[4], chart + table)


def _render_section_6(ctx: _ReportContext) -> str:
    edges = list(FIXED_BIN_EDGES)
    v1_values = list(ctx.profile_v1.chunks_per_document.values()) if ctx.profile_v1 else []
    v2_values = list(ctx.profile_v2.chunks_per_document.values()) if ctx.profile_v2 else []
    counts_v1 = _binned_counts(v1_values, edges)
    counts_v2 = _binned_counts(v2_values, edges)
    chart = svg_histogram(edges, counts_v1, counts_v2, title="Chunks per document")
    rows = []
    for category in _ANOMALY_CATEGORIES:
        first = getattr(ctx.anomaly_v1, category, 0) if ctx.anomaly_v1 else 0
        second = getattr(ctx.anomaly_v2, category, 0) if ctx.anomaly_v2 else 0
        rows.append([category, str(first), str(second)])
    table = _data_table(["anomaly category", "V1", "V2"], rows)
    return _section_container(6, _SECTION_TITLES[5], chart + table)


def _render_section_7(ctx: _ReportContext) -> str:
    banner = (
        '<div class="dl-alert">'
        + _status_badge(ctx.compat.state)
        + '<span class="dl-alert-text">'
        + esc(ctx.compat.state)
        + "</span></div>"
    )
    space = ctx.compat.space_test
    if space is None:
        space_block = _empty_state("space consistency not computed")
    else:
        space_block = _data_table(
            ["anchors", "cosine p5", "median cosine", "agreement rate"],
            [
                [
                    format_value(space.anchor_count),
                    format_value(space.cosine_p5),
                    format_value(space.cosine_median),
                    format_value(space.agreement_rate),
                ]
            ],
        )
    summary_table = _data_table(
        ["property", "V1", "V2"],
        [
            ["model", _cell(ctx.v1_summary.model), _cell(ctx.v2_summary.model)],
            ["provider", _cell(ctx.v1_summary.provider), _cell(ctx.v2_summary.provider)],
            ["dtype", _cell(ctx.v1_summary.dtype), _cell(ctx.v2_summary.dtype)],
            [
                "similarity_metric",
                _cell(ctx.v1_summary.similarity_metric),
                _cell(ctx.v2_summary.similarity_metric),
            ],
            [
                "coverage",
                format_value(ctx.v1_summary.coverage),
                format_value(ctx.v2_summary.coverage),
            ],
        ],
    )
    return _section_container(7, _SECTION_TITLES[6], banner + space_block + summary_table)


def _render_section_8(ctx: _ReportContext) -> str:
    value = _as_float(_metric_change(ctx.mmd_metric))
    threshold = _mmd_threshold(ctx)
    warning = getattr(threshold, "warning", None) if threshold is not None else None
    critical = getattr(threshold, "critical", None) if threshold is not None else None
    maximum = _threshold_max(value, warning, critical)
    bar = _svg_threshold_linear_bar(
        value=value,
        warning=warning,
        critical=critical,
        maximum=maximum,
        title="Maximum mean discrepancy",
    )
    details = ctx.mmd_metric.details or {}
    rows = [
        ["MMD", format_value(_metric_change(ctx.mmd_metric))],
        ["status", ctx.mmd_metric.status],
        ["estimator", format_value(details.get("estimator"))],
        ["bandwidth sigma", format_value(details.get("sigma"))],
        ["warning threshold", format_value(warning)],
        ["critical threshold", format_value(critical)],
    ]
    sample = ctx.mmd_metric.sample
    if sample is not None:
        rows.append(["V1 subsample", str(sample.v1_n) + " of " + str(sample.v1_of)])
        rows.append(["V2 subsample", str(sample.v2_n) + " of " + str(sample.v2_of)])
        rows.append(["seed", str(sample.seed)])
    table = _data_table(["parameter", "value"], rows)
    return _section_container(8, _SECTION_TITLES[7], bar + table)


def _render_section_9(ctx: _ReportContext) -> str:
    cards = (
        '<div class="dl-kpi-grid">'
        + _kpi_card("V1 duplicate rate", _cell(ctx.dup_v1.duplicate_rate))
        + _kpi_card("V2 duplicate rate", _cell(ctx.dup_v2.duplicate_rate))
        + _kpi_card("V1 groups", str(len(ctx.dup_v1.groups)))
        + _kpi_card("V2 groups", str(len(ctx.dup_v2.groups)))
        + "</div>"
    )
    rows = []
    for group in ctx.dup_v2.groups[:20]:
        rows.append(
            [
                group.representative_id,
                group.class_,
                str(len(group.member_ids)),
                ", ".join(group.member_ids),
            ]
        )
    table = _data_table(["representative_id", "class", "members", "member_ids"], rows)
    return _section_container(9, _SECTION_TITLES[8], cards + table)


def _render_section_10(ctx: _ReportContext) -> str:
    return _section_container(
        10, _SECTION_TITLES[9], _empty_state("not run: not implemented until release 0.3")
    )


def _render_section_11(ctx: _ReportContext) -> str:
    return _section_container(
        11,
        _SECTION_TITLES[10],
        _empty_state("not run: retrieval data unavailable (no evaluation set provided)"),
    )


def _render_section_12(ctx: _ReportContext) -> str:
    details = ctx.mmd_metric.details or {}
    rows = [
        ["MMD point estimate", format_value(_metric_change(ctx.mmd_metric))],
        ["estimator", format_value(details.get("estimator"))],
        ["bandwidth sigma", format_value(details.get("sigma"))],
        ["metric status", ctx.mmd_metric.status],
        ["determinism level", ctx.determinism_level],
        ["borderline decisions", str(ctx.borderline)],
    ]
    sample = ctx.mmd_metric.sample
    if sample is not None:
        rows.append(["V1 subsample", str(sample.v1_n) + " of " + str(sample.v1_of)])
        rows.append(["V2 subsample", str(sample.v2_n) + " of " + str(sample.v2_of)])
        rows.append(["seed", str(sample.seed)])
    body = (
        '<details class="dl-details"><summary>Statistical parameters</summary>'
        + _data_table(["parameter", "value"], rows)
        + "</details>"
    )
    return _section_container(12, _SECTION_TITLES[11], body)


_CONFIG_CATEGORIES = (
    (
        "Input",
        ("schema", "schema_a", "schema_b", "identity", "documents", "chunk_match", "text", "metadata"),
    ),
    ("Embeddings", ("embeddings",)),
    (
        "Analysis",
        ("analysis", "drift", "neighbour", "significance", "cluster", "baseline", "retrieval", "ci", "privacy"),
    ),
    ("Chunking", ("chunking",)),
    ("Duplicates", ("duplicates",)),
    ("Limits", ("limits",)),
)


def _flatten_config(value: Any, prefix: str) -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    if hasattr(value, "__dataclass_fields__"):
        for field in dataclass_fields(value):
            pairs.extend(_flatten_config(getattr(value, field.name), prefix + field.name + "."))
    elif isinstance(value, dict):
        for key in sorted(value.keys(), key=str):
            pairs.extend(_flatten_config(value[key], prefix + str(key) + "."))
    elif isinstance(value, (list, tuple, set, frozenset)):
        pairs.append((prefix[:-1], ", ".join(str(item) for item in value)))
    else:
        pairs.append((prefix[:-1], value))
    return pairs


def _render_section_13(ctx: _ReportContext) -> str:
    parts = []
    for label, names in _CONFIG_CATEGORIES:
        pairs: list[tuple[str, Any]] = []
        for name in names:
            if hasattr(ctx.cfg, name):
                pairs.extend(_flatten_config(getattr(ctx.cfg, name), name + "."))
        parts.append(
            '<details class="dl-accordion"><summary>'
            + esc(label)
            + "</summary>"
            + _key_value_grid(pairs)
            + "</details>"
        )
    return _section_container(13, _SECTION_TITLES[12], "".join(parts))


def _render_section_14(ctx: _ReportContext) -> str:
    rows = [
        ["V1 content digest", ctx.v1_manifest.content_digest],
        ["V2 content digest", ctx.v2_manifest.content_digest],
        ["determinism level", ctx.determinism_level],
        ["borderline decisions", str(ctx.borderline)],
        ["random seed", str(ctx.cfg.analysis.seed)],
        ["unsafe pickle", _bool_text(ctx.unsafe_pickle)],
        ["V1 snapshot id", ctx.v1_manifest.snapshot_id],
        ["V2 snapshot id", ctx.v2_manifest.snapshot_id],
    ]
    for key in sorted(ctx.limits_applied.keys(), key=str):
        rows.append(["limit: " + str(key), ctx.limits_applied[key]])
    return _section_container(14, _SECTION_TITLES[13], _data_table(["item", "value"], rows))


def _render_section_15(ctx: _ReportContext) -> str:
    items = ctx.sink.items()
    if not items:
        return _section_container(15, _SECTION_TITLES[14], _empty_state("no warnings or degraded conditions"))
    rows = [
        [
            item.code.value,
            CODE_CLASS[item.code].value,
            str(item.count),
            format_value(item.dimension),
            item.message,
        ]
        for item in items
    ]
    body = _data_table(["code", "severity", "count", "dimension", "message"], rows)
    if ctx.diagnostic:
        excerpts = "".join(
            '<pre class="dl-pre">' + esc(item.message) + "</pre>" for item in items
        )
        body += (
            '<details class="dl-details" open><summary>Diagnostic message excerpts</summary>'
            + excerpts
            + "</details>"
        )
    return _section_container(15, _SECTION_TITLES[14], body)


def _render_section_16(ctx: _ReportContext) -> str:
    if not ctx.breakdowns:
        return _section_container(
            16,
            _SECTION_TITLES[15],
            _empty_state("not run: no breakdowns requested (--breakdown-by not specified)"),
        )
    parts = []
    for field in sorted(ctx.breakdowns.keys(), key=str):
        rows = []
        for row in ctx.breakdowns[field]:
            status_counts = ", ".join(
                str(key) + "=" + str(value) for key, value in sorted(row.status_counts.items())
            )
            rows.append(
                [
                    row.segment_value,
                    _bool_text(row.is_other),
                    str(row.document_count),
                    status_counts,
                    format_value(row.mmd),
                ]
            )
        parts.append('<h3 class="dl-subtitle">' + esc(field) + "</h3>")
        parts.append(
            _data_table(["segment", "is_other", "documents", "status counts", "MMD"], rows)
        )
    return _section_container(16, _SECTION_TITLES[15], "".join(parts))


def _nav_links() -> str:
    parts = ['<p class="dl-nav-title">Sections</p><ul class="dl-nav-list">']
    for index, title in enumerate(_SECTION_TITLES, start=1):
        parts.append(
            '<li><a class="dl-nav-link" href="#sec-'
            + str(index)
            + '">'
            + esc(title)
            + "</a></li>"
        )
    parts.append("</ul>")
    return "".join(parts)


# --- Assembler & writer ---

def render_html_report(
    *,
    health: HealthReport,
    v1_manifest: Manifest,
    v2_manifest: Manifest,
    document_diff: list[DocumentDiffRow],
    chunk_diff: ChunkDiffResult,
    field_drifts: list[FieldDrift],
    profile_v1: ChunkProfile | None,
    profile_v2: ChunkProfile | None,
    anomaly_v1: AnomalyCounts | None,
    anomaly_v2: AnomalyCounts | None,
    compat: CompatibilityResult,
    v1_summary: EmbeddingSnapshotSummary,
    v2_summary: EmbeddingSnapshotSummary,
    mmd_metric: Metric,
    dup_v1: ExactDuplicateResult,
    dup_v2: ExactDuplicateResult,
    cfg: Config,
    sink: DiagnosticSink,
    determinism_level: str,
    borderline: int,
    limits_applied: dict[str, Any],
    unsafe_pickle: bool,
    top_n_ids: list[str] | None = None,
    diagnostic: bool = False,
    breakdowns: dict[str, list[Any]] | None = None,
) -> str:
    """Render one self-contained HTML report document."""
    ctx = _ReportContext(
        health=health,
        v1_manifest=v1_manifest,
        v2_manifest=v2_manifest,
        document_diff=document_diff,
        chunk_diff=chunk_diff,
        field_drifts=field_drifts,
        profile_v1=profile_v1,
        profile_v2=profile_v2,
        anomaly_v1=anomaly_v1,
        anomaly_v2=anomaly_v2,
        compat=compat,
        v1_summary=v1_summary,
        v2_summary=v2_summary,
        mmd_metric=mmd_metric,
        dup_v1=dup_v1,
        dup_v2=dup_v2,
        cfg=cfg,
        sink=sink,
        determinism_level=determinism_level,
        borderline=borderline,
        limits_applied=limits_applied,
        unsafe_pickle=unsafe_pickle,
        top_n_ids=top_n_ids,
        diagnostic=diagnostic,
        breakdowns=breakdowns,
    )
    sections = [
        _render_section_1(ctx),
        _render_section_2(ctx),
        _render_section_3(ctx),
        _render_section_4(ctx),
        _render_section_5(ctx),
        _render_section_6(ctx),
        _render_section_7(ctx),
        _render_section_8(ctx),
        _render_section_9(ctx),
        _render_section_10(ctx),
        _render_section_11(ctx),
        _render_section_12(ctx),
        _render_section_13(ctx),
        _render_section_14(ctx),
        _render_section_15(ctx),
        _render_section_16(ctx),
    ]
    return (
        "<!DOCTYPE html>"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>DriftLens drift report</title>"
        "<style>"
        + _REPORT_CSS
        + "</style></head><body>"
        '<div class="dl-layout">'
        '<nav class="dl-sidebar" aria-label="Report sections">'
        + _nav_links()
        + "</nav>"
        '<main class="dl-main">'
        + "".join(sections)
        + "</main></div></body></html>"
    )


def write_html_report(html_str: str, path: Path) -> None:
    """Write an HTML report, creating missing parent directories."""
    target = Path(path)
    parent = target.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_str, encoding="utf-8")
