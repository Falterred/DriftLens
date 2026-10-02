"""Self-contained HTML report rendering (Step 42)."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from driftlens.analysis.chunking import AnomalyCounts, ChunkProfile
from driftlens.analysis.compatibility import CompatibilityResult, SpaceConsistencyResult
from driftlens.analysis.content import ChunkDiffResult
from driftlens.analysis.duplicates import ExactDuplicateResult
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.analysis.health import HealthReport, make_metric
from driftlens.cli.compare import build_arg_parser, run_compare
from driftlens.config.schema import default_config
from driftlens.errors import Code, DiagnosticSink
from driftlens.reports.html import (
    _svg_threshold_linear_bar,
    esc,
    render_html_report,
    svg_bar_chart,
    svg_grouped_bar_chart,
    svg_histogram,
    svg_ranked_horizontal_bars,
    write_html_report,
)
from driftlens.schema.manifest import FieldMappingSummary, Manifest

HTML_MODULE = Path(__file__).resolve().parent.parent / "driftlens" / "reports" / "html.py"

EXPECTED_TITLES = (
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


@pytest.fixture()
def workdir():
    base = Path(__file__).resolve().parent / "_reports_html_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _manifest(snapshot_id):
    return Manifest(
        schema_version="1.0",
        snapshot_id=snapshot_id,
        content_digest="digest",
        normalization_version="nfc-ws-1",
        tokenizer_id="simple-v1",
        text_params={"min_overlap_chars": 16, "overlap_window": 2000, "casefold": True},
        driftlens_version="0.2.0",
        created_at="2026-01-01T00:00:00+00:00",
        source_type="jsonl",
        granularity="CHUNK",
        granularity_source="detected",
        document_count=1,
        chunk_count=2,
        embedding_dimension=4,
        embedding_dtype="float32",
        embedding_model="E1",
        embedding_model_counts={"E1": 2},
        embedding_provider=None,
        embedding_metadata_confidence={},
        similarity_metric="cosine",
        embedding_normalized=True,
        chunk_identity_levels={"a": [1]},
        identity_summary={"HIGH": 1, "MEDIUM": 0, "LOW": 0},
        field_mapping=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _summary(model="E1"):
    return EmbeddingSnapshotSummary(
        model=model,
        provider=None,
        dtype="float32",
        similarity_metric="cosine",
        similarity_metric_derived=False,
        normalized=True,
        confidence={},
        invalid_count=0,
        zero_count=0,
        coverage=1.0,
        norm_summary=None,
        embeddings_unavailable=False,
    )


def _compat(state="COMPATIBLE"):
    space = SpaceConsistencyResult(
        state="SAME_SPACE",
        anchor_count=10,
        agreement_rate=1.0,
        cosine_p5=0.99,
        cosine_median=0.999,
        reembedded_share=0.0,
        declared_models_differ=False,
        sample=None,
    )
    return CompatibilityResult(
        state=state,
        space_test=space,
        label_conflict=False,
        dimension_v1=4,
        dimension_v2=4,
        metric_v1="cosine",
        metric_v2="cosine",
        model_v1="E1",
        model_v2="E1",
    )


def _dups(rate=0.0):
    return ExactDuplicateResult(
        groups=[],
        units=[],
        redundant_count=0,
        total_chunks=0,
        duplicate_rate=rate,
        covered_fraction=0.0,
        sample=None,
    )


def _profile():
    return ChunkProfile(
        token_stats=None,
        char_stats=None,
        sentence_stats=None,
        chunks_per_document={},
        overlap_estimate=None,
    )


def _anomalies():
    return AnomalyCounts(
        empty=0,
        very_short=0,
        suspicious_short=0,
        oversized=0,
        over_embedder_limit=0,
        encoding_damage=0,
        anomalous_rate=0.0,
        total=0,
    )


def _render(**overrides):
    kwargs = {
        "health": HealthReport(dimensions={}, overall="UNBASELINED", exceeded=[]),
        "v1_manifest": _manifest("v1"),
        "v2_manifest": _manifest("v2"),
        "document_diff": [],
        "chunk_diff": ChunkDiffResult(pairs=[], unmatched_v1=[], unmatched_v2=[]),
        "field_drifts": [],
        "profile_v1": _profile(),
        "profile_v2": _profile(),
        "anomaly_v1": _anomalies(),
        "anomaly_v2": _anomalies(),
        "compat": _compat(),
        "v1_summary": _summary(),
        "v2_summary": _summary(),
        "mmd_metric": make_metric(
            "mmd",
            value=0.01,
            threshold=None,
            v1=0.01,
            v2=0.01,
            change=0.0,
            details={"sigma": 0.1, "estimator": "linear"},
        ),
        "dup_v1": _dups(),
        "dup_v2": _dups(),
        "cfg": default_config(),
        "sink": DiagnosticSink(),
        "determinism_level": "TOLERANCE",
        "borderline": 0,
        "limits_applied": {},
        "unsafe_pickle": False,
    }
    kwargs.update(overrides)
    return render_html_report(**kwargs)


def test_esc_handles_none():
    assert esc(None) == ""


def test_esc_escapes_html_special_chars():
    assert esc("<a href='x'>&") == "&lt;a href=&#x27;x&#x27;&gt;&amp;"


def test_no_script_tags_anywhere():
    html = _render()
    assert "<script" not in html.lower()


def test_self_contained_no_external_refs():
    html = _render()
    for token in ("http://", "https://", "url(", "<link", '<img src="http'):
        assert token not in html


def test_malicious_input_escaping():
    html = _render(
        v1_manifest=_manifest("<script>alert(1)</script>"),
        v2_manifest=_manifest("<img src=x onerror=alert(2)>"),
        v1_summary=_summary("<b>bold</b>"),
    )
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x onerror=alert(2)>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_no_unescaped_interpolation():
    source = HTML_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed = {
        "esc",
        "format_value",
        "_status_badge",
        "_kpi_card",
        "_section_container",
        "_empty_state",
        "_data_table",
        "_key_value_grid",
        "svg_bar_chart",
        "svg_grouped_bar_chart",
        "svg_histogram",
        "svg_ranked_horizontal_bars",
        "_svg_threshold_linear_bar",
    }
    allowed |= {"_render_section_" + str(i) for i in range(1, 17)}
    for node in ast.walk(tree):
        if isinstance(node, ast.FormattedValue):
            value = node.value
            assert isinstance(value, ast.Call)
            assert isinstance(value.func, ast.Name)
            assert value.func.id in allowed
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "format"


def test_svg_bar_chart_valid():
    svg = svg_bar_chart(["a", "b", "<x>"], [1.0, 2.0, 0.5], title="t")
    assert svg.startswith("<svg")
    assert "viewBox=" in svg
    assert svg.count("<rect") == 3
    assert "&lt;x&gt;" in svg


def test_svg_grouped_bar_chart_two_series():
    svg = svg_grouped_bar_chart(["a", "b"], [1.0, 2.0], [3.0, 4.0])
    assert svg.count("<rect") == 4
    assert "dl-bar-v1" in svg
    assert "dl-bar-v2" in svg


def test_svg_histogram_binning():
    svg = svg_histogram([0, 1, 2, 4], [3, 0, 1], [0, 2, 2])
    assert svg.count("<rect") == 6
    assert "0-1" in svg


def test_svg_handles_empty_and_zero_values():
    empty = svg_bar_chart([], [])
    assert "<svg" in empty and "no data" in empty
    zeros = svg_histogram([0, 1, 2], [0, 0], [0, 0])
    assert "<svg" in zeros
    assert "no data" in svg_ranked_horizontal_bars([], [])


def test_svg_threshold_linear_bar():
    svg = _svg_threshold_linear_bar(value=0.3, warning=0.5, critical=0.8)
    assert "dl-marker" in svg
    assert "dl-threshold-warning" in svg
    assert "dl-threshold-critical" in svg
    assert "measured 0.3" in svg


def test_all_16_sections_present():
    html = _render()
    for index in range(1, 17):
        assert 'id="sec-' + str(index) + '"' in html


def test_canonical_section_names():
    html = _render()
    positions = []
    for title in EXPECTED_TITLES:
        assert title in html
        positions.append(html.index(title))
    assert positions == sorted(positions)


def test_sidebar_navigation_anchors():
    html = _render()
    nav = html.split("<nav", 1)[1].split("</nav>", 1)[0]
    assert nav.count('<a class="dl-nav-link" href="#sec-') == 16
    assert 'href="#sec-16"' in nav


def test_section_not_run_placeholders():
    html = _render()
    assert "not run: not implemented until release 0.3" in html
    assert "not run: retrieval data unavailable (no evaluation set provided)" in html


def test_write_html_report_creates_directories(workdir):
    target = workdir / "nested" / "deep" / "report.html"
    write_html_report("<!DOCTYPE html>", target)
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "<!DOCTYPE html>"


def test_warnings_section_full_list():
    sink = DiagnosticSink()
    sink.add(Code.LOW_POWER, "first warning message")
    sink.add(Code.DOC_ORDER_UNKNOWN, "second warning message")
    html = _render(sink=sink)
    assert "first warning message" in html
    assert "second warning message" in html
    assert "no warnings or degraded conditions" not in html


def _write_jsonl(path, rows):
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def test_cli_compare_with_report_out(workdir):
    rows = [
        {"id": "c1", "text": "hello world", "embedding": [0.1, 0.2, 0.3]},
        {"id": "c2", "text": "goodbye moon", "embedding": [0.3, 0.2, 0.1]},
    ]
    first = workdir / "a.jsonl"
    second = workdir / "b.jsonl"
    _write_jsonl(first, rows)
    _write_jsonl(second, rows)
    out = workdir / "report.html"
    args = build_arg_parser().parse_args(
        ["compare", str(first), str(second), "--report-out", str(out)]
    )
    code = run_compare(args)
    assert code == 0
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert len(html) > 0
