"""Terminal report rendering (Step 32)."""

from __future__ import annotations

from driftlens.analysis.health import (
    HealthReport,
    Metric,
    build_health_report,
)
from driftlens.config.schema import default_config
from driftlens.errors import Code, DiagnosticSink
from driftlens.reports.terminal import (
    format_dimension_block,
    format_metric_row,
    format_status_line,
    format_tag,
    format_value,
    format_warnings_block,
    render_ci_report,
    render_quiet_report,
    render_terminal_report,
)
from driftlens.schema.manifest import FieldMappingSummary, Manifest


def _manifest(snapshot_id):
    return Manifest(
        schema_version="1.0",
        snapshot_id=snapshot_id,
        content_digest="digest",
        normalization_version="nfc-ws-1",
        tokenizer_id="simple-v1",
        text_params={"min_overlap_chars": 16, "overlap_window": 2000, "casefold": True},
        driftlens_version="0.1.0",
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


def _render(health, *, sink=None, isatty=False):
    return render_terminal_report(
        health=health,
        v1_manifest=_manifest("v1"),
        v2_manifest=_manifest("v2"),
        sink=sink if sink is not None else DiagnosticSink(),
        determinism_level="TOLERANCE",
        isatty=isatty,
        seed=42,
    )


def test_format_value_none_empty() -> None:
    assert format_value(None) == ""


def test_format_value_with_unit() -> None:
    assert format_value(0.05, unit="pts") == "0.05 pts"


def test_format_tag_plain_when_not_tty() -> None:
    tag = format_tag("WARN", isatty=False)
    assert tag == "[WARN]"
    assert "\033[" not in tag


def test_format_tag_colored_when_tty() -> None:
    tag = format_tag("WARN", isatty=True)
    assert "\033[" in tag
    assert tag.endswith("\033[0m")


def test_format_tag_pass_empty() -> None:
    assert format_tag("PASS", isatty=False) == ""


def test_format_metric_row_skipped_shape() -> None:
    metric = Metric(name="mmd", status="SKIPPED", reason="MIXED_MODELS")
    assert format_metric_row(metric, isatty=False) == "  mmd SKIPPED (MIXED_MODELS)"


def test_format_metric_row_na_shape() -> None:
    metric = Metric(name="duplicate_rate_near", status="N/A", reason=None)
    assert format_metric_row(metric, isatty=False) == "  duplicate_rate_near n/a"


def test_format_metric_row_pair_only_blanks_v1_v2() -> None:
    metric = Metric(
        name="mmd", v1=None, v2=None, change=0.083, status="UNBASELINED"
    )
    row = format_metric_row(metric, isatty=False)
    start = 2 + 28
    assert row[start : start + 14].strip() == ""
    assert row[start + 14 : start + 28].strip() == ""
    assert row[start + 28 : start + 42].strip() == "0.083"


def test_dimension_omitted_when_na_and_empty() -> None:
    health = build_health_report({}, cfg=default_config())
    output = _render(health)
    assert "SEMANTIC" not in output
    assert "RETRIEVAL" not in output


def test_dimension_included_when_it_has_metrics_even_if_na() -> None:
    health = build_health_report(
        {"semantic": [Metric(name="solo", change=1.5, status="REPORT_ONLY")]},
        cfg=default_config(),
    )
    assert "SEMANTIC" in _render(health)


def test_status_line_unbaselined() -> None:
    health = build_health_report({}, cfg=default_config())
    lines = _render(health).split("\n")
    assert lines[1] == "STATUS: UNBASELINED"


def test_status_line_pass() -> None:
    health = HealthReport(dimensions={}, overall="PASS", exceeded=[])
    assert format_status_line(health) == "STATUS: PASS"


def test_status_line_warn_lists_dimensions() -> None:
    health = HealthReport(
        dimensions={}, overall="WARN", exceeded=["chunking", "duplication"]
    )
    assert format_status_line(health) == "STATUS: WARN CHUNKING, DUPLICATION"


def test_warnings_block_omitted_when_empty() -> None:
    output = _render(build_health_report({}, cfg=default_config()))
    assert "WARNINGS" not in output


def test_warnings_block_truncated_at_10() -> None:
    sink = DiagnosticSink()
    for index in range(15):
        sink.add(Code.MODEL_UNKNOWN, f"message {index}")
    block = format_warnings_block(sink)
    assert block.count("\n") == 11
    assert "+5 more, see --json-out" in block


def test_quiet_report_only_non_pass_dimensions() -> None:
    health = build_health_report(
        {
            "content": [
                Metric(
                    name="content_change_rate",
                    change=0.0,
                    status="PASS",
                    is_threshold_key=True,
                )
            ],
            "chunking": [
                Metric(
                    name="chunk_ks",
                    v1=0.1,
                    v2=0.5,
                    change=0.4,
                    status="WARN",
                    is_threshold_key=True,
                )
            ],
        },
        cfg=default_config(),
    )
    text = render_quiet_report(health)
    assert "CHUNKING" in text
    assert "CONTENT" not in text


def test_ci_report_seven_lines() -> None:
    health = build_health_report({}, cfg=default_config())
    text, exit_code = render_ci_report(health)
    assert text.count("\n") == 6
    assert exit_code == 0


def test_full_report_line_width_budget() -> None:
    health = build_health_report(
        {
            "content": [
                Metric(
                    name="content_change_rate",
                    change=0.25,
                    status="WARN",
                    is_threshold_key=True,
                )
            ],
            "chunking": [
                Metric(
                    name="chunk_ks",
                    v1=0.1,
                    v2=0.6,
                    change=0.5,
                    status="FAIL",
                    is_threshold_key=True,
                )
            ],
            "embedding": [
                Metric(
                    name="mmd",
                    change=0.083,
                    status="PASS",
                    is_threshold_key=True,
                )
            ],
        },
        cfg=default_config(),
    )
    sink = DiagnosticSink()
    sink.add(Code.LABEL_CONFLICT, "labels disagree", dimension="embedding")
    output = _render(health, sink=sink)
    for line in output.split("\n"):
        assert len(line) <= 80
