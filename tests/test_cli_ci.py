"""CI-mode exit codes (Step 49, section 13.5)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from driftlens.__main__ import main
from driftlens.analysis.health import (
    Dimension,
    HealthReport,
    Metric,
    build_health_report,
)
from driftlens.cli.ci import (
    COMPAT_SKIP_REASONS,
    compute_ci_exit_code,
    dimension_gate_contribution,
)
from driftlens.config.schema import DIMENSIONS, default_config
from driftlens.reports.terminal import render_ci_report


@pytest.fixture()
def workdir():
    base = Path(__file__).resolve().parent / "_cli_ci_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _write_jsonl(path, rows):
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _dim(status, reasons=()):
    metrics = [
        Metric(name=f"m{index}", status="SKIPPED", reason=reason)
        for index, reason in enumerate(reasons)
    ]
    return Dimension(name="embedding", metrics=metrics, status=status)


def _metric(name, status, *, v1=None, v2=None, change=None, reason=None):
    return Metric(
        name=name,
        v1=v1,
        v2=v2,
        change=change,
        status=status,
        reason=reason,
        is_threshold_key=True,
    )


def _report_with(statuses):
    dimensions = {
        name: Dimension(name=name, metrics=[], status=statuses.get(name, "N/A"))
        for name in DIMENSIONS
    }
    return HealthReport(dimensions=dimensions, overall="FAIL", exceeded=[])


def test_ungated_dimension_never_contributes() -> None:
    dim = Dimension(name="embedding", metrics=[Metric(name="mmd", status="FAIL")], status="FAIL")
    assert (
        dimension_gate_contribution(
            dim,
            gate_level=None,
            on_skipped="fail",
            on_incompatible="fail",
        )
        == 0
    )


def test_gated_warning_level_warn_gives_1() -> None:
    dim = Dimension(name="duplication", metrics=[Metric(name="duplicate_rate", status="WARN")], status="WARN")
    assert (
        dimension_gate_contribution(
            dim,
            gate_level="warning",
            on_skipped="ignore",
            on_incompatible="ignore",
        )
        == 1
    )


def test_gated_warning_level_fail_gives_2() -> None:
    dim = Dimension(name="duplication", metrics=[Metric(name="duplicate_rate", status="FAIL")], status="FAIL")
    assert (
        dimension_gate_contribution(
            dim,
            gate_level="warning",
            on_skipped="ignore",
            on_incompatible="ignore",
        )
        == 2
    )


def test_gated_critical_level_ignores_warn() -> None:
    dim = Dimension(name="retrieval", metrics=[Metric(name="recall_at_5", status="WARN")], status="WARN")
    assert (
        dimension_gate_contribution(
            dim,
            gate_level="critical",
            on_skipped="ignore",
            on_incompatible="ignore",
        )
        == 0
    )


def test_gated_critical_level_fail_still_gives_2() -> None:
    dim = Dimension(name="retrieval", metrics=[Metric(name="recall_at_5", status="FAIL")], status="FAIL")
    assert (
        dimension_gate_contribution(
            dim,
            gate_level="critical",
            on_skipped="ignore",
            on_incompatible="ignore",
        )
        == 2
    )


def test_unbaselined_always_zero_even_when_gated() -> None:
    dim = Dimension(name="content", metrics=[], status="UNBASELINED")
    assert (
        dimension_gate_contribution(
            dim,
            gate_level="critical",
            on_skipped="fail",
            on_incompatible="fail",
        )
        == 0
    )


def test_skipped_compat_caused_uses_on_incompatible_ignore() -> None:
    dim = _dim("SKIPPED", ["SKIPPED_MODEL_CHANGED"])
    assert (
        dimension_gate_contribution(
            dim,
            gate_level="critical",
            on_skipped="fail",
            on_incompatible="ignore",
        )
        == 0
    )


def test_skipped_non_compat_uses_on_skipped_warn() -> None:
    dim = _dim("SKIPPED", ["INSUFFICIENT_DATA"])
    assert (
        dimension_gate_contribution(
            dim,
            gate_level="critical",
            on_skipped="warn",
            on_incompatible="fail",
        )
        == 1
    )


def test_skipped_mixed_reasons_compat_takes_precedence() -> None:
    dim = _dim("SKIPPED", ["SKIPPED_MIXED_MODELS", "RETRIEVER_UNAVAILABLE"])
    assert "SKIPPED_MIXED_MODELS" in COMPAT_SKIP_REASONS
    assert (
        dimension_gate_contribution(
            dim,
            gate_level="critical",
            on_skipped="fail",
            on_incompatible="ignore",
        )
        == 0
    )


def test_compute_ci_exit_code_max_across_dimensions() -> None:
    health = _report_with({"content": "UNBASELINED", "chunking": "WARN", "metadata": "FAIL"})
    cfg = default_config().ci
    cfg.fail_on = {"content": "warning", "chunking": "warning", "metadata": "critical"}
    assert compute_ci_exit_code(health, cfg=cfg) == 2


def test_compute_ci_exit_code_all_pass_zero() -> None:
    health = _report_with({name: "PASS" for name in DIMENSIONS})
    cfg = default_config().ci
    cfg.fail_on = {name: "critical" for name in DIMENSIONS}
    assert compute_ci_exit_code(health, cfg=cfg) == 0


def test_render_ci_lines_seven_lines_plus_exit_code() -> None:
    health = build_health_report({}, cfg=default_config())
    text, code = render_ci_report(health, cfg=default_config().ci)
    lines = text.split("\n")
    assert len(lines) == 8
    assert lines[-1] == "Exit code: 0"
    assert [line.split()[0] for line in lines[:7]] == [
        name.upper() for name in DIMENSIONS
    ]


def test_render_ci_lines_skipped_shows_reason() -> None:
    health = build_health_report(
        {
            "embedding": [
                Metric(
                    name="mmd",
                    status="SKIPPED",
                    reason="SKIPPED_MODEL_CHANGED",
                )
            ]
        },
        cfg=default_config(),
    )
    text, _ = render_ci_report(health, cfg=default_config().ci)
    assert "EMBEDDING SKIPPED (SKIPPED_MODEL_CHANGED)" in text


def test_render_ci_matches_spec_worked_example_shape() -> None:
    health = build_health_report(
        {
            "content": [_metric("content_change_rate", "PASS", v1=0.0, v2=0.1, change=0.1)],
            "chunking": [_metric("chunk_ks", "PASS", v1=0.1, v2=0.2, change=0.1)],
            "metadata": [_metric("metadata_js", "PASS", v1=0.0, v2=0.1, change=0.1)],
            "embedding": [
                _metric("mmd", "SKIPPED", reason="SKIPPED_MODEL_CHANGED"),
            ],
            "semantic": [
                _metric("semantic_ks", "SKIPPED", reason="SKIPPED_MODEL_CHANGED"),
            ],
            "duplication": [_metric("duplicate_rate", "WARN", v1=0.1, v2=0.2, change=0.1)],
            "retrieval": [_metric("recall_at_5", "FAIL", v1=82.1, v2=76.4, change=-5.7)],
        },
        cfg=default_config(),
    )
    text, code = render_ci_report(health, cfg=default_config().ci)
    assert code == 2
    assert text.split("\n")[-1] == "Exit code: 2"


def test_cli_compare_ci_flag_uses_real_exit_code(workdir) -> None:
    first = workdir / "a.jsonl"
    second = workdir / "b.jsonl"
    _write_jsonl(first, [{"id": "c1", "text": "a"}, {"id": "c2", "text": "b"}])
    long_text = " ".join(["word"] * 40)
    _write_jsonl(
        second,
        [
            {"id": "c1", "text": long_text},
            {"id": "c2", "text": long_text + " tail"},
        ],
    )
    config = workdir / "cfg.yaml"
    config.write_text("ci:\n  fail_on:\n    chunk_ks: 0.0001\n", encoding="utf-8")

    with_ci = main(
        ["compare", str(first), str(second), "--config", str(config), "--ci"]
    )
    without_ci = main(["compare", str(first), str(second), "--config", str(config)])
    assert with_ci == 2
    assert without_ci == 0
