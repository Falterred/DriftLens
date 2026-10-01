"""Analysis health status evaluation (Step 21)."""

from __future__ import annotations

from driftlens.analysis.health import (
    Dimension,
    Metric,
    build_health_report,
    dimension_status,
    evaluate_status,
    make_metric,
    overall_status,
)
from driftlens.config.schema import DIMENSIONS, Threshold, default_config


def test_report_only_never_gets_tagged_status() -> None:
    assert (
        evaluate_status(
            is_threshold_key=False,
            value=0.5,
            threshold=Threshold(0.1, 0.2),
            reason=None,
        )
        == "REPORT_ONLY"
    )


def test_skipped_beats_value_absence_distinction() -> None:
    assert (
        evaluate_status(
            is_threshold_key=True,
            value=None,
            threshold=None,
            reason="INSUFFICIENT_DATA",
        )
        == "SKIPPED"
    )
    assert (
        evaluate_status(is_threshold_key=True, value=None, threshold=None, reason=None)
        == "N/A"
    )


def test_unbaselined_vs_pass() -> None:
    assert (
        evaluate_status(is_threshold_key=True, value=0.05, threshold=None, reason=None)
        == "UNBASELINED"
    )
    assert (
        evaluate_status(
            is_threshold_key=True,
            value=0.05,
            threshold=Threshold(warning=0.1, critical=0.2),
            reason=None,
        )
        == "PASS"
    )


def test_warn_and_fail_boundaries() -> None:
    threshold = Threshold(warning=0.1, critical=0.2)

    def status(value: float) -> str:
        return evaluate_status(
            is_threshold_key=True, value=value, threshold=threshold, reason=None
        )

    assert status(0.1) == "PASS"
    assert status(0.10000001) == "WARN"
    assert status(0.2) == "WARN"
    assert status(0.20000001) == "FAIL"


def test_round9_prevents_float_noise_flap() -> None:
    assert (
        evaluate_status(
            is_threshold_key=True,
            value=0.1 + 1e-12,
            threshold=Threshold(warning=0.1, critical=None),
            reason=None,
        )
        == "PASS"
    )


def test_dimension_status_all_skipped() -> None:
    metrics = [
        make_metric(
            "mmd",
            value=None,
            threshold=None,
            reason="INSUFFICIENT_DATA",
        )
        for _ in range(3)
    ]
    assert dimension_status(metrics) == "SKIPPED"


def test_dimension_status_mixed_skipped_and_pass() -> None:
    metrics = [
        make_metric(
            "mmd", value=None, threshold=None, reason="INSUFFICIENT_DATA"
        ),
        make_metric("mmd", value=0.05, threshold=Threshold(warning=0.1, critical=0.2)),
    ]
    assert dimension_status(metrics) == "PASS"


def test_dimension_status_report_only_excluded() -> None:
    metrics = [
        Metric(name="avg_tokens", status="REPORT_ONLY"),
        Metric(name="empty", status="N/A"),
    ]
    assert dimension_status(metrics) == "N/A"


def test_dimension_status_unbaselined_does_not_mask_warn() -> None:
    metrics = [
        Metric(name="unbaselined", status="UNBASELINED"),
        Metric(name="warned", status="WARN"),
    ]
    assert dimension_status(metrics) == "WARN"


def test_overall_status_no_thresholds_anywhere() -> None:
    dimensions = {
        name: Dimension(name=name, metrics=[], status="N/A") for name in DIMENSIONS
    }
    assert overall_status(dimensions) == ("UNBASELINED", [])


def test_overall_status_pass_requires_at_least_one_pass() -> None:
    dimensions = {
        name: Dimension(name=name, metrics=[], status="UNBASELINED")
        for name in DIMENSIONS
    }
    assert overall_status(dimensions) == ("UNBASELINED", [])


def test_overall_status_fail_dominates() -> None:
    statuses = ["PASS", "WARN", "FAIL", "UNBASELINED", "N/A", "SKIPPED", "PASS"]
    dimensions = {
        name: Dimension(name=name, metrics=[], status=status)
        for name, status in zip(DIMENSIONS, statuses)
    }
    assert overall_status(dimensions) == ("FAIL", ["chunking", "metadata"])


def test_dimension_order_fixed() -> None:
    report = build_health_report({}, cfg=default_config())
    assert list(report.dimensions.keys()) == list(DIMENSIONS)
