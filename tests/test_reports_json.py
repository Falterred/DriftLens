"""JSON report rendering (Step 31)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from driftlens.analysis.health import Dimension, HealthReport, Metric
from driftlens.config.schema import Threshold, default_config
from driftlens.errors import Code, DiagnosticSink
from driftlens.reports.json import (
    BorderlineCounter,
    build_report_json,
    dimension_to_json,
    metric_to_json,
    write_report_json,
)
from driftlens.schema.manifest import FieldMappingSummary, Manifest


@pytest.fixture()
def workdir():
    base = Path(__file__).resolve().parent / "_reports_json_tmp"
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


def _empty_health() -> HealthReport:
    return HealthReport(dimensions={}, overall="UNBASELINED", exceeded=[])


def _build(health, *, sink=None, cfg=None, borderline=None):
    return build_report_json(
        health=health,
        v1_manifest=_manifest("v1"),
        v2_manifest=_manifest("v2"),
        sink=sink if sink is not None else DiagnosticSink(),
        cfg=cfg if cfg is not None else default_config(),
        borderline=borderline if borderline is not None else BorderlineCounter(),
        diff_artifact_paths={},
        driftlens_version="0.1.0",
    )


def test_borderline_counter_within_tolerance() -> None:
    counter = BorderlineCounter()
    counter.note(0.1000001, 0.1)
    assert counter.count == 1


def test_borderline_counter_outside_tolerance() -> None:
    counter = BorderlineCounter()
    counter.note(0.11, 0.1)
    assert counter.count == 0


def test_borderline_counter_zero_bound_safe() -> None:
    counter = BorderlineCounter()
    counter.note(5.0, 0.0)
    assert counter.count == 0


def test_metric_to_json_sanitizes_nan() -> None:
    metric = Metric(name="x", v1=float("nan"), status="N/A")
    assert metric_to_json(metric)["v1"] is None


def test_dimension_to_json_keys_by_name() -> None:
    dimension = Dimension(
        name="content",
        metrics=[Metric(name="mmd"), Metric(name="chunk_ks")],
        status="N/A",
    )
    rendered = dimension_to_json(dimension)
    assert sorted(rendered["metrics"]) == ["chunk_ks", "mmd"]


def test_skipped_top_level_uses_first_skipped_reason() -> None:
    health = HealthReport(
        dimensions={
            "embedding": Dimension(
                name="embedding",
                metrics=[
                    Metric(
                        name="mmd",
                        status="SKIPPED",
                        reason="SKIPPED_MODEL_CHANGED",
                    )
                ],
                status="SKIPPED",
            )
        },
        overall="UNBASELINED",
        exceeded=[],
    )
    report = _build(health)
    assert report["skipped"] == [
        {"dimension": "embedding", "reason": "SKIPPED_MODEL_CHANGED"}
    ]


def test_warnings_include_dimension_field() -> None:
    sink = DiagnosticSink()
    sink.add(Code.LABEL_CONFLICT, "conflict", dimension="embedding")
    sink.add(Code.MODEL_UNKNOWN, "unknown")
    report = _build(_empty_health(), sink=sink)
    by_code = {entry["code"]: entry for entry in report["warnings"]}
    assert by_code["LABEL_CONFLICT"]["dimension"] == "embedding"
    assert by_code["MODEL_UNKNOWN"]["dimension"] is None


def test_config_effective_round_trips_thresholds() -> None:
    cfg = default_config()
    cfg.drift["mmd"] = Threshold(0.1, 0.2)
    report = _build(_empty_health(), cfg=cfg)
    assert report["config"]["effective"]["drift"]["mmd"] == {
        "warning": 0.1,
        "critical": 0.2,
    }


def test_write_report_json_raises_on_raw_nan(workdir) -> None:
    with pytest.raises(ValueError):
        write_report_json({"x": float("nan")}, workdir / "report.json")


def test_determinism_level_tolerance_when_float_metrics_present() -> None:
    health = HealthReport(
        dimensions={
            "embedding": Dimension(
                name="embedding",
                metrics=[Metric(name="mmd", change=0.5)],
                status="WARN",
            )
        },
        overall="WARN",
        exceeded=["embedding"],
    )
    report = _build(health)
    assert report["reproducibility"]["determinism_level"] == "TOLERANCE"


def test_determinism_level_exact_when_no_float_metrics() -> None:
    health = HealthReport(
        dimensions={
            "content": Dimension(
                name="content",
                metrics=[Metric(name="documents_added", change=3)],
                status="N/A",
            )
        },
        overall="UNBASELINED",
        exceeded=[],
    )
    report = _build(health)
    assert report["reproducibility"]["determinism_level"] == "EXACT"


def test_report_schema_version_field() -> None:
    assert _build(_empty_health())["report_schema_version"] == "1.0"


def test_written_file_is_valid_json(workdir) -> None:
    report = _build(_empty_health())
    path = workdir / "report.json"
    write_report_json(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["report_schema_version"] == report["report_schema_version"]
    assert loaded["status"] == report["status"]
