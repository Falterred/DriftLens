"""Regression tests for F-04: list-valued metadata must not crash compare.

Audit evidence (audit-artifacts/FINDINGS.md F-04): when a metadata field’s
dominant type differs across the two sides (list vs absent, list vs string,
or any unhashable vs hashable kind), compare_metadata raised
``TypeError: unhashable type: ‘list’`` and the CLI surfaced a bare
INTERNAL_ERROR (exit 6) instead of a report.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from pathlib import Path

import pytest

from driftlens.adapters.jsonl import load_jsonl
from driftlens.analysis.metadata import compare_metadata
from driftlens.config.schema import default_config, SideOptions
from driftlens.errors import DiagnosticSink
from driftlens.schema.manifest import FieldMappingSummary
from driftlens.schema.snapshot import build_snapshot
from driftlens.security.limits import RuntimeLimits


@pytest.fixture
def tmp_path() -> Path:
    """Project-convention temp dir inside tests/ (sandbox-safe, auto-removed)."""
    path = (
        Path(__file__).parent
        / f"_remediation_f04_tmp_{os.getpid()}_{secrets.token_hex(4)}"
    )
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _build_snapshot(base: Path, rows: list) -> object:
    base.mkdir(parents=True, exist_ok=True)
    source = base / "rows.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    cfg = default_config()
    limits = RuntimeLimits(cfg.limits)
    loaded = load_jsonl(source, limits=limits, side=SideOptions())
    return build_snapshot(
        loaded.records,
        side=SideOptions(),
        cfg=cfg,
        source_type="jsonl",
        sink=DiagnosticSink(),
        limits=limits,
        granularity_override=None,
        field_mapping_summary=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _compare(snap1, snap2):
    cfg = default_config()
    common = {r.record_id for r in snap1.records} & {r.record_id for r in snap2.records}
    return compare_metadata(
        snap1, snap2, cfg=cfg.metadata, drift_cfg=cfg.drift, common_record_ids=common
    )


V1 = [
    {"id": "a", "text": "hello world", "tags": ["x", "y"]},
    {"id": "b", "text": "second doc", "tags": ["z"]},
]


def test_list_vs_absent_field_no_crash(tmp_path: Path) -> None:
    """The exact audit minimal repro: tags present vs entirely absent."""
    v2 = [
        {"id": "a", "text": "hello world"},
        {"id": "b", "text": "second doc"},
    ]
    drifts, _metrics = _compare(
        _build_snapshot(tmp_path / "v1dir", V1),
        _build_snapshot(tmp_path / "v2dir", v2),
    )
    names = {d.name for d in drifts}
    assert "tags" in names


def test_list_vs_string_field_no_crash(tmp_path: Path) -> None:
    """tags as lists on side 1 vs plain strings on side 2."""
    v2 = [
        {"id": "a", "text": "hello world", "tags": "plain"},
        {"id": "b", "text": "second doc", "tags": "plain2"},
    ]
    drifts, _metrics = _compare(
        _build_snapshot(tmp_path / "v1dir", V1),
        _build_snapshot(tmp_path / "v2dir", v2),
    )
    tags_row = next(d for d in drifts if d.name == "tags")
    assert tags_row.type_changed is True


def test_null_metadata_vs_typed_no_crash(tmp_path: Path) -> None:
    """G5 scenario: null category/score on one side vs typed values."""
    v1 = [
        {"id": "n1", "text": "text one", "category": None, "score": None},
        {"id": "n2", "text": "text two", "category": None, "score": None},
    ]
    v2 = [
        {"id": "n1", "text": "text one", "category": "science", "score": 1},
        {"id": "n2", "text": "text two", "category": "tech", "score": 2},
    ]
    drifts, _ = _compare(
        _build_snapshot(tmp_path / "v1dir", v1),
        _build_snapshot(tmp_path / "v2dir", v2),
    )
    by_name = {d.name for d in drifts}
    assert {"category", "score"} <= by_name


def test_dict_vs_string_no_crash(tmp_path: Path) -> None:
    """json-typed (dict) values on one side, strings on the other."""
    v1 = [
        {"id": "a", "text": "hello world", "tags": {"x": 1}},
        {"id": "b", "text": "second doc", "tags": {"y": 2}},
    ]
    v2 = [
        {"id": "a", "text": "hello world", "tags": "plain"},
        {"id": "b", "text": "second doc", "tags": "plain"},
    ]
    drifts, _ = _compare(
        _build_snapshot(tmp_path / "v1dir", v1),
        _build_snapshot(tmp_path / "v2dir", v2),
    )
    tags_row = next(d for d in drifts if d.name == "tags")
    assert tags_row is not None


def test_int_vs_string_no_crash(tmp_path: Path) -> None:
    """Numeric-dominant on side 2 with string stragglers on side 1."""
    v1 = [
        {"id": "a", "text": "hello", "score": "high"},
        {"id": "b", "text": "world", "score": "low"},
    ]
    v2 = [
        {"id": "a", "text": "hello", "score": 5},
        {"id": "b", "text": "world", "score": 3},
    ]
    drifts, _ = _compare(
        _build_snapshot(tmp_path / "v1dir", v1),
        _build_snapshot(tmp_path / "v2dir", v2),
    )
    score_row = next(d for d in drifts if d.name == "score")
    assert score_row.type_changed is True


def test_normal_metadata_still_compared(tmp_path: Path) -> None:
    """Sanity: ordinary scalar metadata keeps its drift semantics."""
    v1 = [
        {"id": "a", "text": "hello", "cat": "x"},
        {"id": "b", "text": "world", "cat": "y"},
    ]
    v2 = [
        {"id": "a", "text": "hello", "cat": "x"},
        {"id": "b", "text": "world", "cat": "z"},
    ]
    drifts, _ = _compare(
        _build_snapshot(tmp_path / "v1dir", v1),
        _build_snapshot(tmp_path / "v2dir", v2),
    )
    cat_row = next(d for d in drifts if d.name == "cat")
    assert cat_row.values_changed_chunk_count == 1
    assert cat_row.distance is not None and cat_row.distance > 0


def test_lists_on_both_sides_still_compared(tmp_path: Path) -> None:
    """List-vs-list keeps working: element and length drift still measured."""
    v1 = [
        {"id": "a", "text": "hello", "tags": ["x", "y"]},
        {"id": "b", "text": "world", "tags": ["x"]},
    ]
    v2 = [
        {"id": "a", "text": "hello", "tags": ["x", "y", "z"]},
        {"id": "b", "text": "world", "tags": ["x"]},
    ]
    drifts, _ = _compare(
        _build_snapshot(tmp_path / "v1dir", v1),
        _build_snapshot(tmp_path / "v2dir", v2),
    )
    tags_rows = [d for d in drifts if d.name == "tags"]
    assert tags_rows, "tags field must still be reported"
