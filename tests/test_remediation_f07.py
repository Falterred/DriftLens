"""Regression tests for F-07: duplicate-record-id snapshot integrity.

Audit evidence (audit-artifacts/FINDINGS.md F-07):
* the record digest cache was populated before duplicate-id renames,
  so snapshot ids were computed from stale pre-rename digests and a
  duplicate-id snapshot failed its own --verify-full read-back;
* the documented identity.on_duplicate_id: error policy was unreachable
  (a second disambiguation pass ran after the first had already suffixed).
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from pathlib import Path

import pytest

from driftlens.adapters.jsonl import load_jsonl
from driftlens.config.schema import default_config, SideOptions
from driftlens.errors import DiagnosticSink, InvalidInputError
from driftlens.schema.manifest import FieldMappingSummary
from driftlens.schema.snapshot import build_snapshot, read_snapshot, write_snapshot
from driftlens.security.limits import RuntimeLimits


@pytest.fixture
def workdir() -> Path:
    path = (
        Path(__file__).parent
        / f"_remediation_f07_tmp_{os.getpid()}_{secrets.token_hex(4)}"
    )
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _build_from_rows(rows: list, workdir: Path):
    """Build a snapshot from raw JSONL rows via the real loader."""
    cfg = default_config()
    limits = RuntimeLimits(cfg.limits)
    src = workdir / "src.jsonl"
    src.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    loaded = load_jsonl(src, limits=limits, side=SideOptions())
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


DUP_ROWS = [
    {"id": "a", "text": "first document"},
    {"id": "a", "text": "second document"},
    {"id": "b", "text": "unique document"},
]


def test_duplicate_id_snapshot_passes_verify_full(workdir: Path) -> None:
    """A duplicate-id snapshot must verify against its own final records."""
    global_workdir = workdir
    cfg = default_config()
    snap = _build_from_rows(DUP_ROWS, workdir)
    out = workdir / "dup_snapshot"
    write_snapshot(snap, out, no_text=False)
    reloaded = read_snapshot(out, limits=RuntimeLimits(cfg.limits), verify_full=True)
    assert reloaded.manifest.snapshot_id == snap.manifest.snapshot_id


def test_duplicate_ids_get_suffix_disambiguation(workdir: Path) -> None:
    """Default policy: duplicates renamed deterministically with a warning."""
    cfg = default_config()
    snap = _build_from_rows(DUP_ROWS, workdir)
    ids = sorted(r.record_id for r in snap.records)
    assert ids == ["a#0", "a#1", "b"]


def test_snapshot_id_covers_final_record_ids(workdir: Path) -> None:
    """The snapshot id must change when the final (renamed) ids change.

    Two corpora differing only in a duplicate’s renamed ids must produce
    different snapshot ids (stale-cache would give equal ids).
    """
    snap1 = _build_from_rows(DUP_ROWS, workdir)
    # Rebuild after changing which text wins rank 0: the final ids are the
    # same, but the record contents differ, so ids must differ anyway.
    other = [
        {"id": "a", "text": "zzz different content zzz"},
        {"id": "a", "text": "second document"},
        {"id": "b", "text": "unique document"},
    ]
    snap2 = _build_from_rows(other, workdir)
    assert snap1.manifest.snapshot_id != snap2.manifest.snapshot_id


def test_error_mode_raises_on_duplicate_ids(workdir: Path) -> None:
    """identity.on_duplicate_id: error must be reachable and effective."""
    cfg = default_config()
    cfg.identity.on_duplicate_id = "error"
    limits = RuntimeLimits(cfg.limits)
    src = workdir / "err_src.jsonl"
    src.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in DUP_ROWS),
        encoding="utf-8",
    )
    loaded = load_jsonl(src, limits=limits, side=SideOptions())
    with pytest.raises(InvalidInputError) as info:
        build_snapshot(
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
    assert "duplicate record ids" in str(info.value)


def test_error_mode_untouched_when_no_duplicates(workdir: Path) -> None:
    """The error policy must not fire for unique ids."""
    cfg = default_config()
    cfg.identity.on_duplicate_id = "error"
    unique = [
        {"id": "x", "text": "one"},
        {"id": "y", "text": "two"},
    ]
    snap = _build_with_cfg(unique, cfg, workdir)
    assert sorted(r.record_id for r in snap.records) == ["x", "y"]


def _build_with_cfg(rows: list, cfg, workdir: Path):
    limits = RuntimeLimits(cfg.limits)
    src = workdir / "cfg_src.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    loaded = load_jsonl(src, limits=limits, side=SideOptions())
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


def test_error_mode_via_cli_config(workdir: Path) -> None:
    """End-to-end: the YAML policy reaches the CLI snapshot command."""
    import subprocess
    import sys

    config = workdir / "dup-error.yml"
    config.write_text(
        "identity:\n  on_duplicate_id: error\n", encoding="utf-8"
    )
    src = workdir / "cli_src.jsonl"
    src.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in DUP_ROWS),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "driftlens",
            "snapshot",
            str(src),
            "--out",
            str(workdir / "cli_snap"),
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 3, result.stderr
    assert "duplicate record ids" in result.stderr
