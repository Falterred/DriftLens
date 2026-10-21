"""Regression tests for F-05: forward-compatible manifest reading.

Audit evidence (audit-artifacts/FINDINGS.md F-05): an unknown manifest
field (written by a newer 1.x writer per docs/schema-versioning.md) made
Manifest(**data) raise TypeError -> bare INTERNAL_ERROR (exit 6), breaking
the documented 1.x forward-compatibility promise.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from pathlib import Path

import pytest

from driftlens.config.schema import default_config, SideOptions
from driftlens.errors import InvalidInputError, UnsupportedFormatError
from driftlens.schema.snapshot import build_snapshot, read_snapshot, write_snapshot
from driftlens.security.limits import RuntimeLimits

from test_snapshot_io import _build, _limits, _record


@pytest.fixture
def workdir() -> Path:
    path = (
        Path(__file__).parent
        / f"_remediation_f05_tmp_{os.getpid()}_{secrets.token_hex(4)}"
    )
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_snapshot(workdir: Path, name: str) -> Path:
    snap = _build([_record("a", "hello world", document_id="d1", chunk_index=0)])
    out = workdir / name
    write_snapshot(snap, out, no_text=False)
    return out


def _mutate_manifest(snapshot_dir: Path, mutate) -> None:
    manifest_path = snapshot_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(data)
    manifest_path.write_text(json.dumps(data), encoding="utf-8")


def test_unknown_manifest_field_is_ignored(workdir: Path) -> None:
    """A newer 1.x writer’s additive field must not break this reader."""
    out = _write_snapshot(workdir, "future")
    _mutate_manifest(out, lambda data: data.update(future_additive_field=123))

    loaded = read_snapshot(out, limits=_limits(), verify_full=False)
    assert loaded.manifest.snapshot_id
    assert loaded.manifest.document_count == 1


def test_unknown_statistics_field_is_ignored(workdir: Path) -> None:
    """statistics.json gets the same 1.x tolerance."""
    out = _write_snapshot(workdir, "future_stats")
    stats_path = out / "statistics.json"
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    data["future_stat_field"] = {"x": 1}
    stats_path.write_text(json.dumps(data), encoding="utf-8")
    # statistics.json is checksummed in the manifest; update the checksum
    # like a real newer 1.x writer would, so only the unknown field differs.
    from driftlens.identity.hashing import sha256_hex

    _mutate_manifest(
        out,
        lambda manifest: manifest["files"].update(
            {"statistics.json": sha256_hex(stats_path.read_bytes())}
        ),
    )

    loaded = read_snapshot(out, limits=_limits(), verify_full=False)
    assert loaded.statistics.document_count == 1


def test_missing_required_field_is_actionable_invalid_input(
    workdir: Path,
) -> None:
    """A genuinely invalid manifest must fail as INVALID_INPUT, not exit 6."""
    out = _write_snapshot(workdir, "missing")
    _mutate_manifest(out, lambda data: data.pop("snapshot_id"))

    with pytest.raises(InvalidInputError):
        read_snapshot(out, limits=_limits(), verify_full=False)


def test_newer_major_version_still_rejected(workdir: Path) -> None:
    """Unsupported schema majors keep exiting with UnsupportedFormatError."""
    out = _write_snapshot(workdir, "major2")
    _mutate_manifest(out, lambda data: data.update(schema_version="2.0"))

    with pytest.raises(UnsupportedFormatError):
        read_snapshot(out, limits=_limits(), verify_full=False)


def test_unknown_field_does_not_break_verify_full(workdir: Path) -> None:
    """Tolerance must not weaken integrity verification."""
    out = _write_snapshot(workdir, "future_verify")
    _mutate_manifest(out, lambda data: data.update(future_additive_field=123))

    loaded = read_snapshot(out, limits=_limits(), verify_full=True)
    assert loaded.manifest.snapshot_id


def test_checksum_mismatch_still_rejected(workdir: Path) -> None:
    """File integrity checks are unaffected by manifest tolerance."""
    out = _write_snapshot(workdir, "tampered")
    records_path = out / "records.parquet"
    records_path.write_bytes(b"tampered")

    with pytest.raises(InvalidInputError):
        read_snapshot(out, limits=_limits(), verify_full=False)
