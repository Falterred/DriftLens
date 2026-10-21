"""Regression tests for F-06: snapshot --out must not delete unrelated data.

Audit evidence (audit-artifacts/FINDINGS.md F-06): a destination directory
containing an unrelated manifest.json was silently rmtree'd, destroying user
files.  Ownership must now be established via schema_version + structural
payload files before any replacement.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from pathlib import Path

import pytest

from driftlens.errors import InvalidInputError
from driftlens.schema.snapshot import write_snapshot, read_snapshot

from test_snapshot_io import _build, _limits, _record


@pytest.fixture
def workdir() -> Path:
    path = Path(__file__).parent / f"_remediation_f06_tmp_{os.getpid()}_{secrets.token_hex(4)}"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _fresh_snapshot_records():
    return [_record("a", "hello world", document_id="d1", chunk_index=0)]


def test_unrelated_manifest_dir_is_protected(workdir) -> None:
    """A directory with an unrelated manifest.json must never be deleted."""
    target = workdir / "not_a_snapshot"
    target.mkdir()
    (target / "manifest.json").write_text(
        json.dumps({"garbage": True}), encoding="utf-8"
    )
    (target / "important.txt").write_text("user data", encoding="utf-8")

    snap = _build(_fresh_snapshot_records())

    with pytest.raises(InvalidInputError) as info:
        write_snapshot(snap, target, no_text=False)

    assert "not empty" in str(info.value) or "not a snapshot" in str(info.value)
    # The user's files must survive untouched.
    assert (target / "important.txt").read_text(encoding="utf-8") == "user data"
    assert json.loads((target / "manifest.json").read_text(encoding="utf-8")) == {
        "garbage": True
    }


def test_matching_schema_but_missing_payload_is_protected(workdir) -> None:
    """schema_version alone is not ownership: payload files are required."""
    target = workdir / "partial_manifest"
    target.mkdir()
    (target / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "snapshot_id": "x"}), encoding="utf-8"
    )
    (target / "important.txt").write_text("keep me", encoding="utf-8")

    snap = _build(_fresh_snapshot_records())

    with pytest.raises(InvalidInputError):
        write_snapshot(snap, target, no_text=False)

    assert (target / "important.txt").read_text(encoding="utf-8") == "keep me"


def test_corrupt_manifest_json_is_protected(workdir) -> None:
    """A corrupt manifest.json must fail safely rather than delete."""
    target = workdir / "corrupt_manifest"
    target.mkdir()
    (target / "manifest.json").write_text("{not json", encoding="utf-8")
    (target / "important.txt").write_text("keep me too", encoding="utf-8")

    snap = _build(_fresh_snapshot_records())

    with pytest.raises(InvalidInputError):
        write_snapshot(snap, target, no_text=False)

    assert (target / "important.txt").exists()


def test_legitimate_snapshot_replacement_still_works(workdir) -> None:
    """A real DriftLens snapshot at the destination can still be replaced."""
    target = workdir / "replaceable"

    first = _build([_record("a", "first version", document_id="d1", chunk_index=0)])
    write_snapshot(first, target, no_text=False)
    assert read_snapshot(target, limits=_limits(), verify_full=False).manifest.snapshot_id

    second = _build([_record("a", "second version", document_id="d1", chunk_index=0)])
    write_snapshot(second, target, no_text=False)

    reloaded = read_snapshot(target, limits=_limits(), verify_full=False)
    assert reloaded.manifest.snapshot_id == second.manifest.snapshot_id


def test_empty_directory_still_usable(workdir) -> None:
    """The pre-existing empty-destination behavior is preserved."""
    target = workdir / "empty_dir"
    target.mkdir()

    snap = _build(_fresh_snapshot_records())
    write_snapshot(snap, target, no_text=False)
    assert (target / "records.parquet").is_file()


def test_nonexistent_destination_still_created(workdir) -> None:
    """The pre-existing create-parents behavior is preserved."""
    target = workdir / "brand" / "new" / "path"

    snap = _build(_fresh_snapshot_records())
    write_snapshot(snap, target, no_text=False)
    assert (target / "records.parquet").is_file()


def test_file_destination_still_rejected(workdir) -> None:
    """An ordinary file at the destination remains protected."""
    target = workdir / "plain_file.txt"
    target.write_text("ordinary file", encoding="utf-8")

    snap = _build(_fresh_snapshot_records())
    with pytest.raises(InvalidInputError):
        write_snapshot(snap, target, no_text=False)
    assert target.read_text(encoding="utf-8") == "ordinary file"
