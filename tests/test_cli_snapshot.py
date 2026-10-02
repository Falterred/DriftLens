"""The snapshot CLI command (Step 36)."""

from __future__ import annotations

import os
import secrets
import shutil
from pathlib import Path

import pytest

from driftlens.__main__ import main
from tests.support import make_rows, write_jsonl


@pytest.fixture()
def workdir():
    base = (
        Path(__file__).resolve().parent
        / f"_cli_snapshot_tmp_{os.getpid()}_{secrets.token_hex(4)}"
    )
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_cli_snapshot_end_to_end(workdir, capsys):
    source = write_jsonl(workdir / "source.jsonl", make_rows(5, seed=1, dim=4))
    out = workdir / "snap"

    assert main(["snapshot", str(source), "--out", str(out)]) == 0
    assert (out / "manifest.json").is_file()
    assert (out / "records.parquet").is_file()
    assert (out / "documents.parquet").is_file()
    assert (out / "statistics.json").is_file()
    assert (out / "embeddings").is_dir()
    assert any((out / "embeddings").iterdir())

    capsys.readouterr()
    assert main(["compare", str(out), str(out)]) == 0
    captured = capsys.readouterr()
    assert "UNBASELINED" in captured.out


def test_cli_snapshot_rejects_snapshot_source(workdir, capsys):
    source = write_jsonl(workdir / "source.jsonl", make_rows(3, seed=2, dim=4))
    out = workdir / "snap"
    assert main(["snapshot", str(source), "--out", str(out)]) == 0
    capsys.readouterr()

    code = main(["snapshot", str(out), "--out", str(workdir / "again")])
    captured = capsys.readouterr()
    assert code == 3
    assert "already a snapshot directory" in captured.err
