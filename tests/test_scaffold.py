"""Scaffold, packaging and test-harness checks (Step 1)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import driftlens

from support import make_rows

ROOT = Path(__file__).resolve().parents[1]


def test_version() -> None:
    assert driftlens.__version__ == "0.1.0"


def test_cli_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "driftlens", "--version"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.stdout == "driftlens 0.1.0\n"
    assert result.returncode == 0


def test_make_rows_deterministic() -> None:
    assert make_rows(50, 7) == make_rows(50, 7)
    assert make_rows(50, 7) != make_rows(50, 8)
