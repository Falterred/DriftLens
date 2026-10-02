"""The baseline build CLI (Step 48)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from driftlens.cli.baseline import run_baseline_build
from driftlens.cli.compare import build_arg_parser


@pytest.fixture()
def workdir():
    base = Path(__file__).resolve().parent / "_cli_baseline_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _write_jsonl(path, rows):
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def test_cli_baseline_build_end_to_end(workdir):
    rows_a = [
        {"id": "c1", "text": "alpha beta gamma delta", "embedding": [0.1, 0.2, 0.3]},
        {"id": "c2", "text": "epsilon zeta eta theta", "embedding": [0.3, 0.2, 0.1]},
    ]
    rows_b = [
        {"id": "c1", "text": "alpha beta gamma delta", "embedding": [0.1, 0.2, 0.4]},
        {"id": "c2", "text": "epsilon zeta eta iota", "embedding": [0.3, 0.2, 0.2]},
    ]
    rows_c = [
        {"id": "c1", "text": "alpha beta gamma epsilon", "embedding": [0.2, 0.2, 0.4]},
        {"id": "c2", "text": "epsilon zeta eta kappa", "embedding": [0.3, 0.1, 0.2]},
    ]
    paths = []
    for name, rows in (("a.jsonl", rows_a), ("b.jsonl", rows_b), ("c.jsonl", rows_c)):
        path = workdir / name
        _write_jsonl(path, rows)
        paths.append(path)

    list_file = workdir / "snapshots.txt"
    list_file.write_text(
        "\n".join(str(path) for path in paths) + "\n", encoding="utf-8"
    )
    config = workdir / "driftlens.yml"
    config.write_text("baseline:\n  min_pairs: 2\n", encoding="utf-8")
    out = workdir / "baseline.json"

    args = build_arg_parser().parse_args(
        [
            "baseline",
            "--list",
            str(list_file),
            "--config",
            str(config),
            "--out",
            str(out),
        ]
    )
    assert run_baseline_build(args) == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pair_count"] == 2
    assert "chunk_ks" in data["thresholds"]
    assert "content_change_rate" in data["thresholds"]
