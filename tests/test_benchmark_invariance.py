"""Item 7: invariance tests (Step 34)."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest

from driftlens.cli.compare import build_arg_parser, run_compare

from fixtures.generator import generate_corpus_pair, write_corpus_as


@pytest.fixture()
def workdir():
    base = Path(__file__).resolve().parent / "_benchmark_invariance_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _run_report(workdir, first, second, extra, name):
    out = workdir / name
    argv = ["compare", str(first), str(second), "--json-out", str(out), *extra]
    assert run_compare(build_arg_parser().parse_args(argv)) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload.pop("created_at", None)
    return payload


def _metric_change(payload, name):
    for dimension in payload["dimensions"].values():
        metric = dimension["metrics"].get(name)
        if metric is not None:
            return metric["change"]
    return None


def test_invariance_row_order_shuffle(workdir, no_network) -> None:
    pair = generate_corpus_pair(42, 1000, "pct10")
    first = write_corpus_as(pair.v1_records, "jsonl", workdir / "v1.jsonl")
    second = write_corpus_as(pair.v2_records, "jsonl", workdir / "v2.jsonl")
    base = _run_report(workdir, first, second, [], "base.json")

    shuffled_v1 = list(pair.v1_records)
    random.Random(7).shuffle(shuffled_v1)
    shuffled_v2 = list(pair.v2_records)
    random.Random(8).shuffle(shuffled_v2)
    first = write_corpus_as(shuffled_v1, "jsonl", workdir / "v1.jsonl")
    second = write_corpus_as(shuffled_v2, "jsonl", workdir / "v2.jsonl")
    other = _run_report(workdir, first, second, [], "other.json")

    assert base["snapshots"] == other["snapshots"]
    for name in ("content_change_rate", "chunk_ks", "mmd"):
        assert _metric_change(base, name) == _metric_change(other, name)


def test_invariance_repeated_runs_byte_identical_json(workdir, no_network) -> None:
    pair = generate_corpus_pair(42, 300, "pct10")
    first = write_corpus_as(pair.v1_records, "jsonl", workdir / "v1.jsonl")
    second = write_corpus_as(pair.v2_records, "jsonl", workdir / "v2.jsonl")
    reports = [
        _run_report(workdir, first, second, [], f"run{index}.json")
        for index in range(3)
    ]
    assert reports[0] == reports[1] == reports[2]
