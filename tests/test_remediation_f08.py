"""Regression tests for F-08: --redact-ids must cover every output path.

Audit evidence (audit-artifacts/FINDINGS.md F-08): --redact-ids redacted
only metric values; raw document ids leaked through the HTML top-changed
table, the --diff-out parquet exports, and the terminal top-N line.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_IDS = ("s1", "s2", "s3", "s4", "s5", "s6")


def _workdir(name: str) -> Path:
    path = (
        Path(__file__).parent
        / f"_remediation_f08_tmp_{os.getpid()}_{secrets.token_hex(4)}_{name}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_corpus(workdir: Path):
    """Self-contained corpus (mirrors the audit's A/C pair) so the tests
    run on a fresh checkout without the local-only audit-artifacts dir."""
    baseline = [
        {"id": "s1", "text": "alpha beta gamma delta epsilon", "score": 5},
        {"id": "s2", "text": "zeta eta theta iota kappa", "score": 4},
        {"id": "s3", "text": "lambda mu nu xi omicron", "score": 3},
        {"id": "s4", "text": "pi rho sigma tau upsilon", "score": 2},
        {"id": "s5", "text": "phi chi psi omega alpha beta", "score": 1},
        {"id": "s6", "text": "gamma delta epsilon zeta eta", "score": 6},
    ]
    modified = [
        {"id": "s1", "text": "alpha beta gamma delta epsilon", "score": 5},
        # s2: small edit -> MODIFIED; s3: unrelated text -> REPLACED
        {"id": "s2", "text": "zeta eta theta iota kappa lambda", "score": 4},
        {"id": "s3", "text": "completely different unrelated content here", "score": 3},
        {"id": "s4", "text": "pi rho sigma tau upsilon", "score": 2},
        {"id": "s5", "text": "phi chi psi omega alpha beta", "score": 1},
        {"id": "s6", "text": "gamma delta epsilon zeta eta", "score": 6},
    ]
    first = workdir / "baseline.jsonl"
    second = workdir / "modified.jsonl"
    first.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in baseline),
        encoding="utf-8",
    )
    second.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in modified),
        encoding="utf-8",
    )
    return first, second


def _run_compare(workdir: Path, *extra: str) -> subprocess.CompletedProcess:
    first, second = _write_corpus(workdir)
    argv = [
        sys.executable,
        "-m",
        "driftlens",
        "compare",
        str(first),
        str(second),
        "--redact-ids",
        *extra,
    ]
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=120, cwd=str(ROOT)
    )


def _leaks(*texts: str) -> list[str]:
    found: list[str] = []
    for text in texts:
        for raw in RAW_IDS:
            if raw in text:
                found.append(raw)
    return sorted(set(found))


def test_redacted_terminal_and_html_and_json_have_no_raw_ids() -> None:
    workdir = _workdir("full")
    try:
        html_path = workdir / "report.html"
        json_path = workdir / "report.json"
        diff_dir = workdir / "diff"
        result = _run_compare(
            workdir,
            "--report-out",
            str(html_path),
            "--json-out",
            str(json_path),
            "--diff-out",
            str(diff_dir),
        )
        assert result.returncode == 0, result.stderr

        html = html_path.read_text(encoding="utf-8")
        report = json_path.read_text(encoding="utf-8")
        assert _leaks(result.stdout, html, report) == []

        docs = pq.read_table(diff_dir / "documents.parquet").to_pylist()
        doc_ids = [row["document_id"] for row in docs]
        assert _leaks(*[str(i) for i in doc_ids]) == []
        chunks = pq.read_table(diff_dir / "chunks.parquet").to_pylist()
        chunk_ids = [row["v1_record_id"] or "" for row in chunks] + [
            row["v2_record_id"] or "" for row in chunks
        ]
        assert _leaks(*chunk_ids) == []
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_redacted_ids_are_stable_across_runs() -> None:
    workdir = _workdir("stable")
    try:
        outs = []
        for run in (1, 2):
            json_path = workdir / f"report{run}.json"
            result = _run_compare(workdir, "--json-out", str(json_path))
            assert result.returncode == 0, result.stderr
            report = json.loads(json_path.read_text(encoding="utf-8"))
            # The report carries a generation timestamp; everything else,
            # including all pseudonyms, must be byte-identical across runs.
            report.pop("created_at", None)
            content = json.dumps(report, sort_keys=True)
            assert _leaks(content) == []
            outs.append(content)
        assert outs[0] == outs[1], "pseudonyms must be deterministic"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_redaction_does_not_break_classification() -> None:
    workdir = _workdir("classify")
    try:
        json_path = workdir / "report.json"
        result = _run_compare(workdir, "--json-out", str(json_path))
        assert result.returncode == 0, result.stderr
        report = json.loads(json_path.read_text(encoding="utf-8"))
        metrics = report["dimensions"]["content"]["metrics"]
        # Same shape as the audit corpus: 1 modified + 1 replaced must
        # survive redaction with the correct counts.
        assert metrics["documents_modified"]["change"] == 1
        assert metrics["documents_replaced"]["change"] == 1
        assert metrics["documents_added"]["change"] == 0
        assert metrics["documents_removed"]["change"] == 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_unredacted_run_still_shows_raw_ids() -> None:
    workdir = _workdir("unredacted")
    try:
        first, second = _write_corpus(workdir)
        argv = [
            sys.executable,
            "-m",
            "driftlens",
            "compare",
            str(first),
            str(second),
        ]
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=120, cwd=str(ROOT)
        )
        assert result.returncode == 0, result.stderr
        assert "s2" in result.stdout and "s3" in result.stdout
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
