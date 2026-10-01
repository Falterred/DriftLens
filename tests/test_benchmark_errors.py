"""Item 8: one fixture per fatal error code (Step 34)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from driftlens.__main__ import main
from driftlens.errors import (
    Code,
    EXIT_CODES,
    IncompatibleError,
    UndefinedCaseError,
)


@pytest.fixture()
def workdir():
    base = Path(__file__).resolve().parent / "_benchmark_errors_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def _invalid_input_fixture(workdir: Path):
    path = workdir / "broken.jsonl"
    path.write_text('{"text": ', encoding="utf-8")
    return ["compare", str(path), str(path)]


def _unsupported_format_fixture(workdir: Path):
    path = workdir / "mystery.xyz"
    path.write_text("hello", encoding="utf-8")
    return ["compare", str(path), str(path)]


def _refuse_pickle_fixture(workdir: Path):
    directory = workdir / "chroma_pickle"
    directory.mkdir()
    (directory / "chroma.sqlite3").write_bytes(b"")
    (directory / "stray.pkl").write_bytes(b"pickle")
    return ["compare", str(directory), str(directory)]


def _limit_exceeded_fixture(workdir: Path):
    row = {"id": "c1", "text": "hello", "embedding": [0.1, 0.2]}
    for index in range(2001):
        row[f"field_{index}"] = index
    path = _write_jsonl(workdir / "many_fields.jsonl", [row])
    return ["compare", str(path), str(path)]


def _strict_compat_fixture(workdir: Path):
    rows = [
        {"id": "c1", "text": "hello world", "embedding": [0.1, 0.2, 0.3]},
        {"id": "c2", "text": "goodbye moon", "embedding": [0.3, 0.2, 0.1]},
    ]
    first = _write_jsonl(workdir / "sa.jsonl", rows)
    second = _write_jsonl(workdir / "sb.jsonl", rows)
    config = workdir / "strict.yaml"
    config.write_text(
        "schema_a:\n  similarity_metric: cosine\n"
        "schema_b:\n  similarity_metric: dot\n",
        encoding="utf-8",
    )
    return [
        "compare",
        str(first),
        str(second),
        "--config",
        str(config),
        "--strict-compat",
    ]


def _dependency_missing_fixture(workdir: Path):
    directory = workdir / "chroma_dep"
    directory.mkdir()
    (directory / "chroma.sqlite3").write_bytes(b"")
    return ["compare", str(directory), str(directory)]


ERROR_FIXTURES = {
    Code.INVALID_INPUT: _invalid_input_fixture,
    Code.UNSUPPORTED_FORMAT: _unsupported_format_fixture,
    Code.REFUSE_UNTRUSTED_PICKLE: _refuse_pickle_fixture,
    Code.LIMIT_EXCEEDED: _limit_exceeded_fixture,
    Code.STRICT_COMPAT_VIOLATION: _strict_compat_fixture,
}

ERROR_RAISERS = {
    Code.NORMALIZATION_MISMATCH: lambda: IncompatibleError(
        Code.NORMALIZATION_MISMATCH, message="normalization mismatch"
    ),
    Code.IDENTITY_UNAVAILABLE: lambda: IncompatibleError(
        Code.IDENTITY_UNAVAILABLE, message="no common identity level"
    ),
    Code.UNDEFINED_CASE: lambda: UndefinedCaseError("degenerate"),
    Code.INTERNAL_ERROR: lambda: RuntimeError("boom"),
}

PROBE_ERRORS = {Code.DEPENDENCY_MISSING}

FATAL_CODES = {code for code in EXIT_CODES if code in ERROR_FIXTURES or code in ERROR_RAISERS or code in PROBE_ERRORS}


def test_error_suite_covers_every_fatal_code() -> None:
    fatal = {
        code
        for code in EXIT_CODES
        if code
        in {
            Code.INVALID_INPUT,
            Code.UNSUPPORTED_FORMAT,
            Code.REFUSE_UNTRUSTED_PICKLE,
            Code.DEPENDENCY_MISSING,
            Code.LIMIT_EXCEEDED,
            Code.NORMALIZATION_MISMATCH,
            Code.IDENTITY_UNAVAILABLE,
            Code.STRICT_COMPAT_VIOLATION,
            Code.INTERNAL_ERROR,
            Code.UNDEFINED_CASE,
        }
    }
    assert FATAL_CODES == fatal


@pytest.mark.parametrize("code", sorted(FATAL_CODES, key=lambda item: item.value))
def test_fatal_error_fixture(code, workdir, monkeypatch, capsys) -> None:
    if code in ERROR_FIXTURES:
        argv = ERROR_FIXTURES[code](workdir)
    elif code in PROBE_ERRORS:
        monkeypatch.setattr("driftlens.cli.compare._probe", lambda name: False)
        argv = _dependency_missing_fixture(workdir)
    else:
        raiser = ERROR_RAISERS[code]

        def boom(_args):
            raise raiser()

        monkeypatch.setattr("driftlens.__main__.run_compare", boom)
        argv = ["compare", "a", "b"]

    expected_exit = EXIT_CODES[code]
    assert main(argv) == expected_exit
    captured = capsys.readouterr()
    assert f"DRIFTLENS ERROR {code.value} (exit {expected_exit})" in captured.err
