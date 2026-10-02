"""The compare CLI pipeline (Step 33)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from driftlens import __version__
from driftlens.__main__ import main
from driftlens.cli.compare import build_arg_parser, run_compare
from driftlens.errors import FatalErrors, InvalidInputError


@pytest.fixture()
def workdir():
    base = Path(__file__).resolve().parent / "_cli_compare_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _write_jsonl(path, rows):
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _run(argv):
    return run_compare(build_arg_parser().parse_args(argv))


def _valid_pair(workdir):
    rows = [
        {"id": "c1", "text": "hello world", "embedding": [0.1, 0.2, 0.3]},
        {"id": "c2", "text": "goodbye moon", "embedding": [0.3, 0.2, 0.1]},
    ]
    first = workdir / "a.jsonl"
    second = workdir / "b.jsonl"
    _write_jsonl(first, rows)
    _write_jsonl(second, rows)
    return first, second


def test_pipeline_order_format_detection_before_dependency_check(workdir, monkeypatch):
    monkeypatch.setattr("driftlens.cli.compare._probe", lambda name: False)
    chroma_dir = workdir / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "chroma.sqlite3").write_text("", encoding="utf-8")
    missing = workdir / "missing.jsonl"

    with pytest.raises(FatalErrors) as info:
        _run(["compare", str(missing), str(chroma_dir)])
    assert info.value.exit_code == 3


def test_both_sides_format_errors_collected(workdir, capsys):
    missing_a = workdir / "missing_a.jsonl"
    missing_b = workdir / "missing_b.jsonl"
    code = main(["compare", str(missing_a), str(missing_b)])
    captured = capsys.readouterr()
    assert code == 3
    assert "missing_a" in captured.err
    assert "missing_b" in captured.err


def test_unsafe_pickle_rejected_under_ci(workdir):
    missing = workdir / "nope.jsonl"
    args = build_arg_parser().parse_args(
        [
            "compare",
            str(missing),
            str(missing),
            "--allow-unsafe-pickle",
            "--ci",
        ]
    )
    with pytest.raises(InvalidInputError) as info:
        run_compare(args)
    assert info.value.exit_code == 3


def test_unimplemented_flag_baseline_rejected(workdir):
    first, second = _valid_pair(workdir)
    with pytest.raises(InvalidInputError) as info:
        _run(["compare", str(first), str(second), "--baseline", "foo.json"])
    assert "--baseline" in str(info.value)
    assert "not available" in str(info.value)


def test_simple_compare_end_to_end_pass(workdir, capsys):
    first, second = _valid_pair(workdir)
    code = _run(["compare", str(first), str(second)])
    captured = capsys.readouterr()
    assert code == 0
    assert "STATUS: UNBASELINED" in captured.out


def test_simple_compare_with_threshold_warn(workdir, capsys):
    first = workdir / "a.jsonl"
    second = workdir / "b.jsonl"
    _write_jsonl(first, [{"id": "c1", "text": "a"}, {"id": "c2", "text": "b"}])
    long_text = " ".join(["word"] * 40)
    _write_jsonl(
        second,
        [
            {"id": "c1", "text": long_text},
            {"id": "c2", "text": long_text + " tail"},
        ],
    )
    config = workdir / "cfg.yaml"
    config.write_text("drift:\n  chunk_ks:\n    warning: 0.01\n", encoding="utf-8")

    code = _run(["compare", str(first), str(second), "--config", str(config)])
    captured = capsys.readouterr()
    assert code == 0
    assert "STATUS: WARN" in captured.out
    assert "CHUNKING" in captured.out


def test_json_out_written(workdir):
    first, second = _valid_pair(workdir)
    out = workdir / "out.json"
    code = _run(["compare", str(first), str(second), "--json-out", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["report_schema_version"] == "1.0"


def test_report_out_rejected_at_v01(workdir):
    first, second = _valid_pair(workdir)
    out = workdir / "out.html"
    with pytest.raises(InvalidInputError) as info:
        _run(["compare", str(first), str(second), "--report-out", str(out)])
    assert "release 0.2" in str(info.value)


def test_verify_full_noop_always_passes_at_v01(workdir):
    first, second = _valid_pair(workdir)
    assert _run(["compare", str(first), str(second), "--verify-full"]) == 0


def test_expect_a_mismatch_raises(workdir):
    first, second = _valid_pair(workdir)
    with pytest.raises(InvalidInputError) as info:
        _run(["compare", str(first), str(second), "--expect-a", "sha256:0000"])
    assert "--expect-a" in str(info.value)


def test_main_catches_unexpected_exception_as_exit_6_no_traceback(
    workdir, monkeypatch, capsys
):
    def boom(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr("driftlens.__main__.run_compare", boom)
    code = main(["compare", "a", "b"])
    captured = capsys.readouterr()
    assert code == 6
    assert "Traceback" not in captured.err
    assert "boom" not in captured.err


def test_main_version_flag(capsys):
    code = main(["--version"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == f"driftlens {__version__}"


def test_snapshot_subcommand_is_available(workdir, capsys):
    code = main(
        ["snapshot", str(workdir / "missing.jsonl"), "--out", str(workdir / "out")]
    )
    captured = capsys.readouterr()
    assert code == 3
    assert "not available" not in captured.err
    assert "does not exist" in captured.err
