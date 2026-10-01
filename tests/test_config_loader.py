"""Configuration loader: YAML parsing, fail_on processing and layering (Step 4)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from driftlens.config.loader import effective_side_options, load_config
from driftlens.errors import InvalidInputError


@pytest.fixture
def workdir():
    directory = Path(__file__).resolve().parent / "_config_loader_tmp"
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_key_rejected(workdir) -> None:
    path = _write(workdir, "driftlens.yaml", "analysis:\n  sead: 1\n")
    with pytest.raises(InvalidInputError) as excinfo:
        load_config(path, {}, workdir)
    assert "analysis.sead" in str(excinfo.value)


def test_unsafe_pickle_key(workdir) -> None:
    path = _write(workdir, "driftlens.yaml", "allow_unsafe_pickle: true\n")
    with pytest.raises(InvalidInputError):
        load_config(path, {}, workdir)


def test_fail_on_numeric(workdir) -> None:
    path = _write(workdir, "driftlens.yaml", "ci:\n  fail_on:\n    mmd: 0.25\n")
    cfg = load_config(path, {}, workdir)
    assert cfg.drift["mmd"].critical == 0.25
    assert "mmd" in cfg.ci.critical_gates
    assert cfg.ci.fail_on["embedding"] == "critical"

    conflict = _write(
        workdir,
        "conflict.yaml",
        "drift:\n  mmd:\n    critical: 0.3\nci:\n  fail_on:\n    mmd: 0.25\n",
    )
    with pytest.raises(InvalidInputError):
        load_config(conflict, {}, workdir)


def test_semantic_collision(workdir) -> None:
    dimension = _write(
        workdir, "dimension.yaml", "ci:\n  fail_on:\n    semantic: critical\n"
    )
    cfg = load_config(dimension, {}, workdir)
    assert cfg.ci.fail_on["semantic"] == "critical"

    threshold = _write(
        workdir, "threshold.yaml", "ci:\n  fail_on:\n    semantic: 0.4\n"
    )
    cfg = load_config(threshold, {}, workdir)
    assert cfg.drift["semantic"].critical == 0.4


def test_side_layering(workdir) -> None:
    path = _write(
        workdir,
        "driftlens.yaml",
        "schema:\n  text_field: a\nschema_b:\n  text_field: b\n",
    )
    cfg = load_config(path, {}, workdir)

    side_a = effective_side_options(cfg, "a", {"schema.text_field": "c"})
    side_b = effective_side_options(cfg, "b", {"schema.text_field": "c"})
    assert side_a.text_field == "c"
    assert side_b.text_field == "c"

    side_b_suffixed = effective_side_options(cfg, "b", {"schema_b.text_field": "d"})
    assert side_b_suffixed.text_field == "d"


def test_precedence_cli_over_file(workdir) -> None:
    path = _write(workdir, "driftlens.yaml", "analysis:\n  seed: 1\n")
    cfg = load_config(path, {"analysis.seed": 2}, workdir)
    assert cfg.analysis.seed == 2
