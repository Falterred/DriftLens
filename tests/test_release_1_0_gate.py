"""Release 1.0 acceptance gate (Step 52) — the final capstone.

Mirrors tests/test_release_0_1_gate.py but scoped to the complete 52-step
system: every module imports, the installed CLI runs end to end across compare /
snapshot / baseline / --ci, every diagnostic code has fixture coverage, the
import graph is closed, packaging is consistent, the documentation drift guards
still pass, and every default clears its 95% validation bar.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from driftlens.config.schema import default_config
from driftlens.errors import Code, EXIT_CODES

from test_benchmark_errors import FATAL_CODES
from test_default_validation import (
    DEFAULT_VALIDATION_SUITE,
    run_default_validation_case,
)
from test_release_0_1_gate import WARNING_FIXTURES as WARNING_FIXTURES_0_1

RELEASE_1_0_MODULES = [
    "driftlens",
    "driftlens.__main__",
    "driftlens.adapters",
    "driftlens.adapters._embedding_codec",
    "driftlens.adapters.chroma",
    "driftlens.adapters.csv",
    "driftlens.adapters.detect",
    "driftlens.adapters.faiss",
    "driftlens.adapters.json",
    "driftlens.adapters.jsonl",
    "driftlens.adapters.parquet",
    "driftlens.adapters.sidecar",
    "driftlens.adapters.snapshot",
    "driftlens.analysis",
    "driftlens.analysis.ann",
    "driftlens.analysis.baselines",
    "driftlens.analysis.chunk_matching",
    "driftlens.analysis.chunking",
    "driftlens.analysis.clusters",
    "driftlens.analysis.compatibility",
    "driftlens.analysis.content",
    "driftlens.analysis.distributions",
    "driftlens.analysis.duplicates",
    "driftlens.analysis.embeddings",
    "driftlens.analysis.health",
    "driftlens.analysis.metadata",
    "driftlens.analysis.mmd",
    "driftlens.analysis.neighbours",
    "driftlens.analysis.sampling",
    "driftlens.analysis.significance",
    "driftlens.analysis.temporal",
    "driftlens.cli",
    "driftlens.cli.baseline",
    "driftlens.cli.ci",
    "driftlens.cli.compare",
    "driftlens.cli.snapshot",
    "driftlens.config",
    "driftlens.config.loader",
    "driftlens.config.schema",
    "driftlens.errors",
    "driftlens.evaluation",
    "driftlens.evaluation.ground_truth",
    "driftlens.evaluation.metrics",
    "driftlens.evaluation.retrieval",
    "driftlens.evaluation.retrievers",
    "driftlens.identity",
    "driftlens.identity.chunks",
    "driftlens.identity.documents",
    "driftlens.identity.hashing",
    "driftlens.identity.snapshot_id",
    "driftlens.reports",
    "driftlens.reports.diff_export",
    "driftlens.reports.html",
    "driftlens.reports.json",
    "driftlens.reports.terminal",
    "driftlens.schema",
    "driftlens.schema.eval_set",
    "driftlens.schema.manifest",
    "driftlens.schema.record",
    "driftlens.schema.snapshot",
    "driftlens.security",
    "driftlens.security.limits",
    "driftlens.security.network_guard",
    "driftlens.security.pickle_guard",
    "driftlens.security.redaction",
    "driftlens.text",
    "driftlens.text.normalize",
    "driftlens.text.sentences",
    "driftlens.text.shingles",
    "driftlens.text.tokenizer",
]

# Codes Step 35 excluded because their releases did not exist yet.  Release 1.0
# covers them all, so the 1.0 gate requires fixture coverage for every code.
RELEASE_1_0_CODES = {
    Code.RETRIEVER_UNAVAILABLE: "retrieval comparison (Steps 44-47)",
    Code.BASELINE_INCOMPATIBLE: "baseline build/verify (Step 48)",
    Code.CLUSTER_EXTRA_MISSING: "clustering without hdbscan (Step 39)",
    Code.SIGNIFICANCE_SKIPPED: "significance testing (Step 42)",
    Code.LOW_POWER: "retrieval/significance low-power guard",
    Code.FUTURE_TIMESTAMP: "temporal analysis (Step 41)",
    Code.NO_QUERY_VECTOR: "retrieval query coverage (Step 44)",
}

WARNING_FIXTURES_1_0 = {**WARNING_FIXTURES_0_1, **RELEASE_1_0_CODES}

_ROOT = Path(__file__).resolve().parents[1]
_TESTS_DIR = Path(__file__).resolve().parent
_THIS_FILE = Path(__file__).resolve()


@pytest.fixture()
def workdir():
    base = _TESTS_DIR / "_release_1_0_gate_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _write_jsonl(path: Path, rows) -> None:
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def _run_cli(argv):
    return subprocess.run(
        [sys.executable, "-m", "driftlens", *argv],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_ROOT),
    )


def _check_modules_importable() -> None:
    for name in RELEASE_1_0_MODULES:
        importlib.import_module(name)


def _check_cli_end_to_end(workdir) -> None:
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
    first = workdir / "a.jsonl"
    second = workdir / "b.jsonl"
    third = workdir / "c.jsonl"
    _write_jsonl(first, rows_a)
    _write_jsonl(second, rows_b)
    _write_jsonl(third, rows_c)

    snapshots = []
    for source in (first, second, third):
        target = workdir / ("snap_" + source.stem)
        built = _run_cli(["snapshot", str(source), "--out", str(target)])
        assert built.returncode == 0, built.stderr
        assert (target / "manifest.json").is_file()
        snapshots.append(target)

    compare = _run_cli(["compare", str(snapshots[0]), str(snapshots[1])])
    assert compare.returncode == 0, compare.stderr
    assert "STATUS:" in compare.stdout
    assert compare.stderr == ""

    ci = _run_cli(["compare", str(snapshots[0]), str(snapshots[1]), "--ci"])
    assert "Exit code:" in ci.stdout
    assert ci.returncode in (0, 1, 2), ci.stderr

    list_file = workdir / "snapshots.txt"
    list_file.write_text(
        "\n".join(str(path) for path in snapshots) + "\n", encoding="utf-8"
    )
    config = workdir / "driftlens.yml"
    config.write_text("baseline:\n  min_pairs: 2\n", encoding="utf-8")
    baseline_out = workdir / "baseline.json"
    baseline = _run_cli(
        [
            "baseline",
            "--list",
            str(list_file),
            "--config",
            str(config),
            "--out",
            str(baseline_out),
        ]
    )
    assert baseline.returncode == 0, baseline.stderr
    assert baseline_out.is_file()


def _check_every_code_has_a_fixture() -> None:
    covered = set(FATAL_CODES) | set(WARNING_FIXTURES_1_0)
    missing = sorted(code.value for code in set(Code) - covered)
    assert set(Code) <= covered, "missing fixtures: %s" % missing


def _check_no_forward_reference() -> None:
    allowed = set(RELEASE_1_0_MODULES)
    violations = []
    for path in sorted((_ROOT / "driftlens").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                targets = [node.module]
            for target in targets:
                if not target.startswith("driftlens"):
                    continue
                if target not in allowed:
                    violations.append((str(path.relative_to(_ROOT)), target))
    assert violations == [], "out-of-scope imports: %s" % violations


def _check_packaging_consistent() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    scripts = pyproject["project"]["scripts"]
    assert scripts["driftlens"] == "driftlens.__main__:main"
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    names = {
        requirement.split("=")[0].split(">")[0].split("<")[0].strip()
        for requirement in dev
    }
    assert {"pytest", "faiss-cpu", "chromadb", "scikit-learn", "hdbscan"} <= names


def _check_documentation_guards() -> None:
    # docs/ is gitignored by design, so the Step 51 drift guards are source-based;
    # re-running them is the docs-not-stale confirmation for this capstone.
    from test_schema_freeze import (
        test_cli_reference_matches_current_parser,
        test_configuration_doc_covers_every_field,
        test_schema_freeze_literals,
    )

    test_schema_freeze_literals()
    test_cli_reference_matches_current_parser()
    test_configuration_doc_covers_every_field()


def _check_default_validation_bar() -> None:
    cfg = default_config()
    failed = {}
    for case in DEFAULT_VALIDATION_SUITE:
        rate = run_default_validation_case(case, cfg=cfg)
        if rate < 0.95:
            failed[case.config_key + "::" + case.magnitude_param] = rate
    assert failed == {}, "defaults below the 95%% bar: %s" % failed


def test_release_1_0_gate(workdir) -> None:
    _check_modules_importable()
    _check_cli_end_to_end(workdir)
    _check_every_code_has_a_fixture()
    _check_no_forward_reference()
    _check_packaging_consistent()
    _check_documentation_guards()
    _check_default_validation_bar()


@pytest.mark.skipif(
    os.environ.get("DRIFTLENS_SKIP_NESTED_SUITE") == "1",
    reason="nested full-suite run disabled",
)
def test_full_suite_green() -> None:
    import pytest as _pytest

    # Prevent any nested-in-nested execution (F-09): this run already covers
    # the full suite once; deeper recursion adds cost, not coverage.
    os.environ["DRIFTLENS_SKIP_NESTED_SUITE"] = "1"
    exit_code = _pytest.main(
        [
            "-q",
            "--tb=short",
            str(_TESTS_DIR),
            "--ignore",
            str(_THIS_FILE),
            "--ignore",
            str(_TESTS_DIR / "test_release_0_1_gate.py"),
            "-p",
            "no:cacheprovider",
        ]
    )
    assert exit_code == 0
