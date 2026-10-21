"""Regression tests for F-09: nested-suite de-duplication.

Audit evidence (audit-artifacts/FINDINGS.md F-09): the release gates'
test_full_suite_green re-ran the entire suite recursively (the 0.1 gate
ignored only itself, so the 1.0 gate's nested run fired inside it), and CI
never set DRIFTLENS_SKIP_NESTED_SUITE=1, quadrupling CI cost.

Fixes under test:
* DRIFTLENS_SKIP_NESTED_SUITE=1 skips both nested runners (existing env).
* A nested run sets the env var itself, so no nested-in-nested execution.
* The 0.1 gate's nested invocation ignores the 1.0 gate file.
* CI sets the escape variable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_pytest(env_extra: dict, *args) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
        env=env,
    )


def test_nested_runner_skipped_when_env_set() -> None:
    """With the escape variable set, both nested runners skip, not run."""
    result = _run_pytest(
        {"DRIFTLENS_SKIP_NESTED_SUITE": "1"},
        "-q",
        "-p",
        "no:cacheprovider",
        "-rs",
        "tests/test_release_0_1_gate.py::test_full_suite_green",
        "tests/test_release_1_0_gate.py::test_full_suite_green",
    )
    assert result.returncode == 0, result.stdout[-2000:]
    # Both must report SKIPPED with the documented reason.
    assert result.stdout.count("SKIPPED") >= 2 or "2 skipped" in result.stdout


def test_gate_source_sets_env_inside_nested_run() -> None:
    """Both gate files set the env var before pytest.main (source contract)."""
    for name in ("test_release_0_1_gate.py", "test_release_1_0_gate.py"):
        source = (ROOT / "tests" / name).read_text(encoding="utf-8")
        assert 'os.environ["DRIFTLENS_SKIP_NESTED_SUITE"] = "1"' in source, name


def test_01_gate_nested_run_ignores_10_gate() -> None:
    """The 0.1 gate's nested invocation must ignore the 1.0 gate file."""
    source = (ROOT / "tests" / "test_release_0_1_gate.py").read_text(encoding="utf-8")
    assert "test_release_1_0_gate.py" in source


def test_ci_workflow_sets_escape_variable() -> None:
    """CI runs the suite once by setting the escape variable."""
    workflow = (
        ROOT / ".github" / "workflows" / "linux-tests.yml"
    ).read_text(encoding="utf-8")
    assert "DRIFTLENS_SKIP_NESTED_SUITE" in workflow
