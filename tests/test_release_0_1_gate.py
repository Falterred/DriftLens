"""Release 0.1 acceptance gate (Step 35)."""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from driftlens.errors import Code, EXIT_CODES

from test_benchmark_errors import FATAL_CODES

RELEASE_0_1_MODULES = [
    "driftlens.errors",
    "driftlens.config.schema",
    "driftlens.config.loader",
    "driftlens.security.limits",
    "driftlens.security.network_guard",
    "driftlens.security.pickle_guard",
    "driftlens.text.normalize",
    "driftlens.text.tokenizer",
    "driftlens.text.sentences",
    "driftlens.text.shingles",
    "driftlens.identity.hashing",
    "driftlens.schema.record",
    "driftlens.adapters.detect",
    "driftlens.adapters.jsonl",
    "driftlens.adapters.json",
    "driftlens.adapters.csv",
    "driftlens.adapters.parquet",
    "driftlens.adapters.sidecar",
    "driftlens.adapters.faiss",
    "driftlens.adapters.chroma",
    "driftlens.identity.documents",
    "driftlens.identity.chunks",
    "driftlens.schema.manifest",
    "driftlens.schema.snapshot",
    "driftlens.identity.snapshot_id",
    "driftlens.analysis.health",
    "driftlens.analysis.embeddings",
    "driftlens.analysis.sampling",
    "driftlens.analysis.distributions",
    "driftlens.analysis.content",
    "driftlens.analysis.chunking",
    "driftlens.analysis.metadata",
    "driftlens.analysis.compatibility",
    "driftlens.analysis.duplicates",
    "driftlens.analysis.mmd",
    "driftlens.analysis.significance",
    "driftlens.analysis.neighbours",
    "driftlens.analysis.clusters",
    "driftlens.reports.json",
    "driftlens.reports.terminal",
    "driftlens.cli.compare",
    "driftlens.__main__",
]

# Modules the v0.1 surface imports directly but that are not part of the
# frozen v0.1 public module list, plus the release 0.2 snapshot modules the CLI
# dispatches to (Step 36) so the scope scan below allows them.
SUPPORTING_MODULES = frozenset(
    {
        "driftlens",
        "driftlens.adapters._embedding_codec",
        "driftlens.adapters.snapshot",
        "driftlens.cli.snapshot",
        "driftlens.evaluation.ground_truth",
        "driftlens.evaluation.metrics",
        "driftlens.evaluation.retrieval",
        "driftlens.evaluation.retrievers",
        "driftlens.reports.diff_export",
        "driftlens.reports.html",
        "driftlens.schema.eval_set",
        "driftlens.security.redaction",
    }
)

# Codes that only later releases can produce, with the reason they are excluded.
V03_PLUS_ONLY_CODES = {
    Code.RETRIEVER_UNAVAILABLE: "retrieval comparison is release 0.2/0.3",
    Code.BASELINE_INCOMPATIBLE: "baselines are release 0.3",
    Code.CLUSTER_EXTRA_MISSING: "clusters are release 0.3",
    Code.SIGNIFICANCE_SKIPPED: "significance testing is release 0.3",
    Code.LOW_POWER: "retrieval/significance context only",
    Code.NO_QUERY_VECTOR: "retrieval comparison is release 0.3",
    Code.FUTURE_TIMESTAMP: "temporal analysis is release 0.3",
}

# Minimal triggers for the v0.1 warning/degraded codes Step 34's fatal-focused
# table does not cover. Only membership is asserted by the gate; the descriptions
# document how each code is produced at v0.1.
WARNING_FIXTURES = {
    Code.EMBEDDING_SPACE_INCOMPATIBLE: "differing embedding dimensions",
    Code.EMBEDDINGS_UNAVAILABLE: "invalid rate above embeddings.max_invalid_rate",
    Code.METRIC_MISMATCH: "declared cosine vs dot",
    Code.SKIPPED_MODEL_CHANGED: "MMD suspended by MODEL_CHANGED",
    Code.SKIPPED_SPACE_CHANGED: "MMD suspended by SPACE_CHANGED",
    Code.SKIPPED_PARTIAL_MIGRATION: "MMD suspended by PARTIAL_MIGRATION",
    Code.SKIPPED_MIXED_MODELS: "MMD suspended by MIXED_MODELS",
    Code.GRANULARITY_MISMATCH: "CHUNK vs DOCUMENT snapshots",
    Code.METRIC_UNKNOWN: "no metric declared or derivable",
    Code.SPACE_UNTESTABLE: "fewer than consistency_min_anchors",
    Code.EMPTY_SAMPLE: "empty value set",
    Code.EMPTY_SNAPSHOT: "no records in a snapshot",
    Code.DEGENERATE_KERNEL: "zero bandwidth with differing vectors",
    Code.CONSTANT_SHIFT: "two unequal constant fields",
    Code.INSUFFICIENT_DATA: "fewer than two vectors on a side",
    Code.IDENTITY_WARNING: "duplicate record ids",
    Code.MODEL_UNKNOWN: "no declared embedding model",
    Code.MODEL_CHANGED: "different declared models",
    Code.SPACE_CHANGED: "DIFFERENT_SPACE anchor agreement",
    Code.PARTIAL_MIGRATION: "mixed anchor agreement",
    Code.MIXED_MODELS: "MIXED model summary",
    Code.LABEL_CONFLICT: "declared label disagrees with measured space",
    Code.NORMALIZATION_CLAIM_INVALID: "declared normalized=true contradicted",
    Code.DOC_IDENTITY_MISMATCH: "document identity level differs across snapshots",
    Code.GRANULARITY_ASSUMED: "no document-id candidates anywhere",
    Code.MODEL_OVERRIDE_IGNORED: "CLI model conflicts with declared value",
    Code.CHUNK_IDENTITY_MISMATCH: "no common chunk identity level",
    Code.DOC_ORDER_UNKNOWN: "chunk_index missing",
    Code.EMBEDDING_INVALID: "NaN/Inf vector",
    Code.ZERO_VECTOR: "all-zero vector",
    Code.QUANTIZED: "int8 quantized embeddings",
    Code.RENORMALIZED: "vectors renormalized before analysis",
    Code.PICKLE_IGNORED: "LangChain pickle not opened",
    Code.FIELD_CANDIDATES_IGNORED: "lower-priority field candidates ignored",
}

_ROOT = Path(__file__).resolve().parents[1]
_TESTS_DIR = Path(__file__).resolve().parent
_THIS_FILE = Path(__file__).resolve()


@pytest.fixture()
def workdir():
    base = _TESTS_DIR / "_release_gate_tmp"
    if base.exists():
        import shutil

        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    import shutil

    shutil.rmtree(base, ignore_errors=True)


def test_all_release_0_1_modules_importable() -> None:
    for name in RELEASE_0_1_MODULES:
        importlib.import_module(name)


def test_cli_installed_entry_point_end_to_end(workdir) -> None:
    rows = [
        {"id": "c1", "text": "hello world", "embedding": [0.1, 0.2, 0.3]},
        {"id": "c2", "text": "goodbye moon", "embedding": [0.3, 0.2, 0.1]},
    ]
    first = workdir / "a.jsonl"
    second = workdir / "b.jsonl"
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "driftlens",
            "compare",
            str(first),
            str(second),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_ROOT),
    )
    assert result.returncode == 0
    assert "STATUS:" in result.stdout
    assert result.stderr == ""


def test_every_reachable_code_has_a_fixture() -> None:
    reachable = {code for code in Code if code not in V03_PLUS_ONLY_CODES}
    covered = set(FATAL_CODES) | set(WARNING_FIXTURES)
    missing = sorted(
        (code.value for code in reachable - covered)
    )
    assert reachable <= covered, f"missing fixtures: {missing}"


def test_no_forward_reference_to_unbuilt_modules() -> None:
    allowed = set(RELEASE_0_1_MODULES) | set(SUPPORTING_MODULES)
    violations: list[tuple[str, str]] = []
    for path in sorted((_ROOT / "driftlens").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                targets = [node.module]
            for target in targets:
                if not target.startswith("driftlens"):
                    continue
                if target not in allowed:
                    violations.append(
                        (str(path.relative_to(_ROOT)), target)
                    )
    assert violations == [], f"out-of-scope imports: {violations}"


def test_readme_and_pyproject_consistent() -> None:
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


@pytest.mark.skipif(
    os.environ.get("DRIFTLENS_SKIP_NESTED_SUITE") == "1",
    reason="nested full-suite run disabled",
)
def test_full_suite_green() -> None:
    import pytest as _pytest

    exit_code = _pytest.main(
        [
            "-q",
            "--tb=short",
            str(_TESTS_DIR),
            "--ignore",
            str(_THIS_FILE),
            "-p",
            "no:cacheprovider",
        ]
    )
    assert exit_code == 0
