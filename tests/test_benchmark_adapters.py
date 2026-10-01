"""Item 5: adapter and generator conformance fixtures (Step 34)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from driftlens.adapters.chroma import load_chroma
from driftlens.adapters.csv import load_csv
from driftlens.adapters.faiss import load_faiss
from driftlens.adapters.jsonl import load_jsonl
from driftlens.adapters.parquet import load_parquet
from driftlens.analysis.content import compare_content
from driftlens.config.schema import SideOptions, default_config
from driftlens.errors import (
    DiagnosticSink,
    RefuseUntrustedPickleError,
    UnsupportedFormatError,
)
from driftlens.schema.manifest import FieldMappingSummary
from driftlens.schema.snapshot import build_snapshot
from driftlens.security.limits import RuntimeLimits

from fixtures.embedding_fixtures import (
    chroma_directory_with_pickle,
    faiss_pq_index,
    normalized_vectors,
)
from fixtures.generator import generate_corpus_pair, write_corpus_as


@pytest.fixture()
def workdir():
    base = Path(__file__).resolve().parent / "_benchmark_adapters_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _summary(mapping) -> FieldMappingSummary:
    roles = {}
    for role in (
        "text",
        "document_id",
        "chunk_id",
        "embedding",
        "chunk_index",
        "model",
        "provider",
        "dtype",
        "metric",
    ):
        value = getattr(mapping, role, None)
        if value is not None:
            roles[role] = value
    return FieldMappingSummary(roles=roles, ignored=dict(mapping.ignored))


def _snapshot(loaded):
    cfg = default_config()
    return build_snapshot(
        loaded.records,
        side=SideOptions(),
        cfg=cfg,
        source_type=loaded.source_type,
        sink=DiagnosticSink(),
        limits=RuntimeLimits(cfg.limits),
        granularity_override=None,
        field_mapping_summary=_summary(loaded.field_mapping),
        unsafe_pickle=False,
    )


def _load(fmt, path):
    limits = RuntimeLimits(default_config().limits)
    side = SideOptions()
    if fmt == "jsonl":
        return load_jsonl(path, side=side, limits=limits)
    if fmt == "csv":
        return load_csv(path, side=side, limits=limits)
    if fmt == "parquet":
        return load_parquet(path, side=side, limits=limits)
    raise ValueError(fmt)


def _pair_snapshots(workdir, pair):
    first = write_corpus_as(pair.v1_records, "jsonl", workdir / "v1.jsonl")
    second = write_corpus_as(pair.v2_records, "jsonl", workdir / "v2.jsonl")
    return _snapshot(_load("jsonl", first)), _snapshot(_load("jsonl", second))


def test_generator_deterministic() -> None:
    first = generate_corpus_pair(42, 100, "pct10")
    second = generate_corpus_pair(42, 100, "pct10")
    assert json.dumps(first.v1_records, sort_keys=True) == json.dumps(
        second.v1_records, sort_keys=True
    )
    assert json.dumps(first.v2_records, sort_keys=True) == json.dumps(
        second.v2_records, sort_keys=True
    )


def test_generator_pct10_touches_exactly_10_percent(workdir) -> None:
    pair = generate_corpus_pair(42, 100, "pct10")
    expected = (
        pair.expected.modified + pair.expected.replaced + pair.expected.format_only
    )
    assert expected == 10

    v1, v2 = _pair_snapshots(workdir, pair)
    rows, _chunks, _metrics = compare_content(
        v1, v2, cfg=default_config(), sink=DiagnosticSink()
    )
    changed = sum(
        1
        for row in rows
        if row.status in ("MODIFIED", "REPLACED", "FORMAT_ONLY_CHANGE")
    )
    assert changed == 10


def test_generator_whitespace_only_gives_format_only_change(workdir) -> None:
    pair = generate_corpus_pair(42, 50, "whitespace_only")
    v1, v2 = _pair_snapshots(workdir, pair)
    rows, _chunks, _metrics = compare_content(
        v1, v2, cfg=default_config(), sink=DiagnosticSink()
    )
    assert rows
    assert all(row.status == "FORMAT_ONLY_CHANGE" for row in rows)


def test_generator_replaced_gives_replaced_status(workdir) -> None:
    pair = generate_corpus_pair(42, 50, "replaced")
    v1, v2 = _pair_snapshots(workdir, pair)
    rows, _chunks, _metrics = compare_content(
        v1, v2, cfg=default_config(), sink=DiagnosticSink()
    )
    statuses = [row.status for row in rows]
    assert "MODIFIED" not in statuses
    assert statuses.count("REPLACED") == pair.expected.replaced


def test_cross_format_snapshot_id_equality(workdir) -> None:
    records = generate_corpus_pair(seed=1, size=20, change_profile="none").v1_records
    snapshot_ids = []
    for fmt, extension in (("jsonl", "jsonl"), ("parquet", "parquet"), ("csv", "csv")):
        path = write_corpus_as(records, fmt, workdir / f"corpus.{extension}")
        snapshot_ids.append(_snapshot(_load(fmt, path)).manifest.snapshot_id)
    assert len(set(snapshot_ids)) == 1


@pytest.mark.extras
@pytest.mark.xfail(
    reason="IndexPQ reconstructs in this faiss build (Step 17 sanctioned xfail)",
    strict=False,
)
def test_faiss_pq_fixture_actually_refuses(workdir) -> None:
    import faiss

    vectors = normalized_vectors(50, 16)
    index = faiss_pq_index(vectors)
    path = workdir / "pq.faiss"
    faiss.write_index(index, str(path))
    with pytest.raises(UnsupportedFormatError):
        load_faiss(
            path,
            side=SideOptions(),
            limits=RuntimeLimits(default_config().limits),
            allow_lossy_vectors=False,
            allow_unsafe_pickle=False,
        )


@pytest.mark.extras
def test_chroma_pickle_fixture_actually_refuses(workdir) -> None:
    directory = chroma_directory_with_pickle(workdir / "chroma_pickle")
    with pytest.raises(RefuseUntrustedPickleError):
        load_chroma(
            directory,
            side=SideOptions(),
            limits=RuntimeLimits(default_config().limits),
        )
