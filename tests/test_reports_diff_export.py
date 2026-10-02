"""Diff and breakdown export writers (Step 41)."""

from __future__ import annotations

import os
import secrets
import shutil
import types
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from driftlens.analysis.content import ChunkDiffResult, DocumentDiffRow
from driftlens.analysis.health import Metric
from driftlens.config.schema import default_config
from driftlens.reports.diff_export import (
    compute_breakdown,
    top_n_changed_documents,
    write_chunks_diff,
    write_documents_diff,
)
from driftlens.reports.json import metric_to_json
from driftlens.reports.terminal import format_category_labels
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import Record
from driftlens.schema.snapshot import (
    DocumentRow,
    DocumentTable,
    Snapshot,
    StatisticsAccumulator,
)


@pytest.fixture()
def workdir() -> Path:
    # A unique directory under tests/ (not pytest's tmp_path) keeps the
    # file-sandbox from tripping over pytest's mode-0o700 basetemp.
    base = Path(__file__).parent / f"_diff_export_tmp_{os.getpid()}_{secrets.token_hex(6)}"
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _manifest(record_ids: list[str]) -> Manifest:
    return Manifest(
        schema_version="1.0",
        snapshot_id="snap",
        content_digest="digest",
        normalization_version="nfc-ws-1",
        tokenizer_id="simple-v1",
        text_params={
            "min_overlap_chars": 16,
            "overlap_window": 2000,
            "casefold": True,
        },
        driftlens_version="0.1.0",
        created_at="2026-01-01T00:00:00+00:00",
        source_type="jsonl",
        granularity="CHUNK",
        granularity_source="detected",
        document_count=len(record_ids),
        chunk_count=len(record_ids),
        embedding_dimension=None,
        embedding_dtype=None,
        embedding_model=None,
        embedding_model_counts={},
        embedding_provider=None,
        embedding_metadata_confidence={},
        similarity_metric="cosine",
        embedding_normalized=None,
        chunk_identity_levels={record_id: [4] for record_id in record_ids},
        identity_summary={"HIGH": len(record_ids), "MEDIUM": 0, "LOW": 0},
        field_mapping=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _record(
    record_id: str,
    *,
    document_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Record:
    return Record(
        record_id=record_id,
        document_id=document_id if document_id is not None else record_id,
        metadata=metadata or {},
    )


def _snapshot(records: list[Record]) -> Snapshot:
    rows = {
        record.document_id: DocumentRow(
            document_id=record.document_id,
            chunk_count=1,
            identity_confidence="HIGH",
        )
        for record in records
    }
    return Snapshot(
        manifest=_manifest([record.record_id for record in records]),
        records=list(records),
        documents=DocumentTable(rows=rows),
        statistics=StatisticsAccumulator(),
    )


def _doc_rows(ids: list[str], *, status: str = "UNCHANGED") -> list[DocumentDiffRow]:
    return [
        DocumentDiffRow(
            document_id=document_id,
            status=status,  # type: ignore[arg-type]
            metadata_changed=False,
            timestamp_inconsistent=False,
            reconstruction_confidence_v1=None,
            reconstruction_confidence_v2=None,
            similarity=None,
        )
        for document_id in ids
    ]


def test_write_documents_diff_no_text_by_default(workdir: Path) -> None:
    path = workdir / "documents.parquet"
    write_documents_diff(
        _doc_rows(["d1"]),
        path=path,
        diagnostic=False,
        v1_text_by_doc=None,
        v2_text_by_doc=None,
    )
    schema = pq.read_schema(path)
    assert "text_excerpt_v1" not in schema.names
    assert "text_excerpt_v2" not in schema.names


def test_write_documents_diff_diagnostic_truncates_to_300_chars(
    workdir: Path,
) -> None:
    path = workdir / "documents.parquet"
    write_documents_diff(
        _doc_rows(["d1"]),
        path=path,
        diagnostic=True,
        v1_text_by_doc={"d1": "x" * 1000},
        v2_text_by_doc={"d1": "y" * 1000},
    )
    table = pq.read_table(path)
    assert "text_excerpt_v1" in table.column_names
    assert len(table.column("text_excerpt_v1")[0].as_py()) == 300
    assert len(table.column("text_excerpt_v2")[0].as_py()) == 300


def test_write_chunks_diff_added_removed_nullable_sides(workdir: Path) -> None:
    removed = _record("r1", document_id="d1")
    added = _record("r2", document_id="d2")
    result = ChunkDiffResult(
        pairs=[(removed, None, "REMOVED"), (None, added, "ADDED")],
        unmatched_v1=[],
        unmatched_v2=[],
    )
    path = workdir / "chunks.parquet"
    write_chunks_diff(result, path=path, diagnostic=False)
    table = pq.read_table(path)
    assert table.column("v1_record_id").to_pylist() == ["r1", None]
    assert table.column("v2_record_id").to_pylist() == [None, "r2"]


def test_compute_breakdown_top20_plus_other(workdir: Path) -> None:
    ids = [f"d{index:02d}" for index in range(25)]
    v1 = _snapshot(
        [_record(doc, metadata={"source": f"src-{doc}"}) for doc in ids]
    )
    v2 = _snapshot(
        [_record(doc, metadata={"source": f"src-{doc}"}) for doc in ids]
    )
    rows = compute_breakdown(
        "source",
        v1=v1,
        v2=v2,
        document_diff=_doc_rows(ids, status="MODIFIED"),
        cfg=default_config(),
    )
    assert len(rows) == 21
    other = rows[-1]
    assert other.is_other is True
    assert other.segment_value == "other"
    assert other.document_count == 5


def test_compute_breakdown_mmd_only_above_500(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_mmd(v1_vectors, v2_vectors, v1_ids, v2_ids, *, cfg, fixed_bandwidth, seed):
        calls.append((len(v1_ids), len(v2_ids)))
        return types.SimpleNamespace(mmd=0.75)

    monkeypatch.setattr("driftlens.reports.diff_export.compute_mmd", fake_mmd)

    def build(count: int) -> Snapshot:
        return _snapshot(
            [_record(f"r{index:04d}", metadata={"source": "s"}) for index in range(count)]
        )

    small_v1, small_v2 = build(400), build(400)
    small_rows = compute_breakdown(
        "source",
        v1=small_v1,
        v2=small_v2,
        document_diff=_doc_rows([record.record_id for record in small_v2.records]),
        cfg=default_config(),
    )
    assert small_rows[0].mmd is None
    assert calls == []

    large_v1, large_v2 = build(600), build(600)
    large_rows = compute_breakdown(
        "source",
        v1=large_v1,
        v2=large_v2,
        document_diff=_doc_rows([record.record_id for record in large_v2.records]),
        cfg=default_config(),
    )
    assert large_rows[0].mmd == 0.75
    assert len(calls) == 1


def test_top_n_changed_excludes_unchanged() -> None:
    rows = _doc_rows(["d1", "d2", "d3"], status="MODIFIED")
    rows += _doc_rows(["d4", "d5"], status="UNCHANGED")
    assert top_n_changed_documents(rows, n=10) == ["d1", "d2", "d3"]


def test_top_n_changed_tier_ordering() -> None:
    rows = _doc_rows(["m"], status="MODIFIED")
    rows += _doc_rows(["a"], status="ADDED")
    rows += _doc_rows(["r"], status="REPLACED")
    assert top_n_changed_documents(rows, n=2) == ["r", "m"]


def test_privacy_display_cap_independent_of_analysis_cap() -> None:
    labels = {f"cat-{index:02d}": 100 - index for index in range(50)}
    metric = Metric(name="metadata:categories", details={"category_labels": labels})
    rows = format_category_labels(metric)
    category_rows = [row for row in rows if not row.strip().startswith("+")]
    assert len(category_rows) == 20
    assert rows[-1].strip() == "+30 more"

    payload = metric_to_json(metric)
    assert len(payload["details"]["category_labels"]) == 20
    assert payload["details"]["category_labels_truncated"] is True
