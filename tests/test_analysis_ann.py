"""Optional FAISS-backed ANN search (Step 50)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from driftlens.analysis.ann import build_ann_index
from driftlens.analysis.duplicates import (
    DuplicateUnit,
    compare_duplicates,
    compute_semantic_groups,
)
from driftlens.config.schema import DuplicatesCfg, default_config
from driftlens.errors import Code, DiagnosticSink
from driftlens.evaluation.retrievers import DenseProvidedIndex
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import (
    DocumentTable,
    Snapshot,
    StatisticsAccumulator,
)


def _embedding_record(record_id, vector, *, metric="cosine", validity="ok"):
    record = Record(record_id=record_id)
    array = np.asarray(vector, dtype=float)
    record.embedding = EmbeddingInfo(
        vector=array,
        dtype="float64",
        dimension=array.shape[0],
        norm=float(np.linalg.norm(array)),
        validity=validity,
        model=None,
        provider=None,
        similarity_metric=metric,
        normalized=False,
    )
    record._embedding_digest_hex = record_id
    return record


def _unit(record_id, *, text=None, vector=None, metric="cosine",
          document_id=None, chunk_index=None):
    record = Record(record_id=record_id, text=text, normalized_text=text)
    if text is not None:
        record.normalized_text_hash = text
    record.document_id = document_id
    record.chunk_index = chunk_index
    if vector is not None:
        record.embedding = _embedding_record(record_id, vector, metric=metric).embedding
    return DuplicateUnit(unit_id=record_id, representative=record, members=[record])


def _manifest(record_ids, *, similarity_metric=None):
    return Manifest(
        schema_version="1.0",
        snapshot_id="snap",
        content_digest="digest",
        normalization_version="nfc-ws-1",
        tokenizer_id="simple-v1",
        text_params={"min_overlap_chars": 16, "overlap_window": 2000, "casefold": True},
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
        similarity_metric=similarity_metric,
        embedding_normalized=None,
        chunk_identity_levels={record_id: [4] for record_id in record_ids},
        identity_summary={"HIGH": len(record_ids), "MEDIUM": 0, "LOW": 0},
        field_mapping=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _snapshot(records, *, similarity_metric=None):
    return Snapshot(
        manifest=_manifest(
            [record.record_id for record in records],
            similarity_metric=similarity_metric,
        ),
        records=list(records),
        documents=DocumentTable(rows={}),
        statistics=StatisticsAccumulator(),
    )


def _paired_records(pair_count):
    records = []
    for index in range(pair_count):
        angle = 2.0 * np.pi * index / pair_count
        vector = [float(np.cos(angle)), float(np.sin(angle))]
        records.append(_embedding_record("a%d" % index, vector))
        records.append(_embedding_record("b%d" % index, vector))
    return records


def _paired_units(pair_count):
    units = []
    for record in _paired_records(pair_count):
        units.append(
            _unit(record.record_id, vector=list(record.embedding.vector))
        )
    return units


@pytest.fixture()
def workdir():
    base = Path(__file__).resolve().parent / "_ann_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    yield base
    shutil.rmtree(base, ignore_errors=True)


def test_build_ann_index_single_threaded(monkeypatch):
    faiss = pytest.importorskip("faiss")
    calls = []
    real = faiss.omp_set_num_threads

    def _spy(value):
        calls.append(value)
        return real(value)

    monkeypatch.setattr(faiss, "omp_set_num_threads", _spy)
    prior = faiss.omp_get_max_threads()
    vectors = np.random.RandomState(0).rand(64, 8)
    build_ann_index(vectors, metric="l2", seed=7)
    assert calls[0] == 1
    assert calls[-1] == prior


def test_ann_search_returns_valid_indices():
    pytest.importorskip("faiss")
    vectors = np.random.RandomState(1).rand(64, 8)
    index = build_ann_index(vectors, metric="cosine", seed=3)
    hits = index.search(vectors[:3], top_k=5)
    assert len(hits) == 3
    for row in hits:
        for position, score in row:
            assert 0 <= position < len(vectors)
            assert np.isfinite(score)


def test_ann_search_may_return_fewer_than_top_k():
    pytest.importorskip("faiss")
    vectors = np.random.RandomState(2).rand(8, 4)
    index = build_ann_index(vectors, metric="cosine", seed=4)
    hits = index.search(vectors[:1], top_k=50)
    assert len(hits) == 1
    assert 0 < len(hits[0]) <= len(vectors)
    assert len(hits[0]) < 50


def test_ann_cosine_normalizes_before_indexing():
    pytest.importorskip("faiss")
    vectors = np.asarray([[3.0, 0.0], [0.0, 5.0], [1.0, 1.0]])
    index = build_ann_index(vectors, metric="cosine", seed=0)
    hits = index.search(np.asarray([[6.0, 0.0]]), top_k=1)[0]
    assert hits[0][0] == 0
    assert hits[0][1] == pytest.approx(1.0, abs=0.05)


def test_semantic_duplicates_default_unaffected_by_ann_code_existing():
    units = [
        _unit("a", text="alpha", vector=[1.0, 0.0]),
        _unit("b", text="beta", vector=[1.0, 0.0]),
        _unit("c", text="gamma", vector=[0.0, 1.0]),
    ]
    baseline, _ = compute_semantic_groups(
        units,
        cfg=DuplicatesCfg(),
        threshold=0.9,
        similarity_metric_known=True,
        seed=42,
    )
    plain, _ = compute_semantic_groups(
        units,
        cfg=DuplicatesCfg(),
        threshold=0.9,
        similarity_metric_known=True,
        seed=42,
        use_ann=False,
    )
    assert [group.member_ids for group in plain] == [["a", "b"]]
    assert [group.member_ids for group in plain] == [
        group.member_ids for group in baseline
    ]

    params = []
    small, _ = compute_semantic_groups(
        units,
        cfg=DuplicatesCfg(max_exact_vectors=10),
        threshold=0.9,
        similarity_metric_known=True,
        seed=42,
        use_ann=True,
        ann_params_out=params,
    )
    assert [group.member_ids for group in small] == [["a", "b"]]
    assert params == []


def test_semantic_duplicates_ann_path_finds_known_duplicates():
    pytest.importorskip("faiss")
    pair_count = 20
    units = _paired_units(pair_count)
    params = []
    groups, _ = compute_semantic_groups(
        units,
        cfg=DuplicatesCfg(max_exact_vectors=4),
        threshold=0.98,
        similarity_metric_known=True,
        seed=42,
        use_ann=True,
        ann_params_out=params,
    )
    assert params, "the ANN path should have run"
    found = set()
    for group in groups:
        members = group.member_ids
        for index in range(len(members)):
            for other in range(index + 1, len(members)):
                found.add(frozenset({members[index], members[other]}))
    planted = {
        frozenset({"a%d" % index, "b%d" % index}) for index in range(pair_count)
    }
    recall = len(found & planted) / len(planted)
    assert recall >= 0.9


def test_dense_retrieval_ann_path_gated_by_threshold(monkeypatch):
    from driftlens.evaluation import retrievers as module

    built = []

    class _StubAnn:
        def search(self, query_vectors, *, top_k):
            return [[(1, 0.5)]]

    def _fake_build(vectors, *, metric, seed):
        built.append((tuple(vectors.shape), metric, seed))
        return _StubAnn()

    monkeypatch.setattr(module, "ann_available", lambda: True)
    monkeypatch.setattr(module, "build_ann_index", _fake_build)
    records = [
        _embedding_record("r1", [1.0, 0.0]),
        _embedding_record("r2", [0.0, 1.0]),
    ]
    exact = DenseProvidedIndex(
        records, metric="cosine", use_ann=True, ann_threshold=100
    )
    assert exact._ann is None
    assert built == []

    approx = DenseProvidedIndex(
        records, metric="cosine", use_ann=True, ann_threshold=0
    )
    assert built == [((2, 2), "cosine", 0)]
    results = approx.search(np.asarray([1.0, 0.0]), top_k=1)
    assert results[0].record_id == "r2"


def test_ann_unavailable_falls_back_silently(workdir, monkeypatch):
    from driftlens.__main__ import main
    from driftlens.cli import compare as compare_module

    captured = []

    class _CapturingSink(DiagnosticSink):
        def __init__(self):
            super().__init__()
            captured.append(self)

    monkeypatch.setattr(compare_module, "DiagnosticSink", _CapturingSink)
    monkeypatch.setattr(compare_module, "ann_available", lambda: False)

    rows = [
        {"id": "c1", "text": "hello world", "embedding": [0.1, 0.2, 0.3]},
        {"id": "c2", "text": "goodbye moon", "embedding": [0.3, 0.2, 0.1]},
    ]
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    first = workdir / "a.jsonl"
    second = workdir / "b.jsonl"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")

    code = main(["compare", str(first), str(second), "--ann"])
    assert code == 0
    assert captured
    assert captured[-1].has(Code.ANN_UNAVAILABLE_FALLBACK_EXACT)


def test_ann_build_params_recorded_in_details():
    pytest.importorskip("faiss")
    records = _paired_records(20)
    snapshot = _snapshot(records, similarity_metric="cosine")
    _, _, metrics = compare_duplicates(
        snapshot,
        snapshot,
        cfg=DuplicatesCfg(max_exact_vectors=4),
        drift_cfg=default_config().drift,
        near_threshold=0.5,
        semantic_threshold=0.98,
        similarity_metric_known=True,
        use_ann=True,
    )
    by_name = {metric.name: metric for metric in metrics}
    details = by_name["duplicate_rate_semantic"].details
    assert details is not None
    params = details["ann"]
    assert params["index_type"] == "IVFFlat"
    assert params["seed"] == 42
    assert set(params) == {"index_type", "nlist", "nprobe", "seed"}
    assert params["nlist"] >= 1
    assert params["nprobe"] >= 1
