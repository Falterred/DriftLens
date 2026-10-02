"""Retriever tests (Step 39)."""

from __future__ import annotations

import numpy as np
import pytest

from driftlens.config.schema import Bm25Cfg
from driftlens.evaluation.retrievers import BM25Index, DenseProvidedIndex
from driftlens.schema.record import EmbeddingInfo, Record


def _record(
    record_id: str,
    text: str | None = None,
    *,
    document_id: str | None = None,
    vector: list[float] | None = None,
    validity: str = "ok",
) -> Record:
    embedding = EmbeddingInfo(
        vector=np.asarray(vector, dtype=np.float64) if vector is not None else None,
        dtype="float64" if vector is not None else None,
        dimension=len(vector) if vector is not None else None,
        norm=None,
        validity=validity if vector is not None else None,
        model=None,
        provider=None,
        similarity_metric=None,
        normalized=None,
    )
    return Record(
        record_id=record_id,
        document_id=document_id if document_id is not None else record_id,
        text=text,
        normalized_text=text,
        embedding=embedding,
    )


def _bm25(*records: Record) -> BM25Index:
    return BM25Index(list(records), cfg=Bm25Cfg(), casefold=True)


def test_bm25_deterministic_scoring() -> None:
    index = _bm25(
        _record("r1", "alpha beta"),
        _record("r2", "beta gamma", document_id="d2"),
    )
    first = index.search("alpha beta", top_k=10)
    second = index.search("alpha beta", top_k=10)
    assert first == second
    assert [result.record_id for result in first]
    assert [result.record_id for result in first] == [result.record_id for result in second]


def test_bm25_empty_when_no_vocabulary_overlap() -> None:
    index = _bm25(_record("r1", "alpha beta"))
    assert index.search("zzz qqq", top_k=10) == []


def test_bm25_zero_score_document_still_included() -> None:
    index = _bm25(_record("r1", "alpha"), _record("r2", "beta"))
    results = index.search("alpha", top_k=10)
    assert [result.record_id for result in results] == ["r1", "r2"]
    assert results[1].score == 0.0


def test_bm25_ties_broken_by_record_id() -> None:
    index = _bm25(_record("r2", "alpha beta"), _record("r1", "alpha beta"))
    results = index.search("alpha", top_k=10)
    assert [result.record_id for result in results] == ["r1", "r2"]
    assert results[0].score == results[1].score


def test_bm25_per_snapshot_corpus_stats() -> None:
    first = _bm25(_record("a", "alpha"))
    second = _bm25(
        _record("a", "alpha"),
        _record("b", "alpha"),
        _record("c", "beta"),
    )
    first_score = next(
        result.score for result in first.search("alpha", top_k=10)
        if result.record_id == "a"
    )
    second_score = next(
        result.score for result in second.search("alpha", top_k=10)
        if result.record_id == "a"
    )
    assert first_score != second_score


def test_dense_cosine_scoring() -> None:
    index = DenseProvidedIndex([_record("r1", vector=[3.0, 4.0])], metric="cosine")
    results = index.search(np.asarray([6.0, 8.0]), top_k=10)
    assert results[0].record_id == "r1"
    assert results[0].score == pytest.approx(1.0)


def test_dense_l2_scoring_is_negative_squared_distance() -> None:
    index = DenseProvidedIndex(
        [_record("r1", vector=[1.0, 1.0]), _record("r2", vector=[5.0, 5.0])],
        metric="l2",
    )
    results = index.search(np.asarray([1.0, 1.0]), top_k=10)
    assert [result.record_id for result in results] == ["r1", "r2"]
    assert results[0].score == pytest.approx(0.0)
    assert results[1].score == pytest.approx(-32.0)
