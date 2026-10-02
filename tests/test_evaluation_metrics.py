"""Retrieval metric tests (Step 39)."""

from __future__ import annotations

import pytest

from driftlens.evaluation.metrics import (
    average_top_k_similarity,
    chunk_recall_at_k,
    hit_rate_at_k,
    mrr_at_kmax,
    recall_at_k,
    to_document_level,
)
from driftlens.evaluation.retrievers import RankedResult


def _chunk(record_id: str, document_id: str, score: float = 1.0) -> RankedResult:
    return RankedResult(record_id=record_id, document_id=document_id, score=score)


def test_to_document_level_dedupes_preserving_rank() -> None:
    result = to_document_level(
        [_chunk("c1", "A"), _chunk("c2", "B"), _chunk("c3", "A"), _chunk("c4", "C")]
    )
    assert result.document_ranked == ["A", "B", "C"]
    assert [chunk.record_id for chunk in result.chunk_ranked] == ["c1", "c2", "c3", "c4"]


def test_recall_at_k_none_when_no_relevant() -> None:
    assert recall_at_k(["d1", "d2"], set(), 5) is None


def test_recall_at_k_basic() -> None:
    ranked = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d1", "d2", "d9", "d10"}
    assert recall_at_k(ranked, relevant, 5) == 0.5


def test_hit_rate_at_k_binary() -> None:
    assert hit_rate_at_k(["d1", "d2"], {"d2"}, 5) == 1.0
    assert hit_rate_at_k(["d1", "d2"], {"d9"}, 5) == 0.0


def test_mrr_first_relevant_at_rank_3() -> None:
    chunks = [_chunk("c1", "A"), _chunk("c2", "B"), _chunk("c3", "R"), _chunk("c4", "C")]
    assert mrr_at_kmax(chunks, {"R"}, 10) == pytest.approx(1 / 3)


def test_mrr_uses_raw_chunk_rank_not_document_deduped_rank() -> None:
    chunks = [_chunk("c1", "A"), _chunk("c2", "A"), _chunk("c3", "A"), _chunk("c4", "R")]
    deduped = to_document_level(chunks).document_ranked
    assert deduped == ["A", "R"]
    assert mrr_at_kmax(chunks, {"R"}, 10) == pytest.approx(0.25)


def test_mrr_zero_when_not_found() -> None:
    chunks = [_chunk("c1", "A"), _chunk("c2", "B")]
    assert mrr_at_kmax(chunks, {"R"}, 10) == 0.0


def test_chunk_recall_requires_relevant_chunk_ids() -> None:
    assert chunk_recall_at_k([_chunk("c1", "A")], set(), 5) is None


def test_average_top_k_similarity_empty_ranked_list() -> None:
    assert average_top_k_similarity([], 5) is None
