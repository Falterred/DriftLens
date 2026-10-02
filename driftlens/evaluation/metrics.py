"""Retrieval evaluation metrics over ranked chunks (Step 39, section 11.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from driftlens.evaluation.retrievers import RankedResult


@dataclass
class DocumentLevelResult:
    """A chunk-ranked list plus its document-deduplicated projection."""

    chunk_ranked: list[RankedResult]
    document_ranked: list[str]


def to_document_level(chunk_results: Sequence[RankedResult]) -> DocumentLevelResult:
    """Deduplicate ranked chunks to documents, keeping first-occurrence order."""

    seen: set[str] = set()
    document_ranked: list[str] = []
    for result in chunk_results:
        if result.document_id in seen:
            continue
        seen.add(result.document_id)
        document_ranked.append(result.document_id)
    return DocumentLevelResult(
        chunk_ranked=list(chunk_results), document_ranked=document_ranked
    )


def recall_at_k(
    document_ranked: Sequence[str], relevant_document_ids: set[str], k: int
) -> float | None:
    if not relevant_document_ids:
        return None
    found = len(set(document_ranked[:k]) & relevant_document_ids)
    return found / len(relevant_document_ids)


def hit_rate_at_k(
    document_ranked: Sequence[str], relevant_document_ids: set[str], k: int
) -> float | None:
    if not relevant_document_ids:
        return None
    return 1.0 if set(document_ranked[:k]) & relevant_document_ids else 0.0


def mrr_at_kmax(
    chunk_ranked: Sequence[RankedResult], relevant_document_ids: set[str], kmax: int
) -> float:
    for rank, result in enumerate(chunk_ranked[:kmax], start=1):
        if result.document_id in relevant_document_ids:
            return 1.0 / rank
    return 0.0


def chunk_recall_at_k(
    chunk_ranked: Sequence[RankedResult], relevant_chunk_ids: set[str], k: int
) -> float | None:
    if not relevant_chunk_ids:
        return None
    found_ids = {result.record_id for result in chunk_ranked[:k]}
    return len(found_ids & relevant_chunk_ids) / len(relevant_chunk_ids)


def average_top_k_similarity(chunk_ranked: Sequence[RankedResult], k: int) -> float | None:
    top = chunk_ranked[:k]
    if not top:
        return None
    return sum(result.score for result in top) / len(top)
