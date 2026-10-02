"""Lexical and dense retrieval indexes over snapshot records (Step 39, section 11.1)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from driftlens.analysis.ann import AnnIndex, ann_available, build_ann_index
from driftlens.analysis.neighbours import score_pair
from driftlens.config.schema import Bm25Cfg
from driftlens.schema.record import Record
from driftlens.text.normalize import normalize_text
from driftlens.text.tokenizer import tokenize


@dataclass
class RankedResult:
    """One retrieved record with its owning document and similarity score."""

    record_id: str
    document_id: str
    score: float


def _owning_document_id(record: Record) -> str:
    return record.document_id if record.document_id is not None else record.record_id


class BM25Index:
    """Okapi BM25 over one snapshot's own corpus statistics (section 11.1)."""

    def __init__(
        self, records: Sequence[Record], *, cfg: Bm25Cfg, casefold: bool
    ) -> None:
        self._k1 = cfg.k1
        self._b = cfg.b
        self._casefold = casefold
        self._doc_len: dict[str, int] = {}
        self._term_freq: dict[str, Counter[str]] = {}
        self._doc_freq: dict[str, int] = {}
        self._document_id: dict[str, str] = {}
        for record in records:
            text = record.normalized_text
            if text is None:
                text = record.text
            if text is None:
                continue
            tokens = self._tokenize(text)
            self._doc_len[record.record_id] = len(tokens)
            self._term_freq[record.record_id] = Counter(tokens)
            self._document_id[record.record_id] = _owning_document_id(record)
            for term in set(tokens):
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._n = len(self._doc_len)
        self._vocabulary = set(self._doc_freq)
        self._avgdl = sum(self._doc_len.values()) / self._n if self._n else 0.0

    def _tokenize(self, text: str) -> list[str]:
        tokens = tokenize(text)
        if self._casefold:
            return [token.casefold() for token in tokens]
        return list(tokens)

    def search(self, query: str, *, top_k: int) -> list[RankedResult]:
        if self._n == 0:
            return []
        query_terms = set(self._tokenize(normalize_text(query)))
        if not (query_terms & self._vocabulary):
            return []
        results: list[RankedResult] = []
        for record_id, frequencies in self._term_freq.items():
            doc_len = self._doc_len[record_id]
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency <= 0:
                    continue
                document_frequency = self._doc_freq.get(term, 0)
                idf = math.log(
                    (self._n - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                    + 1.0
                )
                denominator = frequency + self._k1 * (
                    1.0 - self._b + self._b * doc_len / self._avgdl
                )
                score += idf * (frequency * (self._k1 + 1.0)) / denominator
            results.append(
                RankedResult(
                    record_id=record_id,
                    document_id=self._document_id[record_id],
                    score=score,
                )
            )
        results.sort(key=lambda result: (-result.score, result.record_id))
        return results[:top_k]


class DenseProvidedIndex:
    """Exhaustive dense search over each record's provided embedding (section 11.1)."""

    def __init__(
        self,
        records: Sequence[Record],
        *,
        metric: Literal["cosine", "dot", "l2"],
        use_ann: bool = False,
        ann_threshold: int | None = None,
        ann_seed: int = 0,
    ) -> None:
        self._metric = metric
        self._entries: list[tuple[str, str, np.ndarray]] = []
        for record in records:
            embedding = record.embedding
            if embedding.vector is None or embedding.validity != "ok":
                continue
            vector = np.asarray(embedding.vector, dtype=np.float64)
            if not np.any(vector != 0.0):
                continue
            self._entries.append(
                (record.record_id, _owning_document_id(record), vector)
            )
        self._ann: AnnIndex | None = None
        if (
            use_ann
            and ann_threshold is not None
            and len(self._entries) > ann_threshold
            and ann_available()
        ):
            matrix = np.asarray(
                [vector for _, _, vector in self._entries], dtype=np.float64
            )
            self._ann = build_ann_index(matrix, metric=metric, seed=ann_seed)

    def search(self, query_vector: np.ndarray, *, top_k: int) -> list[RankedResult]:
        if self._ann is not None:
            hits = self._ann.search(
                np.asarray(query_vector, dtype=np.float64), top_k=top_k
            )[0]
            return [
                RankedResult(
                    record_id=self._entries[index][0],
                    document_id=self._entries[index][1],
                    score=score,
                )
                for index, score in hits
            ]
        query = np.asarray(query_vector, dtype=np.float64)
        results: list[RankedResult] = []
        for record_id, document_id, vector in self._entries:
            score = score_pair(query, vector, self._metric)
            results.append(
                RankedResult(record_id=record_id, document_id=document_id, score=score)
            )
        results.sort(key=lambda result: (-result.score, result.record_id))
        return results[:top_k]
