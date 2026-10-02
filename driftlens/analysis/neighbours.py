"""Neighbour-overlap analysis between two snapshots (Step 46, section 8.9).

Each snapshot is searched in its own embedding space: a query is a chunk from
the snapshot itself, and its neighbours are the closest other members of that
snapshot's own sampled pool.  The two per-snapshot neighbour sets are then
compared with Jaccard similarity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from driftlens.analysis.compatibility import (
    ANALYSES_SUSPENDED_BY_STATE,
    CompatibilityResult,
    suspension_reason_code,
)
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.analysis.health import Metric, SampleInfo, make_metric
from driftlens.analysis.sampling import hash_threshold_sample, to_sample_info
from driftlens.config.schema import Config
from driftlens.identity.chunks import all_chunk_identity_keys
from driftlens.schema.record import Record
from driftlens.schema.snapshot import Snapshot
from driftlens.text.shingles import jaccard

if TYPE_CHECKING:
    from driftlens.analysis.content import ComparisonLevelResult

_DEFAULT_METRIC = "neighbour_overlap_mean"
_MIN_QUERIES = 30


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def score_pair(
    a: np.ndarray, b: np.ndarray, metric: Literal["cosine", "dot", "l2"]
) -> float:
    """Shared pairwise scoring formula for the three supported metrics."""

    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if metric == "cosine":
        return float(np.dot(_l2_normalize(first), _l2_normalize(second)))
    if metric == "dot":
        return float(np.dot(first, second))
    difference = first - second
    return -float(np.dot(difference, difference))


@dataclass
class NeighbourPool:
    record_ids: list[str]
    vectors: np.ndarray


def build_pool(
    records: Sequence[Record], *, seed: int, sample_size: int
) -> NeighbourPool:
    """Valid, non-zero embeddings, reduced to sample_size by hash sampling."""

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for record in records:
        embedding = record.embedding
        if embedding.vector is None or embedding.validity != "ok":
            continue
        vector = np.asarray(embedding.vector, dtype=np.float64)
        if not np.any(vector != 0.0):
            continue
        ids.append(record.record_id)
        vectors.append(vector)

    if len(ids) > sample_size:
        kept = hash_threshold_sample(
            seed, sample_size, ids, v2_record_ids=None
        ).v1_ids
        reduced_ids: list[str] = []
        reduced_vectors: list[np.ndarray] = []
        for record_id, vector in zip(ids, vectors):
            if record_id in kept:
                reduced_ids.append(record_id)
                reduced_vectors.append(vector)
        ids = reduced_ids
        vectors = reduced_vectors

    if not vectors:
        return NeighbourPool(record_ids=[], vectors=np.empty((0, 0), dtype=np.float64))
    return NeighbourPool(record_ids=ids, vectors=np.vstack(vectors))


class NeighbourSearchIndex:
    """Exact, exhaustive nearest-neighbour search within one snapshot's pool."""

    def __init__(
        self, pool: NeighbourPool, *, metric: Literal["cosine", "dot", "l2"]
    ) -> None:
        self._metric = metric
        self._ids = list(pool.record_ids)
        vectors = np.asarray(pool.vectors, dtype=np.float64)
        if metric == "cosine" and vectors.size:
            norms = np.linalg.norm(vectors, axis=1)
            safe = np.where(norms == 0.0, 1.0, norms)
            vectors = vectors / safe[:, None]
        self._vectors = vectors
        self._index = {record_id: index for index, record_id in enumerate(self._ids)}

    def top_k_other_than(self, query_record_id: str, *, k: int) -> set[str]:
        """The top k pool members other than the query, ties by record_id."""

        query_index = self._index.get(query_record_id)
        if query_index is None:
            return set()
        scored: list[tuple[float, str]] = []
        for index, record_id in enumerate(self._ids):
            if index == query_index:
                continue
            score = score_pair(
                self._vectors[query_index], self._vectors[index], self._metric
            )
            scored.append((score, record_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return {record_id for _score, record_id in scored[:k]}


def _key_at_level(record: Record, level: int) -> str | None:
    keys = all_chunk_identity_keys(
        record,
        document_identity_confidence=None,
        duplicate_rank_within_document=0,
    )
    return keys.get(level)


def select_queries(
    v1_pool: NeighbourPool,
    v2_pool: NeighbourPool,
    *,
    comparison_level: "ComparisonLevelResult",
    v1_records_by_id: dict[str, Record],
    v2_records_by_id: dict[str, Record],
    max_queries: int,
    seed: int,
) -> tuple[list[tuple[str, str]], SampleInfo]:
    """Select paired v1/v2 record ids from the two pools' identity intersection."""

    v1_keyed: dict[str, str] = {}
    for record_id in v1_pool.record_ids:
        record = v1_records_by_id.get(record_id)
        if record is None:
            continue
        key = _key_at_level(record, comparison_level.level)
        if key is not None and key not in v1_keyed:
            v1_keyed[key] = record_id
    v2_keyed: dict[str, str] = {}
    for record_id in v2_pool.record_ids:
        record = v2_records_by_id.get(record_id)
        if record is None:
            continue
        key = _key_at_level(record, comparison_level.level)
        if key is not None and key not in v2_keyed:
            v2_keyed[key] = record_id

    shared = sorted(set(v1_keyed) & set(v2_keyed))
    if len(shared) > max_queries:
        sample = hash_threshold_sample(seed, max_queries, shared, v2_record_ids=None)
        shared = [key for key in shared if key in sample.v1_ids]
        info = to_sample_info(sample, side="v1")
    else:
        info = SampleInfo(
            v1_n=len(shared),
            v1_of=len(shared),
            v2_n=0,
            v2_of=0,
            seed=seed,
        )
    pairs = [(v1_keyed[key], v2_keyed[key]) for key in shared]
    return pairs, info


def _suspended_metric(compat: CompatibilityResult) -> Metric:
    reason_code = suspension_reason_code(compat.state)
    reason = reason_code.value if reason_code is not None else compat.state
    return make_metric(_DEFAULT_METRIC, value=None, threshold=None, reason=reason)


def compare_neighbours(
    v1: Snapshot,
    v2: Snapshot,
    *,
    compat: CompatibilityResult,
    v1_summary: EmbeddingSnapshotSummary,
    v2_summary: EmbeddingSnapshotSummary,
    comparison_level: "ComparisonLevelResult",
    cfg: Config,
) -> Metric:
    """Mean/median neighbour-set Jaccard overlap across paired queries."""

    if v1_summary.similarity_metric is None or v2_summary.similarity_metric is None:
        return make_metric(
            _DEFAULT_METRIC, value=None, threshold=None, reason="METRIC_UNKNOWN"
        )
    if "neighbours" in ANALYSES_SUSPENDED_BY_STATE[compat.state]:
        return _suspended_metric(compat)

    v1_pool = build_pool(
        v1.records, seed=cfg.analysis.seed, sample_size=cfg.analysis.sample_size
    )
    v2_pool = build_pool(
        v2.records, seed=cfg.analysis.seed, sample_size=cfg.analysis.sample_size
    )
    v1_records_by_id = {record.record_id: record for record in v1.records}
    v2_records_by_id = {record.record_id: record for record in v2.records}
    queries, query_sample = select_queries(
        v1_pool,
        v2_pool,
        comparison_level=comparison_level,
        v1_records_by_id=v1_records_by_id,
        v2_records_by_id=v2_records_by_id,
        max_queries=cfg.neighbour.max_queries,
        seed=cfg.analysis.seed,
    )
    if len(queries) < _MIN_QUERIES:
        return make_metric(
            _DEFAULT_METRIC,
            value=None,
            threshold=None,
            reason="INSUFFICIENT_DATA",
            sample=query_sample,
        )

    index_v1 = NeighbourSearchIndex(
        v1_pool, metric=v1_summary.similarity_metric  # type: ignore[arg-type]
    )
    index_v2 = NeighbourSearchIndex(
        v2_pool, metric=v2_summary.similarity_metric  # type: ignore[arg-type]
    )

    overlaps: list[float] = []
    for v1_id, v2_id in queries:
        neighbours_v1 = index_v1.top_k_other_than(v1_id, k=cfg.neighbour.k)
        neighbours_v2 = index_v2.top_k_other_than(v2_id, k=cfg.neighbour.k)
        overlaps.append(
            jaccard(frozenset(neighbours_v1), frozenset(neighbours_v2))
        )

    mean_jaccard = float(np.mean(overlaps))
    median_jaccard = float(np.median(overlaps))
    return make_metric(
        _DEFAULT_METRIC,
        value=mean_jaccard,
        threshold=None,
        change=mean_jaccard,
        sample=query_sample,
        details={
            "k": cfg.neighbour.k,
            "num_queries": len(queries),
            "pool_size_v1": len(v1_pool.record_ids),
            "pool_size_v2": len(v2_pool.record_ids),
            "median": median_jaccard,
        },
    )
