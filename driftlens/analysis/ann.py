"""Optional FAISS-backed approximate nearest-neighbour search (Step 50).

ANN is an opt-in acceleration for the semantic-duplicate stage and dense
retrieval: it is only used when a caller explicitly asks for it and the pool
exceeds the configured exact-search threshold.  ``faiss`` is imported lazily
so this module stays importable when the optional ``[faiss]`` extra is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Fixed over-fetch constant for semantic-duplicate candidates: ANN hits are
# approximate, so a small margin above the exact neighbour count is searched.
SEMANTIC_ANN_TOP_K = 50


@dataclass
class AnnBuildParams:
    """Deterministic-build parameters recorded alongside an ANN result."""

    index_type: str
    nlist: int | None
    nprobe: int | None
    seed: int


@dataclass
class AnnIndex:
    """A trained FAISS IVF index plus the parameters that produced it."""

    build_params: AnnBuildParams
    metric: Literal["cosine", "dot", "l2"]
    _index: object

    def search(
        self, query_vectors: np.ndarray, *, top_k: int
    ) -> list[list[tuple[int, float]]]:
        """Return up to ``top_k`` (row index, score) hits per query row.

        The result is APPROXIMATE and a row may legitimately contain fewer
        than ``top_k`` hits; the FAISS ``-1`` no-result sentinel is filtered
        out so every returned index is a valid row into the indexed array.
        """
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if self.metric == "cosine":
            queries = _l2_normalize(queries)
        scores, indices = self._index.search(queries, max(1, int(top_k)))
        results: list[list[tuple[int, float]]] = []
        for row_scores, row_indices in zip(scores, indices):
            hits: list[tuple[int, float]] = []
            for score, index in zip(row_scores, row_indices):
                if index < 0:
                    continue
                hits.append((int(index), float(score)))
            results.append(hits)
        return results


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def ann_available() -> bool:
    """True when the optional ``faiss`` extra can be imported."""
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def build_ann_index(
    vectors: np.ndarray,
    *,
    metric: Literal["cosine", "dot", "l2"],
    seed: int,
) -> AnnIndex:
    """Train a single-threaded FAISS IVF index over ``vectors``.

    ``metric="cosine"`` normalizes the inputs first and scores with an inner
    product; ``"dot"`` and ``"l2"`` map directly onto FAISS inner-product and
    L2 metrics.  Thread state is restored in a ``finally`` block so this call
    has no process-wide side effect.
    """
    import faiss

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if metric == "cosine":
        matrix = _l2_normalize(matrix)
    dimension = int(matrix.shape[1])
    nlist = max(1, int(math.sqrt(len(matrix))))

    if metric == "l2":
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    else:
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
        )

    prior_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        np.random.seed(seed)
        seed_fn = getattr(faiss, "seed", None)
        if callable(seed_fn):
            seed_fn(seed)
        index.train(matrix)
        index.add(matrix)
    finally:
        faiss.omp_set_num_threads(prior_threads)

    nprobe = min(nlist, max(1, nlist // 8))
    index.nprobe = nprobe
    params = AnnBuildParams(
        index_type="IVFFlat", nlist=nlist, nprobe=nprobe, seed=seed
    )
    return AnnIndex(build_params=params, metric=metric, _index=index)
