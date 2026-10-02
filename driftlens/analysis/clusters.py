"""HDBSCAN cluster analysis (Step 47, section 9)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from driftlens.analysis.compatibility import (
    ANALYSES_SUSPENDED_BY_STATE,
    CompatibilityResult,
    suspension_reason_code,
)
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.analysis.health import Metric, make_metric, round9
from driftlens.analysis.sampling import hash_threshold_sample
from driftlens.config.schema import ClusterCfg, Config
from driftlens.errors import Code, DiagnosticSink
from driftlens.schema.snapshot import Snapshot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sklearn.decomposition import PCA

_DIMENSION = "semantic"
_AUTO_MIN_CLUSTER_SIZE_FLOOR = 15
_AUTO_MIN_CLUSTER_SIZE_SHARE = 0.001

TransitionKind = Literal["STABLE", "SHIFTED", "SPLIT", "MERGED", "NEW", "DISAPPEARED"]
SimilarityKind = Literal["jaccard", "centroid_cosine"]


@dataclass
class ClusterAssignment:
    record_id: str
    label: int  # -1 = noise


@dataclass
class ClusteringResult:
    assignments: list[ClusterAssignment]
    noise_fraction: float
    min_cluster_size: int


@dataclass
class ClusterTransition:
    kind: TransitionKind
    v1_labels: list[int]
    v2_labels: list[int]
    similarity: float | None
    similarity_kind: SimilarityKind | None


def fit_pca(pooled_vectors: np.ndarray, *, n_components: int) -> "PCA":
    """Fit one shared full-SVD PCA on the pooled sample of both snapshots."""
    from sklearn.decomposition import PCA

    pooled = np.asarray(pooled_vectors, dtype=np.float64)
    components = min(n_components, pooled.shape[1], pooled.shape[0] - 1)
    pca = PCA(n_components=components, svd_solver="full")
    pca.fit(pooled)
    return pca


def resolve_min_cluster_size(cfg: ClusterCfg, n: int) -> int:
    """Auto min_cluster_size = max(15, round(0.001 * n)); integers override."""
    if cfg.min_cluster_size == "auto":
        return max(
            _AUTO_MIN_CLUSTER_SIZE_FLOOR,
            round(_AUTO_MIN_CLUSTER_SIZE_SHARE * n),
        )
    return int(cfg.min_cluster_size)


def _canonical_labels(
    labels: np.ndarray, record_ids: Sequence[str]
) -> dict[int, int]:
    members: dict[int, list[str]] = {}
    for record_id, label in zip(record_ids, labels):
        if int(label) == -1:
            continue
        members.setdefault(int(label), []).append(record_id)
    ordered = sorted(members.items(), key=lambda pair: (-len(pair[1]), min(pair[1])))
    return {old: new for new, (old, _members) in enumerate(ordered)}


def run_hdbscan(
    vectors: np.ndarray, record_ids: Sequence[str], *, cfg: ClusterCfg
) -> ClusteringResult:
    """Cluster one snapshot in record_id order with canonicalized labels."""
    import hdbscan

    first = np.asarray(vectors, dtype=np.float64)
    order = sorted(range(len(record_ids)), key=lambda index: record_ids[index])
    sorted_ids = [record_ids[index] for index in order]
    n = len(sorted_ids)
    min_cluster_size = resolve_min_cluster_size(cfg, n)
    if n == 0:
        return ClusteringResult([], 0.0, min_cluster_size)

    sorted_vectors = first[order]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_cluster_size,
        cluster_selection_method="eom",
        allow_single_cluster=False,
    )
    labels = np.asarray(clusterer.fit_predict(sorted_vectors))
    noise_fraction = float(np.sum(labels == -1)) / n
    mapping = _canonical_labels(labels, sorted_ids)
    assignments = [
        ClusterAssignment(
            record_id=record_id,
            label=(-1 if int(label) == -1 else mapping[int(label)]),
        )
        for record_id, label in zip(sorted_ids, labels)
    ]
    return ClusteringResult(assignments, noise_fraction, min_cluster_size)


def cluster_centroid(
    vectors: np.ndarray,
    assignment: ClusteringResult,
    label: int,
    *,
    l2_normalize_first: bool,
) -> np.ndarray:
    """Mean of the member vectors (optionally L2-normalized first)."""
    first = np.asarray(vectors, dtype=np.float64)
    members = [
        first[index]
        for index, item in enumerate(assignment.assignments)
        if item.label == label
    ]
    if not members:
        width = first.shape[1] if first.ndim == 2 else 0
        return np.zeros(width, dtype=np.float64)
    stacked = np.vstack(members)
    if l2_normalize_first:
        norms = np.linalg.norm(stacked, axis=1)
        safe = np.where(norms == 0.0, 1.0, norms)
        stacked = stacked / safe[:, None]
    return np.mean(stacked, axis=0)


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    norm_first = float(np.linalg.norm(first))
    norm_second = float(np.linalg.norm(second))
    if norm_first == 0.0 or norm_second == 0.0:
        return 0.0
    return float(np.dot(first, second) / (norm_first * norm_second))


def _members_by_label(result: ClusteringResult) -> dict[int, set[str]]:
    members: dict[int, set[str]] = {}
    for item in result.assignments:
        if item.label == -1:
            continue
        members.setdefault(item.label, set()).add(item.record_id)
    return members


def _ordered_vectors(
    vectors_by_id: dict[str, np.ndarray], result: ClusteringResult
) -> np.ndarray:
    rows = [
        np.asarray(vectors_by_id[item.record_id], dtype=np.float64)
        for item in result.assignments
    ]
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(rows)


def classify_cluster_transitions(
    v1_result: ClusteringResult,
    v2_result: ClusteringResult,
    v1_vectors_by_id: dict[str, np.ndarray],
    v2_vectors_by_id: dict[str, np.ndarray],
    *,
    cfg: ClusterCfg,
    cosine_space: bool,
) -> list[ClusterTransition]:
    """Four-phase cascade over shrinking unclassified cluster pools."""
    v1_members = _members_by_label(v1_result)
    v2_members = _members_by_label(v2_result)
    v1_labels = sorted(v1_members)
    v2_labels = sorted(v2_members)
    v1_sizes = {label: len(v1_members[label]) for label in v1_labels}
    v2_sizes = {label: len(v2_members[label]) for label in v2_labels}
    v1_vectors = _ordered_vectors(v1_vectors_by_id, v1_result)
    v2_vectors = _ordered_vectors(v2_vectors_by_id, v2_result)

    # Phase A - similarity matrix and Hungarian assignment.
    similarity = np.zeros((len(v1_labels), len(v2_labels)), dtype=np.float64)
    kinds: list[list[SimilarityKind | None]] = [
        [None for _ in v2_labels] for _ in v1_labels
    ]
    for row, label1 in enumerate(v1_labels):
        for column, label2 in enumerate(v2_labels):
            common = len(v1_members[label1] & v2_members[label2])
            smallest = min(v1_sizes[label1], v2_sizes[label2])
            if smallest > 0 and common >= cfg.min_common * smallest:
                union = v1_sizes[label1] + v2_sizes[label2] - common
                similarity[row, column] = common / union if union else 0.0
                kinds[row][column] = "jaccard"
            else:
                centroid1 = cluster_centroid(
                    v1_vectors,
                    v1_result,
                    label1,
                    l2_normalize_first=cosine_space,
                )
                centroid2 = cluster_centroid(
                    v2_vectors,
                    v2_result,
                    label2,
                    l2_normalize_first=cosine_space,
                )
                cosine = _cosine(centroid1, centroid2)
                if cosine >= cfg.centroid_cosine:
                    similarity[row, column] = cosine
                    kinds[row][column] = "centroid_cosine"

    matched: list[tuple[int, int]] = []
    if v1_labels and v2_labels:
        cost = -similarity.copy()
        rows, columns = np.indices(cost.shape)
        scale = 1e-9 / (cost.size + 1)
        cost = cost + scale * (rows * cost.shape[1] + columns + 1)
        row_indices, column_indices = linear_sum_assignment(cost)
        matched = [
            (int(row), int(column))
            for row, column in zip(row_indices, column_indices)
            if similarity[row, column] > 0.0
        ]

    transitions: list[ClusterTransition] = []
    consumed_v1: set[int] = set()
    consumed_v2: set[int] = set()

    # Phase B - stability classification of the assigned pairs.
    for row, column in matched:
        label1 = v1_labels[row]
        label2 = v2_labels[column]
        size_ratio = v2_sizes[label2] / v1_sizes[label1]
        kind = kinds[row][column]
        if not (cfg.size_ratio_min <= size_ratio <= cfg.size_ratio_max):
            transition = "SHIFTED"
        elif kind == "jaccard" and similarity[row, column] >= cfg.stable_jaccard:
            transition = "STABLE"
        elif (
            kind == "centroid_cosine"
            and similarity[row, column] >= cfg.stable_centroid_cosine
        ):
            transition = "STABLE"
        else:
            transition = "SHIFTED"
        transitions.append(
            ClusterTransition(
                kind=transition,  # type: ignore[arg-type]
                v1_labels=[label1],
                v2_labels=[label2],
                similarity=round9(float(similarity[row, column])),
                similarity_kind=kind,
            )
        )
        consumed_v1.add(label1)
        consumed_v2.add(label2)

    # Phase C - split and merge on the remaining unmatched clusters.
    for label1 in v1_labels:
        if label1 in consumed_v1:
            continue
        holders = [
            label2
            for label2 in v2_labels
            if label2 not in consumed_v2
            and len(v1_members[label1] & v2_members[label2]) / v1_sizes[label1]
            >= cfg.split_share
        ]
        if len(holders) >= 2:
            transitions.append(
                ClusterTransition("SPLIT", [label1], holders, None, None)
            )
            consumed_v1.add(label1)
            consumed_v2.update(holders)

    for label2 in v2_labels:
        if label2 in consumed_v2:
            continue
        holders = [
            label1
            for label1 in v1_labels
            if label1 not in consumed_v1
            and len(v2_members[label2] & v1_members[label1]) / v2_sizes[label2]
            >= cfg.split_share
        ]
        if len(holders) >= 2:
            transitions.append(
                ClusterTransition("MERGED", holders, [label2], None, None)
            )
            consumed_v2.add(label2)
            consumed_v1.update(holders)

    # Phase D - remainder.
    for label1 in v1_labels:
        if label1 not in consumed_v1:
            transitions.append(
                ClusterTransition("DISAPPEARED", [label1], [], None, None)
            )
    for label2 in v2_labels:
        if label2 not in consumed_v2:
            transitions.append(ClusterTransition("NEW", [], [label2], None, None))

    return transitions


def _valid_vectors(snapshot: Snapshot) -> tuple[np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    for record in snapshot.records:
        if record.embedding.validity != "ok" or record.embedding.vector is None:
            continue
        vector = np.asarray(record.embedding.vector, dtype=np.float64)
        if float(np.linalg.norm(vector)) == 0.0:
            continue
        vectors.append(vector)
        ids.append(record.record_id)
    if not vectors:
        return np.empty((0, 0), dtype=np.float64), []
    return np.vstack(vectors), ids


def _subset(
    vectors: np.ndarray, ids: Sequence[str], kept: frozenset[str]
) -> tuple[np.ndarray, list[str]]:
    width = vectors.shape[1] if vectors.ndim == 2 else 0
    rows = [vectors[index] for index, record_id in enumerate(ids) if record_id in kept]
    kept_ids = [record_id for record_id in ids if record_id in kept]
    if not rows:
        return np.empty((0, width), dtype=np.float64), []
    return np.asarray(rows, dtype=np.float64), kept_ids


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    safe = np.where(norms == 0.0, 1.0, norms)
    return vectors / safe[:, None]


def _skipped(threshold, reason: str) -> Metric:
    return make_metric(_DIMENSION, value=None, threshold=threshold, reason=reason)


def compare_clusters(
    v1: Snapshot,
    v2: Snapshot,
    *,
    compat: CompatibilityResult,
    v1_summary: EmbeddingSnapshotSummary,
    v2_summary: EmbeddingSnapshotSummary,
    cfg: Config,
    sink: DiagnosticSink,
) -> list[Metric]:
    """The SEMANTIC dimension's metric list (section 9)."""
    threshold = cfg.drift["semantic"]

    if cfg.cluster.mode == "off":
        return []

    if cfg.cluster.mode != "on":
        try:
            import hdbscan  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:
            sink.add(
                Code.CLUSTER_EXTRA_MISSING,
                "cluster analysis requires the [cluster] extra",
                dimension=_DIMENSION,
            )
            return [_skipped(threshold, Code.CLUSTER_EXTRA_MISSING.value)]

    if "clusters" in ANALYSES_SUSPENDED_BY_STATE[compat.state]:
        code = suspension_reason_code(compat.state)
        reason = code.value if code is not None else compat.state
        if code is not None:
            sink.add(
                code,
                "cluster analysis suspended by compatibility state",
                dimension=_DIMENSION,
            )
        return [_skipped(threshold, reason)]

    cosine_space = (
        v1_summary.similarity_metric == "cosine"
        and v2_summary.similarity_metric == "cosine"
    )
    v1_vectors, v1_ids = _valid_vectors(v1)
    v2_vectors, v2_ids = _valid_vectors(v2)
    sample = hash_threshold_sample(
        cfg.analysis.seed, cfg.analysis.sample_size, v1_ids, v2_ids
    )
    v1_vectors, v1_ids = _subset(v1_vectors, v1_ids, sample.v1_ids)
    v2_vectors, v2_ids = _subset(v2_vectors, v2_ids, sample.v2_ids)

    for vectors in (v1_vectors, v2_vectors):
        count = int(vectors.shape[0])
        if count < 2 * resolve_min_cluster_size(cfg.cluster, count):
            return [_skipped(threshold, "INSUFFICIENT_DATA")]

    if cosine_space:
        v1_vectors = _normalize_rows(v1_vectors)
        v2_vectors = _normalize_rows(v2_vectors)

    pooled = np.vstack([v1_vectors, v2_vectors])
    pca = fit_pca(pooled, n_components=cfg.cluster.pca_dims)
    v1_reduced = pca.transform(v1_vectors)
    v2_reduced = pca.transform(v2_vectors)

    v1_result = run_hdbscan(v1_reduced, v1_ids, cfg=cfg.cluster)
    v2_result = run_hdbscan(v2_reduced, v2_ids, cfg=cfg.cluster)

    v1_nonnoise = sorted(
        {item.label for item in v1_result.assignments if item.label != -1}
    )
    v2_nonnoise = sorted(
        {item.label for item in v2_result.assignments if item.label != -1}
    )
    if not v1_nonnoise or not v2_nonnoise:
        return [_skipped(threshold, "N/A")]

    v1_by_id = {record_id: v1_reduced[index] for index, record_id in enumerate(v1_ids)}
    v2_by_id = {record_id: v2_reduced[index] for index, record_id in enumerate(v2_ids)}
    transitions = classify_cluster_transitions(
        v1_result,
        v2_result,
        v1_by_id,
        v2_by_id,
        cfg=cfg.cluster,
        cosine_space=cosine_space,
    )

    stable_v1 = {
        label
        for transition in transitions
        if transition.kind == "STABLE"
        for label in transition.v1_labels
    }
    rate = sum(1 for label in v1_nonnoise if label not in stable_v1) / len(v1_nonnoise)
    return [
        make_metric(
            _DIMENSION,
            value=round9(rate),
            threshold=threshold,
            details={
                "noise_fraction_v1": v1_result.noise_fraction,
                "noise_fraction_v2": v2_result.noise_fraction,
                "transitions": [asdict(transition) for transition in transitions],
            },
        )
    ]
