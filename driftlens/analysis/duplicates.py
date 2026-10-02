"""Duplicate grouping: exact (Step 29), near and semantic (Step 37).

The three stages share one shape: each takes the previous stage's unit list and
returns (groups, next_units), so exact -> near -> semantic chains uniformly.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from driftlens.analysis.health import Metric, SampleInfo, make_metric
from driftlens.analysis.sampling import hash_threshold_sample, to_sample_info
from driftlens.config.schema import DuplicatesCfg, TextCfg, Threshold
from driftlens.identity.hashing import sample_hash
from driftlens.schema.record import Record
from driftlens.schema.snapshot import Snapshot
from driftlens.text.shingles import jaccard, shingle_key, shingles
from driftlens.text.tokenizer import tokenize

# One past the maximum possible 256-bit hash, so an empty shingle set can never
# LSH-bucket with a real, non-empty-shingled unit.
_MINHASH_SENTINEL = 2 ** 256
_SHINGLE_N = 5


@dataclass
class DuplicateUnit:
    unit_id: str
    representative: Record
    members: list[Record]


@dataclass
class DuplicateGroup:
    representative_id: str
    member_ids: list[str]
    class_: Literal["exact", "near", "semantic"]


@dataclass
class ExactDuplicateResult:
    groups: list[DuplicateGroup]
    units: list[DuplicateUnit]
    redundant_count: int
    total_chunks: int
    duplicate_rate: float
    covered_fraction: float
    sample: SampleInfo | None


@dataclass
class ExtendedDuplicateResult(ExactDuplicateResult):
    near_groups: list[DuplicateGroup]
    semantic_groups: list[DuplicateGroup]
    near_units: list[DuplicateUnit]
    semantic_units: list[DuplicateUnit]
    redundant_near: int
    redundant_semantic: int
    duplicate_rate_exact: float
    duplicate_rate_near: float
    duplicate_rate_semantic: float
    duplicate_rate_total: float


def _grouping_key(record: Record) -> str | None:
    if record.text is not None:
        return record.normalized_text_hash
    digest = getattr(record, "_embedding_digest_hex", None)
    if digest is not None:
        return digest
    return None


def compute_exact_groups(
    records: Sequence[Record], *, cfg: DuplicatesCfg, seed: int
) -> ExactDuplicateResult:
    """Group records by their exact text/embedding hash (stage 1 only)."""
    total_chunks = len(records)

    sample: SampleInfo | None = None
    if total_chunks > cfg.max_exact_vectors:
        sampled = hash_threshold_sample(
            seed,
            cfg.max_exact_vectors,
            [record.record_id for record in records],
            v2_record_ids=None,
        )
        considered = [
            record for record in records if record.record_id in sampled.v1_ids
        ]
        covered_fraction = len(considered) / total_chunks
        sample = to_sample_info(sampled)
    else:
        considered = list(records)
        covered_fraction = 1.0

    buckets: dict[str, list[Record]] = defaultdict(list)
    for record in considered:
        key = _grouping_key(record)
        if key is not None:
            buckets[key].append(record)

    groups: list[DuplicateGroup] = []
    units: list[DuplicateUnit] = []
    for members in buckets.values():
        ordered = sorted(members, key=lambda record: record.record_id)
        representative = ordered[0]
        units.append(
            DuplicateUnit(
                unit_id=representative.record_id,
                representative=representative,
                members=ordered,
            )
        )
        if len(ordered) >= 2:
            groups.append(
                DuplicateGroup(
                    representative_id=representative.record_id,
                    member_ids=[record.record_id for record in ordered],
                    class_="exact",
                )
            )

    units.sort(key=lambda unit: unit.unit_id)
    groups.sort(key=lambda group: group.representative_id)

    redundant_count = sum(len(group.member_ids) - 1 for group in groups)
    duplicate_rate = redundant_count / total_chunks if total_chunks else 0.0

    return ExactDuplicateResult(
        groups=groups,
        units=units,
        redundant_count=redundant_count,
        total_chunks=total_chunks,
        duplicate_rate=duplicate_rate,
        covered_fraction=covered_fraction,
        sample=sample,
    )


def minhash_signature(
    shingle_set: frozenset[tuple[str, ...]],
    *,
    seed: int,
    permutations: int = 128,
) -> tuple[int, ...]:
    """MinHash signature under the configured number of hash functions.

    Each permutation index derives its own hash from the single seed via
    sample_hash (Step 11), so the signature inherits that primitive's exact
    determinism with no second hashing mechanism to validate.  An empty shingle
    set gets the maximal sentinel value for every permutation.
    """

    if not shingle_set:
        return tuple(_MINHASH_SENTINEL for _ in range(permutations))
    signature: list[int] = []
    for index in range(permutations):
        prefix = f"minhash:{index}:"
        signature.append(
            min(
                sample_hash(seed, prefix + shingle_key(shingle).hex())
                for shingle in shingle_set
            )
        )
    return tuple(signature)


def lsh_band_rows(threshold: float) -> int:
    """Band size: 4 rows when the threshold is at least 0.5, else 2."""
    return 4 if threshold >= 0.5 else 2


def lsh_candidate_pairs(
    signatures: dict[str, tuple[int, ...]], *, r: int
) -> set[frozenset[str]]:
    """Unit ids sharing a band bucket, as deduped candidate pairs."""
    if not signatures or r <= 0:
        return set()
    signature_length = len(next(iter(signatures.values())))
    n_bands = signature_length // r
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for unit_id, signature in signatures.items():
        for band in range(n_bands):
            start = band * r
            buckets[(band, tuple(signature[start : start + r]))].append(unit_id)

    candidates: set[frozenset[str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for left, right in itertools.combinations(members, 2):
            candidates.add(frozenset({left, right}))
    return candidates


def _build_adjacent_exclusion_set(
    units: Sequence[DuplicateUnit], cfg: DuplicatesCfg
) -> set[frozenset[str]]:
    """Unit pairs that share a document and are within one chunk index."""
    if not cfg.ignore_adjacent_chunks:
        return set()
    by_document: dict[str, list[DuplicateUnit]] = defaultdict(list)
    for unit in units:
        representative = unit.representative
        if (
            representative.document_id is not None
            and representative.chunk_index is not None
        ):
            by_document[representative.document_id].append(unit)
    adjacent: set[frozenset[str]] = set()
    for members in by_document.values():
        for left, right in itertools.combinations(members, 2):
            difference = abs(
                left.representative.chunk_index - right.representative.chunk_index
            )
            if difference <= 1:
                adjacent.add(frozenset({left.unit_id, right.unit_id}))
    return adjacent


def _merge_units(
    units_by_id: dict[str, DuplicateUnit], component: Sequence[str]
) -> DuplicateUnit:
    members: list[Record] = []
    representatives: list[Record] = []
    for unit_id in component:
        unit = units_by_id[unit_id]
        members.extend(unit.members)
        representatives.append(unit.representative)
    representative = min(representatives, key=lambda record: record.record_id)
    return DuplicateUnit(
        unit_id=representative.record_id,
        representative=representative,
        members=members,
    )


def _build_groups_and_units(
    units: Sequence[DuplicateUnit],
    verified: set[frozenset[str]],
    class_: Literal["near", "semantic"],
) -> tuple[list[DuplicateGroup], list[DuplicateUnit]]:
    """Turn verified unit pairs into groups plus the next stage's unit list."""
    units_by_id = {unit.unit_id: unit for unit in units}
    graph: dict[str, set[str]] = {unit_id: set() for unit_id in units_by_id}
    for pair in verified:
        left, right = tuple(pair)
        graph[left].add(right)
        graph[right].add(left)

    components: list[list[str]] = []
    seen: set[str] = set()
    for node in sorted(graph):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(graph[current]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        if len(component) >= 2:
            components.append(sorted(component))

    grouped_ids: set[str] = set()
    groups: list[DuplicateGroup] = []
    next_units: list[DuplicateUnit] = []
    for component in components:
        merged = _merge_units(units_by_id, component)
        next_units.append(merged)
        grouped_ids.update(component)
        groups.append(
            DuplicateGroup(
                representative_id=merged.representative.record_id,
                member_ids=list(component),
                class_=class_,
            )
        )
    for unit in units:
        if unit.unit_id not in grouped_ids:
            next_units.append(unit)

    groups.sort(key=lambda group: group.representative_id)
    next_units.sort(key=lambda unit: unit.unit_id)
    return groups, next_units


def compute_near_groups(
    units: Sequence[DuplicateUnit],
    *,
    cfg: DuplicatesCfg,
    threshold: float,
    seed: int,
    text_cfg: TextCfg | None = None,
) -> tuple[list[DuplicateGroup], list[DuplicateUnit]]:
    """MinHash/LSH candidates verified by exact Jaccard (stage 2)."""
    if text_cfg is None:
        text_cfg = TextCfg()
    units_list = list(units)
    adjacent = _build_adjacent_exclusion_set(units_list, cfg)

    shingle_sets: dict[str, frozenset[tuple[str, ...]]] = {}
    for unit in units_list:
        representative = unit.representative
        if representative.text is None:
            continue
        source = (
            representative.normalized_text
            if representative.normalized_text is not None
            else representative.text
        )
        shingle_sets[unit.unit_id] = shingles(
            tokenize(source), _SHINGLE_N, casefold=text_cfg.casefold
        )

    signatures = {
        unit_id: minhash_signature(
            shingle_set, seed=seed, permutations=cfg.minhash_permutations
        )
        for unit_id, shingle_set in shingle_sets.items()
    }
    candidates = lsh_candidate_pairs(signatures, r=lsh_band_rows(threshold))

    verified: set[frozenset[str]] = set()
    for pair in candidates:
        if pair in adjacent:
            continue
        left, right = tuple(pair)
        if jaccard(shingle_sets[left], shingle_sets[right]) >= threshold:
            verified.add(pair)

    return _build_groups_and_units(units_list, verified, "near")


def _normalized_vector(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return array / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def compute_semantic_groups(
    units: Sequence[DuplicateUnit],
    *,
    cfg: DuplicatesCfg,
    threshold: float,
    similarity_metric_known: bool,
    seed: int,
) -> tuple[list[DuplicateGroup], list[DuplicateUnit]]:
    """Exhaustive cosine on L2-normalized representative vectors (stage 3)."""
    units_list = list(units)
    if not similarity_metric_known:
        return [], units_list

    embed_units: list[DuplicateUnit] = []
    for unit in units_list:
        embedding = unit.representative.embedding
        if embedding.validity != "ok" or embedding.vector is None:
            continue
        if float(np.linalg.norm(np.asarray(embedding.vector, dtype=float))) == 0.0:
            continue
        embed_units.append(unit)

    if len(embed_units) > cfg.max_exact_vectors:
        sampled = hash_threshold_sample(
            seed,
            cfg.max_exact_vectors,
            [unit.unit_id for unit in embed_units],
            v2_record_ids=None,
        )
        embed_units = [
            unit for unit in embed_units if unit.unit_id in sampled.v1_ids
        ]

    adjacent = _build_adjacent_exclusion_set(embed_units, cfg)
    vectors = {
        unit.unit_id: _normalized_vector(unit.representative.embedding.vector)
        for unit in embed_units
    }

    verified: set[frozenset[str]] = set()
    for left, right in itertools.combinations(embed_units, 2):
        pair = frozenset({left.unit_id, right.unit_id})
        if pair in adjacent:
            continue
        if _cosine(vectors[left.unit_id], vectors[right.unit_id]) >= threshold:
            verified.add(pair)

    return _build_groups_and_units(units_list, verified, "semantic")


def compute_all_duplicate_stages(
    records: Sequence[Record],
    *,
    cfg: DuplicatesCfg,
    near_threshold: float,
    semantic_threshold: float,
    similarity_metric_known: bool,
    seed: int,
    text_cfg: TextCfg | None = None,
) -> ExtendedDuplicateResult:
    """Exact (Step 29) -> near -> semantic, chained over one unit list."""
    if text_cfg is None:
        text_cfg = TextCfg()
    exact_result = compute_exact_groups(records, cfg=cfg, seed=seed)
    near_groups, near_units = compute_near_groups(
        exact_result.units,
        cfg=cfg,
        threshold=near_threshold,
        seed=seed,
        text_cfg=text_cfg,
    )
    semantic_groups, semantic_units = compute_semantic_groups(
        near_units,
        cfg=cfg,
        threshold=semantic_threshold,
        similarity_metric_known=similarity_metric_known,
        seed=seed,
    )

    total = exact_result.total_chunks
    redundant_near = sum(len(group.member_ids) - 1 for group in near_groups)
    redundant_semantic = sum(len(group.member_ids) - 1 for group in semantic_groups)
    redundant_total = (
        exact_result.redundant_count + redundant_near + redundant_semantic
    )
    if total:
        duplicate_rate_exact = exact_result.redundant_count / total
        duplicate_rate_near = redundant_near / total
        duplicate_rate_semantic = redundant_semantic / total
        duplicate_rate_total = redundant_total / total
    else:
        duplicate_rate_exact = duplicate_rate_near = 0.0
        duplicate_rate_semantic = duplicate_rate_total = 0.0

    return ExtendedDuplicateResult(
        groups=exact_result.groups,
        units=exact_result.units,
        redundant_count=exact_result.redundant_count,
        total_chunks=total,
        duplicate_rate=duplicate_rate_total,
        covered_fraction=exact_result.covered_fraction,
        sample=exact_result.sample,
        near_groups=near_groups,
        semantic_groups=semantic_groups,
        near_units=near_units,
        semantic_units=semantic_units,
        redundant_near=redundant_near,
        redundant_semantic=redundant_semantic,
        duplicate_rate_exact=duplicate_rate_exact,
        duplicate_rate_near=duplicate_rate_near,
        duplicate_rate_semantic=duplicate_rate_semantic,
        duplicate_rate_total=duplicate_rate_total,
    )


def _count_metric(name: str, v1_value, v2_value):
    return make_metric(
        name,
        value=v2_value,
        threshold=None,
        v1=v1_value,
        v2=v2_value,
        change=v2_value - v1_value,
    )


def compare_duplicates(
    v1: Snapshot,
    v2: Snapshot,
    *,
    cfg: DuplicatesCfg,
    drift_cfg: dict[str, Threshold | None],
    seed: int = 42,
    text_cfg: TextCfg | None = None,
    near_threshold: float = 0.90,
    semantic_threshold: float = 0.98,
    similarity_metric_known: bool = False,
) -> tuple[ExtendedDuplicateResult, ExtendedDuplicateResult, list[Metric]]:
    """All three duplicate classes for both snapshots plus DUPLICATION metrics."""
    v1_result = compute_all_duplicate_stages(
        v1.records,
        cfg=cfg,
        near_threshold=near_threshold,
        semantic_threshold=semantic_threshold,
        similarity_metric_known=similarity_metric_known,
        seed=seed,
        text_cfg=text_cfg,
    )
    v2_result = compute_all_duplicate_stages(
        v2.records,
        cfg=cfg,
        near_threshold=near_threshold,
        semantic_threshold=semantic_threshold,
        similarity_metric_known=similarity_metric_known,
        seed=seed,
        text_cfg=text_cfg,
    )

    rate_change = v2_result.duplicate_rate_total - v1_result.duplicate_rate_total
    metrics: list[Metric] = [
        make_metric(
            "duplicate_rate",
            value=v2_result.duplicate_rate_total,
            threshold=drift_cfg.get("duplicate_rate"),
            v1=v1_result.duplicate_rate_total,
            v2=v2_result.duplicate_rate_total,
            change=rate_change,
        ),
        make_metric(
            "duplicate_rate_exact",
            value=v2_result.duplicate_rate_exact,
            threshold=drift_cfg.get("duplicate_rate_exact"),
            v1=v1_result.duplicate_rate_exact,
            v2=v2_result.duplicate_rate_exact,
            change=v2_result.duplicate_rate_exact - v1_result.duplicate_rate_exact,
        ),
        make_metric(
            "duplicate_rate_near",
            value=v2_result.duplicate_rate_near,
            threshold=drift_cfg.get("duplicate_rate_near"),
            v1=v1_result.duplicate_rate_near,
            v2=v2_result.duplicate_rate_near,
            change=v2_result.duplicate_rate_near - v1_result.duplicate_rate_near,
        ),
        make_metric(
            "duplicate_rate_semantic",
            value=v2_result.duplicate_rate_semantic,
            threshold=drift_cfg.get("duplicate_rate_semantic"),
            v1=v1_result.duplicate_rate_semantic,
            v2=v2_result.duplicate_rate_semantic,
            change=v2_result.duplicate_rate_semantic
            - v1_result.duplicate_rate_semantic,
        ),
        _count_metric(
            "duplicate_groups",
            len(v1_result.groups)
            + len(v1_result.near_groups)
            + len(v1_result.semantic_groups),
            len(v2_result.groups)
            + len(v2_result.near_groups)
            + len(v2_result.semantic_groups),
        ),
        _count_metric(
            "redundant_chunks",
            v1_result.redundant_count
            + v1_result.redundant_near
            + v1_result.redundant_semantic,
            v2_result.redundant_count
            + v2_result.redundant_near
            + v2_result.redundant_semantic,
        ),
        make_metric(
            "duplicate_covered_fraction",
            value=v2_result.covered_fraction,
            threshold=None,
            v1=v1_result.covered_fraction,
            v2=v2_result.covered_fraction,
            change=v2_result.covered_fraction - v1_result.covered_fraction,
        ),
    ]
    return v1_result, v2_result, metrics
