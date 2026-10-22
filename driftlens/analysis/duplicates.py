"""Duplicate grouping: exact (Step 29), near and semantic (Step 37).

The three stages share one shape: each takes the previous stage's unit list and
returns (groups, next_units), so exact -> near -> semantic chains uniformly.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from driftlens.analysis.ann import (
    SEMANTIC_ANN_TOP_K,
    AnnBuildParams,
    ann_available,
    build_ann_index,
)
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
    near_sample: SampleInfo | None = None


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
    # P0: the shingle key encoding is permutation-independent; compute it
    # once per shingle instead of once per (permutation, shingle) pair. The
    # hash inputs are identical to the previous per-permutation form, so
    # signatures are byte-identical.
    key_hexes = [shingle_key(shingle).hex() for shingle in shingle_set]
    signature: list[int] = []
    for index in range(permutations):
        prefix = f"minhash:{index}:"
        signature.append(
            min(sample_hash(seed, prefix + key_hex) for key_hex in key_hexes)
        )
    return tuple(signature)


_UNIVERSAL_MASK = (1 << 64) - 1


def _universal_hash_params(seed: int, permutations: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic (a_i, b_i) pairs for the universal-hash MinHash family.

    Each permutation draws its multipliers from sample_hash under the same
    seed discipline, so signatures remain reproducible across runs and hosts.
    """

    a_list: list[int] = []
    b_list: list[int] = []
    for index in range(permutations):
        material = f"minhash:{index}:"
        draw = sample_hash(seed, material)
        a_list.append((draw >> 128) & _UNIVERSAL_MASK | 1)
        b_list.append(draw & _UNIVERSAL_MASK)
    return (
        np.asarray(a_list, dtype=np.uint64),
        np.asarray(b_list, dtype=np.uint64),
    )


def minhash_signatures_vectorized(
    key_hex_lists: list[list[str]],
    *,
    seed: int,
    permutations: int = 128,
) -> list[tuple[int, ...]]:
    """Universal-hash MinHash signatures for many units in batched form.

    h_i(x) = (a_i * h0(x) + b_i) mod 2^64 with h0 = sample_hash(seed, key)
    per shingle. One sha256 per shingle plus numpy broadcasting replaces
    one sha256 per (permutation, shingle). Preserves the MinHash property:
    each h_i is a 2-universal hash, so Pr[h_i(x) == h_i(y)] == Jaccard(x, y).
    """

    if not key_hex_lists:
        return []
    a, b = _universal_hash_params(seed, permutations)
    out: list[tuple[int, ...]] = []
    for key_hexes in key_hex_lists:
        if not key_hexes:
            out.append(tuple(_MINHASH_SENTINEL for _ in range(permutations)))
            continue
        base = np.asarray(
            [sample_hash(seed, key_hex) & _UNIVERSAL_MASK for key_hex in key_hexes],
            dtype=np.uint64,
        )
        # (permutations, shingles): h_i = (a_i * h0 + b_i) mod 2^64
        signature = ((base[None, :] * a[:, None] + b[:, None]) & _UNIVERSAL_MASK).min(axis=1)
        out.append(tuple(int(value) for value in signature))
    return out


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

    considered_units = units_list
    sample = None
    effective_permutations = cfg.minhash_permutations
    if cfg.max_units is not None and len(units_list) > cfg.max_units:
        # P3 (revised): unit subsampling breaks pairwise duplicate detection
        # (both endpoints of a pair must be retained), so large pools get an
        # adaptive permutation cap instead of a unit sample. 64 permutations
        # was A/B-verified metric-identical on the benchmark corpus; the
        # reduction is reported as a sample annotation for transparency.
        effective_permutations = min(cfg.minhash_permutations, 64)
        sample = SampleInfo(
            v1_n=effective_permutations,
            v1_of=cfg.minhash_permutations,
            v2_n=0,
            v2_of=0,
            seed=seed,
        )

    shingle_sets: dict[str, frozenset[tuple[str, ...]]] = {}
    for unit in considered_units:
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

    if getattr(cfg, "minhash_mode", "universal") == "universal":
        key_hex_lists = [
            [shingle_key(shingle).hex() for shingle in shingle_set]
            for shingle_set in shingle_sets.values()
        ]
        signature_values = minhash_signatures_vectorized(
            key_hex_lists, seed=seed, permutations=effective_permutations
        )
        signatures = dict(zip(shingle_sets.keys(), signature_values))
    else:
        signatures = {
            unit_id: minhash_signature(
                shingle_set, seed=seed, permutations=effective_permutations
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

    groups, next_units = _build_groups_and_units(units_list, verified, "near")
    return groups, next_units, sample


def _normalized_vector(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return array / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def _semantic_groups_ann(
    units_list: list[DuplicateUnit],
    embed_units: list[DuplicateUnit],
    *,
    cfg: DuplicatesCfg,
    threshold: float,
    seed: int,
    ann_params_out: list[AnnBuildParams] | None,
) -> tuple[list[DuplicateGroup], list[DuplicateUnit]]:
    """Opt-in ANN semantic grouping over every eligible representative."""
    adjacent = _build_adjacent_exclusion_set(embed_units, cfg)
    matrix = np.asarray(
        [
            _normalized_vector(unit.representative.embedding.vector)
            for unit in embed_units
        ],
        dtype=float,
    )
    index = build_ann_index(matrix, metric="cosine", seed=seed)
    if ann_params_out is not None:
        ann_params_out.append(index.build_params)

    verified: set[frozenset[str]] = set()
    for position, hits in enumerate(
        index.search(matrix, top_k=SEMANTIC_ANN_TOP_K)
    ):
        left = embed_units[position]
        for neighbour, score in hits:
            if neighbour == position:
                continue
            right = embed_units[neighbour]
            pair = frozenset({left.unit_id, right.unit_id})
            if pair in adjacent:
                continue
            if score >= threshold:
                verified.add(pair)
    return _build_groups_and_units(units_list, verified, "semantic")


def compute_semantic_groups(
    units: Sequence[DuplicateUnit],
    *,
    cfg: DuplicatesCfg,
    threshold: float,
    similarity_metric_known: bool,
    seed: int,
    use_ann: bool = False,
    ann_params_out: list[AnnBuildParams] | None = None,
) -> tuple[list[DuplicateGroup], list[DuplicateUnit]]:
    """Cosine on L2-normalized representative vectors (stage 3).

    The default path is exhaustive; ``use_ann`` opts into a single-threaded
    FAISS IVF index over every eligible representative once the pool reaches
    ``cfg.max_exact_vectors`` (that path does not sample).  The exact stage
    upstream already caps its own pool at that same threshold, so the chained
    pipeline reaches equality rather than strictly exceeding it.
    """
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

    if (
        use_ann
        and embed_units
        and len(embed_units) >= cfg.max_exact_vectors
        and ann_available()
    ):
        return _semantic_groups_ann(
            units_list,
            embed_units,
            cfg=cfg,
            threshold=threshold,
            seed=seed,
            ann_params_out=ann_params_out,
        )

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
    use_ann: bool = False,
    ann_params_out: list[AnnBuildParams] | None = None,
) -> ExtendedDuplicateResult:
    """Exact (Step 29) -> near -> semantic, chained over one unit list."""
    if text_cfg is None:
        text_cfg = TextCfg()
    exact_result = compute_exact_groups(records, cfg=cfg, seed=seed)
    near_groups, near_units, near_sample = compute_near_groups(
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
        use_ann=use_ann,
        ann_params_out=ann_params_out,
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
        near_sample=near_sample,
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
    use_ann: bool = False,
) -> tuple[ExtendedDuplicateResult, ExtendedDuplicateResult, list[Metric]]:
    """All three duplicate classes for both snapshots plus DUPLICATION metrics."""
    ann_params_v1: list[AnnBuildParams] = []
    ann_params_v2: list[AnnBuildParams] = []
    v1_result = compute_all_duplicate_stages(
        v1.records,
        cfg=cfg,
        near_threshold=near_threshold,
        semantic_threshold=semantic_threshold,
        similarity_metric_known=similarity_metric_known,
        seed=seed,
        text_cfg=text_cfg,
        use_ann=use_ann,
        ann_params_out=ann_params_v1,
    )
    v2_result = compute_all_duplicate_stages(
        v2.records,
        cfg=cfg,
        near_threshold=near_threshold,
        semantic_threshold=semantic_threshold,
        similarity_metric_known=similarity_metric_known,
        seed=seed,
        text_cfg=text_cfg,
        use_ann=use_ann,
        ann_params_out=ann_params_v2,
    )

    ann_params = ann_params_v2 or ann_params_v1
    ann_details = {"ann": asdict(ann_params[0])} if ann_params else None

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
            sample=v2_result.near_sample,
            reason="SAMPLED_DUPLICATES" if v2_result.near_sample is not None else None,
        ),
        make_metric(
            "duplicate_rate_semantic",
            value=v2_result.duplicate_rate_semantic,
            threshold=drift_cfg.get("duplicate_rate_semantic"),
            v1=v1_result.duplicate_rate_semantic,
            v2=v2_result.duplicate_rate_semantic,
            change=v2_result.duplicate_rate_semantic
            - v1_result.duplicate_rate_semantic,
            details=ann_details,
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
