"""Split/merge-aware chunk matching (Step 43, section 5.3).

This module implements passes 2-4 of the five-pass chunk comparison
described in section 5.3:

1. identity and hash matches -- already performed by
   driftlens.analysis.content.classify_chunks (Step 25);
2. 1:1 similarity matches -- similarity_1to1_pass;
3. split matches (one v1 chunk, 2+ v2 pieces) -- split_pass;
4. merge matches (2+ v1 chunks, one v2 chunk) -- merge_pass;
5. whatever is left is ADDED/REMOVED.

match_chunks_full orchestrates passes 2-5 per shared document, starting from
Step 25's ChunkDiffResult.unmatched_v1/.unmatched_v2 extension point.  The
preliminary SplitMergeGroup shape sketched in the plan is superseded by the
concrete SplitGroup and MergeGroup below.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from driftlens.analysis.content import (
    ChunkDiffResult,
    ChunkStatus,
    DocumentDiffRow,
    DocumentStatus,
)
from driftlens.config.schema import ChunkMatchCfg, TextCfg
from driftlens.schema.record import Record
from driftlens.text.shingles import coverage, jaccard, shingles
from driftlens.text.tokenizer import tokenize

_SHINGLE_N = 4
_INF = float("inf")
_CONTENT_EQUIVALENT: frozenset[str] = frozenset({"UNCHANGED", "FORMAT_ONLY_CHANGE"})
_SHARED_STATUSES: frozenset[str] = frozenset(
    {"UNCHANGED", "FORMAT_ONLY_CHANGE", "REPLACED", "MODIFIED"}
)


@dataclass
class SplitGroup:
    """One v1 chunk split into two or more v2 pieces (section 5.3 pass 3)."""

    v1_chunk_id: str
    v2_chunk_ids: list[str]


@dataclass
class MergeGroup:
    """Two or more v1 chunks merged into one v2 chunk (section 5.3 pass 4)."""

    v2_chunk_id: str
    v1_chunk_ids: list[str]


@dataclass
class FullChunkMatchResult:
    """The aggregated outcome of all five comparison passes."""

    pairs: list[tuple[Record | None, Record | None, ChunkStatus]] = field(
        default_factory=list
    )
    splits: list[SplitGroup] = field(default_factory=list)
    merges: list[MergeGroup] = field(default_factory=list)
    final_added: list[Record] = field(default_factory=list)
    final_removed: list[Record] = field(default_factory=list)
    rechunked_document_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ChunkMatchOutcome:
    """Group-aware match outcome consumed by ground-truth resolution (11.4)."""

    kind: Literal["one_to_one", "split_part", "merge_part"]
    resolved_id: str | None


def _index_key(chunk_index: int | None) -> float:
    return _INF if chunk_index is None else float(chunk_index)


def _shingle_map(
    records: Sequence[Record], *, casefold: bool
) -> dict[str, frozenset[tuple[str, ...]]]:
    """Shingle sets for every text-bearing record, keyed by record id."""

    result: dict[str, frozenset[tuple[str, ...]]] = {}
    for record in records:
        if record.text is None:
            continue
        text = record.normalized_text
        if text is None:
            text = record.text
        result[record.record_id] = shingles(
            tokenize(text), _SHINGLE_N, casefold=casefold
        )
    return result


def similarity_1to1_pass(
    v1_pool: Sequence[Record],
    v2_pool: Sequence[Record],
    *,
    cfg: ChunkMatchCfg,
    casefold: bool,
) -> tuple[list[tuple[Record, Record]], set[str], set[str]]:
    """Greedy 1:1 similarity matching by descending Jaccard (pass 2).

    Returns the matched pairs plus the matched v1 and v2 record ids.
    Text-less chunks never participate.
    """

    shingle_v1 = _shingle_map(v1_pool, casefold=casefold)
    shingle_v2 = _shingle_map(v2_pool, casefold=casefold)

    scored: list[tuple[float, Record, Record]] = []
    for a in v1_pool:
        set_a = shingle_v1.get(a.record_id)
        if not set_a:
            continue
        for b in v2_pool:
            set_b = shingle_v2.get(b.record_id)
            if not set_b:
                continue
            scored.append((jaccard(set_a, set_b), a, b))

    # Descending J, then ascending chunk_index (None -> inf) then record id.
    scored.sort(
        key=lambda item: (
            -item[0],
            _index_key(item[1].chunk_index),
            _index_key(item[2].chunk_index),
            item[1].record_id,
            item[2].record_id,
        )
    )

    pairs: list[tuple[Record, Record]] = []
    matched_v1: set[str] = set()
    matched_v2: set[str] = set()
    for similarity, a, b in scored:
        if similarity < cfg.min_similarity:
            break
        if a.record_id in matched_v1 or b.record_id in matched_v2:
            continue
        matched_v1.add(a.record_id)
        matched_v2.add(b.record_id)
        pairs.append((a, b))
    return pairs, matched_v1, matched_v2


def _candidate_pieces(
    target: frozenset[tuple[str, ...]],
    pool: Sequence[Record],
    shingle_map: dict[str, frozenset[tuple[str, ...]]],
    *,
    cfg: ChunkMatchCfg,
) -> list[tuple[Record, frozenset[tuple[str, ...]]]]:
    """Pool entries sharing a shingle with target and contributing >= 10%."""

    candidates: list[tuple[Record, frozenset[tuple[str, ...]]]] = []
    for record in pool:
        piece = shingle_map.get(record.record_id)
        if not piece:
            continue
        shared = piece & target
        if not shared:
            continue
        if len(shared) / len(piece) < cfg.split_contribution:
            continue
        candidates.append((record, piece))
    return candidates


def _greedy_cover(
    target: frozenset[tuple[str, ...]],
    candidates: list[tuple[Record, frozenset[tuple[str, ...]]]],
    *,
    cfg: ChunkMatchCfg,
) -> list[Record] | None:
    """Greedily add the best-covering piece until coverage is sufficient.

    Returns the selected records in selection order, or None when no valid
    group of at least two members reaches cfg.coverage of target.
    """

    selected: list[Record] = []
    covered: frozenset[tuple[str, ...]] = frozenset()
    remaining = list(candidates)
    while remaining and not (
        len(selected) >= 2 and coverage(target, covered) >= cfg.coverage
    ):
        best: tuple[Record, frozenset[tuple[str, ...]]] | None = None
        best_key: tuple[float, float, str] | None = None
        for record, piece in remaining:
            gain = len((piece & target) - covered) / len(target)
            key = (-gain, _index_key(record.chunk_index), record.record_id)
            if best_key is None or key < best_key:
                best_key = key
                best = (record, piece)
        if best is None or best_key is None or best_key[0] >= 0.0:
            break
        remaining.remove(best)
        covered = covered | (best[1] & target)
        selected.append(best[0])
    if len(selected) >= 2 and coverage(target, covered) >= cfg.coverage:
        return selected
    return None


def split_pass(
    v1_remaining: Sequence[Record],
    v2_remaining: Sequence[Record],
    *,
    cfg: ChunkMatchCfg,
    casefold: bool,
) -> tuple[list[SplitGroup], set[str], set[str]]:
    """Match unmatched v1 chunks to 2+ unmatched v2 pieces (pass 3)."""

    shingle_v2 = _shingle_map(v2_remaining, casefold=casefold)
    groups: list[SplitGroup] = []
    consumed_v1: set[str] = set()
    consumed_v2: set[str] = set()

    ordered = sorted(
        v1_remaining, key=lambda record: (_index_key(record.chunk_index), record.record_id)
    )
    for a in ordered:
        if a.record_id in consumed_v1:
            continue
        target = _shingle_map([a], casefold=casefold).get(a.record_id, frozenset())
        if not target:
            continue
        pool = [record for record in v2_remaining if record.record_id not in consumed_v2]
        candidates = _candidate_pieces(target, pool, shingle_v2, cfg=cfg)
        selected = _greedy_cover(target, candidates, cfg=cfg)
        if selected is None:
            continue
        groups.append(
            SplitGroup(a.record_id, [record.record_id for record in selected])
        )
        consumed_v1.add(a.record_id)
        consumed_v2.update(record.record_id for record in selected)
    return groups, consumed_v1, consumed_v2


def merge_pass(
    v1_remaining: Sequence[Record],
    v2_remaining: Sequence[Record],
    *,
    cfg: ChunkMatchCfg,
    casefold: bool,
) -> tuple[list[MergeGroup], set[str], set[str]]:
    """Symmetric counterpart of split_pass (pass 4)."""

    shingle_v1 = _shingle_map(v1_remaining, casefold=casefold)
    groups: list[MergeGroup] = []
    consumed_v1: set[str] = set()
    consumed_v2: set[str] = set()

    ordered = sorted(
        v2_remaining, key=lambda record: (_index_key(record.chunk_index), record.record_id)
    )
    for b in ordered:
        if b.record_id in consumed_v2:
            continue
        target = _shingle_map([b], casefold=casefold).get(b.record_id, frozenset())
        if not target:
            continue
        pool = [record for record in v1_remaining if record.record_id not in consumed_v1]
        candidates = _candidate_pieces(target, pool, shingle_v1, cfg=cfg)
        selected = _greedy_cover(target, candidates, cfg=cfg)
        if selected is None:
            continue
        groups.append(
            MergeGroup(b.record_id, [record.record_id for record in selected])
        )
        consumed_v2.add(b.record_id)
        consumed_v1.update(record.record_id for record in selected)
    return groups, consumed_v1, consumed_v2


def compute_rechunked_flag(
    document_id: str,
    *,
    doc_status: DocumentStatus,
    pass1_match_count: int,
    total_v1_chunks: int,
    cfg: ChunkMatchCfg,
) -> bool:
    """RECHUNKED document flag: content-equivalent but identity matches < 50%."""

    _ = document_id
    if doc_status not in _CONTENT_EQUIVALENT:
        return False
    if total_v1_chunks == 0:
        return False
    return (pass1_match_count / total_v1_chunks) < cfg.rechunk_exact_fraction


def match_chunks_full(
    v1_records: Sequence[Record],
    v2_records: Sequence[Record],
    *,
    initial_diff: ChunkDiffResult,
    document_diff: list[DocumentDiffRow],
    cfg: ChunkMatchCfg,
    text_cfg: TextCfg,
) -> FullChunkMatchResult:
    """Run passes 2-5 per shared document, starting from Step 25's output."""

    casefold = text_cfg.casefold
    v1_by_id = {record.record_id: record for record in v1_records}
    v2_by_id = {record.record_id: record for record in v2_records}
    doc_status = {row.document_id: row.status for row in document_diff}
    shared = [
        row.document_id for row in document_diff if row.status in _SHARED_STATUSES
    ]

    all_pairs: list[tuple[Record | None, Record | None, ChunkStatus]] = []
    all_splits: list[SplitGroup] = []
    all_merges: list[MergeGroup] = []
    final_added: list[Record] = []
    final_removed: list[Record] = []
    rechunked: set[str] = set()

    for document_id in shared:
        pool_v1 = [
            record
            for record in initial_diff.unmatched_v1
            if record.document_id == document_id
        ]
        pool_v2 = [
            record
            for record in initial_diff.unmatched_v2
            if record.document_id == document_id
        ]

        sim_pairs, used_v1, used_v2 = similarity_1to1_pass(
            pool_v1, pool_v2, cfg=cfg, casefold=casefold
        )
        remaining_v1 = [
            record for record in pool_v1 if record.record_id not in used_v1
        ]
        remaining_v2 = [
            record for record in pool_v2 if record.record_id not in used_v2
        ]

        splits, split_v1, split_v2 = split_pass(
            remaining_v1, remaining_v2, cfg=cfg, casefold=casefold
        )
        remaining_v1 = [
            record for record in remaining_v1 if record.record_id not in split_v1
        ]
        remaining_v2 = [
            record for record in remaining_v2 if record.record_id not in split_v2
        ]

        merges, merge_v1, merge_v2 = merge_pass(
            remaining_v1, remaining_v2, cfg=cfg, casefold=casefold
        )
        remaining_v1 = [
            record for record in remaining_v1 if record.record_id not in merge_v1
        ]
        remaining_v2 = [
            record for record in remaining_v2 if record.record_id not in merge_v2
        ]

        matched_v1 = set(used_v1) | set(split_v1) | set(merge_v1)
        matched_v2 = set(used_v2) | set(split_v2) | set(merge_v2)

        # Step 25's pass-1 pairs carry through, except any ADDED/REMOVED row
        # that passes 2-4 have now re-matched.
        for first, second, status in initial_diff.pairs:
            if first is not None:
                pair_document = first.document_id
            elif second is not None:
                pair_document = second.document_id
            else:
                continue
            if pair_document != document_id:
                continue
            if (
                first is not None
                and second is None
                and first.record_id in matched_v1
            ):
                continue
            if (
                second is not None
                and first is None
                and second.record_id in matched_v2
            ):
                continue
            all_pairs.append((first, second, status))

        for a, b in sim_pairs:
            all_pairs.append((a, b, "MODIFIED"))

        all_splits.extend(splits)
        for group in splits:
            for piece_id in group.v2_chunk_ids:
                all_pairs.append(
                    (v1_by_id[group.v1_chunk_id], v2_by_id[piece_id], "SPLIT_PART")
                )

        all_merges.extend(merges)
        for group in merges:
            for member_id in group.v1_chunk_ids:
                all_pairs.append(
                    (v1_by_id[member_id], v2_by_id[group.v2_chunk_id], "MERGE_PART")
                )

        final_removed.extend(remaining_v1)
        final_added.extend(remaining_v2)

        if initial_diff.pass1_match_counts:
            pass1_count = initial_diff.pass1_match_counts.get(document_id, 0)
        else:
            pass1_count = sum(
                1
                for first, second, _status in initial_diff.pairs
                if first is not None
                and second is not None
                and first.document_id == document_id
            )
        total_v1_chunks = sum(
            1 for record in v1_records if record.document_id == document_id
        )
        if compute_rechunked_flag(
            document_id,
            doc_status=doc_status[document_id],
            pass1_match_count=pass1_count,
            total_v1_chunks=total_v1_chunks,
            cfg=cfg,
        ):
            rechunked.add(document_id)

    return FullChunkMatchResult(
        pairs=all_pairs,
        splits=all_splits,
        merges=all_merges,
        final_added=final_added,
        final_removed=final_removed,
        rechunked_document_ids=rechunked,
    )


def build_chunk_match_lookup(
    result: FullChunkMatchResult,
) -> Callable[[str], str | None]:
    """Map a v1 record id to a representative matched v2 id, or None."""

    mapping: dict[str, str] = {}
    for first, second, status in result.pairs:
        if first is None or second is None:
            continue
        if status in ("MODIFIED", "SPLIT_PART", "MERGE_PART"):
            mapping[first.record_id] = second.record_id
    # A split's representative is the piece contributing the most coverage;
    # split_pass selects pieces in descending marginal coverage order, so the
    # first id is that piece.
    for group in result.splits:
        if group.v2_chunk_ids:
            mapping[group.v1_chunk_id] = group.v2_chunk_ids[0]

    def lookup(v1_record_id: str) -> str | None:
        return mapping.get(v1_record_id)

    return lookup


def build_group_lookup(
    result: FullChunkMatchResult,
) -> Callable[[str], ChunkMatchOutcome | None]:
    """Map a v1 record id to a group-aware ChunkMatchOutcome."""

    outcomes: dict[str, ChunkMatchOutcome] = {}
    for first, second, status in result.pairs:
        if first is None or second is None:
            continue
        if status == "MODIFIED":
            outcomes[first.record_id] = ChunkMatchOutcome(
                "one_to_one", second.record_id
            )
        elif status == "SPLIT_PART":
            outcomes[first.record_id] = ChunkMatchOutcome(
                "split_part", second.record_id
            )
        elif status == "MERGE_PART":
            outcomes[first.record_id] = ChunkMatchOutcome(
                "merge_part", second.record_id
            )
    for group in result.splits:
        if group.v2_chunk_ids:
            outcomes[group.v1_chunk_id] = ChunkMatchOutcome(
                "split_part", group.v2_chunk_ids[0]
            )

    def lookup(v1_record_id: str) -> ChunkMatchOutcome | None:
        return outcomes.get(v1_record_id)

    return lookup
