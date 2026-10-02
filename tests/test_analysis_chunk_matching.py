"""Chunk matching passes 2-5 and group-aware ground-truth wiring (Step 43)."""

from __future__ import annotations

from driftlens.analysis.chunk_matching import (
    FullChunkMatchResult,
    MergeGroup,
    SplitGroup,
    build_chunk_match_lookup,
    build_group_lookup,
    compute_rechunked_flag,
    match_chunks_full,
    merge_pass,
    similarity_1to1_pass,
    split_pass,
)
from driftlens.analysis.content import ChunkDiffResult, DocumentDiffRow
from driftlens.config.schema import ChunkMatchCfg, TextCfg
from driftlens.evaluation.ground_truth import resolve_ground_truth_chunk
from driftlens.identity.chunks import compute_text_hashes
from driftlens.schema.record import Record

CFG = ChunkMatchCfg()
TEXT_CFG = TextCfg()


def _rec(record_id, text, *, document_id="d1", chunk_index=None, confidence=None):
    record = Record(
        record_id=record_id,
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        identity_confidence=confidence,
    )
    compute_text_hashes(record)
    return record


def test_similarity_pass_greedy_highest_first():
    a1 = _rec("a1", "alpha beta gamma delta epsilon", chunk_index=0)
    a2 = _rec("a2", "alpha beta gamma delta zeta eta", chunk_index=1)
    a3 = _rec("a3", "one two three four five six", chunk_index=2)
    b1 = _rec("b1", "alpha beta gamma delta zeta", chunk_index=0)
    b2 = _rec("b2", "alpha beta gamma delta zeta eta", chunk_index=1)
    b3 = _rec("b3", "one two three four seven eight", chunk_index=2)

    pairs, matched_v1, matched_v2 = similarity_1to1_pass(
        [a1, a2, a3], [b1, b2, b3], cfg=CFG, casefold=True
    )

    assert [(a.record_id, b.record_id) for a, b in pairs] == [("a2", "b2")]
    assert matched_v1 == {"a2"}
    assert matched_v2 == {"b2"}


def test_similarity_pass_tie_break():
    # Both v1 chunks are identical to b1; the lower chunk_index wins even
    # though its record_id sorts last, so index is the tie-break key.
    a_high = _rec("z1", "t1 t2 t3 t4 t5 t6", chunk_index=0)
    a_low = _rec("a1", "t1 t2 t3 t4 t5 t6", chunk_index=1)
    b1 = _rec("b1", "t1 t2 t3 t4 t5 t6", chunk_index=0)

    pairs, _matched_v1, _matched_v2 = similarity_1to1_pass(
        [a_low, a_high], [b1], cfg=CFG, casefold=True
    )

    assert pairs == [(a_high, b1)]


def test_similarity_pass_below_threshold_not_matched():
    a = _rec("a", "a b c d e")
    b = _rec("b", "a b c d e f g")

    pairs, matched_v1, matched_v2 = similarity_1to1_pass(
        [a], [b], cfg=CFG, casefold=True
    )

    assert pairs == []
    assert matched_v1 == set()
    assert matched_v2 == set()


def test_split_pass_basic():
    a = _rec("a", "w1 w2 w3 w4 w5 w6 w7 w8", chunk_index=0)
    c1 = _rec("c1", "w1 w2 w3 w4 w5", chunk_index=0)
    c2 = _rec("c2", "w4 w5 w6 w7 w8", chunk_index=1)

    groups, used_v1, used_v2 = split_pass([a], [c1, c2], cfg=CFG, casefold=True)

    assert len(groups) == 1
    assert groups[0].v1_chunk_id == "a"
    assert groups[0].v2_chunk_ids == ["c1", "c2"]
    assert used_v1 == {"a"}
    assert used_v2 == {"c1", "c2"}


def test_split_pass_rejects_low_contribution_piece():
    a = _rec("a", "w1 w2 w3 w4 w5 w6 w7 w8")
    intruder = " ".join(["w1", "w2", "w3", "w4", "w5"] + [f"u{i}" for i in range(1, 20)])
    c3 = _rec("c3", intruder)

    groups, used_v1, used_v2 = split_pass([a], [c3], cfg=CFG, casefold=True)

    assert groups == []
    assert used_v1 == set()
    assert used_v2 == set()


def test_merge_pass_basic():
    b = _rec("b", "w1 w2 w3 w4 w5 w6 w7 w8", chunk_index=0)
    c1 = _rec("c1", "w1 w2 w3 w4 w5", chunk_index=0)
    c2 = _rec("c2", "w4 w5 w6 w7 w8", chunk_index=1)

    groups, used_v1, used_v2 = merge_pass([c1, c2], [b], cfg=CFG, casefold=True)

    assert len(groups) == 1
    assert groups[0].v2_chunk_id == "b"
    assert groups[0].v1_chunk_ids == ["c1", "c2"]
    assert used_v1 == {"c1", "c2"}
    assert used_v2 == {"b"}


def test_split_merge_use_only_post_similarity_remainder():
    # a and c are identical and therefore consumed by pass 2, so the smaller
    # c2 must NOT be pulled into a split group with the already-matched a.
    a = _rec("a", "w1 w2 w3 w4 w5 w6 w7 w8")
    c = _rec("c", "w1 w2 w3 w4 w5 w6 w7 w8")
    c2 = _rec("c2", "w4 w5 w6 w7 w8")
    initial = ChunkDiffResult(
        pairs=[],
        unmatched_v1=[a],
        unmatched_v2=[c, c2],
        pass1_match_counts={"d1": 1},
    )
    document_diff = [DocumentDiffRow("d1", "UNCHANGED", False, False, "HIGH", "HIGH", 1.0)]

    result = match_chunks_full(
        [a], [c, c2], initial_diff=initial, document_diff=document_diff,
        cfg=CFG, text_cfg=TEXT_CFG,
    )

    assert result.splits == []
    assert [r.record_id for r in result.final_added] == ["c2"]
    assert any(
        first is a and second is c and status == "MODIFIED"
        for first, second, status in result.pairs
    )


def test_rechunked_flag_true_below_threshold():
    assert compute_rechunked_flag(
        "d1", doc_status="UNCHANGED", pass1_match_count=1,
        total_v1_chunks=5, cfg=CFG,
    ) is True


def test_rechunked_flag_false_for_modified_document():
    assert compute_rechunked_flag(
        "d1", doc_status="MODIFIED", pass1_match_count=0,
        total_v1_chunks=5, cfg=CFG,
    ) is False


def test_match_chunks_full_end_to_end():
    u1 = _rec("u1", "base one two three", chunk_index=0)
    u1b = _rec("u1b", "base one two three", chunk_index=0)
    s1 = _rec("s1", "w1 w2 w3 w4 w5 w6 w7 w8", chunk_index=1)
    p1 = _rec("p1", "w1 w2 w3 w4 w5", chunk_index=0)
    p2 = _rec("p2", "w4 w5 w6 w7 w8", chunk_index=1)
    q1 = _rec("q1", "x1 x2 x3 x4 x5", chunk_index=2)
    q2 = _rec("q2", "x4 x5 x6 x7 x8", chunk_index=3)
    m2 = _rec("m2", "x1 x2 x3 x4 x5 x6 x7 x8", chunk_index=2)
    r1 = _rec("r1", "removed only text here", chunk_index=4)
    r2 = _rec("r2", "added only text here", chunk_index=4)

    v1_records = [u1, s1, q1, q2, r1]
    v2_records = [u1b, p1, p2, m2, r2]
    initial = ChunkDiffResult(
        pairs=[(u1, u1b, "UNCHANGED")],
        unmatched_v1=[s1, q1, q2, r1],
        unmatched_v2=[p1, p2, m2, r2],
        pass1_match_counts={"d1": 1},
    )
    document_diff = [DocumentDiffRow("d1", "UNCHANGED", False, False, "HIGH", "HIGH", 1.0)]

    result = match_chunks_full(
        v1_records, v2_records, initial_diff=initial,
        document_diff=document_diff, cfg=CFG, text_cfg=TEXT_CFG,
    )

    assert len(result.splits) == 1
    assert result.splits[0].v1_chunk_id == "s1"
    assert set(result.splits[0].v2_chunk_ids) == {"p1", "p2"}
    assert len(result.merges) == 1
    assert result.merges[0].v2_chunk_id == "m2"
    assert set(result.merges[0].v1_chunk_ids) == {"q1", "q2"}
    assert [r.record_id for r in result.final_removed] == ["r1"]
    assert [r.record_id for r in result.final_added] == ["r2"]
    assert result.rechunked_document_ids == {"d1"}
    assert any(
        first is u1 and second is u1b and status == "UNCHANGED"
        for first, second, status in result.pairs
    )
    assert any(
        first is s1 and second is p1 and status == "SPLIT_PART"
        for first, second, status in result.pairs
    )
    assert any(
        first is q1 and second is m2 and status == "MERGE_PART"
        for first, second, status in result.pairs
    )


def _lookup_fixture_result():
    a = _rec("a", "alpha beta gamma delta")
    b = _rec("b", "alpha beta gamma delta")
    s = _rec("s", "w1 w2 w3 w4 w5 w6 w7 w8")
    p = _rec("p", "w1 w2 w3 w4 w5")
    p2 = _rec("p2", "w4 w5 w6 w7 w8")
    q1 = _rec("q1", "x1 x2 x3 x4 x5")
    q2 = _rec("q2", "x4 x5 x6 x7 x8")
    m = _rec("m", "x1 x2 x3 x4 x5 x6 x7 x8")
    result = FullChunkMatchResult(
        pairs=[
            (a, b, "MODIFIED"),
            (s, p, "SPLIT_PART"),
            (q1, m, "MERGE_PART"),
            (q2, m, "MERGE_PART"),
        ],
        splits=[SplitGroup("s", ["p", "p2"])],
        merges=[MergeGroup("m", ["q1", "q2"])],
    )
    return a, b, s, p, q1, m, result


def test_build_chunk_match_lookup_covers_all_match_kinds():
    a, b, s, p, q1, m, result = _lookup_fixture_result()

    lookup = build_chunk_match_lookup(result)
    assert lookup("a") == "b"
    assert lookup("s") == "p"
    assert lookup("q1") == "m"
    assert lookup("q2") == "m"
    assert lookup("removed") is None

    group_lookup = build_group_lookup(result)
    one_to_one = group_lookup("a")
    assert one_to_one is not None and one_to_one.kind == "one_to_one"
    assert one_to_one.resolved_id == "b"
    split_part = group_lookup("s")
    assert split_part is not None and split_part.kind == "split_part"
    assert split_part.resolved_id == "p"
    merge_part = group_lookup("q1")
    assert merge_part is not None and merge_part.kind == "merge_part"
    assert merge_part.resolved_id == "m"
    assert group_lookup("removed") is None


def test_ground_truth_chunk_resolves_via_split_group():
    s = _rec("s", "w1 w2 w3 w4 w5 w6 w7 w8", confidence="HIGH")
    p1 = _rec("p1", "w1 w2 w3 w4 w5")
    p2 = _rec("p2", "w4 w5 w6 w7 w8")
    result = FullChunkMatchResult(
        pairs=[(s, p1, "SPLIT_PART"), (s, p2, "SPLIT_PART")],
        splits=[SplitGroup("s", ["p1", "p2"])],
    )
    lookup = build_group_lookup(result)

    resolved = resolve_ground_truth_chunk(
        "s", v1_records_by_id={"s": s}, v2_record_ids={"p1", "p2"},
        chunk_match_lookup=lookup,
    )
    assert resolved.state == "PRESENT"
    assert resolved.resolved_id == "p1"

    removed = _rec("r", "totally different removed text", confidence="HIGH")
    gone = resolve_ground_truth_chunk(
        "r", v1_records_by_id={"r": removed}, v2_record_ids=set(),
        chunk_match_lookup=lookup,
    )
    assert gone.state == "REMOVED"
    assert gone.resolved_id is None
