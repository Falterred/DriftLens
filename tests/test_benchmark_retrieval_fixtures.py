"""Step 51 full-scale retrieval eval-set fixture (Release 1.0)."""

from __future__ import annotations

from driftlens.analysis.chunk_matching import (
    FullChunkMatchResult,
    build_chunk_match_lookup,
    split_pass,
)
from driftlens.config.schema import ChunkMatchCfg
from driftlens.evaluation.ground_truth import (
    classify_case,
    resolve_ground_truth_chunk,
    resolve_ground_truth_document,
)
from driftlens.identity.chunks import compute_text_hashes
from driftlens.schema.record import Record
from fixtures.generator import (
    generate_retrieval_eval_fixture,
    retrieval_fixture_plan,
)

CORPUS_SIZE = 12
RATES = {
    "ground_truth_removed_rate": 0.25,
    "edited_rate": 0.25,
    "rechunked_rate": 0.25,
    "partial_gt_rate": 0.25,
}


def _fixture():
    return generate_retrieval_eval_fixture(23, CORPUS_SIZE, **RATES)


def _plan():
    return retrieval_fixture_plan(CORPUS_SIZE, **RATES)


def _records(entries):
    table = {}
    for entry in entries:
        record = Record(
            record_id=entry["id"],
            document_id=entry["document_id"],
            chunk_index=entry["chunk_index"],
            text=entry["text"],
            identity_confidence="HIGH",
        )
        compute_text_hashes(record)
        table[record.record_id] = record
    return table


def _doc_sets(pair):
    v1_docs = {entry["document_id"] for entry in pair.v1_records}
    v2_docs = {entry["document_id"] for entry in pair.v2_records}
    removed = v1_docs - v2_docs
    identity = {doc_id: "HIGH" for doc_id in v1_docs}
    return v1_docs, v2_docs, removed, identity


def test_ground_truth_removed_state():
    pair, eval_set = _fixture()
    _v1_docs, v2_docs, removed, identity = _doc_sets(pair)
    assert len(removed) == _plan().removed_documents

    removed_cases = [
        case
        for case in eval_set.cases
        if case.relevant_document_ids
        and set(case.relevant_document_ids) <= removed
    ]
    assert len(removed_cases) == _plan().removed_documents
    for case in removed_cases:
        items = [
            resolve_ground_truth_document(
                doc_id, v1_document_identity=identity, v2_document_ids=v2_docs
            )
            for doc_id in case.relevant_document_ids
        ]
        assert all(item.state == "REMOVED" for item in items)
        assert classify_case(items, exists_in_v1=True) == "INVALIDATED_GT_REMOVED"


def test_ground_truth_unmappable_state():
    pair, eval_set = _fixture()
    v1_records = _records(pair.v1_records)
    v2_ids = {entry["id"] for entry in pair.v2_records}

    chunk_cases = [case for case in eval_set.cases if case.relevant_chunk_ids]
    assert len(chunk_cases) == (
        _plan().unmappable_chunks + _plan().rechunked_documents
    )
    for case in chunk_cases:
        items = [
            resolve_ground_truth_chunk(
                chunk_id,
                v1_records_by_id=v1_records,
                v2_record_ids=v2_ids,
                chunk_match_lookup=None,
            )
            for chunk_id in case.relevant_chunk_ids
        ]
        assert all(item.state == "UNMAPPABLE" for item in items)
        assert classify_case(items, exists_in_v1=True) == "INVALIDATED_GT_UNMAPPABLE"


def test_partial_state():
    pair, eval_set = _fixture()
    _v1_docs, v2_docs, removed, identity = _doc_sets(pair)

    partial_cases = [
        case
        for case in eval_set.cases
        if len(case.relevant_document_ids) == 2
        and (set(case.relevant_document_ids) & removed)
        and (set(case.relevant_document_ids) - removed)
    ]
    assert len(partial_cases) == _plan().partial_cases
    for case in partial_cases:
        items = [
            resolve_ground_truth_document(
                doc_id, v1_document_identity=identity, v2_document_ids=v2_docs
            )
            for doc_id in case.relevant_document_ids
        ]
        assert {item.state for item in items} == {"PRESENT", "REMOVED"}
        assert classify_case(items, exists_in_v1=True) == "PARTIAL"


def test_rechunked_case_resolves_via_chunk_matching_when_available():
    pair, eval_set = _fixture()
    v1_records = _records(pair.v1_records)
    v2_records = _records(pair.v2_records)
    v2_ids = set(v2_records)

    rechunk_cases = [
        case for case in eval_set.cases if case.query.startswith("rechunked query")
    ]
    assert len(rechunk_cases) == _plan().rechunked_documents
    case = rechunk_cases[0]
    chunk_id = case.relevant_chunk_ids[0]

    without_lookup = resolve_ground_truth_chunk(
        chunk_id,
        v1_records_by_id=v1_records,
        v2_record_ids=v2_ids,
        chunk_match_lookup=None,
    )
    assert without_lookup.state == "UNMAPPABLE"

    v1_record = v1_records[chunk_id]
    v2_chunks = sorted(
        (
            record
            for record in v2_records.values()
            if record.document_id == v1_record.document_id
        ),
        key=lambda record: record.chunk_index,
    )
    assert len(v2_chunks) == 2
    groups, _used_v1, _used_v2 = split_pass(
        [v1_record], v2_chunks, cfg=ChunkMatchCfg(), casefold=True
    )
    assert len(groups) == 1

    lookup = build_chunk_match_lookup(FullChunkMatchResult(pairs=[], splits=groups))
    resolved = resolve_ground_truth_chunk(
        chunk_id,
        v1_records_by_id=v1_records,
        v2_record_ids=v2_ids,
        chunk_match_lookup=lookup,
    )
    assert resolved.state == "PRESENT"
    assert resolved.resolved_id in {record.record_id for record in v2_chunks}
