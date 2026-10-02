"""Ground-truth resolution and case classification tests (Step 40)."""

from __future__ import annotations

from driftlens.analysis.content import ComparisonLevelResult
from driftlens.evaluation.ground_truth import (
    ResolvedGroundTruth,
    classify_case,
    resolve_eval_set,
    resolve_ground_truth_chunk,
    resolve_ground_truth_document,
)
from driftlens.schema.eval_set import EvalCase, EvalSet
from driftlens.schema.record import Record
from driftlens.schema.snapshot import DocumentRow, DocumentTable


def _item(item_id: str, state: str, *, kind: str = "document") -> ResolvedGroundTruth:
    return ResolvedGroundTruth(
        item_id=item_id,
        kind=kind,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        resolved_id=item_id if state == "PRESENT" else None,
    )


def _document_table(*document_ids: str, confidence: str = "HIGH") -> DocumentTable:
    rows = {
        document_id: DocumentRow(
            document_id=document_id,
            chunk_count=1,
            identity_confidence=confidence,
        )
        for document_id in document_ids
    }
    return DocumentTable(rows=rows)


def _case(
    *, documents: tuple[str, ...] = (), chunks: tuple[str, ...] = (), query_id: str = "q1"
) -> EvalCase:
    return EvalCase(
        query_id=query_id,
        query_id_was_explicit=True,
        query="alpha",
        relevant_document_ids=list(documents),
        relevant_chunk_ids=list(chunks),
        query_embeddings=None,
    )


def test_resolve_document_present() -> None:
    resolved = resolve_ground_truth_document(
        "d1", v1_document_identity={"d1": "HIGH"}, v2_document_ids={"d1"}
    )
    assert resolved.state == "PRESENT"
    assert resolved.resolved_id == "d1"
    assert resolved.kind == "document"


def test_resolve_document_removed_high_confidence() -> None:
    resolved = resolve_ground_truth_document(
        "d1", v1_document_identity={"d1": "HIGH"}, v2_document_ids=set()
    )
    assert resolved.state == "REMOVED"
    assert resolved.resolved_id is None


def test_resolve_document_unmappable_low_confidence() -> None:
    resolved = resolve_ground_truth_document(
        "d1", v1_document_identity={"d1": "LOW"}, v2_document_ids=set()
    )
    assert resolved.state == "UNMAPPABLE"
    assert resolved.resolved_id is None


def test_resolve_chunk_unmappable_at_v02_always() -> None:
    records = {"c1": Record(record_id="c1", identity_confidence="HIGH")}
    resolved = resolve_ground_truth_chunk(
        "c1",
        v1_records_by_id=records,
        v2_record_ids=set(),
        chunk_match_lookup=None,
    )
    assert resolved.state == "UNMAPPABLE"
    assert resolved.resolved_id is None
    assert resolved.kind == "chunk"


def test_classify_case_valid() -> None:
    assert classify_case([_item("d1", "PRESENT")], exists_in_v1=True) == "VALID"


def test_classify_case_partial() -> None:
    items = [_item("d1", "PRESENT"), _item("d2", "REMOVED")]
    assert classify_case(items, exists_in_v1=True) == "PARTIAL"


def test_classify_case_invalidated_gt_removed() -> None:
    items = [_item("d1", "REMOVED"), _item("d2", "REMOVED")]
    assert classify_case(items, exists_in_v1=True) == "INVALIDATED_GT_REMOVED"


def test_classify_case_invalidated_gt_unmappable() -> None:
    items = [_item("d1", "REMOVED"), _item("d2", "UNMAPPABLE")]
    assert classify_case(items, exists_in_v1=True) == "INVALIDATED_GT_UNMAPPABLE"


def test_classify_case_invalid_v1() -> None:
    items = [_item("d1", "PRESENT")]
    assert classify_case(items, exists_in_v1=False) == "INVALID_V1"


def test_resolve_eval_set_partial_ground_truth_excludes_removed_from_present_sets() -> None:
    eval_set = EvalSet(cases=[_case(documents=("d1", "d2"))])
    resolved = resolve_eval_set(
        eval_set,
        v1_documents=_document_table("d1", "d2"),
        v2_documents=_document_table("d1"),
        v1_records_by_id={},
        v2_records_by_id={},
        comparison_level=ComparisonLevelResult(level=4, hash_kind="text"),
    )
    assert len(resolved) == 1
    assert resolved[0].state == "PARTIAL"
    assert resolved[0].present_document_ids == {"d1"}
    assert resolved[0].present_chunk_ids == set()
