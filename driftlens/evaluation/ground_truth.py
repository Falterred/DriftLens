"""Ground-truth resolution and case classification (Step 40, section 11.4).

Resolution maps a case's v1-side ground-truth ids onto the v2 snapshot.
At release 0.2 chunk matching does not exist yet (Step 43), so every
chunk-id item that is not found by direct identity resolves to
``UNMAPPABLE``; the ``chunk_match_lookup`` hook is accepted so the
Step 43 matcher can be wired in without changing this module's shape.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from driftlens.schema.eval_set import EvalCase, EvalSet
from driftlens.schema.record import Record
from driftlens.schema.snapshot import DocumentTable

if TYPE_CHECKING:
    from driftlens.analysis.content import ComparisonLevelResult

GroundTruthState = Literal["PRESENT", "REMOVED", "UNMAPPABLE"]
CaseState = Literal[
    "VALID",
    "PARTIAL",
    "INVALIDATED_GT_REMOVED",
    "INVALIDATED_GT_UNMAPPABLE",
    "INVALID_V1",
]


@dataclass
class ResolvedGroundTruth:
    """One ground-truth item resolved against the v2 snapshot."""

    item_id: str
    kind: Literal["document", "chunk"]
    state: GroundTruthState
    resolved_id: str | None


@dataclass
class ResolvedCase:
    """One eval case with its resolved ground truth and comparable id sets."""

    case: EvalCase
    state: CaseState
    present_document_ids: set[str]
    present_chunk_ids: set[str]


def resolve_ground_truth_document(
    doc_id: str, *, v1_document_identity: dict[str, str], v2_document_ids: set[str]
) -> ResolvedGroundTruth:
    """Resolve one document ground-truth id (section 11.4).

    The caller guarantees the id exists in v1 (the case-level classifier is
    responsible for routing "absent from v1 entirely" to ``INVALID_V1``).
    """

    if doc_id in v2_document_ids:
        return ResolvedGroundTruth(doc_id, "document", "PRESENT", doc_id)
    confidence = v1_document_identity.get(doc_id)
    if confidence in ("HIGH", "MEDIUM"):
        return ResolvedGroundTruth(doc_id, "document", "REMOVED", None)
    # LOW confidence, or an id absent from the identity map: identity cannot
    # carry the item across, so it is unmappable rather than removed.
    return ResolvedGroundTruth(doc_id, "document", "UNMAPPABLE", None)


def resolve_ground_truth_chunk(
    chunk_record_id: str,
    *,
    v1_records_by_id: dict[str, Record],
    v2_record_ids: set[str],
    chunk_match_lookup: Callable[[str], str | None] | None = None,
) -> ResolvedGroundTruth:
    """Resolve one chunk ground-truth id (section 11.4).

    ``chunk_match_lookup`` is the Step 43 chunk-matching hook. It is ``None`
    for every release-0.2 call site, which makes every non-identity chunk
    item ``UNMAPPABLE`` regardless of its own identity confidence.
    """

    if chunk_record_id in v2_record_ids:
        return ResolvedGroundTruth(
            chunk_record_id, "chunk", "PRESENT", chunk_record_id
        )
    if chunk_match_lookup is not None:
        matched = chunk_match_lookup(chunk_record_id)
        if matched is not None:
            return ResolvedGroundTruth(chunk_record_id, "chunk", "PRESENT", matched)
    record = v1_records_by_id.get(chunk_record_id)
    if record is not None and record.identity_confidence == "LOW":
        # Reserved for the Step 43 lookup: a low-confidence, unmatched chunk
        # stays unmappable once chunk matching exists.
        return ResolvedGroundTruth(chunk_record_id, "chunk", "UNMAPPABLE", None)
    return ResolvedGroundTruth(chunk_record_id, "chunk", "UNMAPPABLE", None)


def classify_case(
    resolved_items: Sequence[ResolvedGroundTruth], *, exists_in_v1: bool
) -> CaseState:
    """Classify one case from its resolved items (section 11.4)."""

    if not exists_in_v1:
        return "INVALID_V1"
    states = {item.state for item in resolved_items}
    if states == {"PRESENT"}:
        return "VALID"
    if "PRESENT" in states:
        return "PARTIAL"
    if "REMOVED" in states and "UNMAPPABLE" not in states:
        return "INVALIDATED_GT_REMOVED"
    # A mix of REMOVED and UNMAPPABLE with nothing PRESENT, a pure UNMAPPABLE
    # set, or (defensively) an empty set: UNMAPPABLE takes priority.
    return "INVALIDATED_GT_UNMAPPABLE"


def resolve_eval_set(
    eval_set: EvalSet,
    *,
    v1_documents: DocumentTable,
    v2_documents: DocumentTable,
    v1_records_by_id: dict[str, Record],
    v2_records_by_id: dict[str, Record],
    comparison_level: ComparisonLevelResult,
) -> list[ResolvedCase]:
    """Resolve every case of an eval set against the v2 snapshot (section 11.4).

    Ids absent from v1 contribute nothing to the resolved items (they cannot
    be resolved) but still count toward ``exists_in_v1``; a case whose every
    listed id is absent from v1 becomes ``INVALID_V1``.
    """

    v1_document_identity = {
        doc_id: row.identity_confidence
        for doc_id, row in v1_documents.rows.items()
    }
    v2_document_ids = set(v2_documents.rows)
    v2_record_ids = set(v2_records_by_id)
    # The comparison level is accepted for the caller contract; v0.2 resolution
    # is identity-only and the Step 43 chunk matcher will consume it.
    _ = comparison_level

    resolved_cases: list[ResolvedCase] = []
    for case in eval_set.cases:
        items = [
            resolve_ground_truth_document(
                doc_id,
                v1_document_identity=v1_document_identity,
                v2_document_ids=v2_document_ids,
            )
            for doc_id in case.relevant_document_ids
            if doc_id in v1_document_identity
        ] + [
            resolve_ground_truth_chunk(
                chunk_id,
                v1_records_by_id=v1_records_by_id,
                v2_record_ids=v2_record_ids,
                chunk_match_lookup=None,
            )
            for chunk_id in case.relevant_chunk_ids
            if chunk_id in v1_records_by_id
        ]
        exists_in_v1 = any(
            doc_id in v1_document_identity
            for doc_id in case.relevant_document_ids
        ) or any(
            chunk_id in v1_records_by_id for chunk_id in case.relevant_chunk_ids
        )
        present_document_ids = {
            item.resolved_id
            for item in items
            if item.kind == "document"
            and item.state == "PRESENT"
            and item.resolved_id is not None
        }
        present_chunk_ids = {
            item.resolved_id
            for item in items
            if item.kind == "chunk"
            and item.state == "PRESENT"
            and item.resolved_id is not None
        }
        resolved_cases.append(
            ResolvedCase(
                case=case,
                state=classify_case(items, exists_in_v1=exists_in_v1),
                present_document_ids=present_document_ids,
                present_chunk_ids=present_chunk_ids,
            )
        )
    return resolved_cases
