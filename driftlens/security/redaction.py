"""Deterministic, snapshot-scoped identifier redaction (Step 41).

Redaction replaces identifier-valued metric values with a short pseudonym
derived from the snapshot id and the raw value (sha256(snapshot_id + US +
value)[:12]).  The same raw value therefore maps to the same pseudonym within
one snapshot, and to a different pseudonym in another snapshot, so reports
cannot be joined on displayed identifiers or labels.

The module is pure: it never reads files or configuration.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from driftlens.analysis.health import Metric
from driftlens.identity.hashing import sha256_hex

if TYPE_CHECKING:  # pragma: no cover - typing only
    from driftlens.analysis.content import ChunkDiffResult, DocumentDiffRow
    from driftlens.analysis.duplicates import ExactDuplicateResult
    from driftlens.schema.record import Record

#: Length of the hex pseudonym prefix (PROJECT_ARCHITECTURE.md privacy rules).
_REDACT_LENGTH = 12

#: Unit separator, matching the hash-input convention used by sample_hash.
_UNIT_SEPARATOR = chr(31)


def _pseudonym(value: str, *, snapshot_id: str) -> str:
    joined = snapshot_id + _UNIT_SEPARATOR + value
    return sha256_hex(joined.encode("utf-8"))[:_REDACT_LENGTH]


def redact_id(value: str, *, snapshot_id: str) -> str:
    """Pseudonymize a record/document identifier within one snapshot."""

    return _pseudonym(value, snapshot_id=snapshot_id)


def redact_label(value: str, *, snapshot_id: str) -> str:
    """Pseudonymize a categorical label within one snapshot.

    The construction is identical to redact_id; the distinct name marks the
    semantic difference (an id and a label with the same text still share a
    pseudonym, which is intentional so pairs stay joinable in one report).
    """

    return _pseudonym(value, snapshot_id=snapshot_id)


def apply_redaction_to_metric(m: Metric, *, snapshot_id: str, redact: bool) -> Metric:
    """Return *m* with identifier values pseudonymized when *redact* is set.

    Non-identifier metrics are returned unchanged, and numeric values are never
    touched.  Identifier metrics redact v1/v2/change; the redactor is chosen by
    metric.identifier_kind so callers can name the semantic role.
    """

    if not redact or not m.is_identifier_valued:
        return m
    pseudonymize = redact_label if m.identifier_kind == "label" else redact_id

    def _one(value: float | str | None) -> float | str | None:
        if isinstance(value, str):
            return pseudonymize(value, snapshot_id=snapshot_id)
        return value

    return replace(
        m,
        v1=_one(m.v1),
        v2=_one(m.v2),
        change=_one(m.change),
    )

def redact_records(records: "list[Record]", *, snapshot_id: str) -> "list[Record]":
    """Return copies of *records* with record_id/document_id pseudonymized (F-08).

    Record identity fields are the only redacted parts; text, embeddings and
    metrics are untouched so classification and report generation keep working.
    """
    out = []
    for record in records:
        new = replace(record)
        new.record_id = redact_id(record.record_id, snapshot_id=snapshot_id)
        if record.document_id is not None:
            new.document_id = redact_id(
                record.document_id, snapshot_id=snapshot_id
            )
        out.append(new)
    return out


def redact_document_diff_rows(
    rows: "list[DocumentDiffRow]", *, snapshot_id: str
) -> "list[DocumentDiffRow]":
    """Document diff rows with pseudonymized document ids (F-08)."""
    return [
        replace(row, document_id=redact_id(row.document_id, snapshot_id=snapshot_id))
        for row in rows
    ]


def redact_chunk_diff_result(
    result: "ChunkDiffResult", *, snapshot_id: str
) -> "ChunkDiffResult":
    """Chunk diff with pseudonymized record ids on both sides (F-08).

    Matching structure is preserved: pairs stay pairs, unmatched lists stay
    unmatched; only the identifier fields change.
    """
    pairs = []
    for first, second, status in result.pairs:
        new_first = None
        new_second = None
        if first is not None:
            new_first = replace(first)
            new_first.record_id = redact_id(
                first.record_id, snapshot_id=snapshot_id
            )
        if second is not None:
            new_second = replace(second)
            new_second.record_id = redact_id(
                second.record_id, snapshot_id=snapshot_id
            )
        pairs.append((new_first, new_second, status))
    unmatched_v1 = redact_records(list(result.unmatched_v1), snapshot_id=snapshot_id)
    unmatched_v2 = redact_records(list(result.unmatched_v2), snapshot_id=snapshot_id)
    return replace(
        result,
        pairs=pairs,
        unmatched_v1=unmatched_v1,
        unmatched_v2=unmatched_v2,
        pass1_match_counts={
            redact_id(doc_id, snapshot_id=snapshot_id): count
            for doc_id, count in result.pass1_match_counts.items()
        },
    )


def redact_duplicate_result(
    result: "ExactDuplicateResult", *, snapshot_id: str
) -> "ExactDuplicateResult":
    """Duplicate groups with pseudonymized representative/member ids (F-08)."""
    from driftlens.analysis.duplicates import DuplicateGroup

    def _redact_group(group):
        return replace(
            group,
            representative_id=redact_id(
                group.representative_id, snapshot_id=snapshot_id
            ),
            member_ids=[
                redact_id(member, snapshot_id=snapshot_id)
                for member in group.member_ids
            ],
        )

    redacted = replace(
        result,
        groups=[_redact_group(g) for g in result.groups],
    )
    for attr in ("near_groups", "semantic_groups"):
        groups = getattr(result, attr, None)
        if groups:
            setattr(redacted, attr, [_redact_group(g) for g in groups])
    return redacted

