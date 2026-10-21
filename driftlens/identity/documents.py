"""Document identity, granularity detection and id disambiguation (Step 19)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Sequence

from driftlens.config.schema import SideOptions
from driftlens.errors import (
    Code,
    DiagnosticSink,
    InvalidInputError,
    UndefinedCaseError,
)
from driftlens.identity.chunks import (
    all_chunk_identity_keys,
    compute_chunk_identity,
    compute_embedding_digest_field,
    compute_text_hashes,
)
from driftlens.identity.hashing import canonical_json, record_digest
from driftlens.schema.record import Record

DOCUMENT_ID_CANDIDATE_NAMES = ["source_id", "source", "file", "path", "url"]

Granularity = Literal["CHUNK", "DOCUMENT"]
GranularitySource = Literal["detected", "assumed", "forced"]

_DocConfidence = Literal["HIGH", "MEDIUM", "LOW"]
_CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


@dataclass
class DocumentIdentityResult:
    """The first available document-identity level for one record."""

    level: int
    document_id: str
    confidence: _DocConfidence


def _identity_hint(record: Record) -> str | None:
    transient = getattr(record, "_identity_source_hint", None)
    if transient is not None:
        return transient
    return record.identity_source_hint


def _metadata_document_value(record: Record) -> str | None:
    lowered = {key.lower(): key for key in record.metadata}
    for name in DOCUMENT_ID_CANDIDATE_NAMES:
        key = lowered.get(name.lower())
        if key is None:
            continue
        value = record.metadata.get(key)
        if value is None:
            continue
        return value if isinstance(value, str) else str(value)
    return None


def compute_document_identity(
    record: Record,
    *,
    adapter_hint: str | None,
    explicit_column_value: str | None,
    explicit_column_is_source_identifier: bool,
    granularity: Granularity,
    chunk_level1_id: str | None,
) -> DocumentIdentityResult:
    """First (highest priority) available document-identity level for a record."""

    if explicit_column_value is not None:
        level = 2 if explicit_column_is_source_identifier else 1
        return DocumentIdentityResult(
            level=level, document_id=explicit_column_value, confidence="HIGH"
        )

    metadata_value = _metadata_document_value(record)
    if metadata_value is not None:
        return DocumentIdentityResult(
            level=3, document_id=metadata_value, confidence="MEDIUM"
        )

    if granularity == "DOCUMENT" and chunk_level1_id is not None:
        return DocumentIdentityResult(
            level=4, document_id=chunk_level1_id, confidence="HIGH"
        )

    if record.normalized_text_hash is not None:
        return DocumentIdentityResult(
            level=5, document_id="cid:" + record.normalized_text_hash, confidence="LOW"
        )

    digest = getattr(record, "_embedding_digest_hex", None)
    if digest is not None:
        return DocumentIdentityResult(
            level=5, document_id="cid:" + digest, confidence="LOW"
        )

    raise UndefinedCaseError(case="document identity")


def detect_granularity(
    records: Sequence[Record], *, forced: Granularity | None
) -> tuple[Granularity, GranularitySource]:
    """Classifier for CHUNK vs DOCUMENT granularity (levels 1-3 and 5 only)."""

    if forced is not None:
        return forced, "forced"

    results = [
        compute_document_identity(
            record,
            adapter_hint=_identity_hint(record),
            explicit_column_value=record.document_id,
            explicit_column_is_source_identifier=False,
            granularity="CHUNK",
            chunk_level1_id=None,
        )
        for record in records
    ]

    if all(result.level == 5 for result in results):
        return "DOCUMENT", "assumed"

    counts: Counter[str] = Counter()
    for record, result in zip(records, results):
        if result.level <= 3:
            counts[result.document_id] += 1

    repeated = any(count > 1 for count in counts.values())
    chunk_indexed = any(
        record.chunk_index is not None and record.chunk_index > 0 for record in records
    )
    if repeated or chunk_indexed:
        return "CHUNK", "detected"
    return "DOCUMENT", "detected"


def _digest_bytes(record: Record) -> bytes:
    cached = getattr(record, "_digest_bytes", None)
    if cached is not None:
        return cached
    digest = record_digest(
        record_id=record.record_id,
        document_id=record.document_id,
        raw_text_hash=record.raw_text_hash,
        normalized_text_hash=record.normalized_text_hash,
        chunk_index=record.chunk_index,
        created_at=record.created_at,
        updated_at=record.updated_at,
        metadata_canonical=canonical_json(record.metadata),
        embedding_model=record.embedding.model,
        embedding_digest_hex=getattr(record, "_embedding_digest_hex", None),
    )
    record._digest_bytes = digest
    return digest


def disambiguate_duplicate_ids(
    records: list[Record],
    *,
    on_duplicate_id: Literal["suffix", "error"],
    sink: DiagnosticSink,
) -> list[Record]:
    """Rename duplicate record ids to id#k ordered by record digest."""

    groups: dict[object, list[Record]] = {}
    order: list[object] = []
    for record in records:
        key = record.record_id
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)

    duplicate_groups = 0
    for key in order:
        group = groups[key]
        if len(group) < 2:
            continue
        duplicate_groups += 1
        for rank, record in enumerate(sorted(group, key=_digest_bytes)):
            record.record_id = f"{key}#{rank}"
            # The cached digest covers the pre-rename record id; the rename
            # invalidates it, otherwise snapshot ids verified against final
            # records mismatch (audit finding F-07).
            if getattr(record, "_digest_bytes", None) is not None:
                del record._digest_bytes

    if duplicate_groups:
        sink.add(Code.IDENTITY_WARNING, f"{duplicate_groups} duplicate record ids")
        if on_duplicate_id == "error":
            raise InvalidInputError(message=f"{duplicate_groups} duplicate record ids")

    return records


def assign_identity(
    records: list[Record],
    *,
    side: SideOptions,
    sink: DiagnosticSink,
    granularity_override: Granularity | None,
    on_duplicate_id: Literal["suffix", "error"] = "suffix",
) -> tuple[list[Record], Granularity, GranularitySource]:
    """Orchestrate hashes -> documents -> granularity -> chunks -> disambiguation."""

    for record in records:
        compute_text_hashes(record)
        compute_embedding_digest_field(record)

    granularity, granularity_source = detect_granularity(
        records, forced=granularity_override
    )
    if granularity_source == "assumed":
        sink.add(Code.GRANULARITY_ASSUMED, "document granularity assumed")

    level1: dict[int, str | None] = {}
    for record in records:
        keys = all_chunk_identity_keys(
            record,
            document_identity_confidence=None,
            duplicate_rank_within_document=None,
        )
        key = keys.get(1)
        level1[id(record)] = key[len("c1:"):] if key is not None else None

    document_confidence: dict[str, str] = {}
    for record in records:
        result = compute_document_identity(
            record,
            adapter_hint=_identity_hint(record),
            explicit_column_value=record.document_id,
            explicit_column_is_source_identifier=False,
            granularity=granularity,
            chunk_level1_id=level1[id(record)],
        )
        record.document_id = result.document_id
        record._document_identity_level = result.level
        current = document_confidence.get(result.document_id)
        if current is None or _CONFIDENCE_RANK[result.confidence] < _CONFIDENCE_RANK[current]:
            document_confidence[result.document_id] = result.confidence

    ranks: dict[int, int | None] = {}
    groups: dict[tuple, list[Record]] = {}
    for record in records:
        if record.normalized_text_hash is None:
            ranks[id(record)] = None
            continue
        groups.setdefault((record.document_id, record.normalized_text_hash), []).append(
            record
        )
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda item: (
                item.chunk_index if item.chunk_index is not None else math.inf,
                _digest_bytes(item),
            ),
        )
        for rank, record in enumerate(ordered):
            ranks[id(record)] = rank

    for record in records:
        result = compute_chunk_identity(
            record,
            document_identity_confidence=document_confidence.get(record.document_id),
            duplicate_rank_within_document=ranks.get(id(record)),
        )
        record.identity_source = f"chunk_level_{result.level}"
        record.identity_confidence = result.confidence

    disambiguate_duplicate_ids(
        records, on_duplicate_id=on_duplicate_id, sink=sink
    )
    return records, granularity, granularity_source
