"""Document reconstruction and content diffing (Step 25)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from driftlens.analysis.compatibility import match_records_by_identity_key
from driftlens.analysis.health import Metric, make_metric
from driftlens.config.schema import Config, DocumentsCfg, TextCfg
from driftlens.errors import Code, DiagnosticSink, IncompatibleError
from driftlens.identity.documents import Granularity
from driftlens.identity.hashing import sha256_hex
from driftlens.schema.manifest import Manifest
from driftlens.schema.record import Record
from driftlens.schema.snapshot import DocumentTable, Snapshot
from driftlens.text.normalize import normalize_text
from driftlens.text.shingles import jaccard, shingles
from driftlens.text.tokenizer import tokenize

DocumentStatus = Literal[
    "ADDED",
    "REMOVED",
    "UNCHANGED",
    "FORMAT_ONLY_CHANGE",
    "REPLACED",
    "MODIFIED",
]
ChunkStatus = Literal[
    "UNCHANGED",
    "FORMAT_ONLY_CHANGE",
    "MOVED",
    "MODIFIED",
    "ADDED",
    "REMOVED",
    "SPLIT_PART",
    "MERGE_PART",
]


@dataclass
class ReconstructedDocument:
    document_id: str
    text: str | None
    document_raw_hash: str
    document_content_hash: str
    reconstruction_confidence: Literal["HIGH", "LOW"]
    order_unknown: bool


@dataclass
class DocumentDiffRow:
    document_id: str
    status: DocumentStatus
    metadata_changed: bool
    timestamp_inconsistent: bool
    reconstruction_confidence_v1: str | None
    reconstruction_confidence_v2: str | None
    similarity: float | None


@dataclass
class ComparisonLevelResult:
    level: int
    hash_kind: Literal["text", "embedding"]


@dataclass
class ChunkDiffResult:
    pairs: list[tuple[Record | None, Record | None, ChunkStatus]]
    unmatched_v1: list[Record]
    unmatched_v2: list[Record]
    # Step 43 extension point: per-document count of pass-1 (identity/hash)
    # matches, used to compute the RECHUNKED document flag. Additive and
    # defaulted so every existing construction site keeps working.
    pass1_match_counts: dict[str, int] = field(default_factory=dict)


def _content_hash(text: str) -> str:
    stripped = re.sub(r"\s+", "", normalize_text(text))
    return sha256_hex(stripped.encode("utf-8"))


def _embedding_digest(record: Record) -> str:
    return getattr(record, "_embedding_digest_hex", None) or ""


def _overlap_length(a: str, b: str, *, window: int, min_chars: int) -> int:
    max_length = min(len(a), len(b), window)
    for length in range(max_length, 0, -1):
        if a.endswith(b[:length]):
            return length if length >= min_chars else 0
    return 0


def _join_with_overlap(ordered: Sequence[Record], cfg: TextCfg) -> str:
    first = ordered[0]
    combined = first.text or ""
    previous = first.normalized_text
    if previous is None:
        previous = normalize_text(first.text or "")

    for record in ordered[1:]:
        current_raw = record.text or ""
        current = record.normalized_text
        if current is None:
            current = normalize_text(current_raw)
        overlap = _overlap_length(
            previous, current, window=cfg.overlap_window, min_chars=cfg.min_overlap_chars
        )
        if overlap:
            combined += current_raw[overlap:]
        else:
            combined += "\n" + current_raw
        previous = current
    return combined


def reconstruct_document_text(
    document_id: str,
    chunk_records: Sequence[Record],
    *,
    granularity: Granularity,
    cfg: TextCfg,
) -> ReconstructedDocument:
    """Reconstruct one document's text and raw/content hashes (section 4.6)."""
    if granularity == "DOCUMENT":
        record = chunk_records[0]
        text = record.text
        if text is not None:
            raw_hash = sha256_hex(text.encode("utf-8"))
            content_hash = _content_hash(text)
        else:
            raw_hash = sha256_hex(_embedding_digest(record).encode("utf-8"))
            content_hash = raw_hash
        return ReconstructedDocument(
            document_id=document_id,
            text=text,
            document_raw_hash=raw_hash,
            document_content_hash=content_hash,
            reconstruction_confidence="HIGH",
            order_unknown=False,
        )

    order_unknown = any(record.chunk_index is None for record in chunk_records)
    if order_unknown:
        ordered = sorted(chunk_records, key=lambda record: record.record_id)
    else:
        ordered = sorted(
            chunk_records, key=lambda record: (record.chunk_index or 0)
        )

    has_text = any(record.text is not None for record in chunk_records)
    if not has_text:
        digest_text = "".join(_embedding_digest(record) for record in ordered)
        raw_hash = sha256_hex(digest_text.encode("utf-8"))
        return ReconstructedDocument(
            document_id=document_id,
            text=None,
            document_raw_hash=raw_hash,
            document_content_hash=raw_hash,
            reconstruction_confidence="LOW" if order_unknown else "HIGH",
            order_unknown=order_unknown,
        )

    if order_unknown:
        combined = "\n".join(record.text or "" for record in ordered)
    else:
        combined = _join_with_overlap(ordered, cfg)

    return ReconstructedDocument(
        document_id=document_id,
        text=combined,
        document_raw_hash=sha256_hex(combined.encode("utf-8")),
        document_content_hash=_content_hash(combined),
        reconstruction_confidence="LOW" if order_unknown else "HIGH",
        order_unknown=order_unknown,
    )


def _common_levels(manifest: Manifest) -> set[int]:
    sets = [set(levels) for levels in manifest.chunk_identity_levels.values()]
    if not sets:
        return set()
    return set.intersection(*sets)


def determine_comparison_level(
    v1_manifest: Manifest,
    v2_manifest: Manifest,
    v1_records: Sequence[Record],
    v2_records: Sequence[Record],
) -> ComparisonLevelResult:
    """Best chunk-identity level common to every record in both snapshots."""
    both_common = _common_levels(v1_manifest) & _common_levels(v2_manifest)
    if not both_common:
        raise IncompatibleError(
            Code.IDENTITY_UNAVAILABLE,
            message=(
                "no chunk identity level is available for every record in both "
                "snapshots"
            ),
        )
    level = min(both_common)

    if level == 4:
        records = list(v1_records) + list(v2_records)
        hash_kind: Literal["text", "embedding"] = (
            "text" if records and all(r.text is not None for r in records) else "embedding"
        )
    else:
        hash_kind = "text"
    return ComparisonLevelResult(level=level, hash_kind=hash_kind)


def _worst_doc_level(records: Sequence[Record]) -> int | None:
    levels = [
        level
        for level in (
            getattr(record, "_document_identity_level", None) for record in records
        )
        if level is not None
    ]
    return max(levels) if levels else None


def check_doc_identity_mismatch(
    v1_records: Sequence[Record],
    v2_records: Sequence[Record],
    sink: DiagnosticSink,
) -> None:
    """Emit DOC_IDENTITY_MISMATCH when the snapshots' worst levels differ."""
    v1_worst = _worst_doc_level(v1_records)
    v2_worst = _worst_doc_level(v2_records)
    if v1_worst != v2_worst:
        sink.add(
            Code.DOC_IDENTITY_MISMATCH,
            f"document identity level {v1_worst} in v1 vs {v2_worst} in v2",
        )


def diff_documents(
    v1_docs: DocumentTable,
    v2_docs: DocumentTable,
    v1_recon: dict[str, ReconstructedDocument],
    v2_recon: dict[str, ReconstructedDocument],
    *,
    cfg: DocumentsCfg,
    metadata_changed_ids: set[str] | None = None,
) -> list[DocumentDiffRow]:
    """Classify every document id in the union of the two tables (section 5.1)."""
    union = list(v1_docs.rows)
    union.extend(doc_id for doc_id in v2_docs.rows if doc_id not in v1_docs.rows)
    changed_ids = metadata_changed_ids or set()
    rows: list[DocumentDiffRow] = []

    for document_id in union:
        in_v1 = document_id in v1_docs.rows
        in_v2 = document_id in v2_docs.rows
        recon1 = v1_recon.get(document_id)
        recon2 = v2_recon.get(document_id)
        similarity: float | None = None

        if in_v2 and not in_v1:
            status: DocumentStatus = "ADDED"
        elif in_v1 and not in_v2:
            status = "REMOVED"
        else:
            assert recon1 is not None and recon2 is not None
            if recon1.document_raw_hash == recon2.document_raw_hash:
                status = "UNCHANGED"
            elif recon1.document_content_hash == recon2.document_content_hash:
                status = "FORMAT_ONLY_CHANGE"
            elif recon1.text is not None and recon2.text is not None:
                similarity = jaccard(
                    shingles(tokenize(recon1.text), 5),
                    shingles(tokenize(recon2.text), 5),
                )
                status = "REPLACED" if similarity < cfg.replace_threshold else "MODIFIED"
            else:
                status = "MODIFIED"

        rows.append(
            DocumentDiffRow(
                document_id=document_id,
                status=status,
                metadata_changed=document_id in changed_ids,
                timestamp_inconsistent=False,
                reconstruction_confidence_v1=(
                    recon1.reconstruction_confidence if recon1 is not None else None
                ),
                reconstruction_confidence_v2=(
                    recon2.reconstruction_confidence if recon2 is not None else None
                ),
                similarity=similarity,
            )
        )
    return rows


def _raw_hashes_equal(a: Record, b: Record) -> bool:
    if a.raw_text_hash is not None or b.raw_text_hash is not None:
        return a.raw_text_hash == b.raw_text_hash
    return _embedding_digest(a) == _embedding_digest(b)


def classify_chunks(
    v1_records: Sequence[Record],
    v2_records: Sequence[Record],
    *,
    comparison_level: ComparisonLevelResult,
    shared_documents: set[str],
) -> ChunkDiffResult:
    """v0.1 chunk classification: orders 1-3, the identity half of 4, and 6."""
    v1_filtered = [
        record for record in v1_records if record.document_id in shared_documents
    ]
    v2_filtered = [
        record for record in v2_records if record.document_id in shared_documents
    ]
    matched = match_records_by_identity_key(
        v1_filtered, v2_filtered, comparison_level
    )
    matched_v1 = {id(first) for first, _second in matched}
    matched_v2 = {id(second) for _first, second in matched}

    pairs: list[tuple[Record | None, Record | None, ChunkStatus]] = []
    unmatched_v1: list[Record] = []
    unmatched_v2: list[Record] = []
    pass1_match_counts: dict[str, int] = {}

    for first, second in matched:
        if _raw_hashes_equal(first, second):
            index_equal = (
                first.chunk_index == second.chunk_index
                or first.chunk_index is None
                or second.chunk_index is None
            )
            status: ChunkStatus = "UNCHANGED" if index_equal else "MOVED"
        elif first.normalized_text_hash == second.normalized_text_hash:
            status = "FORMAT_ONLY_CHANGE"
        else:
            status = "MODIFIED"
        pairs.append((first, second, status))
        if first.document_id is not None:
            pass1_match_counts[first.document_id] = (
                pass1_match_counts.get(first.document_id, 0) + 1
            )

    remaining_v2 = sorted(
        (record for record in v2_filtered if id(record) not in matched_v2),
        key=lambda record: record.record_id,
    )
    for record in remaining_v2:
        pairs.append((None, record, "ADDED"))
        unmatched_v2.append(record)

    remaining_v1 = sorted(
        (record for record in v1_filtered if id(record) not in matched_v1),
        key=lambda record: record.record_id,
    )
    for record in remaining_v1:
        pairs.append((record, None, "REMOVED"))
        unmatched_v1.append(record)

    return ChunkDiffResult(
        pairs=pairs,
        unmatched_v1=unmatched_v1,
        unmatched_v2=unmatched_v2,
        pass1_match_counts=pass1_match_counts,
    )


def _reconstruct_all(
    snapshot: Snapshot, cfg: Config
) -> dict[str, ReconstructedDocument]:
    grouped: dict[str, list[Record]] = {}
    for record in snapshot.records:
        if record.document_id is None:
            continue
        grouped.setdefault(record.document_id, []).append(record)

    recon: dict[str, ReconstructedDocument] = {}
    for document_id in snapshot.documents.rows:
        records = grouped.get(document_id, [])
        if not records:
            continue
        recon[document_id] = reconstruct_document_text(
            document_id, records, granularity=snapshot.manifest.granularity, cfg=cfg.text
        )
    return recon


def _content_metrics(
    rows: Sequence[DocumentDiffRow],
    chunk_result: ChunkDiffResult,
    cfg: Config,
) -> list[Metric]:
    document_counts = Counter(row.status for row in rows)
    numerator = (
        document_counts["ADDED"]
        + document_counts["REMOVED"]
        + document_counts["MODIFIED"]
        + document_counts["REPLACED"]
    )
    total = len(rows)
    rate = numerator / total if total else 0.0

    metrics = [
        make_metric(
            "content_change_rate",
            value=rate,
            threshold=cfg.drift.get("content_change_rate"),
        )
    ]
    for status, name in (
        ("ADDED", "documents_added"),
        ("REMOVED", "documents_removed"),
        ("MODIFIED", "documents_modified"),
        ("REPLACED", "documents_replaced"),
        ("FORMAT_ONLY_CHANGE", "documents_format_only_change"),
        ("UNCHANGED", "documents_unchanged"),
    ):
        metrics.append(make_metric(name, value=document_counts[status], threshold=None))

    chunk_counts = Counter(status for _first, _second, status in chunk_result.pairs)
    for status, name in (
        ("ADDED", "chunks_added"),
        ("REMOVED", "chunks_removed"),
        ("MODIFIED", "chunks_modified"),
        ("MOVED", "chunks_moved"),
        ("FORMAT_ONLY_CHANGE", "chunks_format_only_change"),
        ("UNCHANGED", "chunks_unchanged"),
    ):
        metrics.append(make_metric(name, value=chunk_counts[status], threshold=None))
    return metrics


def compare_content(
    v1: Snapshot,
    v2: Snapshot,
    *,
    cfg: Config,
    sink: DiagnosticSink,
) -> tuple[list[DocumentDiffRow], ChunkDiffResult, list[Metric]]:
    """Orchestrate document reconstruction, diffing and content metrics."""
    check_doc_identity_mismatch(v1.records, v2.records, sink)
    comparison_level = determine_comparison_level(
        v1.manifest, v2.manifest, v1.records, v2.records
    )

    v1_best = min(_common_levels(v1.manifest)) if _common_levels(v1.manifest) else comparison_level.level
    v2_best = min(_common_levels(v2.manifest)) if _common_levels(v2.manifest) else comparison_level.level
    if comparison_level.level > v1_best or comparison_level.level > v2_best:
        sink.add(
            Code.CHUNK_IDENTITY_MISMATCH,
            (
                f"comparison level {comparison_level.level} is lower than a "
                f"snapshot's own best level ({v1_best} in v1, {v2_best} in v2)"
            ),
        )

    v1_recon = _reconstruct_all(v1, cfg)
    v2_recon = _reconstruct_all(v2, cfg)
    rows = diff_documents(
        v1.documents, v2.documents, v1_recon, v2_recon, cfg=cfg.documents
    )

    if v1.manifest.granularity != v2.manifest.granularity:
        sink.add(
            Code.GRANULARITY_MISMATCH,
            (
                f"granularity {v1.manifest.granularity} in v1 vs "
                f"{v2.manifest.granularity} in v2"
            ),
        )
        chunk_result = ChunkDiffResult(pairs=[], unmatched_v1=[], unmatched_v2=[])
    elif v1.manifest.granularity == "CHUNK":
        shared = set(v1.documents.rows) & set(v2.documents.rows)
        chunk_result = classify_chunks(
            v1.records,
            v2.records,
            comparison_level=comparison_level,
            shared_documents=shared,
        )
    else:
        chunk_result = ChunkDiffResult(pairs=[], unmatched_v1=[], unmatched_v2=[])

    metrics = _content_metrics(rows, chunk_result, cfg)
    return rows, chunk_result, metrics
