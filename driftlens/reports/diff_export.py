"""Diff and breakdown export writers (Step 41).

Each writer emits a self-contained Parquet artifact for one comparison
surface.  Identifiers are exported verbatim here; the report layer is
responsible for redaction.  Text excerpts are opt-in via the diagnostic flag.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from driftlens.analysis.chunking import ChunkProfile, compute_chunk_profile
from driftlens.analysis.content import ChunkDiffResult, DocumentDiffRow
from driftlens.analysis.mmd import compute_mmd
from driftlens.config.schema import Config
from driftlens.evaluation.ground_truth import ResolvedCase
from driftlens.schema.snapshot import Snapshot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from driftlens.analysis.duplicates import ExtendedDuplicateResult

#: Diagnostic document excerpts are truncated to this many characters.
DIAGNOSTIC_EXCERPT_CHARS = 300

#: Number of metadata segment values shown before collapsing into "other".
BREAKDOWN_TOP_VALUES = 20

#: Minimum records per side before a breakdown segment computes MMD.
BREAKDOWN_MMD_MIN_RECORDS = 500

_DOCUMENT_TIER = {
    "REPLACED": 4,
    "MODIFIED": 3,
    "ADDED": 2,
    "REMOVED": 2,
    "FORMAT_ONLY_CHANGE": 1,
    "UNCHANGED": 0,
}


def _string(values: list[Any]) -> pa.Array:
    return pa.array(values, type=pa.string())


def _int32(values: list[Any]) -> pa.Array:
    return pa.array(values, type=pa.int32())


def _bool(values: list[Any]) -> pa.Array:
    return pa.array(values, type=pa.bool_())


def _float64(values: list[Any]) -> pa.Array:
    return pa.array(values, type=pa.float64())


def _string_list(values: list[Any]) -> pa.Array:
    return pa.array(values, type=pa.list_(pa.string()))


def _write_table(columns: dict[str, pa.Array], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _excerpt(text: str | None) -> str | None:
    if not text:
        return None
    return text[:DIAGNOSTIC_EXCERPT_CHARS]


def write_documents_diff(
    rows: list[DocumentDiffRow],
    *,
    path: Path,
    diagnostic: bool,
    v1_text_by_doc: dict[str, str] | None,
    v2_text_by_doc: dict[str, str] | None,
) -> None:
    """Write one row per document diff row.

    Text excerpt columns are added only in diagnostic mode; because the columns
    are omitted from the Arrow schema entirely, a default export carries no
    document text.  A missing document entry and an empty document both become
    Arrow null excerpts.
    """

    columns: dict[str, pa.Array] = {
        "document_id": _string([row.document_id for row in rows]),
        "status": _string([row.status for row in rows]),
        "metadata_changed": _bool([row.metadata_changed for row in rows]),
        "timestamp_inconsistent": _bool(
            [row.timestamp_inconsistent for row in rows]
        ),
        "reconstruction_confidence_v1": _string(
            [row.reconstruction_confidence_v1 for row in rows]
        ),
        "reconstruction_confidence_v2": _string(
            [row.reconstruction_confidence_v2 for row in rows]
        ),
        "similarity": _float64([row.similarity for row in rows]),
    }
    if diagnostic:
        source_v1 = v1_text_by_doc or {}
        source_v2 = v2_text_by_doc or {}
        columns["text_excerpt_v1"] = _string(
            [_excerpt(source_v1.get(row.document_id)) for row in rows]
        )
        columns["text_excerpt_v2"] = _string(
            [_excerpt(source_v2.get(row.document_id)) for row in rows]
        )
    _write_table(columns, path)


def write_chunks_diff(
    pairs: ChunkDiffResult, *, path: Path, diagnostic: bool
) -> None:
    """Write one row per chunk pair; a missing side stays Arrow null."""

    del diagnostic  # reserved for future chunk-position detail
    rows = pairs.pairs
    _write_table(
        {
            "v1_record_id": _string(
                [first.record_id if first is not None else None for first, _, _ in rows]
            ),
            "v2_record_id": _string(
                [
                    second.record_id if second is not None else None
                    for _, second, _ in rows
                ]
            ),
            "status": _string([status for _, _, status in rows]),
            "position_changed": _bool([status == "MOVED" for _, _, status in rows]),
            "group_id": _string([None for _ in rows]),
        },
        path,
    )


def write_duplicates_diff(
    result: "ExtendedDuplicateResult", *, path: Path, side: Literal["v1", "v2"]
) -> None:
    """Write one row per duplicate group across all three classes.

    The path is the diff output directory; the file is named
    duplicates_<side>.parquet so both sides can be written side by side.
    """

    group_rows: list[tuple[str, str, list[str]]] = []
    for class_name, groups in (
        ("exact", getattr(result, "groups", [])),
        ("near", getattr(result, "near_groups", [])),
        ("semantic", getattr(result, "semantic_groups", [])),
    ):
        for group in groups:
            group_rows.append(
                (
                    f"{class_name}:{group.representative_id}",
                    class_name,
                    list(group.member_ids),
                )
            )
    out_path = Path(path) / f"duplicates_{side}.parquet"
    _write_table(
        {
            "group_id": _string([row[0] for row in group_rows]),
            "class_": _string([row[1] for row in group_rows]),
            "member_ids": _string_list([row[2] for row in group_rows]),
        },
        out_path,
    )


def write_retrieval_cases_diff(
    resolved: list[ResolvedCase],
    *,
    path: Path,
    rank_v1: dict[str, int | None] | None = None,
    rank_v2: dict[str, int | None] | None = None,
) -> None:
    """Write one row per eval case.

    rank_v1/rank_v2 map a query id to the raw chunk rank of its first relevant
    document, or None when it is not retrievable within kmax.  The remaining
    columns are placeholders that Step 43 will populate.
    """

    ranks_v1 = rank_v1 or {}
    ranks_v2 = rank_v2 or {}
    _write_table(
        {
            "query_id": _string([item.case.query_id for item in resolved]),
            "state": _string([item.state for item in resolved]),
            "rank_first_relevant_v1": _int32(
                [ranks_v1.get(item.case.query_id) for item in resolved]
            ),
            "rank_first_relevant_v2": _int32(
                [ranks_v2.get(item.case.query_id) for item in resolved]
            ),
            "status_of_relevant_documents": _string([None for _ in resolved]),
            "partial_ground_truth_removed": _bool([None for _ in resolved]),
            "metadata_changed_on_relevant": _bool([None for _ in resolved]),
            "chunk_anomaly_flags": _string_list([None for _ in resolved]),
            "added_chunks_above_best": _int32([None for _ in resolved]),
            "compatibility_state": _string([None for _ in resolved]),
        },
        path,
    )


@dataclass
class BreakdownRow:
    """One metadata segment: status mix, chunk profile and (above 500) MMD."""

    segment_value: str
    is_other: bool
    document_count: int
    status_counts: dict[str, int]
    chunk_profile: dict[str, float] | None
    mmd: float | None


def _document_field_values(snapshot: Snapshot, field: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for record in snapshot.records:
        if record.document_id in values:
            continue
        value = record.metadata.get(field)
        if value is None:
            continue
        values[record.document_id] = str(value)
    return values


def _segment_records(snapshot: Snapshot, document_ids: set[str]) -> list[Any]:
    return [
        record for record in snapshot.records if record.document_id in document_ids
    ]


def _ok_vectors(records: list[Any]) -> tuple[np.ndarray, list[str]]:
    vectors: list[list[float]] = []
    ids: list[str] = []
    for record in records:
        embedding = record.embedding
        if embedding is None or embedding.validity != "ok":
            continue
        if embedding.vector is None:
            continue
        vector = np.asarray(embedding.vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        vectors.append(vector.tolist())
        ids.append(record.record_id)
    if not vectors:
        return np.empty((0, 0), dtype=np.float64), []
    return np.asarray(vectors, dtype=np.float64), ids


def _chunk_profile_dict(profile: ChunkProfile) -> dict[str, float] | None:
    out: dict[str, float] = {}
    for prefix, stats in (
        ("token_", profile.token_stats),
        ("char_", profile.char_stats),
        ("sentence_", profile.sentence_stats),
    ):
        if stats:
            for key, value in stats.items():
                out[f"{prefix}{key}"] = float(value)
    overlap = profile.overlap_estimate
    if overlap is not None:
        out["overlap_share"] = float(overlap.share_with_overlap)
        out["overlap_median_l"] = float(overlap.median_l)
        out["overlap_p95_l"] = float(overlap.p95_l)
    return out or None


def _segment_row(
    segment_value: str,
    *,
    is_other: bool,
    document_ids: set[str],
    document_count: int,
    document_diff: list[DocumentDiffRow],
    v1: Snapshot,
    v2: Snapshot,
    cfg: Config,
) -> BreakdownRow:
    status_counts = Counter(
        row.status for row in document_diff if row.document_id in document_ids
    )
    segment_v1 = _segment_records(v1, document_ids)
    segment_v2 = _segment_records(v2, document_ids)
    profile = _chunk_profile_dict(
        compute_chunk_profile(segment_v1, v1.documents, seed=cfg.analysis.seed)
    )
    mmd_value: float | None = None
    if (
        len(segment_v1) >= BREAKDOWN_MMD_MIN_RECORDS
        and len(segment_v2) >= BREAKDOWN_MMD_MIN_RECORDS
    ):
        v1_vectors, v1_ids = _ok_vectors(segment_v1)
        v2_vectors, v2_ids = _ok_vectors(segment_v2)
        result = compute_mmd(
            v1_vectors,
            v2_vectors,
            v1_ids,
            v2_ids,
            cfg=cfg.analysis,
            fixed_bandwidth=cfg.analysis.bandwidth,
            seed=cfg.analysis.seed,
        )
        mmd_value = result.mmd
    return BreakdownRow(
        segment_value=segment_value,
        is_other=is_other,
        document_count=document_count,
        status_counts=dict(status_counts),
        chunk_profile=profile,
        mmd=mmd_value,
    )


def compute_breakdown(
    field: str,
    *,
    v1: Snapshot,
    v2: Snapshot,
    document_diff: list[DocumentDiffRow],
    cfg: Config,
) -> list[BreakdownRow]:
    """Group documents by a metadata field, top 20 values plus an other row.

    This is the only surface allowed to print category labels for a field with
    more than PRIVACY_MAX_LABEL_CATEGORIES values; every other report surface
    keeps the hard cap.
    """

    values = _document_field_values(v2, field)
    counts: Counter[str] = Counter(values.values())
    top = counts.most_common(BREAKDOWN_TOP_VALUES)
    top_values = {value for value, _ in top}
    rows: list[BreakdownRow] = []
    for value, count in top:
        segment_docs = {
            doc_id for doc_id, doc_value in values.items() if doc_value == value
        }
        rows.append(
            _segment_row(
                value,
                is_other=False,
                document_ids=segment_docs,
                document_count=count,
                document_diff=document_diff,
                v1=v1,
                v2=v2,
                cfg=cfg,
            )
        )
    other_docs = {
        doc_id for doc_id, doc_value in values.items() if doc_value not in top_values
    }
    rows.append(
        _segment_row(
            "other",
            is_other=True,
            document_ids=other_docs,
            document_count=len(other_docs),
            document_diff=document_diff,
            v1=v1,
            v2=v2,
            cfg=cfg,
        )
    )
    return rows


def write_breakdown(rows: list[BreakdownRow], *, path: Path) -> None:
    """Write a breakdown table; nested fields are JSON-encoded strings."""

    _write_table(
        {
            "segment_value": _string([row.segment_value for row in rows]),
            "is_other": _bool([row.is_other for row in rows]),
            "document_count": _int32([row.document_count for row in rows]),
            "status_counts": _string(
                [json.dumps(row.status_counts) for row in rows]
            ),
            "chunk_profile": _string(
                [
                    None if row.chunk_profile is None else json.dumps(row.chunk_profile)
                    for row in rows
                ]
            ),
            "mmd": _float64([row.mmd for row in rows]),
        },
        path,
    )


def top_n_changed_documents(
    document_diff: list[DocumentDiffRow], *, n: int
) -> list[str]:
    """Return up to n changed document ids ordered by change severity."""

    changed = [row for row in document_diff if row.status != "UNCHANGED"]
    changed.sort(key=lambda row: (-_DOCUMENT_TIER[row.status], row.document_id))
    return [row.document_id for row in changed[:n]]
