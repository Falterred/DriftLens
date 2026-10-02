"""Retrieval-mode selection and comparison orchestration (Step 40, sections 11.1/11.3/11.4)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from driftlens.analysis.compatibility import CompatibilityResult
from driftlens.analysis.content import ComparisonLevelResult
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.analysis.health import Metric, make_metric
from driftlens.config.schema import Config
from driftlens.errors import (
    Code,
    DiagnosticSink,
    InvalidInputError,
    RetrieverUnavailableError,
)
from driftlens.evaluation.ground_truth import (
    CaseState,
    ResolvedCase,
    resolve_eval_set,
)
from driftlens.evaluation.metrics import (
    average_top_k_similarity,
    chunk_recall_at_k,
    hit_rate_at_k,
    mrr_at_kmax,
    recall_at_k,
    to_document_level,
)
from driftlens.evaluation.retrievers import BM25Index, DenseProvidedIndex, RankedResult
from driftlens.schema.eval_set import EvalSet
from driftlens.schema.snapshot import Snapshot

RetrievalMode = Literal["lexical", "dense"]

# Section 8.4/11.1: dense recall is not comparable when either snapshot's
# embedding space is mixed.
_SPACE_MIXED_STATES = ("MIXED_MODELS", "PARTIAL_MIGRATION")

# Section 11.4: below this many comparable cases the retrieval block is
# reported with a LOW_POWER diagnostic.
LOW_POWER_MIN_CASES = 30

_RETRIEVER_UNAVAILABLE = "RETRIEVER_UNAVAILABLE"


@dataclass
class PerModeResult:
    """One retriever's metrics across the whole eval set."""

    retriever: RetrievalMode | None
    metrics: dict[str, Metric]
    case_states: dict[CaseState, int]
    skipped_reason: str | None = None


@dataclass
class RetrievalComparisonResult:
    """The per-mode results of one retrieval comparison run."""

    modes: list[PerModeResult]


@dataclass
class _CaseRetrieval:
    """Per-case ranked output for both snapshots."""

    document_ranked_v1: list[str]
    document_ranked_v2: list[str]
    chunk_ranked_v1: list[RankedResult]
    chunk_ranked_v2: list[RankedResult]
    present_document_ids: set[str]
    present_chunk_ids: set[str]


def choose_retrieval_mode(
    eval_set: EvalSet,
    *,
    requested: str,
    v1_summary: EmbeddingSnapshotSummary,
    v2_summary: EmbeddingSnapshotSummary,
    compat: CompatibilityResult,
    lexical_runnable: bool = True,
) -> list[RetrievalMode]:
    """Decide which retrievers run for the whole eval set (section 11.1).

    ``lexical_runnable`` is pre-computed by the orchestrator from the two
    snapshots (lexical retrieval needs at least one text-bearing record per
    snapshot) so this function stays summary-based.
    """

    dense_available = (
        v1_summary.similarity_metric is not None
        and v2_summary.similarity_metric is not None
        and compat.state not in _SPACE_MIXED_STATES
    )
    every_case_has_both_vectors = all(
        case.query_embeddings is not None
        and case.query_embeddings.v1 is not None
        and case.query_embeddings.v2 is not None
        for case in eval_set.cases
    )
    dense_runnable = dense_available and every_case_has_both_vectors

    if requested == "auto":
        # Dense is preferred when it can run for every case; otherwise lexical.
        # Neither runnable is the DEGRADED/RETRIEVAL-SKIPPED case, not fatal.
        if dense_runnable:
            return ["dense"]
        if lexical_runnable:
            return ["lexical"]
        return []
    if requested == "lexical":
        if not lexical_runnable:
            raise RetrieverUnavailableError(
                message="lexical retrieval requires text on both snapshots"
            )
        return ["lexical"]
    if requested == "dense":
        if not dense_runnable:
            raise RetrieverUnavailableError(
                message=(
                    "dense retrieval requires query vectors for every case and a "
                    "compatible, non-space-mixed embedding space"
                )
            )
        return ["dense"]
    if requested == "both":
        return [
            mode
            for mode in ("lexical", "dense")
            if (mode == "lexical" and lexical_runnable)
            or (mode == "dense" and dense_runnable)
        ]
    raise InvalidInputError(message=f"unknown retrieval mode {requested!r}")


def _has_text(snapshot: Snapshot) -> bool:
    return any(
        record.text is not None or record.normalized_text is not None
        for record in snapshot.records
    )


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _metric(name: str, v1_value: float | None, v2_value: float | None) -> Metric:
    change = (
        None if v1_value is None or v2_value is None else v2_value - v1_value
    )
    return make_metric(
        name,
        value=v2_value,
        threshold=None,
        v1=v1_value,
        v2=v2_value,
        change=change,
    )


def _search_both(
    mode: RetrievalMode,
    case: ResolvedCase,
    *,
    index_v1,
    index_v2,
    top_k: int,
    sink: DiagnosticSink,
) -> tuple[list[RankedResult], list[RankedResult]] | None:
    if mode == "lexical":
        return (
            index_v1.search(case.case.query, top_k=top_k),
            index_v2.search(case.case.query, top_k=top_k),
        )
    embeddings = case.case.query_embeddings
    if embeddings is None or embeddings.v1 is None or embeddings.v2 is None:
        # Defensive: mode selection only picks dense when every case has both
        # vectors, so this branches only if a caller bypasses it.
        sink.add(
            Code.NO_QUERY_VECTOR,
            f"case {case.case.query_id} has no query vector for both snapshots",
        )
        return None
    return (
        index_v1.search(embeddings.v1, top_k=top_k),
        index_v2.search(embeddings.v2, top_k=top_k),
    )


def _evaluate_mode(
    mode: RetrievalMode,
    *,
    v1: Snapshot,
    v2: Snapshot,
    comparable: list[ResolvedCase],
    case_states: dict[CaseState, int],
    cfg: Config,
    v1_summary: EmbeddingSnapshotSummary,
    v2_summary: EmbeddingSnapshotSummary,
    sink: DiagnosticSink,
) -> PerModeResult:
    if mode == "lexical":
        index_v1 = BM25Index(
            v1.records, cfg=cfg.retrieval.bm25, casefold=cfg.text.casefold
        )
        index_v2 = BM25Index(
            v2.records, cfg=cfg.retrieval.bm25, casefold=cfg.text.casefold
        )
    else:
        index_v1 = DenseProvidedIndex(
            v1.records, metric=v1_summary.similarity_metric  # type: ignore[arg-type]
        )
        index_v2 = DenseProvidedIndex(
            v2.records, metric=v2_summary.similarity_metric  # type: ignore[arg-type]
        )

    kmax = max(cfg.retrieval.k)
    rows: list[_CaseRetrieval] = []
    empty_v2 = 0
    for case in comparable:
        ranked = _search_both(
            mode,
            case,
            index_v1=index_v1,
            index_v2=index_v2,
            top_k=kmax,
            sink=sink,
        )
        if ranked is None:
            continue
        chunk_ranked_v1, chunk_ranked_v2 = ranked
        rows.append(
            _CaseRetrieval(
                document_ranked_v1=to_document_level(chunk_ranked_v1).document_ranked,
                document_ranked_v2=to_document_level(chunk_ranked_v2).document_ranked,
                chunk_ranked_v1=chunk_ranked_v1,
                chunk_ranked_v2=chunk_ranked_v2,
                present_document_ids=case.present_document_ids,
                present_chunk_ids=case.present_chunk_ids,
            )
        )
        if not chunk_ranked_v2:
            empty_v2 += 1

    metrics: dict[str, Metric] = {}
    for k in cfg.retrieval.k:
        metrics[f"recall@{k}"] = _metric(
            f"recall@{k}",
            _mean([recall_at_k(row.document_ranked_v1, row.present_document_ids, k) for row in rows]),
            _mean([recall_at_k(row.document_ranked_v2, row.present_document_ids, k) for row in rows]),
        )
        metrics[f"hit_rate@{k}"] = _metric(
            f"hit_rate@{k}",
            _mean([hit_rate_at_k(row.document_ranked_v1, row.present_document_ids, k) for row in rows]),
            _mean([hit_rate_at_k(row.document_ranked_v2, row.present_document_ids, k) for row in rows]),
        )
        metrics[f"chunk_recall@{k}"] = _metric(
            f"chunk_recall@{k}",
            _mean([chunk_recall_at_k(row.chunk_ranked_v1, row.present_chunk_ids, k) for row in rows]),
            _mean([chunk_recall_at_k(row.chunk_ranked_v2, row.present_chunk_ids, k) for row in rows]),
        )
    metrics["mrr"] = _metric(
        "mrr",
        _mean([mrr_at_kmax(row.chunk_ranked_v1, row.present_document_ids, kmax) for row in rows]),
        _mean([mrr_at_kmax(row.chunk_ranked_v2, row.present_document_ids, kmax) for row in rows]),
    )
    metrics["avg_top_k_similarity"] = _metric(
        "avg_top_k_similarity",
        _mean([average_top_k_similarity(row.chunk_ranked_v1, kmax) for row in rows]),
        _mean([average_top_k_similarity(row.chunk_ranked_v2, kmax) for row in rows]),
    )
    empty_rate = empty_v2 / len(rows) if rows else None
    metrics["empty_rate"] = make_metric(
        "empty_rate",
        value=empty_rate,
        threshold=None,
        v1=None,
        v2=empty_rate,
        change=None,
    )
    return PerModeResult(
        retriever=mode,
        metrics=metrics,
        case_states=case_states,
        skipped_reason=None,
    )


def run_retrieval_comparison(
    v1: Snapshot,
    v2: Snapshot,
    eval_set: EvalSet,
    *,
    cfg: Config,
    compat: CompatibilityResult,
    v1_summary: EmbeddingSnapshotSummary,
    v2_summary: EmbeddingSnapshotSummary,
    comparison_level: ComparisonLevelResult,
    sink: DiagnosticSink,
) -> RetrievalComparisonResult:
    """Resolve ground truth and run every selected retriever (section 11.4)."""

    resolved_cases = resolve_eval_set(
        eval_set,
        v1_documents=v1.documents,
        v2_documents=v2.documents,
        v1_records_by_id={record.record_id: record for record in v1.records},
        v2_records_by_id={record.record_id: record for record in v2.records},
        comparison_level=comparison_level,
    )
    comparable = [
        case for case in resolved_cases if case.state in ("VALID", "PARTIAL")
    ]
    case_states = Counter(case.state for case in resolved_cases)
    if len(comparable) < LOW_POWER_MIN_CASES:
        sink.add(
            Code.LOW_POWER,
            f"{len(comparable)} comparable case(s) is below the {LOW_POWER_MIN_CASES}-case power floor",
        )

    modes = choose_retrieval_mode(
        eval_set,
        requested=cfg.retrieval.mode,
        v1_summary=v1_summary,
        v2_summary=v2_summary,
        compat=compat,
        lexical_runnable=_has_text(v1) and _has_text(v2),
    )
    if not modes:
        return RetrievalComparisonResult(
            modes=[
                PerModeResult(
                    retriever=None,
                    metrics={},
                    case_states=case_states,
                    skipped_reason=_RETRIEVER_UNAVAILABLE,
                )
            ]
        )
    return RetrievalComparisonResult(
        modes=[
            _evaluate_mode(
                mode,
                v1=v1,
                v2=v2,
                comparable=comparable,
                case_states=case_states,
                cfg=cfg,
                v1_summary=v1_summary,
                v2_summary=v2_summary,
                sink=sink,
            )
            for mode in modes
        ]
    )
