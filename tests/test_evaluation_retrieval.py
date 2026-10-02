"""Retrieval-mode selection and comparison tests (Step 40)."""

from __future__ import annotations

import numpy as np
import pytest

from driftlens.analysis.compatibility import CompatibilityResult
from driftlens.analysis.content import ComparisonLevelResult
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.config.schema import default_config
from driftlens.errors import Code, DiagnosticSink, RetrieverUnavailableError
from driftlens.evaluation.retrieval import (
    choose_retrieval_mode,
    run_retrieval_comparison,
)
from driftlens.schema.eval_set import EvalCase, EvalSet, QueryEmbeddings
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import (
    DocumentRow,
    DocumentTable,
    Snapshot,
    StatisticsAccumulator,
)

_COMPARISON_LEVEL = ComparisonLevelResult(level=4, hash_kind="text")


def _case(
    query_id: str = "q1",
    *,
    text: str = "alpha",
    vectors: bool = True,
    document_ids: tuple[str, ...] = ("d1",),
    chunk_ids: tuple[str, ...] = (),
) -> EvalCase:
    embeddings = None
    if vectors:
        embeddings = QueryEmbeddings(
            v1=np.asarray([1.0, 0.0], dtype=np.float64),
            v2=np.asarray([1.0, 0.0], dtype=np.float64),
        )
    return EvalCase(
        query_id=query_id,
        query_id_was_explicit=True,
        query=text,
        relevant_document_ids=list(document_ids),
        relevant_chunk_ids=list(chunk_ids),
        query_embeddings=embeddings,
    )


def _summary(*, similarity_metric: str | None = "cosine") -> EmbeddingSnapshotSummary:
    return EmbeddingSnapshotSummary(
        model="m",
        provider="p",
        dtype="float32",
        similarity_metric=similarity_metric,
        similarity_metric_derived=False,
        normalized=False,
        confidence={},
        invalid_count=0,
        zero_count=0,
        coverage=1.0,
        norm_summary=None,
        embeddings_unavailable=False,
    )


def _compat(state: str = "COMPATIBLE") -> CompatibilityResult:
    return CompatibilityResult(
        state=state,  # type: ignore[arg-type]
        space_test=None,
        label_conflict=False,
        dimension_v1=2,
        dimension_v2=2,
        metric_v1="cosine",
        metric_v2="cosine",
        model_v1="m",
        model_v2="m",
    )


def _manifest(record_ids: list[str]) -> Manifest:
    return Manifest(
        schema_version="1.0",
        snapshot_id="snap",
        content_digest="digest",
        normalization_version="nfc-ws-1",
        tokenizer_id="simple-v1",
        text_params={"min_overlap_chars": 16, "overlap_window": 2000, "casefold": True},
        driftlens_version="0.1.0",
        created_at="2026-01-01T00:00:00+00:00",
        source_type="jsonl",
        granularity="CHUNK",
        granularity_source="detected",
        document_count=len(record_ids),
        chunk_count=len(record_ids),
        embedding_dimension=None,
        embedding_dtype=None,
        embedding_model=None,
        embedding_model_counts={},
        embedding_provider=None,
        embedding_metadata_confidence={},
        similarity_metric="cosine",
        embedding_normalized=None,
        chunk_identity_levels={record_id: [4] for record_id in record_ids},
        identity_summary={"HIGH": len(record_ids), "MEDIUM": 0, "LOW": 0},
        field_mapping=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _record(
    record_id: str,
    *,
    text: str | None = None,
    document_id: str | None = None,
) -> Record:
    return Record(
        record_id=record_id,
        document_id=document_id if document_id is not None else record_id,
        text=text,
        normalized_text=text,
        embedding=EmbeddingInfo(
            vector=None,
            dtype=None,
            dimension=None,
            norm=None,
            validity=None,
            model=None,
            provider=None,
            similarity_metric=None,
            normalized=None,
        ),
    )


def _snapshot(records: list[Record], *, document_ids: tuple[str, ...] = ()) -> Snapshot:
    rows = {
        document_id: DocumentRow(
            document_id=document_id,
            chunk_count=1,
            identity_confidence="HIGH",
        )
        for document_id in document_ids
    }
    return Snapshot(
        manifest=_manifest([record.record_id for record in records]),
        records=list(records),
        documents=DocumentTable(rows=rows),
        statistics=StatisticsAccumulator(),
    )


def _run(v1: Snapshot, v2: Snapshot, eval_set: EvalSet, sink: DiagnosticSink):
    return run_retrieval_comparison(
        v1,
        v2,
        eval_set,
        cfg=default_config(),
        compat=_compat(),
        v1_summary=_summary(),
        v2_summary=_summary(),
        comparison_level=_COMPARISON_LEVEL,
        sink=sink,
    )


def test_choose_mode_auto_prefers_dense_when_fully_available() -> None:
    modes = choose_retrieval_mode(
        EvalSet(cases=[_case()]),
        requested="auto",
        v1_summary=_summary(),
        v2_summary=_summary(),
        compat=_compat(),
    )
    assert modes == ["dense"]


def test_choose_mode_auto_falls_back_to_lexical() -> None:
    eval_set = EvalSet(cases=[_case("q1", vectors=True), _case("q2", vectors=False)])
    modes = choose_retrieval_mode(
        eval_set,
        requested="auto",
        v1_summary=_summary(),
        v2_summary=_summary(),
        compat=_compat(),
    )
    assert modes == ["lexical"]


def test_choose_mode_dense_explicit_raises_when_unavailable() -> None:
    with pytest.raises(RetrieverUnavailableError) as excinfo:
        choose_retrieval_mode(
            EvalSet(cases=[_case(vectors=False)]),
            requested="dense",
            v1_summary=_summary(),
            v2_summary=_summary(),
            compat=_compat(),
        )
    assert excinfo.value.exit_code == 3


def test_choose_mode_auto_neither_runnable_returns_empty_not_raises() -> None:
    modes = choose_retrieval_mode(
        EvalSet(cases=[_case(vectors=False)]),
        requested="auto",
        v1_summary=_summary(),
        v2_summary=_summary(),
        compat=_compat(),
        lexical_runnable=False,
    )
    assert modes == []


def test_choose_mode_both_partial_availability() -> None:
    modes = choose_retrieval_mode(
        EvalSet(cases=[_case(vectors=False)]),
        requested="both",
        v1_summary=_summary(),
        v2_summary=_summary(),
        compat=_compat(),
    )
    assert modes == ["lexical"]


def test_dense_unavailable_when_space_mixed() -> None:
    with pytest.raises(RetrieverUnavailableError):
        choose_retrieval_mode(
            EvalSet(cases=[_case()]),
            requested="dense",
            v1_summary=_summary(),
            v2_summary=_summary(),
            compat=_compat("MIXED_MODELS"),
        )


def test_run_retrieval_comparison_whole_set_skipped() -> None:
    eval_set = EvalSet(cases=[_case(vectors=False)])
    v1 = _snapshot([_record("r1", document_id="d1")], document_ids=("d1",))
    v2 = _snapshot([_record("r1", document_id="d1")], document_ids=("d1",))
    sink = DiagnosticSink()
    result = _run(v1, v2, eval_set, sink)
    assert len(result.modes) == 1
    assert result.modes[0].retriever is None
    assert result.modes[0].skipped_reason == "RETRIEVER_UNAVAILABLE"
    assert result.modes[0].metrics == {}


def test_run_retrieval_comparison_low_power_diagnostic() -> None:
    eval_set = EvalSet(cases=[_case(vectors=False)])
    v1 = _snapshot(
        [_record("r1", text="alpha beta", document_id="d1")], document_ids=("d1",)
    )
    v2 = _snapshot(
        [_record("r1", text="alpha beta", document_id="d1")], document_ids=("d1",)
    )
    sink = DiagnosticSink()
    result = _run(v1, v2, eval_set, sink)
    assert sink.has(Code.LOW_POWER)
    assert [mode.retriever for mode in result.modes] == ["lexical"]
