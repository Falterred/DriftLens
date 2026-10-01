"""Embedding descriptor resolution and vector analysis (Step 22)."""

from __future__ import annotations

import numpy as np

from driftlens.analysis.embeddings import (
    analyze_embeddings,
    check_normalization,
    resolve_embedding_descriptors,
    validate_vector,
)
from driftlens.config.schema import EmbeddingsCfg, SideOptions
from driftlens.errors import Code, DiagnosticSink
from driftlens.schema.record import EmbeddingInfo, Record


def _record(
    vector=None,
    *,
    record_id="r",
    dtype=None,
    model=None,
    provider=None,
    metric=None,
    normalized=None,
    norm=None,
    validity=None,
    metadata=None,
):
    info = EmbeddingInfo(
        vector=vector,
        dtype=dtype,
        dimension=(len(vector) if vector is not None else None),
        norm=norm,
        validity=validity,
        model=model,
        provider=provider,
        similarity_metric=metric,
        normalized=normalized,
    )
    return Record(
        record_id=record_id, embedding=info, metadata=dict(metadata or {})
    )


def test_source_precedence_row_beats_file_level() -> None:
    record = _record(model="E1")
    result = resolve_embedding_descriptors(
        record,
        file_descriptors={"embedding_model": "OTHER"},
        side=SideOptions(),
    )
    assert result["model"].value == "E1"
    assert result["model"].confidence == "DECLARED"


def test_source_3_metadata_key_fallback() -> None:
    record = _record(metadata={"embedding_model": "E3"})
    result = resolve_embedding_descriptors(
        record, file_descriptors={}, side=SideOptions()
    )
    assert result["model"].value == "E3"
    assert result["model"].confidence == "DECLARED"


def test_source_5_fills_gap_only() -> None:
    record = _record()
    result = resolve_embedding_descriptors(
        record, file_descriptors={}, side=SideOptions(model="E5")
    )
    assert result["model"].value == "E5"
    assert result["model"].confidence == "USER_SUPPLIED"


def test_source_5_conflict_ignored() -> None:
    record = _record(
        np.array([1.0, 0.0], dtype=np.float32), dtype="float32", model="E1"
    )
    sink = DiagnosticSink()
    analyze_embeddings(
        [record],
        file_descriptors={},
        side=SideOptions(model="E5"),
        cfg=EmbeddingsCfg(),
        sink=sink,
    )
    assert record.embedding.model == "E1"
    assert sink.has(Code.MODEL_OVERRIDE_IGNORED)


def test_validate_vector_nan() -> None:
    assert validate_vector(np.array([1.0, np.nan]), "float32") == ("nan_inf", None)


def test_validate_vector_zero() -> None:
    assert validate_vector(np.array([0.0, 0.0]), "float32") == ("zero", 0.0)


def test_validate_vector_ok() -> None:
    assert validate_vector(np.array([3.0, 4.0]), "float32") == ("ok", 5.0)


def test_normalization_claim_invalid() -> None:
    records = []
    for index in range(100):
        norm = 2.0 if index < 5 else 1.0
        records.append(
            _record(
                np.array([norm, 0.0], dtype=np.float32),
                record_id=f"r{index}",
                dtype="float32",
                norm=norm,
                validity="ok",
            )
        )
    check = check_normalization(records, declared=True, cfg=EmbeddingsCfg())
    assert check.claim_invalid is True
    assert check.inferred is False


def test_normalization_undeclared_inferred_true() -> None:
    records = []
    for index in range(1000):
        norm = 0.5 if index < 5 else 1.0
        records.append(
            _record(
                np.array([norm, 0.0], dtype=np.float32),
                record_id=f"r{index}",
                dtype="float32",
                norm=norm,
                validity="ok",
            )
        )
    check = check_normalization(records, declared=None, cfg=EmbeddingsCfg())
    assert check.inferred is True


def test_normalization_int8_excluded() -> None:
    records = [
        _record(
            np.array([2, 0], dtype=np.int8),
            dtype="int8",
            norm=2.0,
            validity="ok",
        )
    ]
    check = check_normalization(records, declared=None, cfg=EmbeddingsCfg())
    assert check.inferred is None
    assert check.measured_share is None


def test_coverage_excludes_only_nan_inf() -> None:
    records = []
    for index in range(10):
        if index < 2:
            vector = np.array([np.nan, 1.0], dtype=np.float32)
        elif index == 2:
            vector = np.array([0.0, 0.0], dtype=np.float32)
        else:
            vector = np.array([1.0, 0.0], dtype=np.float32)
        records.append(_record(vector, record_id=f"r{index}", dtype="float32"))
    summary = analyze_embeddings(
        records,
        file_descriptors={},
        side=SideOptions(),
        cfg=EmbeddingsCfg(),
        sink=DiagnosticSink(),
    )
    assert summary.coverage == 0.8


def test_embeddings_unavailable_threshold() -> None:
    records = []
    for index in range(100):
        if index < 2:
            vector = np.array([np.nan, 1.0], dtype=np.float32)
        else:
            vector = np.array([1.0, 0.0], dtype=np.float32)
        records.append(_record(vector, record_id=f"r{index}", dtype="float32"))
    summary = analyze_embeddings(
        records,
        file_descriptors={},
        side=SideOptions(),
        cfg=EmbeddingsCfg(max_invalid_rate=0.01),
        sink=DiagnosticSink(),
    )
    assert summary.embeddings_unavailable is True


def test_metric_derivation_from_normalized_vectors() -> None:
    records = [
        _record(
            np.array([1.0, 0.0], dtype=np.float32),
            record_id=f"r{index}",
            dtype="float32",
        )
        for index in range(100)
    ]
    summary = analyze_embeddings(
        records,
        file_descriptors={},
        side=SideOptions(),
        cfg=EmbeddingsCfg(),
        sink=DiagnosticSink(),
    )
    assert summary.similarity_metric == "cosine"
    assert summary.similarity_metric_derived is True
    assert summary.confidence["similarity_metric"] == "INFERRED_FROM_DATA"


def test_metric_stays_unknown_when_not_normalized() -> None:
    records = [
        _record(
            np.array([3.0, 4.0], dtype=np.float32),
            record_id=f"r{index}",
            dtype="float32",
        )
        for index in range(10)
    ]
    summary = analyze_embeddings(
        records,
        file_descriptors={},
        side=SideOptions(),
        cfg=EmbeddingsCfg(),
        sink=DiagnosticSink(),
    )
    assert summary.similarity_metric is None


def test_norm_summary_excludes_invalid_and_zero() -> None:
    records = [
        _record(np.array([3.0, 4.0], dtype=np.float32), record_id="ok1", dtype="float32"),
        _record(np.array([0.0, 2.0], dtype=np.float32), record_id="ok2", dtype="float32"),
        _record(np.array([np.nan, 1.0], dtype=np.float32), record_id="bad", dtype="float32"),
        _record(np.array([0.0, 0.0], dtype=np.float32), record_id="zero", dtype="float32"),
    ]
    summary = analyze_embeddings(
        records,
        file_descriptors={},
        side=SideOptions(),
        cfg=EmbeddingsCfg(),
        sink=DiagnosticSink(),
    )
    assert summary.norm_summary is not None
    assert summary.norm_summary["min"] == 2.0
    assert summary.norm_summary["max"] == 5.0
