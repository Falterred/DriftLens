"""Snapshot builder tests (Step 20)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from driftlens.config.schema import SideOptions, default_config
from driftlens.errors import Code, DiagnosticSink, InvalidInputError
from driftlens.identity.snapshot_id import EmbeddingDescriptor, build_descriptor
from driftlens.schema.manifest import SNAPSHOT_SCHEMA_VERSION, FieldMappingSummary
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import build_snapshot
from driftlens.security.limits import RuntimeLimits
from driftlens.text.normalize import normalization_version
from driftlens.text.tokenizer import tokenizer_id


def _record(
    record_id,
    text=None,
    *,
    dim=None,
    model=None,
    provider=None,
    dtype=None,
    metric=None,
    normalized=None,
    document_id=None,
    chunk_index=None,
    row_number=0,
    hint=None,
    model_confidence=None,
):
    vector = np.zeros(dim, dtype=np.float32) if dim is not None else None
    embedding = EmbeddingInfo(
        vector=vector,
        dtype=(dtype or "float32") if vector is not None else None,
        dimension=dim if vector is not None else None,
        norm=float(np.linalg.norm(vector)) if vector is not None else None,
        validity="ok" if vector is not None else None,
        model=model,
        provider=provider,
        similarity_metric=metric,
        normalized=normalized,
    )
    record = Record(
        record_id=record_id,
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        row_number=row_number,
        embedding=embedding,
        identity_source_hint=hint,
    )
    if model_confidence is not None:
        record._embedding_metadata_confidence = {"model": model_confidence}
    return record


def _build(
    records,
    *,
    field_mapping=None,
    granularity_override=None,
    source_type="jsonl",
    unsafe_pickle=False,
    sink=None,
    cfg=None,
    side=None,
    limits=None,
):
    cfg = cfg or default_config()
    side = side or SideOptions()
    limits = limits or RuntimeLimits(cfg.limits)
    sink = sink if sink is not None else DiagnosticSink()
    field_mapping = field_mapping if field_mapping is not None else FieldMappingSummary()
    return build_snapshot(
        records,
        side=side,
        cfg=cfg,
        source_type=source_type,
        sink=sink,
        limits=limits,
        granularity_override=granularity_override,
        field_mapping_summary=field_mapping,
        unsafe_pickle=unsafe_pickle,
    )


def test_dimension_mismatch_names_modal_and_first_row() -> None:
    records = [
        _record(f"r{i}", f"text {i}", dim=768, row_number=i) for i in range(8)
    ]
    records.append(_record("bad", "odd text", dim=512, row_number=5))

    with pytest.raises(InvalidInputError) as info:
        _build(records)

    error = info.value
    assert "768" in str(error)
    assert "512" in str(error)
    assert "row 5" in (error.location or "")


def test_model_aggregation_single_value() -> None:
    records = [
        _record("a", "alpha", dim=8, model="E1"),
        _record("b", "beta", dim=8, model="E1"),
    ]
    snapshot = _build(records)
    assert snapshot.manifest.embedding_model == "E1"


def test_model_aggregation_mixed() -> None:
    records = [
        _record("a", "alpha", dim=8, model="E1"),
        _record("b", "beta", dim=8, model="E1"),
        _record("c", "gamma", dim=8, model="E2"),
    ]
    snapshot = _build(records)
    assert snapshot.manifest.embedding_model == "MIXED"
    assert snapshot.manifest.embedding_model_counts == {"E1": 2, "E2": 1}


def test_model_aggregation_unknown() -> None:
    records = [
        _record("a", "alpha", dim=8),
        _record("b", "beta", dim=8),
    ]
    snapshot = _build(records)
    assert snapshot.manifest.embedding_model is None


def test_confidence_worst_of_ranking() -> None:
    records = [
        _record("a", "alpha", dim=8, model="E1", model_confidence="DECLARED"),
        _record("b", "beta", dim=8, model="E1", model_confidence="INFERRED_FROM_DATA"),
    ]
    snapshot = _build(records)
    assert snapshot.manifest.embedding_metadata_confidence["model"] == "INFERRED_FROM_DATA"


def test_descriptor_excludes_volatile_fields() -> None:
    descriptor = EmbeddingDescriptor(
        model="E1",
        provider="p",
        dtype="float32",
        dimension=8,
        similarity_metric="cosine",
        normalized=True,
    )
    payload = json.loads(
        build_descriptor(
            SNAPSHOT_SCHEMA_VERSION,
            normalization_version(),
            tokenizer_id(),
            "CHUNK",
            descriptor,
        )
    )
    forbidden = {
        "created_at",
        "driftlens_version",
        "field_mapping",
        "text_params",
        "granularity_source",
        "timings",
    }
    assert forbidden.isdisjoint(payload)


def test_snapshot_id_deterministic_across_record_order() -> None:
    def make():
        return [
            _record("a", "alpha", dim=8, model="E1"),
            _record("b", "beta", dim=8, model="E1"),
            _record("c", "gamma", dim=8, model="E1"),
        ]

    first = _build(make())
    second = _build(list(reversed(make())))
    assert first.manifest.snapshot_id == second.manifest.snapshot_id


def test_snapshot_id_changes_with_content() -> None:
    first = _build([_record("a", "alpha"), _record("b", "beta")])
    second = _build([_record("a", "alpha"), _record("b", "gamma")])
    assert first.manifest.snapshot_id != second.manifest.snapshot_id


def test_snapshot_id_stable_across_field_naming() -> None:
    def make():
        return [
            _record("a", "alpha", dim=8, model="E1"),
            _record("b", "beta", dim=8, model="E1"),
        ]

    first = _build(make(), field_mapping=FieldMappingSummary(roles={"text": "content"}))
    second = _build(make(), field_mapping=FieldMappingSummary(roles={"text": "text"}))
    assert first.manifest.snapshot_id == second.manifest.snapshot_id


def test_document_table_worst_confidence() -> None:
    high = _record("a", "hello world", document_id="d1")
    low = _record("b", None, dim=8, document_id="d1", hint="vector_position")

    snapshot = _build([high, low])

    row = snapshot.documents.rows["d1"]
    assert row.identity_confidence == "LOW"


def test_records_returned_sorted_by_final_id() -> None:
    records = [
        _record("z", "zulu"),
        _record("a", "alpha"),
        _record("m", "mike"),
    ]
    snapshot = _build(records)
    ids = [record.record_id for record in snapshot.records]
    assert ids == sorted(ids)


def test_granularity_assumed_propagates_to_manifest() -> None:
    sink = DiagnosticSink()
    snapshot = _build([_record("a", "alpha"), _record("b", "beta")], sink=sink)
    assert snapshot.manifest.granularity_source == "assumed"
    assert sink.has(Code.GRANULARITY_ASSUMED)
