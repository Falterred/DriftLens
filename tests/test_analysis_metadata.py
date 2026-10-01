"""Metadata flattening, schema inference and drift (Step 27)."""

from __future__ import annotations

import pytest

from driftlens.analysis.metadata import (
    build_field_schema,
    compare_metadata,
    flatten_metadata,
    infer_field_type,
    is_identifier_like,
)
from driftlens.config.schema import MetadataCfg, default_config
from driftlens.errors import LimitExceededError
from driftlens.schema.manifest import FieldMappingSummary, Manifest
from driftlens.schema.record import Record
from driftlens.schema.snapshot import DocumentTable, Snapshot, StatisticsAccumulator


def _record(record_id, metadata):
    return Record(record_id=record_id, metadata=dict(metadata))


def _manifest(levels):
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
        document_count=len(levels),
        chunk_count=len(levels),
        embedding_dimension=None,
        embedding_dtype=None,
        embedding_model=None,
        embedding_model_counts={},
        embedding_provider=None,
        embedding_metadata_confidence={},
        similarity_metric=None,
        embedding_normalized=None,
        chunk_identity_levels=levels,
        identity_summary={"HIGH": len(levels), "MEDIUM": 0, "LOW": 0},
        field_mapping=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _snapshot(records):
    return Snapshot(
        manifest=_manifest({record.record_id: [4] for record in records}),
        records=list(records),
        documents=DocumentTable(rows={}),
        statistics=StatisticsAccumulator(),
    )


def test_flatten_nested_dict_within_depth() -> None:
    assert flatten_metadata({"a": {"b": {"c": 1}}}, max_depth=3) == {"a.b.c": 1}


def test_flatten_collapses_beyond_max_depth() -> None:
    assert flatten_metadata(
        {"a": {"b": {"c": {"d": 1}}}}, max_depth=3
    ) == {"a.b.c": {"d": 1}}


def test_flatten_escapes_literal_dots_in_keys() -> None:
    literal = flatten_metadata({"a.b": 1}, max_depth=3)
    nested = flatten_metadata({"a": {"b": 1}}, max_depth=3)
    assert list(literal) == [r"a\.b"]
    assert list(nested) == ["a.b"]
    assert set(literal) != set(nested)


def test_infer_type_int_float_collapse() -> None:
    assert infer_field_type([1, 2.0, 3, 4.5], cfg=MetadataCfg()) == "float"


def test_infer_type_mixed_by_share() -> None:
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, "ten"]
    assert infer_field_type(values, cfg=MetadataCfg()) == "mixed"


def test_infer_type_not_mixed_below_share() -> None:
    values = [1] * 49 + ["x"]
    assert infer_field_type(values, cfg=MetadataCfg()) == "int"


def test_infer_type_datetime_string_detected() -> None:
    values = ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]
    assert infer_field_type(values, cfg=MetadataCfg()) == "datetime"


def test_identifier_like_string() -> None:
    values = [f"s{index}" for index in range(250)]
    assert is_identifier_like(values, "string", cfg=MetadataCfg()) is True


def test_identifier_like_int_needs_both_conditions() -> None:
    lots = [index % 250 for index in range(500)]
    assert is_identifier_like(lots, "int", cfg=MetadataCfg()) is False
    dense = list(range(250)) + [0] * 10
    assert is_identifier_like(dense, "int", cfg=MetadataCfg()) is True


def test_identifier_like_float_never() -> None:
    values = [float(index) for index in range(250)]
    assert is_identifier_like(values, "float", cfg=MetadataCfg()) is False


def test_field_schema_null_rate() -> None:
    values = [1] * 7 + [None] * 3
    schema = build_field_schema("f", values, cfg=MetadataCfg())
    assert schema.null_rate == 0.3


def test_field_schema_category_labels_omitted_when_identifier_like() -> None:
    values = [f"s{index}" for index in range(250)]
    schema = build_field_schema("f", values, cfg=MetadataCfg())
    assert schema.is_identifier_like is True
    assert schema.category_labels is None


def test_compare_metadata_added_removed_fields() -> None:
    v1 = _snapshot([_record("r1", {"only_v1": 1})])
    v2 = _snapshot([_record("r1", {"only_v2": 2})])
    _drifts, metrics = compare_metadata(
        v1,
        v2,
        cfg=MetadataCfg(),
        drift_cfg=default_config().drift,
        common_record_ids=set(),
    )
    by_name = {metric.name: metric for metric in metrics}
    assert by_name["schema_fields_added"].change == 1
    assert by_name["schema_fields_removed"].change == 1


def test_compare_metadata_mixed_field_distance_none() -> None:
    v1 = _snapshot([_record("r1", {"m": 1}), _record("r2", {"m": "x"})])
    v2 = _snapshot([_record("r1", {"m": 1}), _record("r2", {"m": "x"})])
    drifts, metrics = compare_metadata(
        v1,
        v2,
        cfg=MetadataCfg(),
        drift_cfg=default_config().drift,
        common_record_ids=set(),
    )
    drift = next(item for item in drifts if item.name == "m")
    assert drift.distance is None
    assert any(metric.name.startswith("type_share_js") for metric in metrics)


def test_compare_metadata_list_field_dual_distance() -> None:
    v1 = _snapshot(
        [
            _record("r1", {"tags": ["a", "b"]}),
            _record("r2", {"tags": ["a", "c"]}),
        ]
    )
    v2 = _snapshot(
        [
            _record("r1", {"tags": ["a", "b", "c"]}),
            _record("r2", {"tags": ["b", "c"]}),
        ]
    )
    _drifts, metrics = compare_metadata(
        v1,
        v2,
        cfg=MetadataCfg(),
        drift_cfg=default_config().drift,
        common_record_ids=set(),
    )
    by_name = {metric.name: metric for metric in metrics}
    assert by_name["metadata_ks"].change is not None
    assert by_name["metadata_js"].change is not None


def test_compare_metadata_values_changed_uses_canonical_equality() -> None:
    v1 = _snapshot([_record("r1", {"x": 1})])
    v2 = _snapshot([_record("r1", {"x": 1.0})])
    drifts, _metrics = compare_metadata(
        v1,
        v2,
        cfg=MetadataCfg(),
        drift_cfg=default_config().drift,
        common_record_ids={"r1"},
    )
    drift = next(item for item in drifts if item.name == "x")
    assert drift.values_changed_chunk_count == 1


def test_max_fields_limit_exceeded() -> None:
    metadata = {f"f{index}": index for index in range(6)}
    with pytest.raises(LimitExceededError):
        compare_metadata(
            _snapshot([_record("r1", metadata)]),
            _snapshot([_record("r1", metadata)]),
            cfg=MetadataCfg(max_fields=5),
            drift_cfg=default_config().drift,
            common_record_ids=set(),
        )
