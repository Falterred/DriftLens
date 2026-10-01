"""Compatibility state determination and space consistency (Step 28)."""

from __future__ import annotations

import numpy as np
import pytest

from driftlens.analysis.compatibility import (
    CompatibilityResult,
    apply_strict_compat,
    determine_compatibility_state,
    run_space_consistency_test,
    suspension_reason_code,
)
from driftlens.analysis.content import ComparisonLevelResult
from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.config.schema import EmbeddingsCfg, default_config
from driftlens.errors import Code, DiagnosticSink, IncompatibleError
from driftlens.schema.record import EmbeddingInfo, Record

LEVEL = ComparisonLevelResult(level=1, hash_kind="text")


def _summary(*, model=None, metric=None):
    return EmbeddingSnapshotSummary(
        model=model,
        provider=None,
        dtype="float32",
        similarity_metric=metric,
        similarity_metric_derived=False,
        normalized=None,
        confidence={},
        invalid_count=0,
        zero_count=0,
        coverage=1.0,
        norm_summary=None,
        embeddings_unavailable=False,
    )


def _record(record_id, vector, *, text_hash="h", model=None):
    value = np.asarray(vector, dtype=np.float32)
    info = EmbeddingInfo(
        vector=value,
        dtype="float32",
        dimension=len(value),
        norm=float(np.linalg.norm(value)),
        validity="ok",
        model=model,
        provider=None,
        similarity_metric=None,
        normalized=None,
    )
    record = Record(record_id=record_id, embedding=info)
    record.raw_text_hash = text_hash
    return record


def _pair_records(count, *, second_vector=(1.0, 0.0), model1=None, model2=None):
    first = [_record(f"r{index}", (1.0, 0.0), model=model1) for index in range(count)]
    second = [_record(f"r{index}", second_vector, model=model2) for index in range(count)]
    return first, second


def _result(state):
    return CompatibilityResult(
        state=state,
        space_test=None,
        label_conflict=False,
        dimension_v1=None,
        dimension_v2=None,
        metric_v1=None,
        metric_v2=None,
        model_v1=None,
        model_v2=None,
    )


def test_dimension_mismatch_rule1_short_circuits() -> None:
    result = determine_compatibility_state(
        _summary(),
        _summary(),
        [],
        [],
        v1_dimension=768,
        v2_dimension=1536,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    assert result.state == "EMBEDDING_SPACE_INCOMPATIBLE"
    assert result.space_test is None


def test_mixed_models_rule2() -> None:
    first, second = _pair_records(1)
    result = determine_compatibility_state(
        _summary(model="MIXED"),
        _summary(model="E1"),
        first,
        second,
        v1_dimension=2,
        v2_dimension=2,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    assert result.state == "MIXED_MODELS"
    assert result.space_test is not None


def test_different_space_rule3() -> None:
    first, second = _pair_records(30, second_vector=(0.0, 1.0))
    space = run_space_consistency_test(
        first, second, comparison_level=LEVEL, cfg=EmbeddingsCfg(), seed=42
    )
    assert space.state == "DIFFERENT_SPACE"

    result = determine_compatibility_state(
        _summary(model="E1"),
        _summary(model="E1"),
        first,
        second,
        v1_dimension=2,
        v2_dimension=2,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    assert result.state == "SPACE_CHANGED"


def test_metric_mismatch_rule4() -> None:
    first, second = _pair_records(30)
    result = determine_compatibility_state(
        _summary(model="E1", metric="cosine"),
        _summary(model="E1", metric="dot"),
        first,
        second,
        v1_dimension=2,
        v2_dimension=2,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    assert result.state == "METRIC_MISMATCH"


def test_compatible_rule5() -> None:
    first, second = _pair_records(30)
    result = determine_compatibility_state(
        _summary(model="E1", metric="cosine"),
        _summary(model="E1", metric="cosine"),
        first,
        second,
        v1_dimension=2,
        v2_dimension=2,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    assert result.state == "COMPATIBLE"


def test_model_changed_rule6() -> None:
    first, second = _pair_records(5)
    result = determine_compatibility_state(
        _summary(model="E1"),
        _summary(model="E2"),
        first,
        second,
        v1_dimension=2,
        v2_dimension=2,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    assert result.state == "MODEL_CHANGED"


def test_model_unknown_rule6() -> None:
    first, second = _pair_records(5)
    result = determine_compatibility_state(
        _summary(model="E1"),
        _summary(model=None),
        first,
        second,
        v1_dimension=2,
        v2_dimension=2,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    assert result.state == "MODEL_UNKNOWN"


def test_untestable_same_trusted_model_compatible() -> None:
    first, second = _pair_records(5)
    result = determine_compatibility_state(
        _summary(model="E1"),
        _summary(model="E1"),
        first,
        second,
        v1_dimension=2,
        v2_dimension=2,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=DiagnosticSink(),
    )
    assert result.state == "COMPATIBLE"


def test_label_conflict_same_model_different_space() -> None:
    first, second = _pair_records(30, second_vector=(0.0, 1.0))
    sink = DiagnosticSink()
    determine_compatibility_state(
        _summary(model="E1"),
        _summary(model="E1"),
        first,
        second,
        v1_dimension=2,
        v2_dimension=2,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=sink,
    )
    assert sink.has(Code.LABEL_CONFLICT)


def test_label_conflict_different_model_same_space() -> None:
    first, second = _pair_records(30)
    sink = DiagnosticSink()
    determine_compatibility_state(
        _summary(model="E1"),
        _summary(model="E2"),
        first,
        second,
        v1_dimension=2,
        v2_dimension=2,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=sink,
    )
    assert sink.has(Code.LABEL_CONFLICT)


def test_no_label_conflict_when_consistent() -> None:
    first, second = _pair_records(30)
    sink = DiagnosticSink()
    determine_compatibility_state(
        _summary(model="E1"),
        _summary(model="E1"),
        first,
        second,
        v1_dimension=2,
        v2_dimension=2,
        comparison_level=LEVEL,
        cfg=default_config(),
        sink=sink,
    )
    assert not sink.has(Code.LABEL_CONFLICT)


def test_space_untestable_below_min_anchors() -> None:
    first, second = _pair_records(5)
    space = run_space_consistency_test(
        first, second, comparison_level=LEVEL, cfg=EmbeddingsCfg(), seed=42
    )
    assert space.state == "SPACE_UNTESTABLE"


def test_anchor_sampling_applied_above_max() -> None:
    first, second = _pair_records(6000)
    space = run_space_consistency_test(
        first,
        second,
        comparison_level=LEVEL,
        cfg=EmbeddingsCfg(consistency_max_anchors=5000),
        seed=42,
    )
    assert space.anchor_count <= 5000
    assert space.sample is not None


def test_strict_compat_raises_on_non_compatible() -> None:
    with pytest.raises(IncompatibleError) as info:
        apply_strict_compat(_result("MODEL_CHANGED"), strict_compat=True)
    assert info.value.exit_code == 4


def test_strict_compat_allows_model_unknown() -> None:
    apply_strict_compat(_result("MODEL_UNKNOWN"), strict_compat=True)


def test_suspension_reason_mapping() -> None:
    assert suspension_reason_code("MODEL_CHANGED") == Code.SKIPPED_MODEL_CHANGED
    assert suspension_reason_code("COMPATIBLE") is None
