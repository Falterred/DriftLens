"""Embedding-space compatibility and suspension states (Step 28)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from driftlens.analysis.embeddings import EmbeddingSnapshotSummary
from driftlens.analysis.health import SampleInfo
from driftlens.analysis.sampling import hash_threshold_sample, to_sample_info
from driftlens.config.schema import Config, EmbeddingsCfg
from driftlens.errors import Code, DiagnosticSink, IncompatibleError
from driftlens.identity.chunks import all_chunk_identity_keys
from driftlens.schema.record import Record

if TYPE_CHECKING:
    from driftlens.analysis.content import ComparisonLevelResult

SpaceState = Literal[
    "SAME_SPACE", "DIFFERENT_SPACE", "PARTIAL_MIGRATION", "SPACE_UNTESTABLE"
]

CompatibilityState = Literal[
    "COMPATIBLE",
    "MODEL_UNKNOWN",
    "MODEL_CHANGED",
    "SPACE_CHANGED",
    "PARTIAL_MIGRATION",
    "MIXED_MODELS",
    "EMBEDDING_SPACE_INCOMPATIBLE",
    "METRIC_MISMATCH",
]

_WARNING_STATE_CODES = {
    "MODEL_UNKNOWN": Code.MODEL_UNKNOWN,
    "MODEL_CHANGED": Code.MODEL_CHANGED,
    "SPACE_CHANGED": Code.SPACE_CHANGED,
    "PARTIAL_MIGRATION": Code.PARTIAL_MIGRATION,
    "MIXED_MODELS": Code.MIXED_MODELS,
}

ANALYSES_SUSPENDED_BY_STATE: dict[CompatibilityState, frozenset[str]] = {
    "COMPATIBLE": frozenset(),
    "MODEL_UNKNOWN": frozenset(),
    "MODEL_CHANGED": frozenset({"mmd", "clusters"}),
    "SPACE_CHANGED": frozenset({"mmd", "clusters"}),
    "PARTIAL_MIGRATION": frozenset({"mmd", "clusters", "neighbours"}),
    "MIXED_MODELS": frozenset({"mmd", "clusters", "neighbours"}),
    "EMBEDDING_SPACE_INCOMPATIBLE": frozenset({"mmd", "clusters"}),
    "METRIC_MISMATCH": frozenset({"mmd", "clusters"}),
}

_SUSPENSION_CODES = {
    "MODEL_CHANGED": Code.SKIPPED_MODEL_CHANGED,
    "SPACE_CHANGED": Code.SKIPPED_SPACE_CHANGED,
    "PARTIAL_MIGRATION": Code.SKIPPED_PARTIAL_MIGRATION,
    "MIXED_MODELS": Code.SKIPPED_MIXED_MODELS,
}


@dataclass
class SpaceConsistencyResult:
    state: SpaceState
    anchor_count: int
    agreement_rate: float | None
    cosine_p5: float | None
    cosine_median: float | None
    reembedded_share: float | None
    declared_models_differ: bool
    sample: SampleInfo | None


@dataclass
class CompatibilityResult:
    state: CompatibilityState
    space_test: SpaceConsistencyResult | None
    label_conflict: bool
    dimension_v1: int | None
    dimension_v2: int | None
    metric_v1: str | None
    metric_v2: str | None
    model_v1: str | None
    model_v2: str | None


def _key_at_level(record: Record, level: int) -> str | None:
    keys = all_chunk_identity_keys(
        record,
        document_identity_confidence=None,
        duplicate_rank_within_document=0,
    )
    return keys.get(level)


def match_records_by_identity_key(
    v1_records: Sequence[Record],
    v2_records: Sequence[Record],
    level: "ComparisonLevelResult",
) -> list[tuple[Record, Record]]:
    """Clean 1:1 join on the identity key at the comparison level."""
    v1_by_key: dict[str, Record] = {}
    for record in v1_records:
        key = _key_at_level(record, level.level)
        if key is not None:
            v1_by_key[key] = record
    v2_by_key: dict[str, Record] = {}
    for record in v2_records:
        key = _key_at_level(record, level.level)
        if key is not None:
            v2_by_key[key] = record
    common = sorted(set(v1_by_key) & set(v2_by_key))
    return [(v1_by_key[key], v2_by_key[key]) for key in common]


def _aggregate_model(records: Sequence[Record]) -> str | None:
    models = {
        record.embedding.model
        for record in records
        if record.embedding.vector is not None and record.embedding.model is not None
    }
    if not models:
        return None
    if len(models) == 1:
        return next(iter(models))
    return "MIXED"


def _cosine(a: Record, b: Record) -> float:
    first = np.asarray(a.embedding.vector, dtype=np.float64)
    second = np.asarray(b.embedding.vector, dtype=np.float64)
    denominator = float(np.linalg.norm(first)) * float(np.linalg.norm(second))
    return float(np.dot(first, second) / denominator)


def run_space_consistency_test(
    v1_records: Sequence[Record],
    v2_records: Sequence[Record],
    *,
    comparison_level: "ComparisonLevelResult",
    cfg: EmbeddingsCfg,
    seed: int,
) -> SpaceConsistencyResult:
    """Measure cross-snapshot vector agreement over shared-text anchors."""
    pairs = match_records_by_identity_key(v1_records, v2_records, comparison_level)
    anchors = [
        (first, second)
        for first, second in pairs
        if first.raw_text_hash is not None
        and first.raw_text_hash == second.raw_text_hash
        and first.embedding.validity == "ok"
        and second.embedding.validity == "ok"
        and first.embedding.norm != 0.0
        and second.embedding.norm != 0.0
    ]

    sample_info: SampleInfo | None = None
    if len(anchors) > cfg.consistency_max_anchors:
        sampled = hash_threshold_sample(
            seed,
            cfg.consistency_max_anchors,
            [first.record_id for first, _second in anchors],
            v2_record_ids=None,
        )
        anchors = [
            pair for pair in anchors if pair[0].record_id in sampled.v1_ids
        ]
        sample_info = to_sample_info(sampled, side="v1")

    declared_models_differ = False
    model1 = _aggregate_model(v1_records)
    model2 = _aggregate_model(v2_records)
    if (
        model1 is not None
        and model2 is not None
        and model1 != model2
        and "MIXED" not in (model1, model2)
    ):
        declared_models_differ = True

    if len(anchors) < cfg.consistency_min_anchors:
        return SpaceConsistencyResult(
            state="SPACE_UNTESTABLE",
            anchor_count=len(anchors),
            agreement_rate=None,
            cosine_p5=None,
            cosine_median=None,
            reembedded_share=None,
            declared_models_differ=declared_models_differ,
            sample=sample_info,
        )

    cosines = [_cosine(first, second) for first, second in anchors]
    agreeing = sum(1 for value in cosines if value >= cfg.consistency_cosine)
    agreement_rate = agreeing / len(anchors)
    cosine_p5 = float(np.percentile(cosines, 5))
    cosine_median = float(np.median(cosines))

    reembedded_share: float | None = None
    if agreement_rate >= cfg.consistency_same:
        state: SpaceState = "SAME_SPACE"
    elif agreement_rate <= cfg.consistency_different:
        state = "DIFFERENT_SPACE"
    else:
        state = "PARTIAL_MIGRATION"
        reembedded_share = 1 - agreement_rate

    return SpaceConsistencyResult(
        state=state,
        anchor_count=len(anchors),
        agreement_rate=agreement_rate,
        cosine_p5=cosine_p5,
        cosine_median=cosine_median,
        reembedded_share=reembedded_share,
        declared_models_differ=declared_models_differ,
        sample=sample_info,
    )


def determine_compatibility_state(
    v1_summary: EmbeddingSnapshotSummary,
    v2_summary: EmbeddingSnapshotSummary,
    v1_records: Sequence[Record],
    v2_records: Sequence[Record],
    *,
    v1_dimension: int | None,
    v2_dimension: int | None,
    comparison_level: "ComparisonLevelResult",
    cfg: Config,
    sink: DiagnosticSink,
) -> CompatibilityResult:
    """Section 8.4's ordered six-rule state determination."""
    space_test: SpaceConsistencyResult | None = None

    # Rule 1 - dimensions differ: the space test is not attempted numerically.
    if (
        v1_dimension is not None
        and v2_dimension is not None
        and v1_dimension != v2_dimension
    ):
        state: CompatibilityState = "EMBEDDING_SPACE_INCOMPATIBLE"
    else:
        # Phase 1: the space test is always attempted when dimensions are usable.
        space_test = run_space_consistency_test(
            v1_records,
            v2_records,
            comparison_level=comparison_level,
            cfg=cfg.embeddings,
            seed=cfg.analysis.seed,
        )
        if v1_summary.model == "MIXED" or v2_summary.model == "MIXED":
            # Rule 2
            state = "MIXED_MODELS"
        elif space_test.state == "DIFFERENT_SPACE":
            # Rule 3
            state = "SPACE_CHANGED"
        elif space_test.state == "PARTIAL_MIGRATION":
            # Rule 3
            state = "PARTIAL_MIGRATION"
        elif (
            v1_summary.similarity_metric is not None
            and v2_summary.similarity_metric is not None
            and v1_summary.similarity_metric != v2_summary.similarity_metric
        ):
            # Rule 4 - checked before rule 5's COMPATIBLE grant.
            state = "METRIC_MISMATCH"
        elif space_test.state == "SAME_SPACE":
            # Rule 5
            state = "COMPATIBLE"
        else:
            # Rule 6 - untestable space falls back to declared model labels.
            if (
                v1_summary.model is not None
                and v2_summary.model is not None
                and v1_summary.model != v2_summary.model
            ):
                state = "MODEL_CHANGED"
            elif v1_summary.model is None or v2_summary.model is None:
                state = "MODEL_UNKNOWN"
            else:
                state = "COMPATIBLE"

    models_equal = (
        v1_summary.model is not None
        and v2_summary.model is not None
        and v1_summary.model == v2_summary.model
    )
    models_differ = (
        v1_summary.model is not None
        and v2_summary.model is not None
        and v1_summary.model != v2_summary.model
    )
    space_state = space_test.state if space_test is not None else None
    label_conflict = (models_equal and space_state == "DIFFERENT_SPACE") or (
        models_differ and space_state == "SAME_SPACE"
    )
    if label_conflict:
        sink.add(
            Code.LABEL_CONFLICT,
            "declared model labels disagree with the measured embedding space",
        )

    warning_code = _WARNING_STATE_CODES.get(state)
    if warning_code is not None:
        sink.add(warning_code, f"compatibility state is {state}")

    return CompatibilityResult(
        state=state,
        space_test=space_test,
        label_conflict=label_conflict,
        dimension_v1=v1_dimension,
        dimension_v2=v2_dimension,
        metric_v1=v1_summary.similarity_metric,
        metric_v2=v2_summary.similarity_metric,
        model_v1=v1_summary.model,
        model_v2=v2_summary.model,
    )


def apply_strict_compat(result: CompatibilityResult, *, strict_compat: bool) -> None:
    """Raise the exit-4 STRICT_COMPAT_VIOLATION for a non-compatible state."""
    if strict_compat and result.state not in ("COMPATIBLE", "MODEL_UNKNOWN"):
        raise IncompatibleError(
            Code.STRICT_COMPAT_VIOLATION,
            message=f"--strict-compat: compatibility state is {result.state}",
        )


def suspension_reason_code(state: CompatibilityState) -> Code | None:
    """The SKIPPED_* code tagging a metric suspended by this state."""
    return _SUSPENSION_CODES.get(state)
