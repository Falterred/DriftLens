"""Embedding-space descriptor resolution and vector validation (Step 22)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from driftlens.adapters.detect import (
    DTYPE_CANDIDATES,
    METRIC_CANDIDATES,
    MODEL_CANDIDATES,
    PROVIDER_CANDIDATES,
)
from driftlens.config.schema import EmbeddingsCfg, SideOptions
from driftlens.errors import Code, DiagnosticSink
from driftlens.schema.record import Record

CONFIDENCE_RANK = {
    "UNKNOWN": 0,
    "INFERRED_FROM_DATA": 1,
    "USER_SUPPLIED": 2,
    "ADAPTER_METADATA": 3,
    "DECLARED": 4,
}

NUMPY_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
}

_NORMALIZED_CANDIDATES = ["embedding_normalized", "normalized"]
_DESCRIPTOR_NAMES = ("model", "provider", "dtype", "similarity_metric")
_NORMALIZED_KEYS = ("model", "provider", "dtype", "similarity_metric", "normalized")


@dataclass
class ResolvedDescriptor:
    value: str | None
    confidence: str
    # Internal marker: sources 1-4 held a value that disagreed with source 5.
    # Consumed by analyze_embeddings' aggregation pass (not part of the public
    # two-field contract in step 22; see NOTE).
    override_ignored: bool = False


@dataclass
class NormalizationCheck:
    declared: bool | None
    measured_share: float | None
    inferred: bool | None
    claim_invalid: bool


@dataclass
class EmbeddingSnapshotSummary:
    model: str | None
    provider: str | None
    dtype: str | None
    similarity_metric: str | None
    similarity_metric_derived: bool
    normalized: bool | None
    confidence: dict[str, str]
    invalid_count: int
    zero_count: int
    coverage: float
    norm_summary: dict[str, float] | None
    embeddings_unavailable: bool


def _metadata_lookup(metadata: dict[str, Any], candidates: Sequence[str]) -> str | None:
    lowered = {str(key).lower(): key for key in metadata}
    for name in candidates:
        key = lowered.get(name.lower())
        if key is None:
            continue
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    return None


def _resolve_normalized(
    record: Record, file_descriptors: dict[str, str]
) -> tuple[bool | None, str]:
    if record.embedding.normalized is not None:
        return record.embedding.normalized, "DECLARED"
    file_value = _parse_bool(file_descriptors.get("embedding_normalized"))
    if file_value is not None:
        return file_value, "DECLARED"
    metadata_value = _parse_bool(
        _metadata_lookup(record.metadata, _NORMALIZED_CANDIDATES)
    )
    if metadata_value is not None:
        return metadata_value, "DECLARED"
    hint = getattr(record, "_adapter_metadata_hint", None) or {}
    hint_value = _parse_bool(hint.get("normalized"))
    if hint_value is not None:
        return hint_value, "ADAPTER_METADATA"
    return None, "UNKNOWN"


def resolve_embedding_descriptors(
    record: Record,
    *,
    file_descriptors: dict[str, str],
    side: SideOptions,
) -> dict[str, ResolvedDescriptor]:
    """Run the full five-source descriptor chain for one record (pure)."""
    hint = getattr(record, "_adapter_metadata_hint", None) or {}
    specs = (
        (
            "model",
            MODEL_CANDIDATES,
            ("embedding_model",),
            record.embedding.model,
            hint.get("model"),
            getattr(side, "model", None),
        ),
        (
            "provider",
            PROVIDER_CANDIDATES,
            ("embedding_provider",),
            record.embedding.provider,
            hint.get("provider"),
            getattr(side, "provider", None),
        ),
        (
            "dtype",
            DTYPE_CANDIDATES,
            ("embedding_dtype",),
            record.embedding.dtype,
            hint.get("dtype"),
            getattr(side, "dtype", None),
        ),
        (
            "similarity_metric",
            METRIC_CANDIDATES,
            ("similarity_metric",),
            record.embedding.similarity_metric,
            hint.get("similarity_metric"),
            getattr(side, "similarity_metric", None),
        ),
    )

    resolved: dict[str, ResolvedDescriptor] = {}
    for name, candidates, file_keys, row_value, hint_value, side_value in specs:
        found: str | None = None
        confidence = "UNKNOWN"
        override_ignored = False

        if row_value is not None:
            found = str(row_value)
            confidence = "DECLARED"
        else:
            for file_key in file_keys:
                file_value = file_descriptors.get(file_key)
                if file_value is not None:
                    found = str(file_value)
                    confidence = "DECLARED"
                    break

        if found is None:
            metadata_value = _metadata_lookup(record.metadata, candidates)
            if metadata_value is not None:
                found = metadata_value
                confidence = "DECLARED"

        if found is None and hint_value is not None:
            found = str(hint_value)
            confidence = "ADAPTER_METADATA"

        if found is None:
            if side_value is not None:
                found = str(side_value)
                confidence = "USER_SUPPLIED"
        elif side_value is not None and str(side_value) != found:
            override_ignored = True

        resolved[name] = ResolvedDescriptor(
            value=found, confidence=confidence, override_ignored=override_ignored
        )
    return resolved


def validate_vector(vector: np.ndarray, dtype: str) -> tuple[str, float | None]:
    """Return (validity, norm) for one vector (step 22 implementation logic 2)."""
    values = np.asarray(vector, dtype=np.float64)
    if np.any(np.isnan(values) | np.isinf(values)):
        return ("nan_inf", None)
    if np.all(values == 0):
        return ("zero", 0.0)
    return ("ok", float(np.linalg.norm(values)))


def _snapshot_dtype(records: Sequence[Record]) -> str | None:
    dtypes = {
        record.embedding.dtype
        for record in records
        if record.embedding.vector is not None
    }
    non_null = {dtype for dtype in dtypes if dtype is not None}
    if len(non_null) == 1:
        return next(iter(non_null))
    return None


def check_normalization(
    records: Sequence[Record],
    *,
    declared: bool | None,
    cfg: EmbeddingsCfg,
) -> NormalizationCheck:
    """Check a declared normalized claim and/or infer normalization (logic 3)."""
    dtype = _snapshot_dtype(records)
    if dtype == "int8":
        return NormalizationCheck(
            declared=declared, measured_share=None, inferred=None, claim_invalid=False
        )

    norms = [
        record.embedding.norm
        for record in records
        if record.embedding.validity == "ok"
    ]
    tolerance = (
        cfg.normalized_tolerance["float16"]
        if dtype == "float16"
        else cfg.normalized_tolerance["float32"]
    )
    if norms:
        compliant = sum(1 for norm in norms if abs(norm - 1.0) <= tolerance)
        measured_share: float | None = compliant / len(norms)
    else:
        measured_share = None

    if declared is True:
        claim_invalid = measured_share is not None and (1 - measured_share) > 0.01
        inferred = False if claim_invalid else True
    elif declared is False:
        claim_invalid = False
        inferred = False
    else:
        claim_invalid = False
        inferred = measured_share is not None and measured_share >= cfg.normalized_share

    return NormalizationCheck(
        declared=declared,
        measured_share=measured_share,
        inferred=inferred,
        claim_invalid=claim_invalid,
    )


def _distinct(values: list[str]) -> str | None:
    distinct = sorted(set(values))
    if not distinct:
        return None
    if len(distinct) == 1:
        return distinct[0]
    return "MIXED"


def _three_way_bool(values: list[bool]) -> bool | None:
    distinct = set(values)
    if len(distinct) == 1:
        return next(iter(distinct))
    return None


def analyze_embeddings(
    records: Sequence[Record],
    *,
    file_descriptors: dict[str, str],
    side: SideOptions,
    cfg: EmbeddingsCfg,
    sink: DiagnosticSink,
) -> EmbeddingSnapshotSummary:
    """Resolve descriptors, validate vectors and summarize one snapshot."""
    embedding_records = [r for r in records if r.embedding.vector is not None]

    if not embedding_records:
        return EmbeddingSnapshotSummary(
            model=None,
            provider=None,
            dtype=None,
            similarity_metric=None,
            similarity_metric_derived=False,
            normalized=None,
            confidence={name: "UNKNOWN" for name in _NORMALIZED_KEYS},
            invalid_count=0,
            zero_count=0,
            coverage=0.0,
            norm_summary=None,
            embeddings_unavailable=False,
        )

    per_record: list[dict[str, ResolvedDescriptor]] = []
    normalized_values: list[bool] = []
    override_count = 0

    for record in embedding_records:
        resolved = resolve_embedding_descriptors(
            record, file_descriptors=file_descriptors, side=side
        )
        per_record.append(resolved)
        override_count += sum(
            1 for descriptor in resolved.values() if descriptor.override_ignored
        )

        record.embedding.model = resolved["model"].value
        record.embedding.provider = resolved["provider"].value
        record.embedding.dtype = resolved["dtype"].value  # type: ignore[assignment]
        record.embedding.similarity_metric = resolved["similarity_metric"].value  # type: ignore[assignment]

        normalized_value, _ = _resolve_normalized(record, file_descriptors)
        if normalized_value is not None:
            normalized_values.append(normalized_value)

        validity, norm = validate_vector(record.embedding.vector, record.embedding.dtype or "float32")
        record.embedding.validity = validity  # type: ignore[assignment]
        record.embedding.norm = norm

    declared = _three_way_bool(normalized_values) if normalized_values else None
    norm_check = check_normalization(records, declared=declared, cfg=cfg)

    model = _distinct([r["model"].value for r in per_record if r["model"].value])
    provider = _distinct([r["provider"].value for r in per_record if r["provider"].value])
    dtype = _distinct([r["dtype"].value for r in per_record if r["dtype"].value])

    metric_values = [r["similarity_metric"].value for r in per_record if r["similarity_metric"].value]
    similarity_metric = _distinct(metric_values)
    similarity_metric_derived = False
    if similarity_metric is None and norm_check.inferred is True:
        similarity_metric = "cosine"
        similarity_metric_derived = True

    invalid_count = sum(1 for r in embedding_records if r.embedding.validity == "nan_inf")
    zero_count = sum(1 for r in embedding_records if r.embedding.validity == "zero")
    total = len(embedding_records)
    coverage = (total - invalid_count) / total
    invalid_rate = invalid_count / total
    embeddings_unavailable = invalid_rate > cfg.max_invalid_rate

    norms = [
        r.embedding.norm
        for r in embedding_records
        if r.embedding.validity == "ok"
    ]
    if norms:
        norm_summary: dict[str, float] | None = {
            "min": float(min(norms)),
            "median": float(np.median(norms)),
            "p95": float(np.percentile(norms, 95)),
            "max": float(max(norms)),
        }
    else:
        norm_summary = None

    confidence: dict[str, str] = {}
    for name in _DESCRIPTOR_NAMES:
        worst = min(
            (r[name].confidence for r in per_record),
            key=lambda level: CONFIDENCE_RANK.get(level, 0),
        )
        confidence[name] = worst
    worst_normalized = min(
        (_resolve_normalized(r, file_descriptors)[1] for r in embedding_records),
        key=lambda level: CONFIDENCE_RANK.get(level, 0),
    )
    confidence["normalized"] = worst_normalized
    if similarity_metric_derived:
        confidence["similarity_metric"] = "INFERRED_FROM_DATA"

    if norm_check.claim_invalid and norm_check.measured_share is not None:
        invalid_share = 1 - norm_check.measured_share
        sink.add(
            Code.NORMALIZATION_CLAIM_INVALID,
            f"{invalid_share:.1%} of vectors violate the declared normalized=true claim",
        )

    if override_count:
        sink.add(
            Code.MODEL_OVERRIDE_IGNORED,
            f"--embedding-model ignored for {override_count} record(s); differs from declared value",
            count=override_count,
        )

    return EmbeddingSnapshotSummary(
        model=model,
        provider=provider,
        dtype=dtype,
        similarity_metric=similarity_metric,
        similarity_metric_derived=similarity_metric_derived,
        normalized=norm_check.inferred,
        confidence=confidence,
        invalid_count=invalid_count,
        zero_count=zero_count,
        coverage=coverage,
        norm_summary=norm_summary,
        embeddings_unavailable=embeddings_unavailable,
    )
