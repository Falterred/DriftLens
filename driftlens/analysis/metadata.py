"""Metadata flattening, schema inference and drift (Step 27)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from driftlens.analysis.distributions import (
    js_divergence_categorical,
    js_divergence_histogram,
    ks_statistic,
    mean_median_variance_quantiles,
    numeric_js_bin_edges,
    wasserstein1_normalized,
)
from driftlens.analysis.health import Metric, make_metric, round9
from driftlens.config.schema import MetadataCfg, Threshold
from driftlens.errors import LimitExceededError
from driftlens.identity.hashing import canonical_json
from driftlens.schema.record import Record, is_iso8601_datetime_string
from driftlens.schema.snapshot import Snapshot

FieldType = Literal[
    "bool", "int", "float", "string", "datetime", "list", "json", "mixed"
]


@dataclass
class FieldSchema:
    name: str
    type: FieldType
    null_rate: float
    distinct_count: int
    is_identifier_like: bool
    category_labels: dict[str, int] | None


@dataclass
class FieldDrift:
    name: str
    type_v1: FieldType | None
    type_v2: FieldType | None
    type_changed: bool
    null_rate_v1: float
    null_rate_v2: float
    null_rate_change: float
    distance: float | None
    distance_kind: Literal["js", "ks", "wasserstein"] | None
    distance_reason: str | None
    values_changed_chunk_count: int


def _is_scalar_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        item is None or isinstance(item, (bool, int, float, str)) for item in value
    )


def flatten_metadata(metadata: dict[str, Any], *, max_depth: int) -> dict[str, Any]:
    """Flatten nested metadata into dotted keys up to max_depth levels."""
    flattened: dict[str, Any] = {}

    def walk(prefix: str, mapping: dict[str, Any], depth: int) -> None:
        for key, value in mapping.items():
            escaped = str(key).replace(".", "\\.")
            full = f"{prefix}.{escaped}" if prefix else escaped
            level = depth + 1
            if isinstance(value, dict) and level < max_depth:
                walk(full, value, level)
            else:
                flattened[full] = value

    walk("", metadata, 0)
    return flattened


def _type_of(value: Any) -> FieldType:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "datetime" if is_iso8601_datetime_string(value) else "string"
    if isinstance(value, list):
        return "list" if _is_scalar_list(value) else "json"
    if isinstance(value, dict):
        return "json"
    return "string"


def infer_field_type(values: Sequence[Any], *, cfg: MetadataCfg) -> FieldType:
    """Dominant observed type with the int/float collapse and mixed share rule."""
    non_null = [value for value in values if value is not None]
    if not non_null:
        return "string"

    type_counts: Counter[FieldType] = Counter(_type_of(value) for value in non_null)
    if set(type_counts) == {"int", "float"}:
        type_counts = Counter(
            {"float": type_counts["int"] + type_counts["float"]}
        )

    dominant, dominant_n = type_counts.most_common(1)[0]
    total = len(non_null)
    runner_up_share = (total - dominant_n) / total
    if runner_up_share >= cfg.mixed_type_share:
        return "mixed"
    return dominant


def is_identifier_like(
    values: Sequence[Any], field_type: FieldType, *, cfg: MetadataCfg
) -> bool:
    """Whether a string/int field has too many distinct values to enumerate."""
    if field_type not in ("string", "int"):
        return False
    non_null = [value for value in values if value is not None]
    distinct = len(set(non_null))
    if field_type == "string":
        return distinct > cfg.max_categories
    if not non_null:
        return False
    uniqueness_ratio = distinct / len(non_null)
    return distinct > cfg.max_categories and uniqueness_ratio >= 0.95


def _distinct_count(non_null: Sequence[Any]) -> int:
    try:
        return len(set(non_null))
    except TypeError:
        return len({canonical_json(value) for value in non_null})


def build_field_schema(
    field_name: str, all_values: Sequence[Any | None], *, cfg: MetadataCfg
) -> FieldSchema:
    """Per-field type, null rate, distinct count and labels."""
    total = len(all_values)
    non_null = [value for value in all_values if value is not None]
    null_rate = round9((total - len(non_null)) / total) if total else 0.0
    field_type = infer_field_type(all_values, cfg=cfg)
    distinct_count = _distinct_count(non_null)
    id_like = is_identifier_like(non_null, field_type, cfg=cfg)

    category_labels: dict[str, int] | None = None
    if not id_like and field_type != "json" and distinct_count <= cfg.max_categories:
        category_labels = dict(
            Counter(canonical_json(value).decode("utf-8") for value in non_null)
        )

    return FieldSchema(
        name=field_name,
        type=field_type,
        null_rate=null_rate,
        distinct_count=distinct_count,
        is_identifier_like=id_like,
        category_labels=category_labels,
    )


def _flatten_snapshot(
    snapshot: Snapshot, cfg: MetadataCfg
) -> dict[str, dict[str, Any]]:
    return {
        record.record_id: flatten_metadata(record.metadata, max_depth=cfg.max_depth)
        for record in snapshot.records
    }


def _field_values(
    snapshot: Snapshot, flat_by_id: dict[str, dict[str, Any]]
) -> dict[str, list[Any | None]]:
    names: set[str] = set()
    for flat in flat_by_id.values():
        names.update(flat)
    return {
        name: [flat_by_id[record.record_id].get(name) for record in snapshot.records]
        for name in names
    }


def _type_shares(values: Sequence[Any]) -> dict[str, int]:
    return dict(Counter(_type_of(value) for value in values if value is not None))


def _epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _changed_count(
    field: str,
    v1_flat: dict[str, dict[str, Any]],
    v2_flat: dict[str, dict[str, Any]],
    common_record_ids: set[str],
) -> int:
    changed = 0
    for record_id in common_record_ids:
        first = v1_flat.get(record_id, {}).get(field)
        second = v2_flat.get(record_id, {}).get(field)
        if canonical_json(first) != canonical_json(second):
            changed += 1
    return changed


def _drift(
    field: str,
    schema1: FieldSchema,
    schema2: FieldSchema,
    changed: int,
    distance: float | None,
    kind: Literal["js", "ks", "wasserstein"] | None,
    reason: str | None,
) -> FieldDrift:
    return FieldDrift(
        name=field,
        type_v1=schema1.type,
        type_v2=schema2.type,
        type_changed=schema1.type != schema2.type,
        null_rate_v1=schema1.null_rate,
        null_rate_v2=schema2.null_rate,
        null_rate_change=schema2.null_rate - schema1.null_rate,
        distance=distance,
        distance_kind=kind,
        distance_reason=reason,
        values_changed_chunk_count=changed,
    )


def compare_metadata(
    v1: Snapshot,
    v2: Snapshot,
    *,
    cfg: MetadataCfg,
    drift_cfg: dict[str, Threshold | None],
    common_record_ids: set[str],
) -> tuple[list[FieldDrift], list[Metric]]:
    """Per-field drift rows plus the METADATA dimension's aggregate metrics."""
    v1_flat = _flatten_snapshot(v1, cfg)
    v2_flat = _flatten_snapshot(v2, cfg)
    v1_values = _field_values(v1, v1_flat)
    v2_values = _field_values(v2, v2_flat)

    all_fields = sorted(set(v1_values) | set(v2_values))
    if len(all_fields) > cfg.max_fields:
        raise LimitExceededError(flag="metadata.max_fields")

    drifts: list[FieldDrift] = []
    extra_metrics: list[Metric] = []
    js_values: list[float] = []
    ks_values: list[float] = []

    for field in all_fields:
        values1 = v1_values.get(field, [None] * len(v1.records))
        values2 = v2_values.get(field, [None] * len(v2.records))
        schema1 = build_field_schema(field, values1, cfg=cfg)
        schema2 = build_field_schema(field, values2, cfg=cfg)
        changed = _changed_count(field, v1_flat, v2_flat, common_record_ids)

        non1 = [value for value in values1 if value is not None]
        non2 = [value for value in values2 if value is not None]
        distance: float | None = None
        kind: Literal["js", "ks", "wasserstein"] | None = None
        reason: str | None = None
        field_js: float | None = None
        field_ks: float | None = None
        list_extra: list[FieldDrift] = []

        if schema2.type == "mixed":
            reason = "mixed (type shares only)"
            shares1 = _type_shares(values1)
            shares2 = _type_shares(values2)
            share_value, share_reason = js_divergence_categorical(shares1, shares2)
            extra_metrics.append(
                make_metric(
                    f"type_share_js:{field}",
                    value=share_value,
                    threshold=None,
                    reason=share_reason,
                )
            )
        elif schema2.is_identifier_like:
            reason = "identifier-like"
        elif schema2.type in ("bool", "string"):
            value, value_reason = js_divergence_categorical(
                Counter(non1), Counter(non2)
            )
            distance = value
            kind = "js"
            reason = value_reason
            field_js = value
        elif schema2.type in ("int", "float", "datetime"):
            if schema2.type == "datetime":
                first = [_epoch(value) for value in non1]
                second = [_epoch(value) for value in non2]
            else:
                first = [float(value) for value in non1]
                second = [float(value) for value in non2]
            value, value_reason = ks_statistic(first, second)
            distance = value
            kind = "ks"
            reason = value_reason
            field_ks = value

            pooled = first + second
            wasserstein, wasserstein_reason = wasserstein1_normalized(first, second)
            if pooled:
                edges = numeric_js_bin_edges(pooled)
                histogram_js, _histogram_reason = js_divergence_histogram(
                    first, second, edges=edges
                )
            else:
                histogram_js = None
            extra_metrics.append(
                make_metric(
                    f"metadata_wasserstein:{field}",
                    value=wasserstein,
                    threshold=None,
                    reason=wasserstein_reason,
                    details={
                        "summary": mean_median_variance_quantiles(pooled),
                        "histogram_js": histogram_js,
                    },
                )
            )
        elif schema2.type == "list":
            lengths1 = [len(value) for value in non1]
            lengths2 = [len(value) for value in non2]
            value, value_reason = ks_statistic(lengths1, lengths2)
            distance = value
            kind = "ks"
            reason = value_reason
            field_ks = value

            elements1 = Counter(
                element for value in non1 for element in value
            )
            elements2 = Counter(
                element for value in non2 for element in value
            )
            distinct_elements = len(set(elements1) | set(elements2))
            if distinct_elements <= cfg.max_categories:
                element_js, element_reason = js_divergence_categorical(
                    elements1, elements2
                )
                if element_js is not None:
                    field_js = element_js
                    list_extra.append(
                        _drift(
                            field,
                            schema1,
                            schema2,
                            changed,
                            element_js,
                            "js",
                            element_reason,
                        )
                    )
        elif schema2.type == "json":
            reason = "json (collapsed)"

        drifts.append(_drift(field, schema1, schema2, changed, distance, kind, reason))
        drifts.extend(list_extra)
        if field_js is not None:
            js_values.append(field_js)
        if field_ks is not None:
            ks_values.append(field_ks)

    metrics: list[Metric] = [
        make_metric(
            "metadata_js",
            value=max(js_values) if js_values else None,
            threshold=drift_cfg.get("metadata_js"),
        ),
        make_metric(
            "metadata_ks",
            value=max(ks_values) if ks_values else None,
            threshold=drift_cfg.get("metadata_ks"),
        ),
        make_metric(
            "metadata_null_rate",
            value=max((abs(d.null_rate_change) for d in drifts), default=0.0),
            threshold=drift_cfg.get("metadata_null_rate"),
        ),
        make_metric(
            "fields_modified",
            value=sum(1 for d in drifts if d.values_changed_chunk_count > 0),
            threshold=None,
        ),
        make_metric(
            "schema_fields_added",
            value=len(set(v2_values) - set(v1_values)),
            threshold=None,
        ),
        make_metric(
            "schema_fields_removed",
            value=len(set(v1_values) - set(v2_values)),
            threshold=None,
        ),
        make_metric(
            "type_changed_fields",
            value=sum(1 for d in drifts if d.type_changed),
            threshold=None,
        ),
    ]
    metrics.extend(extra_metrics)
    return drifts, metrics
