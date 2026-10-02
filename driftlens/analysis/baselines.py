"""Baseline construction and application (Step 48, section 12.4).

A baseline freezes per-threshold-key warning/critical levels derived from the
same analysis pipeline a live comparison runs.  It deliberately reimplements
nothing: every value comes from the Step 25-30/43-47 compare_* functions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from driftlens.analysis.chunking import compare_chunking
from driftlens.analysis.clusters import compare_clusters
from driftlens.analysis.compatibility import (
    ANALYSES_SUSPENDED_BY_STATE,
    determine_compatibility_state,
)
from driftlens.analysis.content import compare_content, determine_comparison_level
from driftlens.analysis.duplicates import compare_duplicates
from driftlens.analysis.embeddings import analyze_embeddings
from driftlens.analysis.health import Metric, round9
from driftlens.analysis.metadata import compare_metadata
from driftlens.analysis.mmd import compare_mmd, median_heuristic_bandwidth
from driftlens.config.schema import THRESHOLD_KEYS, Config, SideOptions, Threshold
from driftlens.errors import Code, DiagnosticSink, InvalidInputError
from driftlens.evaluation.retrieval import run_retrieval_comparison
from driftlens.schema.eval_set import EvalSet
from driftlens.schema.manifest import Manifest
from driftlens.schema.snapshot import Snapshot
from driftlens.security.limits import loads_limited

BaselineMethod = Literal["mean_std", "quantile"]

# Level-type keys report the later snapshot's own level, not a change.  Every
# other key reports a change already.  Both categories read a different Metric
# field (v2 versus change), but together they cover all 14 threshold keys.
LEVEL_TYPE_KEYS = frozenset(
    {
        "duplicate_rate",
        "duplicate_rate_exact",
        "duplicate_rate_near",
        "duplicate_rate_semantic",
        "anomalous_chunk_rate",
    }
)

_METRIC_KEYS = tuple(THRESHOLD_KEYS)


@dataclass
class BaselineFile:
    """A serializable threshold baseline built from consecutive snapshots."""

    format_version: str = "1.0"
    thresholds: dict[str, Threshold] = field(default_factory=dict)
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    similarity_metric: str | None = None
    tokenizer_id: str = ""
    normalization_version: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    pair_count: int = 0
    method: BaselineMethod = "mean_std"


def _require_uniform(
    snapshots: Sequence[Snapshot],
    field_name: str,
    getter,
    *,
    skip_none: bool,
) -> None:
    values: list[Any] = []
    for snapshot in snapshots:
        value = getter(snapshot)
        if skip_none and value is None:
            continue
        values.append(value)
    if len(set(values)) > 1:
        raise InvalidInputError(
            location="baseline",
            message=f"baseline snapshots disagree on {field_name}",
            hint="rebuild the baseline from snapshots produced with one configuration",
        )


def validate_baseline_snapshots(snapshots: Sequence[Snapshot]) -> None:
    """Reject a build list whose snapshots are not mutually comparable."""
    _require_uniform(
        snapshots,
        "normalization_version",
        lambda snapshot: snapshot.manifest.normalization_version,
        skip_none=False,
    )
    _require_uniform(
        snapshots,
        "tokenizer_id",
        lambda snapshot: snapshot.manifest.tokenizer_id,
        skip_none=False,
    )
    # Embedding fields are only meaningful for snapshots that carry vectors; a
    # text-only snapshot contributes None and is excluded from the check.
    _require_uniform(
        snapshots,
        "embedding_dimension",
        lambda snapshot: snapshot.manifest.embedding_dimension,
        skip_none=True,
    )
    _require_uniform(
        snapshots,
        "similarity_metric",
        lambda snapshot: snapshot.manifest.similarity_metric,
        skip_none=True,
    )


def _mmd_vectors(snapshot: Snapshot, *, normalize: bool) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for record in snapshot.records:
        if record.embedding.validity != "ok" or record.embedding.vector is None:
            continue
        vector = np.asarray(record.embedding.vector, dtype=np.float64)
        if normalize:
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                continue
            vector = vector / norm
        vectors.append(vector)
    if not vectors:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(vectors)


def _embedding_pair(
    v1: Snapshot,
    v2: Snapshot,
    *,
    cfg: Config,
    sink: DiagnosticSink,
):
    side = SideOptions()
    v1_summary = analyze_embeddings(
        v1.records, file_descriptors={}, side=side, cfg=cfg.embeddings, sink=sink
    )
    v2_summary = analyze_embeddings(
        v2.records, file_descriptors={}, side=side, cfg=cfg.embeddings, sink=sink
    )
    comparison_level = determine_comparison_level(
        v1.manifest, v2.manifest, v1.records, v2.records
    )
    compat = determine_compatibility_state(
        v1_summary,
        v2_summary,
        v1.records,
        v2.records,
        v1_dimension=v1.manifest.embedding_dimension,
        v2_dimension=v2.manifest.embedding_dimension,
        comparison_level=comparison_level,
        cfg=cfg,
        sink=sink,
    )
    return v1_summary, v2_summary, comparison_level, compat


def compute_fixed_mmd_bandwidth(
    newest_pair: tuple[Snapshot, Snapshot], *, cfg: Config
) -> float | None:
    """Median-heuristic bandwidth from the newest pair, or None if suspended."""
    v1, v2 = newest_pair
    sink = DiagnosticSink()
    v1_summary, v2_summary, _level, compat = _embedding_pair(
        v1, v2, cfg=cfg, sink=sink
    )
    if "mmd" in ANALYSES_SUSPENDED_BY_STATE.get(compat.state, frozenset()):
        return None
    normalize = (
        v1_summary.similarity_metric == "cosine"
        and v2_summary.similarity_metric == "cosine"
    )
    parts: list[np.ndarray] = []
    for snapshot in (v1, v2):
        vectors = _mmd_vectors(snapshot, normalize=normalize)
        if vectors.size:
            parts.append(vectors)
    if not parts:
        return 0.0
    pooled = np.vstack(parts)
    return float(median_heuristic_bandwidth(pooled, seed=cfg.analysis.seed))


def compute_threshold_from_values(
    values: Sequence[float], *, method: str, min_pairs: int
) -> Threshold | None:
    """Warning/critical levels from accumulated values, or None if too few."""
    series = [float(value) for value in values]
    if len(series) < min_pairs:
        return None
    array = np.asarray(series, dtype=np.float64)
    if method == "mean_std":
        mean = float(np.mean(array))
        std = float(np.std(array, ddof=0))
        warning_raw = mean + 2.0 * std
        critical_raw = mean + 3.0 * std
    elif method == "quantile":
        warning_raw = float(np.percentile(array, 90))
        critical_raw = float(np.percentile(array, 99))
    else:
        raise InvalidInputError(
            location="baseline", message=f"unknown baseline method {method!r}"
        )
    return Threshold(
        warning=round9(max(0.0, warning_raw)),
        critical=round9(max(0.0, critical_raw)),
    )


def _metric_baseline_value(metric: Metric, key: str) -> float | None:
    if key in LEVEL_TYPE_KEYS and isinstance(metric.v2, (int, float)):
        return float(metric.v2)
    if isinstance(metric.change, (int, float)):
        return float(metric.change)
    return None


def _retrieval_drop(result, cfg: Config) -> float | None:
    if not result.modes:
        return None
    primary = result.modes[0]
    if primary.retriever is None:
        return None
    metric = primary.metrics.get(cfg.retrieval.primary_metric)
    if metric is None or metric.status in ("SKIPPED", "N/A"):
        return None
    if not isinstance(metric.v1, (int, float)) or not isinstance(metric.v2, (int, float)):
        return None
    return float(metric.v1) - float(metric.v2)


def _collect_pair_metrics(
    v1: Snapshot,
    v2: Snapshot,
    *,
    cfg: Config,
    compat,
    v1_summary,
    v2_summary,
    comparison_level,
    eval_set: EvalSet | None,
    sink: DiagnosticSink,
) -> tuple[dict[str, Metric], float | None]:
    metrics: list[Metric] = []
    _docs, _chunks, content_metrics = compare_content(v1, v2, cfg=cfg, sink=sink)
    metrics.extend(content_metrics)
    metrics.extend(list(compare_chunking(v1, v2, cfg=cfg)))
    common_record_ids = {record.record_id for record in v1.records} & {
        record.record_id for record in v2.records
    }
    _drifts, metadata_metrics = compare_metadata(
        v1,
        v2,
        cfg=cfg.metadata,
        drift_cfg=cfg.drift,
        common_record_ids=common_record_ids,
    )
    metrics.extend(metadata_metrics)
    _dups_v1, _dups_v2, duplication_metrics = compare_duplicates(
        v1,
        v2,
        cfg=cfg.duplicates,
        drift_cfg=cfg.drift,
        seed=cfg.analysis.seed,
        text_cfg=cfg.text,
        near_threshold=cfg.analysis.near_duplicate_threshold,
        semantic_threshold=cfg.analysis.duplicate_threshold,
        similarity_metric_known=(
            v1.manifest.similarity_metric is not None
            and v2.manifest.similarity_metric is not None
        ),
    )
    metrics.extend(duplication_metrics)
    metrics.append(
        compare_mmd(
            v1,
            v2,
            compat=compat,
            v1_summary=v1_summary,
            v2_summary=v2_summary,
            cfg=cfg,
            sink=sink,
        )
    )
    if cfg.cluster.mode != "off":
        metrics.extend(
            compare_clusters(
                v1,
                v2,
                compat=compat,
                v1_summary=v1_summary,
                v2_summary=v2_summary,
                cfg=cfg,
                sink=sink,
            )
        )
    by_name: dict[str, Metric] = {}
    for metric in metrics:
        by_name[metric.name] = metric
    retrieval_drop: float | None = None
    if eval_set is not None:
        retrieval_result = run_retrieval_comparison(
            v1,
            v2,
            eval_set,
            cfg=cfg,
            compat=compat,
            v1_summary=v1_summary,
            v2_summary=v2_summary,
            comparison_level=comparison_level,
            sink=sink,
        )
        retrieval_drop = _retrieval_drop(retrieval_result, cfg)
    return by_name, retrieval_drop


def build_baseline(
    snapshots: Sequence[Snapshot], *, eval_set: EvalSet | None, cfg: Config
) -> BaselineFile:
    """Run the analysis pipeline over consecutive pairs and freeze thresholds."""
    ordered = list(snapshots)
    validate_baseline_snapshots(ordered)
    sink = DiagnosticSink()

    fixed_bandwidth: float | None = None
    if len(ordered) >= 2:
        fixed_bandwidth = compute_fixed_mmd_bandwidth(
            (ordered[-2], ordered[-1]), cfg=cfg
        )

    build_cfg = cfg
    if fixed_bandwidth is not None:
        build_cfg = replace(
            cfg, analysis=replace(cfg.analysis, bandwidth=fixed_bandwidth)
        )

    accumulators: dict[str, list[float]] = {key: [] for key in _METRIC_KEYS}
    for index in range(max(0, len(ordered) - 1)):
        v1, v2 = ordered[index], ordered[index + 1]
        v1_summary, v2_summary, comparison_level, compat = _embedding_pair(
            v1, v2, cfg=build_cfg, sink=sink
        )
        by_name, retrieval_drop = _collect_pair_metrics(
            v1,
            v2,
            cfg=build_cfg,
            compat=compat,
            v1_summary=v1_summary,
            v2_summary=v2_summary,
            comparison_level=comparison_level,
            eval_set=eval_set,
            sink=sink,
        )
        for key in _METRIC_KEYS:
            if key == "retrieval_drop":
                if retrieval_drop is not None:
                    accumulators[key].append(retrieval_drop)
                continue
            metric = by_name.get(key)
            if metric is None or metric.status in ("SKIPPED", "N/A"):
                continue
            value = _metric_baseline_value(metric, key)
            if value is not None:
                accumulators[key].append(value)

    thresholds: dict[str, Threshold] = {}
    for key in _METRIC_KEYS:
        threshold = compute_threshold_from_values(
            accumulators[key],
            method=cfg.baseline.method,
            min_pairs=cfg.baseline.min_pairs,
        )
        if threshold is not None:
            thresholds[key] = threshold

    reference = ordered[-1].manifest if ordered else None
    settings = {
        "retrieval.primary_metric": cfg.retrieval.primary_metric,
        "retrieval.k": list(cfg.retrieval.k),
        "ci.retrieval_drop_mode": cfg.ci.retrieval_drop_mode,
        "sample_size": cfg.analysis.sample_size,
        "seed": cfg.analysis.seed,
        "bandwidth": fixed_bandwidth,
    }
    return BaselineFile(
        format_version="1.0",
        thresholds=thresholds,
        embedding_model=reference.embedding_model if reference else None,
        embedding_dimension=reference.embedding_dimension if reference else None,
        similarity_metric=reference.similarity_metric if reference else None,
        tokenizer_id=reference.tokenizer_id if reference else "",
        normalization_version=reference.normalization_version if reference else "",
        settings=settings,
        pair_count=max(0, len(ordered) - 1),
        method=cfg.baseline.method,
    )


def _baseline_incompatible(
    cfg: Config, baseline: BaselineFile, manifest: Manifest
) -> bool:
    if baseline.embedding_dimension is not None and manifest.embedding_dimension not in (
        None,
        baseline.embedding_dimension,
    ):
        return True
    if baseline.similarity_metric is not None and manifest.similarity_metric not in (
        None,
        baseline.similarity_metric,
    ):
        return True
    if baseline.tokenizer_id and manifest.tokenizer_id not in (
        None,
        baseline.tokenizer_id,
    ):
        return True
    if baseline.normalization_version and manifest.normalization_version not in (
        None,
        baseline.normalization_version,
    ):
        return True
    if baseline.embedding_model is None:
        if manifest.embedding_model is not None:
            return True
    elif manifest.embedding_model != baseline.embedding_model:
        return True
    current = {
        "retrieval.primary_metric": cfg.retrieval.primary_metric,
        "retrieval.k": list(cfg.retrieval.k),
        "ci.retrieval_drop_mode": cfg.ci.retrieval_drop_mode,
        "sample_size": cfg.analysis.sample_size,
        "seed": cfg.analysis.seed,
        "bandwidth": cfg.analysis.bandwidth,
    }
    for key, value in current.items():
        if baseline.settings.get(key) != value:
            return True
    return False


def apply_baseline(
    cfg: Config,
    baseline: BaselineFile,
    *,
    v1_manifest: Manifest,
    v2_manifest: Manifest,
    sink: DiagnosticSink,
) -> Config:
    """Return cfg with unset thresholds filled from the baseline.

    An incompatible baseline never raises: the config is returned unchanged and
    a BASELINE_INCOMPATIBLE diagnostic is recorded so the run stays explicit.
    """
    if _baseline_incompatible(cfg, baseline, v1_manifest):
        sink.add(
            Code.BASELINE_INCOMPATIBLE,
            "baseline was built for a different configuration or embedding space",
            dimension=None,
        )
        return cfg

    new_drift: dict[str, Threshold | None] = {}
    for key in _METRIC_KEYS:
        explicit = cfg.drift.get(key)
        if explicit is not None:
            new_drift[key] = explicit
            continue
        fallback = baseline.thresholds.get(key)
        new_drift[key] = replace(fallback) if fallback is not None else None
    return replace(cfg, drift=new_drift)


def write_baseline(baseline: BaselineFile, path: str | Path) -> None:
    """Serialize a baseline to JSON, creating parent directories as needed."""
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(baseline), indent=2, sort_keys=True)
    target.write_text(payload + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> BaselineFile:
    """Read and validate a baseline JSON file."""
    target = Path(path)
    if not target.exists():
        raise InvalidInputError(
            location=str(target), message="baseline file not found"
        )
    data = loads_limited(target.read_text(encoding="utf-8"), location=str(target))
    thresholds = {
        key: Threshold(**value) for key, value in data.get("thresholds", {}).items()
    }
    return BaselineFile(
        format_version=data.get("format_version", "1.0"),
        thresholds=thresholds,
        embedding_model=data.get("embedding_model"),
        embedding_dimension=data.get("embedding_dimension"),
        similarity_metric=data.get("similarity_metric"),
        tokenizer_id=data.get("tokenizer_id", ""),
        normalization_version=data.get("normalization_version", ""),
        settings=dict(data.get("settings", {})),
        pair_count=int(data.get("pair_count", 0)),
        method=data.get("method", "mean_std"),
    )
