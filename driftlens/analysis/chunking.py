"""Chunk-profile, overlap and anomaly analysis (Step 26)."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from driftlens.analysis.distributions import (
    FIXED_BIN_EDGES,
    js_divergence_histogram,
    ks_statistic,
    mean_median_variance_quantiles,
    wasserstein1_normalized,
)
from driftlens.analysis.health import Metric, SampleInfo, make_metric
from driftlens.analysis.sampling import hash_threshold_sample, to_sample_info
from driftlens.config.schema import ChunkingCfg, Config
from driftlens.schema.record import Record
from driftlens.schema.snapshot import DocumentTable, Snapshot
from driftlens.text.tokenizer import tokenize

AnomalyCategory = Literal[
    "empty", "very_short", "suspicious_short", "oversized", "over_embedder_limit"
]

_INCLUDED_CATEGORIES = {"empty", "very_short", "oversized", "over_embedder_limit"}
_PAIR_SEPARATOR = "\x1f"
_OVERLAP_MIN_TOKENS = 3
_OVERLAP_PAIR_CAP = 100_000


@dataclass
class ChunkAnomalyFlags:
    size_category: AnomalyCategory | None
    encoding_damage: bool


@dataclass
class OverlapEstimate:
    share_with_overlap: float
    median_l: float
    p95_l: float
    median_l_over_tokens_a: float
    sample: SampleInfo


@dataclass
class ChunkProfile:
    token_stats: dict[str, float] | None
    char_stats: dict[str, float] | None
    sentence_stats: dict[str, float] | None
    chunks_per_document: dict[str, int]
    overlap_estimate: OverlapEstimate | None


@dataclass
class AnomalyCounts:
    empty: int
    very_short: int
    suspicious_short: int
    oversized: int
    over_embedder_limit: int
    encoding_damage: int
    anomalous_rate: float
    total: int


def classify_chunk_anomaly(record: Record, *, cfg: ChunkingCfg) -> ChunkAnomalyFlags:
    """First-match size cascade plus the orthogonal encoding-damage flag."""
    text = record.text
    if text is None:
        return ChunkAnomalyFlags(
            size_category=None, encoding_damage=bool(record.encoding_damage)
        )

    size_category: AnomalyCategory | None = None
    count = record.token_count
    if count is not None:
        if count == 0:
            size_category = "empty"
        elif count < cfg.short:
            size_category = "very_short"
        elif count < cfg.suspicious_short:
            size_category = "suspicious_short"
        elif cfg.embedding_max_tokens is not None:
            adjusted = count * cfg.token_safety_factor
            if adjusted > cfg.embedding_max_tokens:
                size_category = "over_embedder_limit"
            elif count > cfg.oversized:
                size_category = "oversized"
        elif count > cfg.oversized:
            size_category = "oversized"

    return ChunkAnomalyFlags(
        size_category=size_category, encoding_damage=bool(record.encoding_damage)
    )


def _token_overlap(
    tokens_a: Sequence[str], tokens_b: Sequence[str], *, min_tokens: int
) -> int:
    max_length = min(len(tokens_a), len(tokens_b))
    for length in range(max_length, min_tokens - 1, -1):
        if list(tokens_a[-length:]) == list(tokens_b[:length]):
            return length
    return 0


def compute_overlap_estimate(
    records: Sequence[Record], *, seed: int
) -> OverlapEstimate | None:
    """Section 5.3 adjacent-chunk token overlap, sampled to 100,000 pairs."""
    if not any(record.chunk_index is not None for record in records):
        return None

    grouped: dict[object, list[Record]] = {}
    for record in records:
        grouped.setdefault(record.document_id, []).append(record)

    pairs: list[tuple[Record, Record]] = []
    for group in grouped.values():
        indexed = sorted(
            (record for record in group if record.chunk_index is not None),
            key=lambda record: record.chunk_index,
        )
        for first, second in zip(indexed, indexed[1:]):
            pairs.append((first, second))

    pair_keys = [
        first.record_id + _PAIR_SEPARATOR + second.record_id for first, second in pairs
    ]
    sample = hash_threshold_sample(seed, _OVERLAP_PAIR_CAP, pair_keys)
    kept = [
        pair
        for pair, key in zip(pairs, pair_keys)
        if key in sample.v1_ids
    ]

    overlaps: list[int] = []
    ratios: list[float] = []
    for first, second in kept:
        tokens_a = tokenize(first.normalized_text or first.text or "")
        tokens_b = tokenize(second.normalized_text or second.text or "")
        length = _token_overlap(tokens_a, tokens_b, min_tokens=_OVERLAP_MIN_TOKENS)
        overlaps.append(length)
        if tokens_a:
            ratios.append(length / len(tokens_a))

    if overlaps:
        share = sum(1 for length in overlaps if length > 0) / len(overlaps)
        median_l = float(statistics.median(overlaps))
        p95_l = float(np.percentile(overlaps, 95))
    else:
        share = 0.0
        median_l = 0.0
        p95_l = 0.0
    ratio_median = float(statistics.median(ratios)) if ratios else 0.0

    return OverlapEstimate(
        share_with_overlap=share,
        median_l=median_l,
        p95_l=p95_l,
        median_l_over_tokens_a=ratio_median,
        sample=to_sample_info(sample, side="v1"),
    )


def compute_chunk_profile(
    records: Sequence[Record], documents: DocumentTable, *, seed: int = 42
) -> ChunkProfile:
    """Descriptive profile plus the per-document counts and overlap estimate."""
    token_stats = mean_median_variance_quantiles(
        [record.token_count for record in records if record.token_count is not None]
    )
    char_stats = mean_median_variance_quantiles(
        [record.char_count for record in records if record.char_count is not None]
    )
    sentence_stats = mean_median_variance_quantiles(
        [
            record.sentence_count
            for record in records
            if record.sentence_count is not None
        ]
    )
    return ChunkProfile(
        token_stats=token_stats,
        char_stats=char_stats,
        sentence_stats=sentence_stats,
        chunks_per_document={
            document_id: row.chunk_count for document_id, row in documents.rows.items()
        },
        overlap_estimate=compute_overlap_estimate(records, seed=seed),
    )


def compute_anomaly_counts(
    records: Sequence[Record], *, cfg: ChunkingCfg
) -> AnomalyCounts:
    """Per-chunk OR of the included anomaly dimensions (suspicious excluded)."""
    flags = [classify_chunk_anomaly(record, cfg=cfg) for record in records]
    categories = Counter(
        flag.size_category for flag in flags if flag.size_category is not None
    )
    encoding_damage = sum(1 for flag in flags if flag.encoding_damage)
    included = sum(
        1
        for flag in flags
        if flag.size_category in _INCLUDED_CATEGORIES or flag.encoding_damage
    )
    total = len(records)
    return AnomalyCounts(
        empty=categories["empty"],
        very_short=categories["very_short"],
        suspicious_short=categories["suspicious_short"],
        oversized=categories["oversized"],
        over_embedder_limit=categories["over_embedder_limit"],
        encoding_damage=encoding_damage,
        anomalous_rate=(included / total) if total else 0.0,
        total=total,
    )


def _value_list(records: Sequence[Record], attribute: str) -> list[float]:
    return [
        float(value)
        for value in (getattr(record, attribute) for record in records)
        if value is not None
    ]


def _distance_metric(name: str, v1_values, v2_values, *, threshold):
    value, reason = ks_statistic(v1_values, v2_values)
    return make_metric(name, value=value, threshold=threshold, reason=reason)


def _wasserstein_metric(name: str, v1_values, v2_values, *, threshold):
    value, reason = wasserstein1_normalized(v1_values, v2_values)
    return make_metric(name, value=value, threshold=threshold, reason=reason)


def _count_metric(name: str, v1_count: int, v2_count: int, *, approximate: bool = False):
    return make_metric(
        name,
        value=v2_count,
        threshold=None,
        v1=v1_count,
        v2=v2_count,
        change=v2_count - v1_count,
        approximate=approximate,
    )


def compare_chunking(v1: Snapshot, v2: Snapshot, *, cfg: Config) -> list[Metric]:
    """Build the CHUNKING dimension's metric list."""
    v1_profile = compute_chunk_profile(
        v1.records, v1.documents, seed=cfg.analysis.seed
    )
    v2_profile = compute_chunk_profile(
        v2.records, v2.documents, seed=cfg.analysis.seed
    )
    v1_anomalies = compute_anomaly_counts(v1.records, cfg=cfg.chunking)
    v2_anomalies = compute_anomaly_counts(v2.records, cfg=cfg.chunking)

    v1_tokens = _value_list(v1.records, "token_count")
    v2_tokens = _value_list(v2.records, "token_count")
    v1_chars = _value_list(v1.records, "char_count")
    v2_chars = _value_list(v2.records, "char_count")
    v1_sentences = _value_list(v1.records, "sentence_count")
    v2_sentences = _value_list(v2.records, "sentence_count")

    metrics: list[Metric] = [
        _distance_metric(
            "chunk_ks", v1_tokens, v2_tokens, threshold=cfg.drift.get("chunk_ks")
        ),
        _wasserstein_metric(
            "chunk_wasserstein",
            v1_tokens,
            v2_tokens,
            threshold=cfg.drift.get("chunk_wasserstein"),
        ),
        _distance_metric("char_ks", v1_chars, v2_chars, threshold=None),
        _wasserstein_metric(
            "char_wasserstein", v1_chars, v2_chars, threshold=None
        ),
        _distance_metric("sentence_ks", v1_sentences, v2_sentences, threshold=None),
        _wasserstein_metric(
            "sentence_wasserstein", v1_sentences, v2_sentences, threshold=None
        ),
    ]

    js_value, js_reason = js_divergence_histogram(
        list(v1_profile.chunks_per_document.values()),
        list(v2_profile.chunks_per_document.values()),
        edges=FIXED_BIN_EDGES,
    )
    metrics.append(
        make_metric("chunks_per_document_js", value=js_value, threshold=None, reason=js_reason)
    )
    v1_counts = list(v1_profile.chunks_per_document.values())
    v2_counts = list(v2_profile.chunks_per_document.values())
    metrics.append(
        _count_metric(
            "chunks_per_document_mean",
            round(sum(v1_counts) / len(v1_counts)) if v1_counts else 0,
            round(sum(v2_counts) / len(v2_counts)) if v2_counts else 0,
        )
    )
    metrics.append(
        _count_metric(
            "chunks_per_document_median",
            round(statistics.median(v1_counts)) if v1_counts else 0,
            round(statistics.median(v2_counts)) if v2_counts else 0,
        )
    )

    metrics.append(
        make_metric(
            "anomalous_chunk_rate",
            value=v2_anomalies.anomalous_rate,
            threshold=cfg.drift.get("anomalous_chunk_rate"),
            v1=v1_anomalies.anomalous_rate,
            v2=v2_anomalies.anomalous_rate,
            change=v2_anomalies.anomalous_rate - v1_anomalies.anomalous_rate,
        )
    )

    for name, attribute, approximate in (
        ("empty_chunks", "empty", False),
        ("very_short_chunks", "very_short", False),
        ("suspicious_short_chunks", "suspicious_short", False),
        ("oversized_chunks", "oversized", False),
        ("over_embedder_limit_chunks", "over_embedder_limit", True),
        ("encoding_damage_chunks", "encoding_damage", False),
    ):
        metrics.append(
            _count_metric(
                name,
                getattr(v1_anomalies, attribute),
                getattr(v2_anomalies, attribute),
                approximate=approximate,
            )
        )

    v1_overlap = v1_profile.overlap_estimate
    v2_overlap = v2_profile.overlap_estimate
    if v1_overlap is not None and v2_overlap is not None:
        for name, attribute in (
            ("overlap_share", "share_with_overlap"),
            ("overlap_median_l", "median_l"),
            ("overlap_p95_l", "p95_l"),
            ("overlap_median_l_over_tokens_a", "median_l_over_tokens_a"),
        ):
            first = getattr(v1_overlap, attribute)
            second = getattr(v2_overlap, attribute)
            metrics.append(
                make_metric(
                    name,
                    value=second,
                    threshold=None,
                    v1=first,
                    v2=second,
                    change=second - first,
                )
            )

    return metrics
