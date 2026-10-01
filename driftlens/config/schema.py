"""Configuration schema: dataclasses, defaults and validation for DriftLens."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from driftlens.errors import InvalidInputError, LimitExceededError

DIMENSIONS = (
    "content",
    "chunking",
    "metadata",
    "embedding",
    "semantic",
    "duplication",
    "retrieval",
)

DIMENSION_OF_KEY = {
    "mmd": "embedding",
    "chunk_ks": "chunking",
    "chunk_wasserstein": "chunking",
    "anomalous_chunk_rate": "chunking",
    "metadata_js": "metadata",
    "metadata_ks": "metadata",
    "metadata_null_rate": "metadata",
    "content_change_rate": "content",
    "duplicate_rate": "duplication",
    "duplicate_rate_exact": "duplication",
    "duplicate_rate_near": "duplication",
    "duplicate_rate_semantic": "duplication",
    "retrieval_drop": "retrieval",
    "semantic": "semantic",
}

THRESHOLD_KEYS = tuple(DIMENSION_OF_KEY)

PRIVACY_MAX_LABEL_CATEGORIES = 20

_VALID_ON_DUPLICATE_ID = ("suffix", "error")
_VALID_CLUSTER_MODES = ("auto", "on", "off")
_VALID_BASELINE_METHODS = ("mean_std", "quantile")
_VALID_RETRIEVAL_MODES = ("auto", "lexical", "dense", "both")
_VALID_CI_POLICIES = ("warn", "fail", "ignore")
_VALID_RETRIEVAL_DROP_MODES = ("absolute", "relative")
_VALID_SIMILARITY_METRICS = ("cosine", "dot", "l2")
_VALID_TIMESTAMP_UNITS = ("iso", "seconds", "milliseconds")
_VALID_GRANULARITIES = ("AUTO", "CHUNK", "DOCUMENT")
_PRIMARY_METRIC_RE = re.compile(r"^(recall|hit_rate)@(\d+)$")


@dataclass
class Threshold:
    warning: float | None = None
    critical: float | None = None


@dataclass
class SideOptions:
    id_field: str | None = None
    chunk_id_field: str | None = None
    text_field: str | None = None
    embedding_field: str | None = None
    metadata_fields: list[str] | None = None
    model_field: str | None = None
    granularity: str = "AUTO"
    sidecar: str | None = None
    timestamp_unit: str = "iso"
    chroma_collection: str | None = None
    format: str | None = None
    embedding_format: str = "json"
    embedding_prefix: str | None = None
    model: str | None = None
    similarity_metric: str | None = None
    allow_lossy_vectors: bool = False


@dataclass
class EmbeddingsCfg:
    model: str | None = None
    similarity_metric: str | None = None
    min_coverage: float = 0.95
    max_invalid_rate: float = 0.01
    normalized_tolerance: dict[str, float] = field(
        default_factory=lambda: {"float32": 1e-3, "float16": 1e-2}
    )
    normalized_share: float = 0.99
    consistency_cosine: float = 0.99
    consistency_same: float = 0.95
    consistency_different: float = 0.05
    consistency_min_anchors: int = 30
    consistency_max_anchors: int = 5000


@dataclass
class AnalysisCfg:
    sample_size: int = 10000
    seed: int = 42
    duplicate_threshold: float = 0.98
    near_duplicate_threshold: float = 0.90
    bandwidth: float | None = None
    workers: int = 4


@dataclass
class IdentityCfg:
    warn_rate: float = 0.01
    on_duplicate_id: str = "suffix"


@dataclass
class DocumentsCfg:
    replace_threshold: float = 0.20


@dataclass
class ChunkMatchCfg:
    min_similarity: float = 0.60
    coverage: float = 0.80
    split_contribution: float = 0.10
    rechunk_exact_fraction: float = 0.50


@dataclass
class DuplicatesCfg:
    ignore_adjacent_chunks: bool = True
    max_exact_vectors: int = 50000
    minhash_permutations: int = 128


@dataclass
class TextCfg:
    min_overlap_chars: int = 16
    overlap_window: int = 2000
    casefold: bool = True


@dataclass
class ChunkingCfg:
    short: int = 3
    suspicious_short: int = 20
    oversized: int = 2048
    token_safety_factor: float = 1.3
    embedding_max_tokens: int | None = None


@dataclass
class MetadataCfg:
    max_depth: int = 3
    max_categories: int = 200
    max_fields: int = 2000
    mixed_type_share: float = 0.05


@dataclass
class ClusterCfg:
    mode: str = "auto"
    pca_dims: int = 50
    min_cluster_size: int | str = "auto"
    min_common: float = 0.10
    centroid_cosine: float = 0.90
    stable_jaccard: float = 0.70
    stable_centroid_cosine: float = 0.95
    size_ratio_min: float = 0.5
    size_ratio_max: float = 2.0
    split_share: float = 0.20


@dataclass
class NeighbourCfg:
    k: int = 10
    max_queries: int = 2000


@dataclass
class SignificanceCfg:
    permutations: int = 999
    bootstrap: int = 200
    max_n: int = 5000


@dataclass
class BaselineCfg:
    file: str | None = None
    min_pairs: int = 5
    method: str = "mean_std"


@dataclass
class Bm25Cfg:
    k1: float = 1.2
    b: float = 0.75


@dataclass
class RetrievalCfg:
    mode: str = "auto"
    k: list[int] = field(default_factory=lambda: [1, 5, 10])
    primary_metric: str = "recall@5"
    eval_set: str | None = None
    min_similarity: float | None = None
    bm25: Bm25Cfg = field(default_factory=Bm25Cfg)


@dataclass
class PrivacyCfg:
    redact_ids: bool = False
    top_n: int = 10


@dataclass
class LimitsCfg:
    max_records: int = 10_000_000
    max_dimension: int = 16384
    max_text_length: int = 1_000_000
    max_sample_size: int = 50_000
    max_workers: int | str = "auto"
    max_input_bytes: int = 68_719_476_736
    max_memory: float = 0.75


@dataclass
class CiCfg:
    fail_on: dict[str, str] = field(
        default_factory=lambda: {
            "embedding": "critical",
            "retrieval": "critical",
            "duplication": "warning",
        }
    )
    on_skipped: str = "warn"
    on_incompatible: str = "ignore"
    retrieval_drop_mode: str = "absolute"
    critical_gates: set[str] = field(default_factory=set)


@dataclass
class Config:
    schema: SideOptions
    schema_a: dict
    schema_b: dict
    embeddings: EmbeddingsCfg
    analysis: AnalysisCfg
    identity: IdentityCfg
    documents: DocumentsCfg
    chunk_match: ChunkMatchCfg
    duplicates: DuplicatesCfg
    text: TextCfg
    chunking: ChunkingCfg
    metadata: MetadataCfg
    cluster: ClusterCfg
    neighbour: NeighbourCfg
    significance: SignificanceCfg
    drift: dict[str, Threshold | None]
    baseline: BaselineCfg
    retrieval: RetrievalCfg
    privacy: PrivacyCfg
    limits: LimitsCfg
    ci: CiCfg


def default_config() -> Config:
    """Build a configuration in which every block holds its default value."""

    return Config(
        schema=SideOptions(),
        schema_a={},
        schema_b={},
        embeddings=EmbeddingsCfg(),
        analysis=AnalysisCfg(),
        identity=IdentityCfg(),
        documents=DocumentsCfg(),
        chunk_match=ChunkMatchCfg(),
        duplicates=DuplicatesCfg(),
        text=TextCfg(),
        chunking=ChunkingCfg(),
        metadata=MetadataCfg(),
        cluster=ClusterCfg(),
        neighbour=NeighbourCfg(),
        significance=SignificanceCfg(),
        drift={key: None for key in THRESHOLD_KEYS},
        baseline=BaselineCfg(),
        retrieval=RetrievalCfg(),
        privacy=PrivacyCfg(),
        limits=LimitsCfg(),
        ci=CiCfg(),
    )


def _side_option_values(cfg: Config, name: str) -> list[object]:
    values: list[object] = [getattr(cfg.schema, name)]
    for overrides in (cfg.schema_a, cfg.schema_b):
        if isinstance(overrides, dict) and name in overrides:
            values.append(overrides[name])
    return values


def resolved_max_workers(cfg: Config) -> int:
    """Resolve the limits.max_workers setting to a concrete number."""

    limit = cfg.limits.max_workers
    if limit == "auto":
        return min(os.cpu_count() or 1, 64)
    return int(limit)


def _invalid(message: str) -> InvalidInputError:
    return InvalidInputError(message=message, location="config")


def validate_config(cfg: Config) -> None:
    """Validate every enumerated value and bound; raises for the first problem."""

    if cfg.identity.on_duplicate_id not in _VALID_ON_DUPLICATE_ID:
        raise _invalid(f"invalid identity.on_duplicate_id {cfg.identity.on_duplicate_id!r}")
    if cfg.cluster.mode not in _VALID_CLUSTER_MODES:
        raise _invalid(f"invalid cluster.mode {cfg.cluster.mode!r}")
    if cfg.baseline.method not in _VALID_BASELINE_METHODS:
        raise _invalid(f"invalid baseline.method {cfg.baseline.method!r}")
    if cfg.retrieval.mode not in _VALID_RETRIEVAL_MODES:
        raise _invalid(f"invalid retrieval.mode {cfg.retrieval.mode!r}")
    if cfg.ci.on_skipped not in _VALID_CI_POLICIES:
        raise _invalid(f"invalid ci.on_skipped {cfg.ci.on_skipped!r}")
    if cfg.ci.on_incompatible not in _VALID_CI_POLICIES:
        raise _invalid(f"invalid ci.on_incompatible {cfg.ci.on_incompatible!r}")
    if cfg.ci.retrieval_drop_mode not in _VALID_RETRIEVAL_DROP_MODES:
        raise _invalid(f"invalid ci.retrieval_drop_mode {cfg.ci.retrieval_drop_mode!r}")

    for value in _side_option_values(cfg, "granularity"):
        if value is None or str(value).upper() not in _VALID_GRANULARITIES:
            raise _invalid(f"invalid granularity {value!r}")
    for value in _side_option_values(cfg, "timestamp_unit"):
        if value not in _VALID_TIMESTAMP_UNITS:
            raise _invalid(f"invalid timestamp_unit {value!r}")
    for value in _side_option_values(cfg, "similarity_metric"):
        if value is not None and value not in _VALID_SIMILARITY_METRICS:
            raise _invalid(f"invalid similarity_metric {value!r}")

    embeddings_metric = cfg.embeddings.similarity_metric
    if embeddings_metric is not None and embeddings_metric not in _VALID_SIMILARITY_METRICS:
        raise _invalid(f"invalid embeddings.similarity_metric {embeddings_metric!r}")

    primary_metric = cfg.retrieval.primary_metric
    if primary_metric != "mrr":
        matched = _PRIMARY_METRIC_RE.match(primary_metric)
        if matched is None or int(matched.group(2)) not in cfg.retrieval.k:
            raise _invalid(f"invalid retrieval.primary_metric {primary_metric!r}")

    fractions = (
        ("embeddings.min_coverage", cfg.embeddings.min_coverage),
        ("embeddings.max_invalid_rate", cfg.embeddings.max_invalid_rate),
        ("embeddings.normalized_share", cfg.embeddings.normalized_share),
        ("embeddings.consistency_cosine", cfg.embeddings.consistency_cosine),
        ("embeddings.consistency_same", cfg.embeddings.consistency_same),
        ("embeddings.consistency_different", cfg.embeddings.consistency_different),
        ("documents.replace_threshold", cfg.documents.replace_threshold),
        ("identity.warn_rate", cfg.identity.warn_rate),
        ("analysis.duplicate_threshold", cfg.analysis.duplicate_threshold),
        ("analysis.near_duplicate_threshold", cfg.analysis.near_duplicate_threshold),
    )
    for name, value in fractions:
        if not 0.0 <= value <= 1.0:
            raise _invalid(f"{name} must be within [0, 1]")

    positive_counts = (
        ("analysis.sample_size", cfg.analysis.sample_size),
        ("analysis.workers", cfg.analysis.workers),
        ("significance.permutations", cfg.significance.permutations),
        ("significance.bootstrap", cfg.significance.bootstrap),
        ("significance.max_n", cfg.significance.max_n),
        ("baseline.min_pairs", cfg.baseline.min_pairs),
    )
    for name, value in positive_counts:
        if value <= 0:
            raise _invalid(f"{name} must be a positive count")
    for value in cfg.retrieval.k:
        if value <= 0:
            raise _invalid("retrieval.k values must be positive counts")

    if cfg.analysis.workers > resolved_max_workers(cfg):
        raise LimitExceededError(flag="--max-workers")
    if cfg.analysis.sample_size > cfg.limits.max_sample_size:
        raise LimitExceededError(flag="--max-sample-size")

    for key, threshold in cfg.drift.items():
        if threshold is None:
            continue
        if (
            threshold.warning is not None
            and threshold.critical is not None
            and threshold.warning > threshold.critical
        ):
            raise _invalid(f"drift.{key}: warning exceeds critical")
