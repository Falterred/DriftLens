"""Snapshot manifest shape (Step 20)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from driftlens.identity.documents import Granularity, GranularitySource

SNAPSHOT_SCHEMA_VERSION = "1.0"


@dataclass
class FieldMappingSummary:
    """Role -> source key summary of the adapter's field mapping."""

    roles: dict[str, str] = field(default_factory=dict)
    ignored: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Manifest:
    """Snapshot-level descriptors and identity aggregates."""

    schema_version: str
    snapshot_id: str
    content_digest: str
    normalization_version: str
    tokenizer_id: str
    text_params: dict[str, int | bool]
    driftlens_version: str
    created_at: str
    source_type: str
    granularity: Granularity
    granularity_source: GranularitySource
    document_count: int
    chunk_count: int
    embedding_dimension: int | None
    embedding_dtype: str | None
    embedding_model: str | None
    embedding_model_counts: dict[str, int]
    embedding_provider: str | None
    embedding_metadata_confidence: dict[str, str]
    similarity_metric: str | None
    embedding_normalized: bool | None
    chunk_identity_levels: dict[str, list[int]]
    identity_summary: dict[str, int]
    field_mapping: FieldMappingSummary
    unsafe_pickle: bool
    hash_algorithm: str = "sha256"
    files: dict[str, str] = field(default_factory=dict)
    limits_applied: dict[str, Any] = field(default_factory=dict)
