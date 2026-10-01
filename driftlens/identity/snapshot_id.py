"""Snapshot descriptor and identifier construction (Step 20)."""

from __future__ import annotations

from dataclasses import dataclass

from driftlens.identity.documents import Granularity, _digest_bytes
from driftlens.identity.hashing import (
    canonical_json,
    content_digest,
)
from driftlens.identity.hashing import snapshot_id as _snapshot_id
from driftlens.schema.record import Record


@dataclass
class EmbeddingDescriptor:
    """Snapshot-level embedding space descriptors included in the snapshot id."""

    model: str | None
    provider: str | None
    dtype: str | None
    dimension: int | None
    similarity_metric: str | None
    normalized: bool | None


def build_descriptor(
    schema_version: str,
    normalization_version: str,
    tokenizer_id: str,
    granularity: Granularity,
    embedding: EmbeddingDescriptor,
) -> bytes:
    """dl-json-1 canonical bytes of the snapshot identity descriptor."""

    payload = {
        "schema_version": schema_version,
        "normalization_version": normalization_version,
        "tokenizer_id": tokenizer_id,
        "granularity": granularity,
        "embedding": {
            "model": embedding.model,
            "provider": embedding.provider,
            "dtype": embedding.dtype,
            "dimension": embedding.dimension,
            "similarity_metric": embedding.similarity_metric,
            "normalized": embedding.normalized,
        },
    }
    return canonical_json(payload)


def compute_snapshot_id(
    records: list[Record], descriptor_bytes: bytes
) -> tuple[str, bytes]:
    """Return (snapshot id string, raw content digest bytes) for final records."""

    digests = {record.record_id: _digest_bytes(record) for record in records}
    content = content_digest(digests)
    return _snapshot_id(content, descriptor_bytes), content
