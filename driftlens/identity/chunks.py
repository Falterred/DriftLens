"""Chunk-level identity and text/embedding digest fields (Step 19)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from driftlens.errors import UndefinedCaseError
from driftlens.identity.hashing import embedding_digest, sha256_hex
from driftlens.schema.record import Record
from driftlens.text.normalize import check_utf8_clean, normalize_text
from driftlens.text.sentences import sentence_count
from driftlens.text.tokenizer import token_count

CHUNK_LEVEL_KEY_PREFIXES = {1: "c1:", 2: "c2:", 3: "c3:", 4: "c4:"}

_IdentityConfidence = Literal["HIGH", "MEDIUM", "LOW"]
_DAMAGE_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class ChunkIdentityResult:
    """The highest available chunk-identity level for one record."""

    level: int
    key: str
    confidence: _IdentityConfidence


def compute_text_hashes(record: Record) -> None:
    """Populate the text-derived fields on a record from its raw text."""

    if record.text is None:
        return
    text = check_utf8_clean(record.text, location="record text")
    record.raw_text_hash = sha256_hex(text.encode("utf-8"))
    normalized = normalize_text(text)
    record.normalized_text = normalized
    record.normalized_text_hash = sha256_hex(normalized.encode("utf-8"))
    record.token_count = token_count(normalized)
    record.char_count = len(normalized)
    record.sentence_count = sentence_count(normalized)
    record.encoding_damage = bool(_DAMAGE_RE.search(text))


def compute_embedding_digest_field(record: Record) -> None:
    """Cache a transient little-endian embedding digest for identity levels."""

    vector = record.embedding.vector
    if vector is None:
        return
    dtype = record.embedding.dtype or str(vector.dtype)
    dimension = record.embedding.dimension
    if dimension is None:
        dimension = int(vector.shape[-1])
    little = vector.astype(vector.dtype.newbyteorder("<")).tobytes()
    record._embedding_digest_hex = embedding_digest(dtype, dimension, little)


def _identity_hint(record: Record) -> str | None:
    transient = getattr(record, "_identity_source_hint", None)
    if transient is not None:
        return transient
    return record.identity_source_hint


def _explicit_chunk_id(record: Record) -> str | None:
    if record.record_id is None:
        return None
    if _identity_hint(record) == "vector_position":
        return None
    return record.record_id


def all_chunk_identity_keys(
    record: Record,
    *,
    document_identity_confidence: str | None,
    duplicate_rank_within_document: int | None,
) -> dict[int, str]:
    """Every available chunk-identity key for a record, by level."""

    keys: dict[int, str] = {}
    explicit = _explicit_chunk_id(record)
    if explicit is not None:
        keys[1] = CHUNK_LEVEL_KEY_PREFIXES[1] + explicit

    document_id = record.document_id
    if (
        document_id is not None
        and record.chunk_index is not None
        and document_identity_confidence != "LOW"
    ):
        keys[2] = f"{CHUNK_LEVEL_KEY_PREFIXES[2]}{document_id}#{record.chunk_index}"

    if (
        document_id is not None
        and record.normalized_text_hash is not None
        and document_identity_confidence != "LOW"
        and duplicate_rank_within_document is not None
    ):
        keys[3] = (
            f"{CHUNK_LEVEL_KEY_PREFIXES[3]}{document_id}#"
            f"{record.normalized_text_hash}#{duplicate_rank_within_document}"
        )

    if record.normalized_text_hash is not None:
        keys[4] = CHUNK_LEVEL_KEY_PREFIXES[4] + record.normalized_text_hash
    else:
        digest = getattr(record, "_embedding_digest_hex", None)
        if digest is not None:
            keys[4] = CHUNK_LEVEL_KEY_PREFIXES[4] + digest

    return keys


def _confidence_for_level(record: Record, level: int) -> _IdentityConfidence:
    if level == 1:
        return "MEDIUM" if _identity_hint(record) == "indexidmap" else "HIGH"
    if level in (2, 3):
        return "MEDIUM"
    return "LOW"


def compute_chunk_identity(
    record: Record,
    *,
    document_identity_confidence: str | None,
    duplicate_rank_within_document: int | None,
) -> ChunkIdentityResult:
    """First (highest priority) available chunk-identity level for a record."""

    keys = all_chunk_identity_keys(
        record,
        document_identity_confidence=document_identity_confidence,
        duplicate_rank_within_document=duplicate_rank_within_document,
    )
    for level in (1, 2, 3, 4):
        if level in keys:
            return ChunkIdentityResult(
                level=level,
                key=keys[level],
                confidence=_confidence_for_level(record, level),
            )
    raise UndefinedCaseError(case="chunk identity")
