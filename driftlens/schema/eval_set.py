"""Evaluation-set schema, query-id derivation and query-embedding joining (Step 38).

An eval set is a JSON array of query cases (section 11.2). Each case carries a
final query_id (explicit, or derived from the normalized query text), the
raw query text, document- and/or chunk-level ground truth, and optionally
inline query vectors. Query vectors may instead be joined from the
--query-embeddings-a/-b JSONL files at load time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from driftlens.adapters._embedding_codec import decode_embedding_array
from driftlens.errors import InvalidInputError
from driftlens.identity.hashing import sha256_hex
from driftlens.security.limits import RuntimeLimits, loads_limited
from driftlens.text.normalize import normalize_text


@dataclass
class QueryEmbeddings:
    """Optional inline query vectors for both sides of a comparison."""

    v1: np.ndarray | None
    v2: np.ndarray | None


@dataclass
class EvalCase:
    """One query case with its final id and ground truth."""

    query_id: str
    query_id_was_explicit: bool
    query: str
    relevant_document_ids: list[str]
    relevant_chunk_ids: list[str]
    query_embeddings: QueryEmbeddings | None


@dataclass
class EvalSet:
    """A loaded evaluation set."""

    cases: list[EvalCase]


def derive_query_id(query_text: str) -> str:
    """Derive a stable query id from the normalized query text (section 11.2)."""

    normalized = normalize_text(query_text)
    digest = sha256_hex(normalized.encode("utf-8"))
    return "q:" + digest[:12]


def _string_list(obj: dict[str, Any], key: str, *, location: str) -> list[str]:
    """Return a validated list-of-strings field, treating absent/null as empty."""

    if key not in obj or obj[key] is None:
        return []
    value = obj[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise InvalidInputError(
            message=f"{key!r} must be a list of strings", location=location
        )
    return list(value)


def _decode_query_vector(value: Any, *, location: str) -> np.ndarray:
    """Decode one inline or file-supplied query vector (float32 by default)."""

    vector, _dtype = decode_embedding_array(value, declared_dtype=None, location=location)
    return vector


def _read_inline_query_embeddings(
    obj: dict[str, Any], *, location: str
) -> QueryEmbeddings | None:
    if "query_embeddings" not in obj:
        return None
    raw = obj["query_embeddings"]
    if not isinstance(raw, dict):
        raise InvalidInputError(
            message="'query_embeddings' must be an object", location=location
        )
    v1 = None
    if raw.get("v1") is not None:
        v1 = _decode_query_vector(
            raw["v1"], location=f"{location} query_embeddings.v1"
        )
    v2 = None
    if raw.get("v2") is not None:
        v2 = _decode_query_vector(
            raw["v2"], location=f"{location} query_embeddings.v2"
        )
    return QueryEmbeddings(v1=v1, v2=v2)


def load_eval_set(path: Path, *, limits: RuntimeLimits) -> EvalSet:
    """Load and validate an evaluation set from a JSON file (section 11.2)."""

    path = Path(path)
    limits.add_input_bytes(path.stat().st_size, location=str(path))
    top = loads_limited(path.read_text(encoding="utf-8"), location=str(path))
    if not isinstance(top, list):
        raise InvalidInputError(
            message="eval set must be a top-level JSON array", location=str(path)
        )

    cases: list[EvalCase] = []
    for index, obj in enumerate(top, start=1):
        location = f"{path} case {index}"
        if (
            not isinstance(obj, dict)
            or "query" not in obj
            or not isinstance(obj["query"], str)
        ):
            raise InvalidInputError(
                message="missing or invalid 'query' field", location=location
            )

        was_explicit = "query_id" in obj and obj["query_id"] is not None
        if was_explicit:
            if not isinstance(obj["query_id"], str):
                raise InvalidInputError(
                    message="'query_id' must be a string", location=location
                )
            query_id = obj["query_id"]
        else:
            query_id = derive_query_id(obj["query"])

        relevant_documents = _string_list(
            obj, "relevant_document_ids", location=location
        )
        relevant_chunks = _string_list(obj, "relevant_chunk_ids", location=location)
        if not relevant_documents and not relevant_chunks:
            raise InvalidInputError(
                message=(
                    "at least one of relevant_document_ids or relevant_chunk_ids "
                    "is required"
                ),
                location=location,
            )

        cases.append(
            EvalCase(
                query_id=query_id,
                query_id_was_explicit=was_explicit,
                query=obj["query"],
                relevant_document_ids=relevant_documents,
                relevant_chunk_ids=relevant_chunks,
                query_embeddings=_read_inline_query_embeddings(obj, location=location),
            )
        )

    counts = Counter(case.query_id for case in cases)
    duplicates = [query_id for query_id, count in counts.items() if count > 1]
    if duplicates:
        listed = ", ".join(duplicates[:5])
        suffix = "..." if len(duplicates) > 5 else ""
        raise InvalidInputError(
            message=(
                f"{len(duplicates)} duplicate query_id value(s): {listed}{suffix}"
            ),
            location=str(path),
        )

    return EvalSet(cases=cases)


def _load_query_embedding_file(
    path: Path, *, limits: RuntimeLimits
) -> dict[str, np.ndarray]:
    """Load a JSONL query-embedding file into a query_id -> vector map."""

    path = Path(path)
    table: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        for row_number, line_bytes in enumerate(handle, start=1):
            if not line_bytes.strip():
                continue
            location = f"{path} line {row_number}"
            limits.add_input_bytes(len(line_bytes), location=location)
            try:
                line = line_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raise InvalidInputError(
                    message="invalid UTF-8", location=location
                ) from None
            row = loads_limited(line, location=location)
            if not isinstance(row, dict):
                raise InvalidInputError(
                    message="row is not a JSON object", location=location
                )
            query_id = row.get("query_id")
            if not isinstance(query_id, str):
                raise InvalidInputError(
                    message="'query_id' must be a string", location=location
                )
            if row.get("embedding") is None:
                raise InvalidInputError(
                    message="missing 'embedding' field", location=location
                )
            table[query_id] = _decode_query_vector(
                row["embedding"], location=location
            )
    return table


def join_query_embedding_files(
    eval_set: EvalSet,
    *,
    path_a: Path | None,
    path_b: Path | None,
    limits: RuntimeLimits,
) -> EvalSet:
    """Join the --query-embeddings-a/-b JSONL files onto the eval cases."""

    if path_a is None and path_b is None:
        return eval_set

    non_explicit = [
        case.query_id for case in eval_set.cases if not case.query_id_was_explicit
    ]
    if non_explicit:
        raise InvalidInputError(
            message=(
                f"{len(non_explicit)} eval-set case(s) have no explicit query_id, "
                "required when --query-embeddings-a/-b is given"
            )
        )

    file_a = (
        _load_query_embedding_file(path_a, limits=limits)
        if path_a is not None
        else {}
    )
    file_b = (
        _load_query_embedding_file(path_b, limits=limits)
        if path_b is not None
        else {}
    )

    cases: list[EvalCase] = []
    for case in eval_set.cases:
        inline = case.query_embeddings
        inline_v1 = inline.v1 if inline is not None else None
        inline_v2 = inline.v2 if inline is not None else None
        v1 = file_a.get(case.query_id, inline_v1)
        v2 = file_b.get(case.query_id, inline_v2)
        cases.append(
            EvalCase(
                query_id=case.query_id,
                query_id_was_explicit=case.query_id_was_explicit,
                query=case.query,
                relevant_document_ids=list(case.relevant_document_ids),
                relevant_chunk_ids=list(case.relevant_chunk_ids),
                query_embeddings=QueryEmbeddings(v1=v1, v2=v2),
            )
        )

    return EvalSet(cases=cases)
