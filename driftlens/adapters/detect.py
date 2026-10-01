"""Input-format detection and field-role resolution (Step 13)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from driftlens.config.schema import SideOptions
from driftlens.errors import InvalidInputError, UnsupportedFormatError
from driftlens.security.limits import RuntimeLimits, loads_limited

InputFormat = Literal["jsonl", "json", "csv", "parquet", "chroma", "faiss", "snapshot"]
COMPRESSED_EXTENSIONS = frozenset({".gz", ".zip", ".bz2", ".xz", ".zst"})
_FORMATS = frozenset(
    {"jsonl", "json", "csv", "parquet", "chroma", "faiss", "snapshot"}
)
_DIRECTORY_FORMATS = frozenset({"snapshot", "chroma"})

FieldRole = Literal[
    "text",
    "document_id",
    "chunk_id",
    "embedding",
    "chunk_index",
    "timestamps",
    "embedding_model",
    "provider",
    "dtype",
    "metric",
]

TEXT_CANDIDATES = [
    "text",
    "content",
    "page_content",
    "chunk_text",
    "chunk",
    "passage",
    "body",
    "document",
]
DOCUMENT_ID_CANDIDATES = ["document_id", "doc_id", "docid"]
SOURCE_IDENTIFIER_CANDIDATES = ["source_id", "source", "file", "path", "url"]
CHUNK_ID_CANDIDATES = ["chunk_id", "id", "_id", "uuid"]
EMBEDDING_CANDIDATES = [
    "embedding",
    "embeddings",
    "vector",
    "vec",
    "emb",
    "dense_vector",
]
CHUNK_INDEX_CANDIDATES = ["chunk_index", "chunk_idx", "chunk_no", "position", "seq"]
TIMESTAMP_CANDIDATES = ["created_at", "updated_at", "modified_at", "timestamp"]
MODEL_CANDIDATES = ["embedding_model", "model", "model_name", "embedder"]
PROVIDER_CANDIDATES = ["embedding_provider", "provider"]
DTYPE_CANDIDATES = ["embedding_dtype", "dtype"]
METRIC_CANDIDATES = ["similarity_metric", "metric", "distance"]


@dataclass
class FieldMapping:
    """Resolved source-key assignment for every field role."""

    text: str | None = None
    document_id: str | None = None
    document_id_level: int | None = None
    chunk_id: str | None = None
    embedding: str | None = None
    embedding_is_b64: bool = False
    chunk_index: str | None = None
    timestamp_fields: dict[str, str] = field(default_factory=dict)
    model: str | None = None
    provider: str | None = None
    dtype: str | None = None
    metric: str | None = None
    ignored: dict[str, list[str]] = field(default_factory=dict)
    consumed_keys: frozenset[str] = frozenset()
    copied_keys: frozenset[str] = frozenset()


def _decode_utf8(data: bytes, path: Path) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise InvalidInputError("invalid UTF-8", location=str(path)) from None


def _read_text(path: Path, limits: RuntimeLimits) -> str:
    limits.add_input_bytes(path.stat().st_size, location=str(path))
    return _decode_utf8(path.read_bytes(), path)


def detect_format(
    path: Path, *, forced: str | None, limits: RuntimeLimits
) -> InputFormat:
    """Detect the input format of a path, honouring an explicit forced format."""

    path = Path(path)

    if forced is not None:
        value = forced.strip().lower()
        if value not in _FORMATS:
            raise UnsupportedFormatError(
                f"unknown format {forced!r}", location=str(path)
            )
        if value in _DIRECTORY_FORMATS and not path.is_dir():
            raise InvalidInputError(
                f"format {value!r} requires a directory", location=str(path)
            )
        if value == "faiss":
            if not (path.is_dir() or path.is_file()):
                raise InvalidInputError(
                    "format 'faiss' requires a file or directory", location=str(path)
                )
        elif value not in _DIRECTORY_FORMATS and not path.is_file():
            raise InvalidInputError(
                f"format {value!r} requires a file", location=str(path)
            )
        return value  # type: ignore[return-value]

    if not path.exists():
        raise InvalidInputError(
            message="input path does not exist", location=str(path)
        )

    if path.is_dir():
        manifest = path / "manifest.json"
        if manifest.is_file():
            payload: Any = None
            try:
                payload = loads_limited(_read_text(manifest, limits), location=str(manifest))
            except InvalidInputError:
                payload = None
            if isinstance(payload, dict) and "schema_version" in payload:
                return "snapshot"
        if (path / "chroma.sqlite3").is_file():
            return "chroma"
        if (path / "index.faiss").is_file():
            return "faiss"
        raise UnsupportedFormatError(
            message="directory matches no supported input shape", location=str(path)
        )

    suffix = path.suffix.lower()
    if suffix in COMPRESSED_EXTENSIONS:
        raise UnsupportedFormatError(
            message=(
                f"compressed input ({suffix}) is not supported; "
                "decompress before running driftlens"
            ),
            location=str(path),
        )

    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic == b"PAR1":
        return "parquet"

    if suffix in (".faiss", ".index"):
        return "faiss"
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix == ".json":
        payload = loads_limited(_read_text(path, limits), location=str(path))
        if isinstance(payload, list):
            return "json"
        if isinstance(payload, dict):
            for key in ("records", "data", "items"):
                if isinstance(payload.get(key), list):
                    return "json"
        raise UnsupportedFormatError(
            message="JSON file is not a records array or records wrapper",
            location=str(path),
        )
    if suffix in (".csv", ".tsv"):
        return "csv"
    raise UnsupportedFormatError(
        message=f"unrecognized file extension {path.suffix!r}", location=str(path)
    )


def _search(
    by_lower: dict[str, list[str]], candidates: list[str]
) -> tuple[str | None, list[str], list[str]]:
    """Return (winner, lower-priority present keys, same-tier collision keys)."""

    for index, candidate in enumerate(candidates):
        matches = by_lower.get(candidate.lower(), [])
        if not matches:
            continue
        if len(matches) > 1:
            return None, [], list(matches)
        ignored: list[str] = []
        for lower in candidates[index + 1 :]:
            ignored.extend(by_lower.get(lower.lower(), []))
        return matches[0], ignored, []
    return None, [], []


_OVERRIDE_ATTR = {
    "text": "text_field",
    "document_id": "id_field",
    "chunk_id": "chunk_id_field",
    "embedding": "embedding_field",
    "model": "model_field",
}


def resolve_fields(
    available_keys: Iterable[str],
    *,
    granularity_hint: Literal["CHUNK", "DOCUMENT", "AUTO"],
    overrides: SideOptions,
    location: str,
) -> FieldMapping:
    """Resolve each field role to a source key using §2.3 priority order."""

    keys = list(available_keys)
    by_lower: dict[str, list[str]] = {}
    for key in keys:
        by_lower.setdefault(key.lower(), []).append(key)

    ignored: dict[str, list[str]] = {}
    problems: list[str] = []

    def pick(role: str, candidates: list[str]) -> str | None:
        attr = _OVERRIDE_ATTR.get(role)
        override_key = getattr(overrides, attr, None) if attr is not None else None
        if override_key is not None:
            if override_key not in keys:
                problems.append(
                    f"{role}: override field {override_key!r} is not present in input"
                )
                return None
            return override_key
        winner, lower_present, collision = _search(by_lower, candidates)
        if collision:
            problems.append(
                f"{role}: ambiguous field, multiple inputs match at the same "
                f"priority: {', '.join(collision)}"
            )
            return None
        if lower_present:
            ignored[role] = lower_present
        return winner

    mapping = FieldMapping(ignored=ignored)
    consumed: set[str] = set()
    copied: set[str] = set()

    mapping.text = pick("text", TEXT_CANDIDATES)
    if mapping.text is not None:
        consumed.add(mapping.text)

    override_id = overrides.id_field
    if override_id is not None:
        if override_id not in keys:
            problems.append(
                f"document_id: override field {override_id!r} is not present in input"
            )
        else:
            mapping.document_id = override_id
            mapping.document_id_level = 1
            consumed.add(override_id)
    else:
        winner, lower_present, collision = _search(by_lower, DOCUMENT_ID_CANDIDATES)
        if collision:
            problems.append(
                "document_id: ambiguous field, multiple inputs match at the same "
                f"priority: {', '.join(collision)}"
            )
        elif winner is not None:
            mapping.document_id = winner
            mapping.document_id_level = 1
            if lower_present:
                ignored["document_id"] = lower_present
            consumed.add(winner)
        else:
            winner, lower_present, collision = _search(
                by_lower, SOURCE_IDENTIFIER_CANDIDATES
            )
            if collision:
                problems.append(
                    "document_id: ambiguous field, multiple inputs match at the same "
                    f"priority: {', '.join(collision)}"
                )
            elif winner is not None:
                mapping.document_id = winner
                mapping.document_id_level = 2
                if lower_present:
                    ignored["document_id"] = lower_present
                copied.add(winner)

    mapping.chunk_id = pick("chunk_id", CHUNK_ID_CANDIDATES)
    if mapping.chunk_id is not None:
        consumed.add(mapping.chunk_id)

    override_embedding = overrides.embedding_field
    if override_embedding is not None:
        if override_embedding not in keys:
            problems.append(
                f"embedding: override field {override_embedding!r} is not present in input"
            )
        else:
            mapping.embedding = override_embedding
            consumed.add(override_embedding)
    else:
        winner, lower_present, collision = _search(by_lower, EMBEDDING_CANDIDATES)
        if collision:
            problems.append(
                "embedding: ambiguous field, multiple inputs match at the same "
                f"priority: {', '.join(collision)}"
            )
        elif winner is not None:
            mapping.embedding = winner
            if lower_present:
                ignored["embedding"] = lower_present
            consumed.add(winner)
        else:
            b64_matches = by_lower.get("embedding_b64", [])
            if len(b64_matches) > 1:
                problems.append(
                    "embedding: ambiguous field, multiple inputs match at the same "
                    f"priority: {', '.join(b64_matches)}"
                )
            elif b64_matches:
                mapping.embedding = b64_matches[0]
                mapping.embedding_is_b64 = True
                consumed.add(b64_matches[0])

    mapping.chunk_index = pick("chunk_index", CHUNK_INDEX_CANDIDATES)
    if mapping.chunk_index is not None:
        consumed.add(mapping.chunk_index)

    for role, candidates in (
        ("model", MODEL_CANDIDATES),
        ("provider", PROVIDER_CANDIDATES),
        ("dtype", DTYPE_CANDIDATES),
        ("metric", METRIC_CANDIDATES),
    ):
        winner = pick(role, candidates)
        setattr(mapping, role, winner)
        if winner is not None:
            consumed.add(winner)

    for candidate in TIMESTAMP_CANDIDATES:
        matches = by_lower.get(candidate.lower(), [])
        if not matches:
            continue
        if len(matches) > 1:
            problems.append(
                "timestamps: ambiguous field, multiple inputs match at the same "
                f"priority: {', '.join(matches)}"
            )
            continue
        mapping.timestamp_fields[candidate] = matches[0]
        consumed.add(matches[0])

    if mapping.text is None and mapping.embedding is None:
        problems.append("no text or embedding column found")

    if problems:
        raise InvalidInputError(message="; ".join(problems), location=location)

    mapping.consumed_keys = frozenset(consumed)
    mapping.copied_keys = frozenset(copied)
    return mapping
