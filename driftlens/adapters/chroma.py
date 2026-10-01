"""Chroma directory adapter (Step 18)."""

from __future__ import annotations

import gc
import os
import secrets
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from driftlens.adapters.detect import FieldMapping, resolve_fields
from driftlens.adapters.jsonl import LoadedInput
from driftlens.config.schema import SideOptions
from driftlens.errors import (
    DependencyMissingError,
    InvalidInputError,
    RefuseUntrustedPickleError,
    UnsupportedFormatError,
)
from driftlens.identity.hashing import canonical_json
from driftlens.schema.record import EmbeddingInfo, Record, canonicalize_timestamp
from driftlens.security.limits import RuntimeLimits
from driftlens.text.normalize import check_utf8_clean, normalize_text
from driftlens.text.sentences import sentence_count
from driftlens.text.tokenizer import token_count

PAGE_SIZE = 5000
PKL_SUFFIXES = (".pickle", ".pkl")
FIXED_COLUMNS = ("document", "embedding")
REPLACEMENT = chr(0xFFFD)
ALLOWED_CONTROLS = {chr(9), chr(10)}


@dataclass
class ChromaVariant:
    """One supported Chroma on-disk schema."""

    package_range: str
    schema_identifier: str


SUPPORTED_VARIANTS: list[ChromaVariant] = [
    ChromaVariant(
        package_range="==1.5.9",
        schema_identifier='[["embeddings_queue",2],["metadb",6],["sysdb",10]]',
    ),
]


def _opt_text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _encoding_damage(text: str) -> bool:
    if REPLACEMENT in text:
        return True
    return any(ord(ch) < 0x20 and ch not in ALLOWED_CONTROLS for ch in text)


def scan_for_pickle_files(directory: Path) -> list[Path]:
    """Return every pickle file found anywhere under a Chroma directory."""

    found: list[Path] = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(PKL_SUFFIXES):
                found.append(Path(root) / name)
    found.sort()
    return found


def read_schema_identifier(sqlite_path: Path) -> str:
    """Canonical identifier for a Chroma sqlite schema version set."""

    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT dir, MAX(version) FROM migrations GROUP BY dir ORDER BY dir"
        ).fetchall()
    finally:
        connection.close()
    pairs = [[str(directory), int(version)] for directory, version in rows]
    return canonical_json(pairs).decode("utf-8")


def _new_temp_directory() -> Path:
    base = Path(tempfile.gettempdir())
    path = base / f"driftlens-chroma-{os.getpid()}-{secrets.token_hex(8)}"
    path.mkdir()
    os.chmod(path, 0o700)
    return path


def copy_chroma_directory(source: Path, *, limits: RuntimeLimits) -> Path:
    """Copy a Chroma directory into a private temporary directory."""

    total = 0
    for root, _dirs, files in os.walk(source):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    limits.add_input_bytes(total, location=str(source))
    destination = _new_temp_directory()
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def _remove_directory(path: Path) -> None:
    delay = 0.05
    for attempt in range(10):
        try:
            shutil.rmtree(path, ignore_errors=False)
            return
        except PermissionError:
            if attempt == 9:
                raise
            gc.collect()
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


@contextmanager
def temporary_chroma_copy(
    source: Path, *, limits: RuntimeLimits
) -> Iterator[Path]:
    """Yield a private copy of a Chroma directory, removed on exit."""

    copy_path = copy_chroma_directory(source, limits=limits)
    try:
        yield copy_path
    finally:
        _remove_directory(copy_path)


def _collection_name(collection: Any) -> str:
    name = getattr(collection, "name", None)
    if name is None:
        return str(collection)
    return str(name)


def _select_collection(client: Any, side: SideOptions) -> Any:
    collections = list(client.list_collections())
    names = [_collection_name(collection) for collection in collections]
    if side.chroma_collection is not None:
        for collection in collections:
            if _collection_name(collection) == side.chroma_collection:
                return collection
        try:
            return client.get_collection(side.chroma_collection)
        except Exception:
            available = ", ".join(names) if names else "(none)"
            raise InvalidInputError(
                message=(
                    f"collection {side.chroma_collection!r} not found; "
                    f"available collections: {available}"
                )
            ) from None
    if not collections:
        raise InvalidInputError(message="no collections found in Chroma directory")
    if len(collections) > 1:
        raise InvalidInputError(
            message=(
                "multiple collections found; select one with "
                f"--chroma-collection: {', '.join(names)}"
            )
        )
    return collections[0]


def _configuration(collection: Any) -> dict[str, Any]:
    configuration = getattr(collection, "configuration_json", None)
    if isinstance(configuration, dict):
        return configuration
    return {}


def _metric_kind(collection: Any) -> str | None:
    metadata = getattr(collection, "metadata", None) or {}
    space = metadata.get("hnsw:space")
    if space is None:
        hnsw = _configuration(collection).get("hnsw") or {}
        if isinstance(hnsw, dict):
            space = hnsw.get("space")
    if space is None:
        return None
    space = str(space).lower()
    if space == "l2":
        return "l2"
    if space == "ip":
        return "dot"
    if space == "cosine":
        return "cosine"
    return None


def _embedding_model(collection: Any, side: SideOptions) -> str | None:
    metadata = getattr(collection, "metadata", None) or {}
    declared = metadata.get("embedding_function")
    if declared is None:
        declared = _configuration(collection).get("embedding_function")
    name: Any = None
    if isinstance(declared, dict):
        name = declared.get("name")
    elif isinstance(declared, str):
        name = declared
    if isinstance(name, dict):
        name = name.get("name")
    if isinstance(name, str) and name:
        return name
    return side.model


def _shutdown_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass

def _page_rows(collection: Any) -> list[tuple[str, Any, Any, Any]]:
    rows: list[tuple[str, Any, Any, Any]] = []
    offset = 0
    while True:
        page = collection.get(
            include=["embeddings", "documents", "metadatas"],
            limit=PAGE_SIZE,
            offset=offset,
        )
        ids = page.get("ids")
        if ids is None:
            break
        count = len(ids)
        if count == 0:
            break
        documents = page.get("documents")
        embeddings = page.get("embeddings")
        metadatas = page.get("metadatas")
        for index in range(count):
            document = documents[index] if documents is not None else None
            embedding = embeddings[index] if embeddings is not None else None
            metadata = metadatas[index] if metadatas is not None else None
            rows.append((str(ids[index]), document, embedding, metadata))
        offset += count
    return rows


def _timestamp_value(
    metadata: dict[str, Any],
    mapping: FieldMapping,
    role: str,
    *,
    unit: str,
    location: str,
) -> str | None:
    key = mapping.timestamp_fields.get(role)
    if key is None:
        return None
    value = metadata.get(key)
    if value is None:
        return None
    return canonicalize_timestamp(value, unit=unit, location=location)


def load_chroma(
    path: Path, *, side: SideOptions, limits: RuntimeLimits
) -> LoadedInput:
    """Load a Chroma persistence directory into canonical records."""

    path = Path(path)
    pickles = scan_for_pickle_files(path)
    if pickles:
        raise RefuseUntrustedPickleError(
            message=(
                f"{len(pickles)} pickle file(s) found in Chroma directory; "
                "export to JSONL instead"
            )
        )
    sqlite_path = path / "chroma.sqlite3"
    if not sqlite_path.is_file():
        raise InvalidInputError(
            message="chroma.sqlite3 not found", location=str(path)
        )
    schema_id = read_schema_identifier(sqlite_path)
    variant = next(
        (item for item in SUPPORTED_VARIANTS if item.schema_identifier == schema_id),
        None,
    )
    if variant is None:
        raise UnsupportedFormatError(
            message=f"unrecognized Chroma schema identifier {schema_id}",
            location=str(path),
        )

    with temporary_chroma_copy(path, limits=limits) as copy_path:
        client = None
        try:
            try:
                import chromadb
            except ImportError as exc:
                raise DependencyMissingError(
                    message="chromadb is required to read Chroma directories"
                ) from exc
            settings = chromadb.config.Settings(anonymized_telemetry=False)
            client = chromadb.PersistentClient(path=str(copy_path), settings=settings)
            collection = _select_collection(client, side)
            metric = _metric_kind(collection)
            model = _embedding_model(collection, side)
            rows = _page_rows(collection)
        finally:
            if client is not None:
                _shutdown_client(client)
            collection = None
            client = None
            gc.collect()

    rows.sort(key=lambda item: item[0])

    present = [document is not None for _, document, _, _ in rows]
    if any(present) and not all(present):
        raise InvalidInputError(
            message="mixed text presence", location=str(path)
        )

    metadata_keys: list[str] = []
    seen: set[str] = set()
    for _, _, _, metadata in rows:
        for key in metadata or {}:
            if key not in seen:
                seen.add(key)
                metadata_keys.append(key)
    available = list(metadata_keys)
    for fixed in FIXED_COLUMNS:
        if fixed not in available:
            available.append(fixed)
    mapping = resolve_fields(
        available,
        granularity_hint="AUTO",
        overrides=side,
        location=str(path),
    )

    if mapping.document_id is None:
        identity_hint: str | None = None
    elif side.id_field is not None:
        identity_hint = "explicit_override"
    else:
        identity_hint = "chroma_metadata"

    consumed = mapping.consumed_keys
    records: list[Record] = []
    for position, (row_id, document, embedding, metadata) in enumerate(rows, start=1):
        location = f"{path} entry {row_id}"
        text = normalized = tokens = chars = sentences = damage = None
        if document is not None:
            text = check_utf8_clean(document, location=location)
            normalized = normalize_text(text)
            tokens = token_count(text)
            chars = len(text)
            sentences = sentence_count(text)
            damage = _encoding_damage(text)

        if embedding is None:
            info = EmbeddingInfo(
                None, None, None, None, None, model, None, metric, None
            )
        else:
            vector = np.asarray(embedding, dtype=np.float32)
            info = EmbeddingInfo(
                vector=vector,
                dtype="float32",
                dimension=int(vector.shape[0]),
                norm=float(np.linalg.norm(np.asarray(vector, dtype=np.float64))),
                validity=None,
                model=model,
                provider=None,
                similarity_metric=metric,
                normalized=None,
            )

        clean_metadata = {
            key: ({} if value is None else value)
            for key, value in (metadata or {}).items()
        }
        document_id = (
            clean_metadata.get(mapping.document_id)
            if mapping.document_id is not None
            else None
        )
        created_at = _timestamp_value(
            clean_metadata, mapping, "created_at", unit=side.timestamp_unit, location=location
        )
        updated_at = _timestamp_value(
            clean_metadata, mapping, "updated_at", unit=side.timestamp_unit, location=location
        )
        if updated_at is None:
            updated_at = _timestamp_value(
                clean_metadata,
                mapping,
                "modified_at",
                unit=side.timestamp_unit,
                location=location,
            )
        if created_at is None:
            created_at = _timestamp_value(
                clean_metadata,
                mapping,
                "timestamp",
                unit=side.timestamp_unit,
                location=location,
            )
        record_metadata = {
            key: value for key, value in clean_metadata.items() if key not in consumed
        }

        records.append(
            Record(
                record_id=row_id,
                document_id=_opt_text(document_id),
                identity_source_hint=identity_hint,
                text=text,
                normalized_text=normalized,
                token_count=tokens,
                char_count=chars,
                sentence_count=sentences,
                encoding_damage=damage,
                embedding=info,
                metadata=record_metadata,
                created_at=created_at,
                updated_at=updated_at,
                row_number=position,
            )
        )

    return LoadedInput(
        records=records,
        field_mapping=mapping,
        file_descriptors={},
        source_type="chroma",
    )
