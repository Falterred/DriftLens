"""FAISS index adapter (Step 17)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from driftlens.adapters.detect import FieldMapping
from driftlens.adapters.jsonl import LoadedInput
from driftlens.adapters.sidecar import SidecarRow, SidecarTable, load_sidecar
from driftlens.config.schema import SideOptions
from driftlens.errors import (
    Code,
    Diagnostic,
    DiagnosticSink,
    InvalidInputError,
    UnsupportedFormatError,
)
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.security.limits import RuntimeLimits
from driftlens.security.pickle_guard import load_langchain_index_pkl
from driftlens.text.normalize import check_utf8_clean, normalize_text
from driftlens.text.sentences import sentence_count
from driftlens.text.tokenizer import token_count

RECONSTRUCTABLE_INDEX_CLASSES = {
    "IndexFlatL2",
    "IndexFlatIP",
    "IndexIVFFlat",
    "IndexHNSWFlat",
}
WRAPPER_INDEX_CLASSES = {"IndexIDMap", "IndexIDMap2"}

_ALLOWED_CONTROLS = {chr(9), chr(10), chr(0x0D)}
_REPLACEMENT = chr(0xFFFD)


@dataclass
class FaissLoadResult(LoadedInput):
    """Adapter output for a FAISS index, with its extraction provenance."""

    vectors_approximate: bool = False
    embedding_only: bool = False
    unsafe_pickle: bool = False
    diagnostics: list[Diagnostic] = field(default_factory=list)


def _downcast(index: Any) -> Any:
    downcast = getattr(faiss, "downcast_index", None)
    if downcast is None:
        return index
    try:
        return downcast(index)
    except Exception:
        return index


def _unwrap_index(index: Any) -> tuple[Any, bool]:
    inner = getattr(index, "index", None)
    if type(index).__name__ in WRAPPER_INDEX_CLASSES and inner is not None:
        return _downcast(inner), True
    return index, False


def _effective_class_name(index: Any) -> str:
    name = type(index).__name__
    if name == "IndexFlat":
        if getattr(index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
            return "IndexFlatIP"
        return "IndexFlatL2"
    return name


def classify_index(index: Any) -> tuple[str, bool]:
    """Return (chunk_identity_source, is_id_mapped) for an index object."""

    _, id_mapped = _unwrap_index(index)
    if id_mapped:
        return "indexidmap", True
    return "vector_position", False


def reconstruct_all_vectors(index: Any) -> np.ndarray:
    """Reconstruct every vector of a reconstructable index as float32."""

    inner, _ = _unwrap_index(index)
    class_name = _effective_class_name(inner)
    if class_name not in RECONSTRUCTABLE_INDEX_CLASSES:
        raise UnsupportedFormatError(
            message=(
                f"{class_name} is not a reconstructable index type; "
                "use --allow-lossy-vectors"
            )
        )
    vectors = inner.reconstruct_n(0, inner.ntotal)
    return np.asarray(vectors, dtype=np.float32)


def _best_effort_vectors(index: Any) -> np.ndarray:
    inner, _ = _unwrap_index(index)
    class_name = _effective_class_name(inner)
    try:
        return np.asarray(inner.reconstruct_n(0, inner.ntotal), dtype=np.float32)
    except RuntimeError:
        pass
    rows: list[np.ndarray] = []
    try:
        for position in range(int(inner.ntotal)):
            rows.append(np.asarray(inner.reconstruct(position), dtype=np.float32))
    except RuntimeError:
        raise UnsupportedFormatError(
            message=(
                f"{class_name} cannot reconstruct vectors even with "
                "--allow-lossy-vectors"
            )
        ) from None
    if rows:
        return np.vstack(rows)
    return np.zeros((0, int(inner.d)), dtype=np.float32)


def _metric_kind(index: Any) -> str:
    metric_type = getattr(index, "metric_type", None)
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return "dot"
    return "l2"


def _index_ids(index: Any, id_mapped: bool) -> list[int]:
    if id_mapped:
        raw = getattr(index, "id_map", None)
        if raw is not None:
            return [int(value) for value in faiss.vector_to_array(raw)]
    return list(range(int(index.ntotal)))


def _encoding_damage(text: str) -> bool:
    if _REPLACEMENT in text:
        return True
    return any(ord(ch) < 0x20 and ch not in _ALLOWED_CONTROLS for ch in text)


def _text_fields(raw_text: str | None, location: str) -> dict[str, Any]:
    if raw_text is None:
        return {
            "text": None,
            "normalized_text": None,
            "token_count": None,
            "char_count": None,
            "sentence_count": None,
            "encoding_damage": None,
        }
    text = check_utf8_clean(raw_text, location=location)
    return {
        "text": text,
        "normalized_text": normalize_text(text),
        "token_count": token_count(text),
        "char_count": len(text),
        "sentence_count": sentence_count(text),
        "encoding_damage": _encoding_damage(text),
    }


def load_faiss(
    path: Path,
    *,
    side: SideOptions,
    limits: RuntimeLimits,
    allow_lossy_vectors: bool,
    allow_unsafe_pickle: bool,
) -> FaissLoadResult:
    """Load a FAISS index (optionally with a sidecar or LangChain pickle)."""

    path = Path(path)
    if path.is_dir():
        index_file = path / "index.faiss"
        pkl_file = path / "index.pkl"
    else:
        index_file = path
        pkl_file = path.with_name("index.pkl")

    if not index_file.is_file():
        raise InvalidInputError(
            message="faiss index file not found", location=str(path)
        )
    limits.add_input_bytes(index_file.stat().st_size, location=str(index_file))

    index = faiss.read_index(str(index_file))
    metric_kind = _metric_kind(index)
    inner, id_mapped = _unwrap_index(index)
    class_name = _effective_class_name(inner)
    reconstructable = class_name in RECONSTRUCTABLE_INDEX_CLASSES

    vectors_approximate = False
    if reconstructable:
        vectors = reconstruct_all_vectors(index)
    elif allow_lossy_vectors:
        vectors = _best_effort_vectors(index)
        vectors_approximate = True
    else:
        raise UnsupportedFormatError(
            message=(
                f"{class_name} is not a reconstructable index type; "
                "use --allow-lossy-vectors"
            ),
            location=str(index_file),
        )

    dimension = int(index.d)
    ntotal = int(index.ntotal)
    index_ids = _index_ids(index, id_mapped)

    sink = DiagnosticSink()
    unsafe_pickle = False
    pkl_entries: dict[int, Any] | None = None
    pkl_exists = pkl_file.is_file()

    sidecar_table: SidecarTable | None = None
    sidecar_mapping = FieldMapping()
    if side.sidecar is not None:
        sidecar_table = load_sidecar(Path(side.sidecar), side=side, limits=limits)
        sidecar_mapping = sidecar_table.field_mapping

    if sidecar_table is not None:
        if pkl_exists:
            sink.add(
                Code.PICKLE_IGNORED,
                "index.pkl ignored because a sidecar was supplied",
            )
        embedding_only = False
    elif allow_unsafe_pickle and pkl_exists:
        pkl_entries = load_langchain_index_pkl(pkl_file, limits=limits)
        unsafe_pickle = True
        embedding_only = False
    elif pkl_exists:
        sink.add(
            Code.PICKLE_IGNORED,
            "index.pkl present but not opened without --allow-unsafe-pickle",
        )
        embedding_only = True
    else:
        embedding_only = True

    if sidecar_table is not None:
        index_id_set = set(index_ids)
        missing = sum(1 for value in index_ids if value not in sidecar_table)
        unknown = sum(1 for key in sidecar_table if key not in index_id_set)
        if missing or unknown:
            raise InvalidInputError(
                message=(
                    "sidecar/index id mismatch: "
                    f"{missing} missing, {unknown} unknown faiss_id values"
                ),
                location=str(index_file),
            )

    if sidecar_table is not None and sidecar_mapping.chunk_id is not None:
        identity_hint = "sidecar_explicit"
    elif id_mapped:
        identity_hint = "indexidmap"
    else:
        identity_hint = "vector_position"

    records: list[Record] = []
    for position in range(ntotal):
        location = f"{index_file} entry {position}"
        faiss_id = index_ids[position]
        side_row: SidecarRow | None = (
            sidecar_table.get(faiss_id) if sidecar_table is not None else None
        )
        entry = pkl_entries.get(position) if pkl_entries is not None else None

        if (
            side_row is not None
            and sidecar_mapping.chunk_id is not None
            and side_row.record.record_id is not None
        ):
            record_id = side_row.record.record_id
        elif entry is not None:
            record_id = entry.chunk_id
        elif id_mapped:
            record_id = str(faiss_id)
        else:
            record_id = str(position)

        if side_row is not None:
            raw_text = side_row.record.text
            text_fields = _text_fields(
                None, location
            )
            text_fields.update(
                {
                    "normalized_text": side_row.record.normalized_text,
                    "token_count": side_row.record.token_count,
                    "char_count": side_row.record.char_count,
                    "sentence_count": side_row.record.sentence_count,
                    "encoding_damage": side_row.record.encoding_damage,
                }
            )
            if raw_text is not None:
                text_fields["text"] = raw_text
        elif entry is not None:
            text_fields = _text_fields(entry.text, location)
        else:
            text_fields = _text_fields(None, location)

        if side.model is not None:
            model: str | None = side.model
        elif side_row is not None and side_row.record.embedding.model is not None:
            model = side_row.record.embedding.model
        else:
            model = "UNKNOWN"

        if side_row is not None:
            provider = side_row.record.embedding.provider
            metadata = dict(side_row.record.metadata)
            document_id = side_row.record.document_id
            chunk_index = side_row.record.chunk_index
            created_at = side_row.record.created_at
            updated_at = side_row.record.updated_at
            row_number = side_row.record.row_number
        else:
            provider = None
            metadata = dict(entry.metadata) if entry is not None else {}
            document_id = None
            chunk_index = None
            created_at = None
            updated_at = None
            row_number = position

        vector = np.asarray(vectors[position], dtype=np.float32)
        embedding = EmbeddingInfo(
            vector=vector,
            dtype="float32",
            dimension=dimension,
            norm=float(np.linalg.norm(np.asarray(vector, dtype=np.float64))),
            validity=None,
            model=model,
            provider=provider,
            similarity_metric=metric_kind,
            normalized=None,
        )

        records.append(
            Record(
                record_id=record_id,
                document_id=document_id,
                identity_source_hint=identity_hint,
                embedding=embedding,
                metadata=metadata,
                chunk_index=chunk_index,
                created_at=created_at,
                updated_at=updated_at,
                row_number=row_number,
                **text_fields,
            )
        )

    return FaissLoadResult(
        records=records,
        field_mapping=sidecar_mapping,
        file_descriptors={},
        source_type="faiss",
        vectors_approximate=vectors_approximate,
        embedding_only=embedding_only,
        unsafe_pickle=unsafe_pickle,
        diagnostics=sink.items(),
    )
