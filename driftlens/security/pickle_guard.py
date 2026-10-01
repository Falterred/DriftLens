"""Restricted unpickling for LangChain index.pkl files (Step 7)."""

from __future__ import annotations

import builtins
import io
import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from driftlens.errors import InvalidInputError
from driftlens.security.limits import RuntimeLimits

ALLOWED_CLASSES: frozenset[tuple[str, str]] = frozenset(
    {
        ("langchain_core.documents.base", "Document"),
        ("langchain_core.documents", "Document"),
        ("langchain_community.docstore.document", "Document"),
        ("langchain.docstore.document", "Document"),
        ("langchain_community.docstore.in_memory", "InMemoryDocstore"),
        ("langchain.docstore.in_memory", "InMemoryDocstore"),
        ("uuid", "UUID"),
    }
)

# "None" is a pickle opcode, not a class reference.
ALLOWED_BUILTINS = frozenset(
    {"dict", "list", "tuple", "str", "int", "float", "bool"}
)


def _stub_setstate(self: Any, state: Any) -> None:
    if isinstance(state, dict) and "__dict__" in state:
        self.__dict__.update(state["__dict__"])
    elif isinstance(state, dict):
        self.__dict__.update(state)
    elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
        self.__dict__.update(state[0])
    else:
        raise InvalidInputError(
            message="unsupported pickle state for an allowlisted class"
        )


_STUBS: dict[str, type] = {}


def _stub_for(name: str) -> type:
    stub = _STUBS.get(name)
    if stub is None:
        stub = type(name, (), {"__setstate__": _stub_setstate})
        _STUBS[name] = stub
    return stub


class RestrictedUnpickler(pickle.Unpickler):
    """An unpickler that resolves only explicitly allowlisted classes."""

    def __init__(self, file: Any, *, location: str | Path | None = None) -> None:
        super().__init__(file)
        self.location = None if location is None else str(location)

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in ALLOWED_CLASSES:
            if module == "uuid" and name == "UUID":
                return uuid.UUID
            return _stub_for(name)
        if module == "builtins" and name in ALLOWED_BUILTINS:
            return getattr(builtins, name)
        raise InvalidInputError(
            message=f"pickle references disallowed class {module}.{name}",
            location=self.location,
        )


@dataclass
class LangChainEntry:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _unexpected() -> InvalidInputError:
    return InvalidInputError(message="unexpected index.pkl structure")


def _doc_attr(doc: Any, name: str) -> Any:
    if isinstance(doc, dict) and name in doc:
        return doc[name]
    return getattr(doc, name, None)


def _docstore_mapping(docstore: Any) -> dict[Any, Any]:
    mapping = getattr(docstore, "_dict", None)
    if isinstance(mapping, dict):
        return mapping
    fallback = getattr(docstore, "__dict__", None)
    if isinstance(fallback, dict):
        return fallback
    if isinstance(docstore, dict):
        return docstore
    raise _unexpected()


def load_langchain_index_pkl(
    path: Path, *, limits: RuntimeLimits
) -> dict[int, LangChainEntry]:
    """Load a LangChain FAISS index.pkl through the restricted unpickler."""

    limits.add_input_bytes(path.stat().st_size, location=str(path))
    data = path.read_bytes()
    unpickler = RestrictedUnpickler(io.BytesIO(data), location=path)
    try:
        payload = unpickler.load()
    except InvalidInputError:
        raise
    except Exception:
        raise InvalidInputError(
            message="unreadable index.pkl", location=str(path)
        ) from None

    if not isinstance(payload, tuple) or len(payload) != 2:
        raise _unexpected()
    docstore, mapping = payload
    if not isinstance(mapping, dict):
        raise _unexpected()
    doc_map = _docstore_mapping(docstore)

    entries: dict[int, LangChainEntry] = {}
    missing = 0
    for position, docstore_id in mapping.items():
        if not isinstance(position, int):
            raise _unexpected()
        if docstore_id not in doc_map:
            missing += 1
            continue
        doc = doc_map[docstore_id]
        page_content = _doc_attr(doc, "page_content")
        metadata = _doc_attr(doc, "metadata")
        if not isinstance(page_content, str) or not isinstance(metadata, dict):
            raise _unexpected()
        entries[position] = LangChainEntry(
            chunk_id=str(docstore_id), text=page_content, metadata=metadata
        )

    if missing:
        raise InvalidInputError(
            message=(
                f"index.pkl mapping has {missing} of {len(mapping)} entries "
                "absent from the docstore"
            )
        )
    return entries
