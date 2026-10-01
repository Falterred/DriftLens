"""Restricted pickle guard (Step 7)."""

from __future__ import annotations

import os
import pickle
import shutil
import sys
import types
from pathlib import Path

import pytest

from driftlens.config.schema import LimitsCfg
from driftlens.errors import InvalidInputError
from driftlens.security.limits import RuntimeLimits
from driftlens.security.pickle_guard import (
    LangChainEntry,
    load_langchain_index_pkl,
)


@pytest.fixture
def workdir():
    directory = Path(__file__).resolve().parent / "_pickle_guard_tmp"
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _load(data: bytes, workdir: Path) -> dict[int, LangChainEntry]:
    path = workdir / "index.pkl"
    path.write_bytes(data)
    return load_langchain_index_pkl(path, limits=RuntimeLimits(LimitsCfg()))


def test_allowlisted_stub_roundtrip(workdir) -> None:
    package_names = [
        "langchain_core",
        "langchain_core.documents",
        "langchain_core.documents.base",
        "langchain_community",
        "langchain_community.docstore",
        "langchain_community.docstore.in_memory",
    ]
    added = {}
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        added[name] = module
        sys.modules[name] = module

    class Document:
        def __init__(self, page_content: str, metadata: dict) -> None:
            self.page_content = page_content
            self.metadata = metadata

    class InMemoryDocstore:
        def __init__(self, _dict: dict) -> None:
            self._dict = _dict

    Document.__module__ = "langchain_core.documents.base"
    Document.__qualname__ = "Document"
    InMemoryDocstore.__module__ = "langchain_community.docstore.in_memory"
    InMemoryDocstore.__qualname__ = "InMemoryDocstore"
    added["langchain_core.documents.base"].Document = Document
    added["langchain_community.docstore.in_memory"].InMemoryDocstore = InMemoryDocstore
    try:
        payload = (
            InMemoryDocstore(
                _dict={"u1": Document(page_content="hi", metadata={"k": 1})}
            ),
            {0: "u1"},
        )
        data = pickle.dumps(payload)
    finally:
        for name in package_names:
            sys.modules.pop(name, None)

    index = _load(data, workdir)
    assert index == {0: LangChainEntry(chunk_id="u1", text="hi", metadata={"k": 1})}


def test_os_system_blocked(workdir, monkeypatch) -> None:
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo x",))

    data = pickle.dumps(Evil())

    calls: list[tuple] = []

    def tripwire(*args, **kwargs):
        calls.append(args)
        return 0

    monkeypatch.setattr(os, "system", tripwire)

    with pytest.raises(InvalidInputError):
        _load(data, workdir)
    assert calls == []


def test_eval_blocked(workdir) -> None:
    class Evil:
        def __reduce__(self):
            return (eval, ("1 + 1",))

    with pytest.raises(InvalidInputError):
        _load(pickle.dumps(Evil()), workdir)


def test_bad_structure(workdir) -> None:
    with pytest.raises(InvalidInputError):
        _load(pickle.dumps([1, 2, 3]), workdir)
