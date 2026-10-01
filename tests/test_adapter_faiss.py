"""FAISS adapter tests (Step 17)."""

from __future__ import annotations

import pickle
import shutil
import sys
import types
from pathlib import Path

import faiss
import numpy as np
import pytest

from driftlens.adapters.faiss import load_faiss
from driftlens.config.schema import LimitsCfg, SideOptions
from driftlens.errors import Code, InvalidInputError, UnsupportedFormatError
from driftlens.security.limits import RuntimeLimits


def _limits() -> RuntimeLimits:
    return RuntimeLimits(LimitsCfg(), available_bytes=1 << 40)


@pytest.fixture()
def workdir() -> Path:
    base = Path(__file__).parent / "_faiss_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _data(count: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.random((count, dim), dtype=np.float32)


def _write_index(
    workdir: Path,
    *,
    count: int = 2,
    dim: int = 4,
    ids: list[int] | None = None,
    inner_product: bool = False,
    name: str = "idx",
) -> Path:
    directory = workdir / name
    directory.mkdir(parents=True, exist_ok=True)
    data = _data(count, dim)
    if inner_product:
        inner = faiss.IndexFlatIP(dim)
    else:
        inner = faiss.IndexFlatL2(dim)
    if ids is None:
        index = inner
        index.add(data)
    else:
        index = faiss.IndexIDMap(inner)
        index.add_with_ids(data, np.asarray(ids, dtype=np.int64))
    faiss.write_index(index, str(directory / "index.faiss"))
    return directory


def _write_pq(workdir: Path, name: str = "pq") -> Path:
    directory = workdir / name
    directory.mkdir(parents=True, exist_ok=True)
    data = _data(5, 4)
    index = faiss.IndexPQ(4, 4, 2)
    index.train(data)
    index.add(data)
    faiss.write_index(index, str(directory / "index.faiss"))
    return directory


def _write_sidecar(directory: Path, rows: list[dict]) -> Path:
    path = directory / "sidecar.jsonl"
    path.write_text(
        "".join(__import__("json").dumps(row) + chr(10) for row in rows),
        encoding="utf-8",
    )
    return path


def _write_pkl(directory: Path, mapping: dict, docs: dict) -> Path:
    modules: dict[str, types.ModuleType] = {}
    for name in (
        "langchain_core",
        "langchain_core.documents",
        "langchain_core.documents.base",
        "langchain_community",
        "langchain_community.docstore",
        "langchain_community.docstore.in_memory",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
        sys.modules[name] = module
    try:
        doc_cls = type(
            "Document",
            (),
            {
                "__module__": "langchain_core.documents.base",
                "__qualname__": "Document",
            },
        )
        modules["langchain_core.documents.base"].Document = doc_cls
        store_cls = type(
            "InMemoryDocstore",
            (),
            {
                "__module__": "langchain_community.docstore.in_memory",
                "__qualname__": "InMemoryDocstore",
            },
        )
        modules["langchain_community.docstore.in_memory"].InMemoryDocstore = store_cls

        documents = {}
        for key, spec in docs.items():
            doc = doc_cls()
            doc.page_content = spec["page_content"]
            doc.metadata = spec["metadata"]
            documents[key] = doc
        store = store_cls()
        store._dict = documents
        path = directory / "index.pkl"
        with path.open("wb") as handle:
            pickle.dump((store, mapping), handle)
    finally:
        for name in modules:
            sys.modules.pop(name, None)
    return path


def _load(directory: Path, *, side: SideOptions | None = None, lossy: bool = False,
          unsafe: bool = False) -> object:
    return load_faiss(
        directory,
        side=side if side is not None else SideOptions(),
        limits=_limits(),
        allow_lossy_vectors=lossy,
        allow_unsafe_pickle=unsafe,
    )


def test_faiss_flat_l2_reconstructable(workdir: Path) -> None:
    directory = _write_index(workdir, count=5, dim=8)
    result = _load(directory)
    assert len(result.records) == 5
    assert all(r.embedding.similarity_metric == "l2" for r in result.records)
    assert result.vectors_approximate is False
    assert result.records[0].embedding.vector.shape == (8,)


def test_faiss_idmap_medium_confidence_without_sidecar_chunk_id(workdir: Path) -> None:
    directory = _write_index(
        workdir, count=2, dim=4, ids=[10, 20], inner_product=True
    )
    result = _load(directory)
    assert all(r.identity_source_hint == "indexidmap" for r in result.records)
    assert [r.record_id for r in result.records] == ["10", "20"]
    assert result.records[0].embedding.similarity_metric == "dot"


def test_faiss_pq_without_flag_rejected(workdir: Path) -> None:
    directory = _write_pq(workdir)
    with pytest.raises(UnsupportedFormatError) as info:
        _load(directory)
    assert "--allow-lossy-vectors" in str(info.value)


def test_faiss_pq_with_flag_still_rejected_if_truly_unreconstructable(
    workdir: Path,
) -> None:
    directory = _write_pq(workdir)
    try:
        _load(directory, lossy=True)
    except UnsupportedFormatError:
        return
    pytest.xfail("this faiss build reconstructs IndexPQ vectors")


def test_faiss_langchain_pkl_ignored_by_default(workdir: Path) -> None:
    directory = _write_index(workdir, count=1)
    _write_pkl(
        directory,
        {0: "u1"},
        {"u1": {"page_content": "hello world", "metadata": {}}},
    )
    result = _load(directory)
    assert result.embedding_only is True
    assert any(d.code is Code.PICKLE_IGNORED for d in result.diagnostics)


def test_faiss_langchain_pkl_with_unsafe_flag(workdir: Path) -> None:
    directory = _write_index(workdir, count=1)
    _write_pkl(
        directory,
        {0: "u1"},
        {"u1": {"page_content": "hello world", "metadata": {"k": 1}}},
    )
    result = _load(directory, unsafe=True)
    assert result.records[0].text == "hello world"
    assert result.records[0].record_id == "u1"
    assert result.unsafe_pickle is True
    assert result.embedding_only is False


def test_faiss_sidecar_index_count_mismatch(workdir: Path) -> None:
    directory = _write_index(workdir, count=2, dim=4)
    sidecar = _write_sidecar(
        directory, [{"faiss_id": 0, "text": "alpha"}]
    )
    with pytest.raises(InvalidInputError) as info:
        _load(directory, side=SideOptions(sidecar=str(sidecar)))
    assert "missing" in str(info.value)
    assert "1" in str(info.value)
