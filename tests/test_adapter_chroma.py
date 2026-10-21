"""Chroma adapter tests (Step 18)."""

from __future__ import annotations

import os
import secrets
import shutil
import sqlite3
import tempfile
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings

import driftlens.adapters.chroma as chroma
from driftlens.adapters.chroma import load_chroma
from driftlens.config.schema import LimitsCfg, SideOptions
from driftlens.errors import (
    InvalidInputError,
    RefuseUntrustedPickleError,
    UnsupportedFormatError,
)
from driftlens.security.limits import RuntimeLimits

pytestmark = pytest.mark.extras


def _limits() -> RuntimeLimits:
    return RuntimeLimits(LimitsCfg(), available_bytes=1 << 40)


def _rmtree_retries(path: Path) -> None:
    """rmtree with bounded retries for Windows sqlite lock release (F-02).

    Cleanup failures are reported (not silently ignored) but never fail the
    test run: they are printed once per leftover directory so a persistent
    leak stays visible instead of vanishing behind ignore_errors=True.
    """
    for attempt in range(6):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            import gc
            import time

            gc.collect()
            time.sleep(0.2 * (attempt + 1))
    try:
        shutil.rmtree(path)
    except OSError:
        print(f"WARNING: chroma test dir left behind (locked): {path}")


@pytest.fixture()
def workdir() -> Path:
    # A globally unique directory per invocation keeps a nested in-process pytest
    # run (the release gate) and any leftover locked directories from colliding.
    base = (
        Path(__file__).parent / f"_chroma_tmp_{os.getpid()}_{secrets.token_hex(6)}"
    )
    base.mkdir(parents=True, exist_ok=True)
    yield base
    _rmtree_retries(base)


def _client(path: Path):
    """Create a PersistentClient whose sqlite lock is released on close.

    Every caller must use the returned client inside _closed_client() or call
    client.close() before the fixture teardown; on Windows the sqlite lock
    otherwise survives until process exit and the temp dir cannot be removed
    (audit finding F-02).
    """
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )


def _temp_leftovers() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob("driftlens-chroma-*")}


def test_pickle_anywhere_in_tree_refused(workdir: Path, monkeypatch) -> None:
    directory = workdir / "pickled"
    nested = directory / "foo"
    nested.mkdir(parents=True)
    (nested / "bar.pkl").write_bytes(b"not a real pickle")
    calls: list[tuple] = []

    def tripwire(*args, **kwargs):
        calls.append(args)
        raise AssertionError("copy attempted")

    monkeypatch.setattr(shutil, "copytree", tripwire)
    with pytest.raises(RefuseUntrustedPickleError) as info:
        load_chroma(directory, side=SideOptions(), limits=_limits())
    assert "pickle file(s) found" in str(info.value)
    assert calls == []


def test_unsupported_schema_identifier(workdir: Path) -> None:
    directory = workdir / "schema"
    directory.mkdir(parents=True)
    connection = sqlite3.connect(str(directory / "chroma.sqlite3"))
    connection.execute("CREATE TABLE migrations (dir TEXT, version INTEGER)")
    connection.execute("INSERT INTO migrations (dir, version) VALUES ('other', 1)")
    connection.commit()
    connection.close()
    with pytest.raises(UnsupportedFormatError) as info:
        load_chroma(directory, side=SideOptions(), limits=_limits())
    assert "unrecognized Chroma schema identifier" in str(info.value)
    assert "other" in str(info.value)


def test_single_collection_auto_selected(workdir: Path) -> None:
    directory = workdir / "single"
    client = _client(directory)
    collection = client.create_collection("docs")
    collection.add(
        ids=["a", "b"],
        documents=["alpha", "beta"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )
    result = load_chroma(directory, side=SideOptions(), limits=_limits())
    assert result.source_type == "chroma"
    assert len(result.records) == 2
    assert sorted(record.record_id for record in result.records) == ["a", "b"]
    client.close()


def test_multi_collection_requires_selection(workdir: Path) -> None:
    directory = workdir / "multi"
    client = _client(directory)
    first = client.create_collection("alpha")
    first.add(ids=["a"], embeddings=[[0.1, 0.2]])
    second = client.create_collection("beta")
    second.add(ids=["b"], embeddings=[[0.3, 0.4]])
    with pytest.raises(InvalidInputError) as info:
        load_chroma(directory, side=SideOptions(), limits=_limits())
    assert "alpha" in str(info.value)
    assert "beta" in str(info.value)
    client.close()


def test_mixed_text_presence_rejected(workdir: Path) -> None:
    directory = workdir / "mixed"
    client = _client(directory)
    collection = client.create_collection("docs")
    collection.add(ids=["a"], documents=["alpha"], embeddings=[[0.1, 0.2]])
    collection.add(ids=["b"], embeddings=[[0.3, 0.4]])
    with pytest.raises(InvalidInputError) as info:
        load_chroma(directory, side=SideOptions(), limits=_limits())
    assert "mixed text presence" in str(info.value)
    client.close()


def test_null_metadata_becomes_empty_map(workdir: Path, monkeypatch) -> None:
    directory = workdir / "nullmeta"
    client = _client(directory)
    client.create_collection("docs")
    monkeypatch.setattr(
        chroma,
        "_page_rows",
        lambda collection: [("a", "alpha", [0.1, 0.2], {"tags": None})],
    )
    result = load_chroma(directory, side=SideOptions(), limits=_limits())
    result = load_chroma(directory, side=SideOptions(), limits=_limits())
    client.close()
    assert result.records[0].metadata["tags"] == {}


def test_metadata_document_id_is_medium_confidence(workdir: Path) -> None:
    directory = workdir / "docid"
    client = _client(directory)
    collection = client.create_collection("docs")
    collection.add(
        ids=["a"],
        documents=["alpha"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"document_id": "d1"}],
    )
    result = load_chroma(directory, side=SideOptions(), limits=_limits())
    assert result.records[0].identity_source_hint == "chroma_metadata"
    client.close()
    assert result.records[0].document_id == "d1"


def test_hnsw_space_mapping(workdir: Path) -> None:
    for space, expected in (("l2", "l2"), ("ip", "dot"), ("cosine", "cosine")):
        directory = workdir / ("space_" + space)
        client = _client(directory)
        collection = client.create_collection("docs", metadata={"hnsw:space": space})
        collection.add(ids=["a"], documents=["alpha"], embeddings=[[0.1, 0.2]])
        result = load_chroma(directory, side=SideOptions(), limits=_limits())
        assert result.records[0].embedding.similarity_metric == expected
        client.close()

    class _Stub:
        metadata: dict = {}
        configuration_json: dict = {}

    assert chroma._metric_kind(_Stub()) is None


def test_row_order_independent_of_paging(workdir: Path, monkeypatch) -> None:
    monkeypatch.setattr(chroma, "PAGE_SIZE", 1)

    def build(name: str, ids: list[str]) -> tuple[Path, object]:
        directory = workdir / name
        client = _client(directory)
        collection = client.create_collection("docs")
        for index, record_id in enumerate(ids):
            collection.add(
                ids=[record_id],
                documents=["text " + record_id],
                embeddings=[[float(index), 0.0]],
            )
        return directory, client

    forward, forward_client = build("forward", ["a", "b", "c"])
    backward, backward_client = build("backward", ["c", "b", "a"])
    first = load_chroma(forward, side=SideOptions(), limits=_limits())
    second = load_chroma(backward, side=SideOptions(), limits=_limits())
    assert [(r.record_id, r.row_number) for r in first.records] == [
        (r.record_id, r.row_number) for r in second.records
    ]
    assert [r.row_number for r in first.records] == [1, 2, 3]
    forward_client.close()
    backward_client.close()


def test_copy_deleted_after_use(workdir: Path) -> None:
    directory = workdir / "copydel"
    client = _client(directory)
    collection = client.create_collection("docs")
    collection.add(ids=["a"], documents=["alpha"], embeddings=[[0.1, 0.2]])
    before = _temp_leftovers()
    load_chroma(directory, side=SideOptions(), limits=_limits())
    client.close()
    assert _temp_leftovers() == before


def test_telemetry_disabled(workdir: Path, monkeypatch) -> None:
    directory = workdir / "telemetry"
    client = _client(directory)
    collection = client.create_collection("docs")
    collection.add(ids=["a"], documents=["alpha"], embeddings=[[0.1, 0.2]])
    captured: dict = {}
    original = chromadb.PersistentClient

    def spy(*args, **kwargs):
        captured["settings"] = kwargs.get("settings")
        return original(*args, **kwargs)

    monkeypatch.setattr(chromadb, "PersistentClient", spy)
    load_chroma(directory, side=SideOptions(), limits=_limits())
    client.close()
    assert captured["settings"].anonymized_telemetry is False
