"""Eval-set schema, query-id derivation and query-embedding joining (Step 38)."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from pathlib import Path

import numpy as np
import pytest

from driftlens.config.schema import LimitsCfg
from driftlens.errors import InvalidInputError
from driftlens.schema.eval_set import (
    EvalCase,
    EvalSet,
    QueryEmbeddings,
    derive_query_id,
    join_query_embedding_files,
    load_eval_set,
)
from driftlens.security.limits import RuntimeLimits


@pytest.fixture()
def workdir() -> Path:
    # A unique directory under tests/ (not pytest's tmp_path) keeps the file-sandbox
    # from tripping over the mode-0o700 basetemp that tmp_path would create.
    base = Path(__file__).parent / f"_eval_set_tmp_{os.getpid()}_{secrets.token_hex(6)}"
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _limits() -> RuntimeLimits:
    return RuntimeLimits(LimitsCfg(), available_bytes=1 << 40)


def _write(workdir: Path, *cases: dict) -> Path:
    path = workdir / "eval_set.json"
    path.write_text(json.dumps(list(cases)), encoding="utf-8")
    return path


def _write_jsonl(workdir: Path, name: str, *rows: dict) -> Path:
    path = workdir / name
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_derive_query_id_normalizes_first() -> None:
    assert derive_query_id("  What  is  X?  ") == derive_query_id("What is X?")


def test_derive_query_id_format() -> None:
    derived = derive_query_id("hello")
    assert derived.startswith("q:")
    assert re.fullmatch(r"q:[0-9a-f]{12}", derived) is not None


def test_load_eval_set_missing_query_field(workdir: Path) -> None:
    path = _write(workdir, {"relevant_document_ids": ["d1"]})
    with pytest.raises(InvalidInputError) as caught:
        load_eval_set(path, limits=_limits())
    assert "case 1" in str(caught.value)


def test_load_eval_set_neither_id_list(workdir: Path) -> None:
    path = _write(
        workdir,
        {"query": "x", "relevant_document_ids": [], "relevant_chunk_ids": []},
    )
    with pytest.raises(InvalidInputError):
        load_eval_set(path, limits=_limits())


def test_load_eval_set_both_id_lists_allowed(workdir: Path) -> None:
    path = _write(
        workdir,
        {
            "query": "x",
            "relevant_document_ids": ["d1"],
            "relevant_chunk_ids": ["c1"],
        },
    )
    result = load_eval_set(path, limits=_limits())
    case = result.cases[0]
    assert case.relevant_document_ids == ["d1"]
    assert case.relevant_chunk_ids == ["c1"]


def test_load_eval_set_duplicate_explicit_ids(workdir: Path) -> None:
    path = _write(
        workdir,
        {"query": "a", "query_id": "q_001", "relevant_document_ids": ["d1"]},
        {"query": "b", "query_id": "q_001", "relevant_chunk_ids": ["c1"]},
    )
    with pytest.raises(InvalidInputError) as caught:
        load_eval_set(path, limits=_limits())
    assert "q_001" in str(caught.value)


def test_load_eval_set_duplicate_derived_ids(workdir: Path) -> None:
    path = _write(
        workdir,
        {"query": "Hello   world", "relevant_document_ids": ["d1"]},
        {"query": "Hello world", "relevant_chunk_ids": ["c1"]},
    )
    with pytest.raises(InvalidInputError) as caught:
        load_eval_set(path, limits=_limits())
    assert derive_query_id("Hello world") in str(caught.value)


def test_load_eval_set_mixed_explicit_and_derived_collision(workdir: Path) -> None:
    collision = derive_query_id("collision text")
    path = _write(
        workdir,
        {"query": "first", "query_id": collision, "relevant_document_ids": ["d1"]},
        {"query": "collision text", "relevant_chunk_ids": ["c1"]},
    )
    with pytest.raises(InvalidInputError) as caught:
        load_eval_set(path, limits=_limits())
    assert collision in str(caught.value)


def _explicit_case(query_id: str, embeddings: QueryEmbeddings | None) -> EvalCase:
    return EvalCase(
        query_id=query_id,
        query_id_was_explicit=True,
        query="query text",
        relevant_document_ids=["d1"],
        relevant_chunk_ids=[],
        query_embeddings=embeddings,
    )


def test_join_query_embeddings_requires_explicit_ids(workdir: Path) -> None:
    eval_set = EvalSet(
        cases=[
            EvalCase(
                query_id="q:derived",
                query_id_was_explicit=False,
                query="x",
                relevant_document_ids=["d1"],
                relevant_chunk_ids=[],
                query_embeddings=None,
            )
        ]
    )
    path = _write_jsonl(workdir, "a.jsonl", {"query_id": "q:derived", "embedding": [1.0, 0.0]})
    with pytest.raises(InvalidInputError):
        join_query_embedding_files(
            eval_set, path_a=path, path_b=None, limits=_limits()
        )


def test_join_query_embeddings_file_overrides_inline(workdir: Path) -> None:
    inline = QueryEmbeddings(v1=np.asarray([0.0, 1.0], dtype=np.float32), v2=None)
    eval_set = EvalSet(cases=[_explicit_case("q_001", inline)])
    path = _write_jsonl(workdir, "a.jsonl", {"query_id": "q_001", "embedding": [1.0, 0.0]})
    result = join_query_embedding_files(
        eval_set, path_a=path, path_b=None, limits=_limits()
    )
    assert result is not eval_set
    embeddings = result.cases[0].query_embeddings
    assert embeddings is not None
    assert np.allclose(embeddings.v1, [1.0, 0.0])
    assert embeddings.v2 is None


def test_join_query_embeddings_unmatched_case_gets_none(workdir: Path) -> None:
    eval_set = EvalSet(cases=[_explicit_case("q_001", None)])
    path = _write_jsonl(workdir, "a.jsonl", {"query_id": "q_other", "embedding": [1.0, 0.0]})
    result = join_query_embedding_files(
        eval_set, path_a=path, path_b=None, limits=_limits()
    )
    embeddings = result.cases[0].query_embeddings
    assert embeddings is not None
    assert embeddings.v1 is None


def test_join_query_embeddings_noop_when_no_files_given() -> None:
    eval_set = EvalSet(cases=[_explicit_case("q_001", None)])
    result = join_query_embedding_files(
        eval_set, path_a=None, path_b=None, limits=_limits()
    )
    assert result is eval_set
