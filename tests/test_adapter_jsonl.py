"""JSONL adapter tests (Step 14)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from driftlens.adapters.jsonl import load_jsonl
from driftlens.config.schema import LimitsCfg, SideOptions
from driftlens.errors import InvalidInputError
from driftlens.security.limits import RuntimeLimits


def _limits() -> RuntimeLimits:
    return RuntimeLimits(LimitsCfg(), available_bytes=1 << 40)


@pytest.fixture()
def workdir() -> Path:
    base = Path(__file__).parent / "_jsonl_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def test_jsonl_basic_load(workdir: Path) -> None:
    path = workdir / "basic.jsonl"
    path.write_text(
        '{"text": "alpha beta", "embedding": [0.1, 0.2, 0.3]}\n'
        '{"text": "gamma", "embedding": [0.4, 0.5, 0.6]}\n'
        '{"text": "delta epsilon zeta", "embedding": [0.7, 0.8, 0.9]}\n',
        encoding="utf-8",
    )
    loaded = load_jsonl(path, side=SideOptions(), limits=_limits())
    assert len(loaded.records) == 3
    assert [r.row_number for r in loaded.records] == [1, 2, 3]
    for record in loaded.records:
        assert record.token_count is not None
        assert record.char_count is not None
        assert record.embedding.dtype == "float32"


def test_jsonl_blank_lines_skipped_but_row_numbers_track_file_lines(
    workdir: Path,
) -> None:
    path = workdir / "blank.jsonl"
    path.write_text(
        '{"text": "a", "embedding": [1, 2]}\n\n{"text": "b", "embedding": [3, 4]}\n',
        encoding="utf-8",
    )
    loaded = load_jsonl(path, side=SideOptions(), limits=_limits())
    assert len(loaded.records) == 2
    assert [r.row_number for r in loaded.records] == [1, 3]


def test_jsonl_embedding_b64_requires_dtype(workdir: Path) -> None:
    path = workdir / "b64.jsonl"
    path.write_text('{"text": "x", "embedding_b64": "AAAA"}\n', encoding="utf-8")
    with pytest.raises(InvalidInputError) as info:
        load_jsonl(path, side=SideOptions(), limits=_limits())
    assert "declared dtype" in str(info.value)


def test_jsonl_nan_inf_tokens_accepted_in_embedding(workdir: Path) -> None:
    path = workdir / "nan.jsonl"
    path.write_text('{"text": "x", "embedding": [1.0, NaN, 2.0]}\n', encoding="utf-8")
    loaded = load_jsonl(path, side=SideOptions(), limits=_limits())
    vector = loaded.records[0].embedding.vector
    assert vector is not None
    assert vector.dtype == np.float32
    assert np.isnan(vector[1])
    assert float(vector[0]) == 1.0


def test_jsonl_empty_row_rejected(workdir: Path) -> None:
    path = workdir / "empty.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(InvalidInputError) as info:
        load_jsonl(path, side=SideOptions(), limits=_limits())
    assert "line 1" in str(info.value)
