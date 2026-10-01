"""Whole-file JSON adapter tests (Step 14)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from driftlens.adapters.json import load_json
from driftlens.config.schema import LimitsCfg, SideOptions
from driftlens.security.limits import RuntimeLimits


def _limits() -> RuntimeLimits:
    return RuntimeLimits(LimitsCfg(), available_bytes=1 << 40)


@pytest.fixture()
def workdir() -> Path:
    base = Path(__file__).parent / "_json_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def test_json_array_shape_load(workdir: Path) -> None:
    path = workdir / "array.json"
    path.write_text(json.dumps([{"text": "a"}, {"text": "b"}]), encoding="utf-8")
    loaded = load_json(path, side=SideOptions(), limits=_limits())
    assert len(loaded.records) == 2
    assert [r.row_number for r in loaded.records] == [1, 2]


def test_json_wrapper_shape_with_descriptors(workdir: Path) -> None:
    path = workdir / "wrapper.json"
    path.write_text(
        json.dumps(
            {
                "records": [{"embedding": [1, 2]}],
                "embedding_model": "E1",
                "embedding_dtype": "float64",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_json(path, side=SideOptions(), limits=_limits())
    assert loaded.file_descriptors == {
        "embedding_model": "E1",
        "embedding_dtype": "float64",
    }
    assert loaded.records[0].embedding.dtype == "float64"


def test_json_metadata_native_types_preserved(workdir: Path) -> None:
    path = workdir / "meta.json"
    path.write_text(
        json.dumps(
            [{"text": "x", "count": 7, "ratio": 1.0, "flag": True}]
        ),
        encoding="utf-8",
    )
    loaded = load_json(path, side=SideOptions(), limits=_limits())
    metadata = loaded.records[0].metadata
    assert metadata["count"] == 7
    assert isinstance(metadata["count"], int)
    assert metadata["ratio"] == 1.0
    assert isinstance(metadata["ratio"], float)
    assert metadata["flag"] is True


def test_json_row_number_is_array_position(workdir: Path) -> None:
    path = workdir / "rows.json"
    path.write_text(json.dumps([{"text": "a"}, {"text": "b"}]), encoding="utf-8")
    loaded = load_json(path, side=SideOptions(), limits=_limits())
    assert loaded.records[1].row_number == 2
