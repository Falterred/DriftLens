"""Sidecar adapter tests (Step 17)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from driftlens.adapters.sidecar import load_sidecar
from driftlens.config.schema import LimitsCfg, SideOptions
from driftlens.errors import InvalidInputError
from driftlens.security.limits import RuntimeLimits


def _limits() -> RuntimeLimits:
    return RuntimeLimits(LimitsCfg(), available_bytes=1 << 40)


@pytest.fixture()
def workdir() -> Path:
    base = Path(__file__).parent / "_sidecar_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + chr(10) for row in rows), encoding="utf-8"
    )
    return path


def test_sidecar_faiss_id_excluded_from_roles(workdir: Path) -> None:
    path = _write(
        workdir / "roles.jsonl",
        [{"faiss_id": 0, "text": "alpha", "id": "c0"}],
    )
    table = load_sidecar(path, side=SideOptions(), limits=_limits())
    assert table.field_mapping.chunk_id != "faiss_id"
    assert table.field_mapping.chunk_id == "id"
    assert table[0].record.record_id == "c0"


def test_sidecar_duplicate_faiss_id_rejected(workdir: Path) -> None:
    path = _write(
        workdir / "dupe.jsonl",
        [
            {"faiss_id": 5, "text": "alpha"},
            {"faiss_id": 5, "text": "beta"},
        ],
    )
    with pytest.raises(InvalidInputError) as info:
        load_sidecar(path, side=SideOptions(), limits=_limits())
    assert "duplicate" in str(info.value)
    assert "1" in str(info.value)


def test_sidecar_embedding_column_rejected(workdir: Path) -> None:
    path = _write(
        workdir / "vec.jsonl",
        [
            {"faiss_id": 0, "text": "alpha", "extra_vec": [0.1, 0.2]},
            {"faiss_id": 1, "text": "beta", "extra_vec": [0.3, 0.4]},
        ],
    )
    with pytest.raises(InvalidInputError) as info:
        load_sidecar(path, side=SideOptions(), limits=_limits())
    assert "extra_vec" in str(info.value)
