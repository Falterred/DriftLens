"""CSV adapter tests (Step 15)."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from driftlens.adapters.csv import load_csv
from driftlens.config.schema import LimitsCfg, SideOptions
from driftlens.errors import InvalidInputError
from driftlens.security.limits import RuntimeLimits


def _limits() -> RuntimeLimits:
    return RuntimeLimits(LimitsCfg(), available_bytes=1 << 40)


@pytest.fixture()
def workdir() -> Path:
    base = Path(__file__).parent / "_csv_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _write(workdir: Path, name: str, text: str) -> Path:
    path = workdir / name
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_json_array_embedding_default(workdir: Path) -> None:
    path = _write(
        workdir,
        "json_array.csv",
        'id,text,embedding\n1,hello,"[0.1,0.2,0.3]"\n',
    )
    loaded = load_csv(path, side=SideOptions(), limits=_limits())
    vector = loaded.records[0].embedding.vector
    assert vector is not None
    assert vector.shape[0] == 3
    assert vector.dtype == np.float32
    assert loaded.records[0].embedding.dtype == "float32"


def test_csv_delimited_embedding_format(workdir: Path) -> None:
    path = _write(
        workdir,
        "delimited.csv",
        'id,text,embedding\n1,hello,"0.1,0.2,0.3"\n',
    )
    loaded = load_csv(
        path,
        side=SideOptions(embedding_format="delimited"),
        limits=_limits(),
    )
    vector = loaded.records[0].embedding.vector
    assert vector is not None
    assert vector.shape[0] == 3
    assert np.allclose(vector, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_csv_wide_columns_contiguous_run(workdir: Path) -> None:
    path = _write(
        workdir,
        "wide.csv",
        "id,text,emb_0,emb_1,emb_2,emb_4\n1,hello,0.1,0.2,0.3,9\n",
    )
    loaded = load_csv(
        path,
        side=SideOptions(embedding_prefix="emb_"),
        limits=_limits(),
    )
    record = loaded.records[0]
    assert record.embedding.vector is not None
    assert record.embedding.vector.shape[0] == 3
    assert record.metadata["emb_4"] == 9
    assert isinstance(record.metadata["emb_4"], int)


def test_csv_wide_columns_missing_zero(workdir: Path) -> None:
    path = _write(workdir, "nozero.csv", "id,emb_1,emb_2\n1,0.1,0.2\n")
    with pytest.raises(InvalidInputError) as info:
        load_csv(
            path,
            side=SideOptions(embedding_prefix="emb_"),
            limits=_limits(),
        )
    assert "emb_0" in str(info.value)


def test_csv_column_count_mismatch(workdir: Path) -> None:
    path = _write(workdir, "short.csv", "id,text\n1\n")
    with pytest.raises(InvalidInputError) as info:
        load_csv(path, side=SideOptions(), limits=_limits())
    message = str(info.value)
    assert "row 1" in message
    assert "expected 2" in message
    assert "found 1" in message


def test_csv_empty_text_cell_is_none(workdir: Path) -> None:
    path = _write(workdir, "emptytext.csv", "id,text\n1,\n")
    loaded = load_csv(path, side=SideOptions(), limits=_limits())
    assert loaded.records[0].text is None
    assert loaded.records[0].record_id == "1"


def test_csv_chunk_index_must_be_int(workdir: Path) -> None:
    path = _write(workdir, "chunkidx.csv", "id,text,chunk_index\n1,hello,abc\n")
    with pytest.raises(InvalidInputError) as info:
        load_csv(path, side=SideOptions(), limits=_limits())
    assert "must be an integer" in str(info.value)


def test_csv_timestamp_unit_seconds(workdir: Path) -> None:
    path = _write(
        workdir, "seconds.csv", "id,text,created_at\n1,hello,1705315200\n"
    )
    loaded = load_csv(
        path,
        side=SideOptions(timestamp_unit="seconds"),
        limits=_limits(),
    )
    expected = datetime.fromtimestamp(1705315200, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    ) + "Z"
    assert loaded.records[0].created_at == expected


def test_csv_metadata_typing_applied(workdir: Path) -> None:
    path = _write(
        workdir, "meta.csv", "id,text,ratio,count\n1,hello,3.0,3\n"
    )
    loaded = load_csv(path, side=SideOptions(), limits=_limits())
    metadata = loaded.records[0].metadata
    assert metadata["ratio"] == 3.0
    assert isinstance(metadata["ratio"], float)
    assert metadata["count"] == 3
    assert isinstance(metadata["count"], int)


def test_csv_row_numbers_exclude_header(workdir: Path) -> None:
    path = _write(workdir, "rows.csv", "id,text\n1,a\n2,b\n")
    loaded = load_csv(path, side=SideOptions(), limits=_limits())
    assert loaded.records[0].row_number == 1
    assert loaded.records[1].row_number == 2
