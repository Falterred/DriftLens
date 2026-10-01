"""Parquet adapter tests (Step 16)."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from driftlens.adapters.parquet import load_parquet
from driftlens.config.schema import LimitsCfg, SideOptions
from driftlens.errors import InvalidInputError
from driftlens.security.limits import RuntimeLimits


def _limits() -> RuntimeLimits:
    return RuntimeLimits(LimitsCfg(), available_bytes=1 << 40)


@pytest.fixture()
def workdir() -> Path:
    base = Path(__file__).parent / "_parquet_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def test_parquet_fixed_size_list_embedding_no_rounding(workdir: Path) -> None:
    path = workdir / "fixed.parquet"
    values = [0.1234567890123, 0.2, 0.3]
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("embedding", pa.list_(pa.float64(), 3)),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(["1"]),
            pa.array(["hello"]),
            pa.array([values], type=pa.list_(pa.float64(), 3)),
        ],
        schema=schema,
    )
    pq.write_table(table, path)

    loaded = load_parquet(path, side=SideOptions(), limits=_limits())
    vector = loaded.records[0].embedding.vector
    assert vector is not None
    assert loaded.records[0].embedding.dtype == "float64"
    assert vector.dtype == np.float64
    assert float(vector[0]) == values[0]


def test_parquet_int8_list_is_metadata_not_embedding(workdir: Path) -> None:
    path = workdir / "int8.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("embedding", pa.list_(pa.int8())),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(["1"]),
            pa.array(["hello"]),
            pa.array([[1, 2, 3]], type=pa.list_(pa.int8())),
        ],
        schema=schema,
    )
    pq.write_table(table, path)

    loaded = load_parquet(path, side=SideOptions(), limits=_limits())
    record = loaded.records[0]
    assert record.embedding.vector is None
    assert record.metadata["embedding"] == [1, 2, 3]


def test_parquet_struct_column_becomes_nested_dict(workdir: Path) -> None:
    path = workdir / "struct.parquet"
    struct_type = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())])
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("meta", struct_type),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(["1"]),
            pa.array(["hello"]),
            pa.array([{"a": 7, "b": "x"}], type=struct_type),
        ],
        schema=schema,
    )
    pq.write_table(table, path)

    loaded = load_parquet(path, side=SideOptions(), limits=_limits())
    assert loaded.records[0].metadata["meta"] == {"a": 7, "b": "x"}


def test_parquet_pandas_metadata_ignored(workdir: Path) -> None:
    path = workdir / "pandas_meta.parquet"
    schema = pa.schema(
        [pa.field("id", pa.string()), pa.field("text", pa.string())],
        metadata={b"pandas": b"\x80\x04\x95not-valid-pickle", b"embedding_model": b"E1"},
    )
    table = pa.Table.from_arrays(
        [pa.array(["1"]), pa.array(["hello"])], schema=schema
    )
    pq.write_table(table, path)

    loaded = load_parquet(path, side=SideOptions(), limits=_limits())
    assert loaded.file_descriptors == {"embedding_model": "E1"}
    assert "pandas" not in loaded.file_descriptors


def test_parquet_timestamp_column_canonicalized(workdir: Path) -> None:
    path = workdir / "ts.parquet"
    stamp = datetime(2024, 1, 15, 8, 0, 0, 123456, tzinfo=timezone.utc)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("created_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(["1"]),
            pa.array(["hello"]),
            pa.array([stamp], type=pa.timestamp("us", tz="UTC")),
        ],
        schema=schema,
    )
    pq.write_table(table, path)

    loaded = load_parquet(path, side=SideOptions(), limits=_limits())
    assert loaded.records[0].created_at == "2024-01-15T08:00:00.123456Z"


def test_parquet_chunk_index_wrong_type_rejected(workdir: Path) -> None:
    path = workdir / "badchunk.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("chunk_index", pa.float64()),
        ]
    )
    table = pa.Table.from_arrays(
        [pa.array(["1"]), pa.array(["hello"]), pa.array([1.0])], schema=schema
    )
    pq.write_table(table, path)

    with pytest.raises(InvalidInputError) as info:
        load_parquet(path, side=SideOptions(), limits=_limits())
    assert "chunk_index must be an integer-typed column" in str(info.value)


def test_parquet_streaming_batches(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = workdir / "groups.parquet"
    schema = pa.schema([pa.field("id", pa.string()), pa.field("text", pa.string())])
    writer = pq.ParquetWriter(path, schema)
    for index in range(3):
        writer.write_table(
            pa.Table.from_arrays(
                [pa.array([str(index + 1)]), pa.array(["t" + str(index + 1)])],
                schema=schema,
            )
        )
    writer.close()

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("ParquetFile.read must not be called")

    monkeypatch.setattr(pq.ParquetFile, "read", _boom)
    loaded = load_parquet(path, side=SideOptions(), limits=_limits())
    assert len(loaded.records) == 3
    assert [record.row_number for record in loaded.records] == [1, 2, 3]


def test_parquet_empty_string_is_kept_not_null(workdir: Path) -> None:
    path = workdir / "emptystr.parquet"
    schema = pa.schema([pa.field("id", pa.string()), pa.field("text", pa.string())])
    table = pa.Table.from_arrays(
        [pa.array(["1"]), pa.array([""], type=pa.string())], schema=schema
    )
    pq.write_table(table, path)

    loaded = load_parquet(path, side=SideOptions(), limits=_limits())
    assert loaded.records[0].text == ""
    assert loaded.records[0].normalized_text == ""
