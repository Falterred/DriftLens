"""Snapshot format write/read tests (Step 36)."""

from __future__ import annotations

import json
import os
import secrets
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from driftlens.adapters.snapshot import load_snapshot_as_input
from driftlens.config.schema import SideOptions, default_config
from driftlens.errors import InvalidInputError, UnsupportedFormatError
from driftlens.identity.hashing import sha256_hex
from driftlens.schema.manifest import FieldMappingSummary
from driftlens.schema.record import EmbeddingInfo, Record
from driftlens.schema.snapshot import build_snapshot, read_snapshot, write_snapshot
from driftlens.security.limits import RuntimeLimits


@pytest.fixture
def workdir():
    path = Path(__file__).parent / f"_snapshot_io_tmp_{os.getpid()}_{secrets.token_hex(4)}"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _record(
    record_id,
    text=None,
    *,
    dim=None,
    model=None,
    document_id=None,
    chunk_index=None,
    row_number=0,
):
    vector = np.zeros(dim, dtype=np.float32) if dim is not None else None
    embedding = EmbeddingInfo(
        vector=vector,
        dtype="float32" if vector is not None else None,
        dimension=dim if vector is not None else None,
        norm=float(np.linalg.norm(vector)) if vector is not None else None,
        validity="ok" if vector is not None else None,
        model=model,
        provider=None,
        similarity_metric=None,
        normalized=None,
    )
    return Record(
        record_id=record_id,
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        row_number=row_number,
        embedding=embedding,
    )


def _build(records):
    from driftlens.errors import DiagnosticSink

    cfg = default_config()
    return build_snapshot(
        records,
        side=SideOptions(),
        cfg=cfg,
        source_type="jsonl",
        sink=DiagnosticSink(),
        limits=RuntimeLimits(cfg.limits),
        granularity_override=None,
        field_mapping_summary=FieldMappingSummary(),
        unsafe_pickle=False,
    )


def _limits():
    return RuntimeLimits(default_config().limits)


def test_write_read_round_trip_preserves_records(workdir):
    snap = _build(
        [
            _record("a", "hello world", dim=4, model="m", document_id="d1", chunk_index=0),
            _record("b", "second text", document_id="d1", chunk_index=1, row_number=1),
        ]
    )
    out = workdir / "round_trip"
    write_snapshot(snap, out, no_text=False)

    loaded = read_snapshot(out, limits=_limits(), verify_full=False)
    original = {record.record_id: record for record in snap.records}
    restored = {record.record_id: record for record in loaded.records}
    assert set(restored) == set(original)
    for record_id, record in original.items():
        got = restored[record_id]
        assert got.text == record.text
        assert got.normalized_text == record.normalized_text
        assert got.normalized_text_hash == record.normalized_text_hash
        assert got.metadata == record.metadata
        assert got.chunk_index == record.chunk_index
        assert got.document_id == record.document_id
        if record.embedding.vector is None:
            assert got.embedding.vector is None
        else:
            assert np.allclose(got.embedding.vector, record.embedding.vector)


def test_no_text_omits_columns(workdir):
    snap = _build([_record("a", "hello", dim=4, model="m", document_id="d1")])

    out = workdir / "no_text"
    write_snapshot(snap, out, no_text=True)
    names = pq.read_schema(out / "records.parquet").names
    assert "text" not in names
    assert "normalized_text" not in names

    out2 = workdir / "with_text"
    write_snapshot(snap, out2, no_text=False)
    names2 = pq.read_schema(out2 / "records.parquet").names
    assert "text" in names2
    assert "normalized_text" in names2


def test_sparse_embedding_storage(workdir):
    uniform = _build(
        [
            _record("a", dim=4, model="m", document_id="d1"),
            _record("b", dim=4, model="m", document_id="d1"),
        ]
    )
    out = workdir / "uniform"
    write_snapshot(uniform, out, no_text=True)
    overrides = pq.read_table(out / "records.parquet").column(
        "embedding_model_override"
    ).to_pylist()
    assert overrides == [None, None]

    mixed = _build(
        [
            _record("a", dim=4, model="m", document_id="d1"),
            _record("b", dim=4, model="m", document_id="d1"),
            _record("c", dim=4, model="x", document_id="d1"),
        ]
    )
    mixed.manifest.embedding_model = "m"
    out2 = workdir / "mixed"
    write_snapshot(mixed, out2, no_text=True)
    overrides2 = pq.read_table(out2 / "records.parquet").column(
        "embedding_model_override"
    ).to_pylist()
    assert [value for value in overrides2 if value is not None] == ["x"]


def test_metadata_round_trips_via_canonical_json(workdir):
    record = _record("a", "hello", dim=4, model="m", document_id="d1")
    record.metadata = {"count": 7, "ratio": 1.0}
    snap = _build([record])

    out = workdir / "metadata"
    write_snapshot(snap, out, no_text=False)
    loaded = read_snapshot(out, limits=_limits(), verify_full=False)

    metadata = loaded.records[0].metadata
    assert metadata == {"count": 7, "ratio": 1.0}
    assert isinstance(metadata["count"], int)
    assert isinstance(metadata["ratio"], float)


def test_checksum_mismatch_detected(workdir, monkeypatch):
    snap = _build([_record("a", "hello", dim=4, model="m", document_id="d1")])
    out = workdir / "checksum"
    write_snapshot(snap, out, no_text=False)

    target = out / "records.parquet"
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))

    def tripwire(*args, **kwargs):
        raise AssertionError("parsing must not run before checksum verification")

    monkeypatch.setattr("pyarrow.parquet.read_table", tripwire)
    with pytest.raises(InvalidInputError, match="checksum mismatch"):
        read_snapshot(out, limits=_limits(), verify_full=False)


def test_unsupported_major_schema_version(workdir):
    snap = _build([_record("a", "hello", dim=4, model="m", document_id="d1")])
    out = workdir / "version"
    write_snapshot(snap, out, no_text=False)

    manifest_path = out / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["schema_version"] = "99.0"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(UnsupportedFormatError) as info:
        read_snapshot(out, limits=_limits(), verify_full=False)
    assert info.value.exit_code == 5


def test_verify_full_detects_tampered_record(workdir):
    snap = _build([_record("a", "hello world", dim=4, model="m", document_id="d1")])
    out = workdir / "tamper"
    write_snapshot(snap, out, no_text=False)

    records_path = out / "records.parquet"
    table = pq.read_table(records_path)
    index = table.schema.get_field_index("normalized_text_hash")
    table = table.set_column(index, "normalized_text_hash", pa.array(["0" * 64], type=pa.string()))
    pq.write_table(table, records_path)

    manifest_path = out / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["files"]["records.parquet"] = sha256_hex(records_path.read_bytes())
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(InvalidInputError, match="snapshot id mismatch"):
        read_snapshot(out, limits=_limits(), verify_full=True)


def test_load_snapshot_as_input_skips_rebuild(workdir, monkeypatch):
    snap = _build([_record("a", "hello", dim=4, model="m", document_id="d1")])
    out = workdir / "skip"
    write_snapshot(snap, out, no_text=False)

    def boom(*args, **kwargs):
        raise AssertionError("assign_identity must not run for a snapshot source")

    monkeypatch.setattr("driftlens.identity.documents.assign_identity", boom)
    loaded = load_snapshot_as_input(out, limits=_limits())
    assert loaded.source_type == "snapshot"
    assert loaded._prebuilt_snapshot.manifest.snapshot_id == snap.manifest.snapshot_id
