"""Input-format detection and field-role resolution (Step 13)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from driftlens.adapters.detect import detect_format, resolve_fields
from driftlens.config.schema import LimitsCfg, SideOptions
from driftlens.errors import InvalidInputError, UnsupportedFormatError
from driftlens.security.limits import RuntimeLimits


def _limits() -> RuntimeLimits:
    return RuntimeLimits(LimitsCfg(), available_bytes=1 << 40)


@pytest.fixture()
def workdir() -> Path:
    base = Path(__file__).parent / "_detect_tmp"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def test_detect_jsonl_by_extension(workdir: Path) -> None:
    path = workdir / "x.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    assert detect_format(path, forced=None, limits=_limits()) == "jsonl"


def test_detect_json_array_shape(workdir: Path) -> None:
    path = workdir / "x.json"
    path.write_text(json.dumps([{"a": 1}]), encoding="utf-8")
    assert detect_format(path, forced=None, limits=_limits()) == "json"


def test_detect_json_records_wrapper(workdir: Path) -> None:
    for key in ("records", "items", "data"):
        path = workdir / f"w_{key}.json"
        path.write_text(json.dumps({key: [{"a": 1}]}), encoding="utf-8")
        assert detect_format(path, forced=None, limits=_limits()) == "json"
    other = workdir / "w_foo.json"
    other.write_text(json.dumps({"foo": [{"a": 1}]}), encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        detect_format(other, forced=None, limits=_limits())


def test_detect_parquet_by_magic_regardless_of_extension(workdir: Path) -> None:
    path = workdir / "x.dat"
    path.write_bytes(b"PAR1" + b"\x00" * 8)
    assert detect_format(path, forced=None, limits=_limits()) == "parquet"


def test_detect_compressed_rejected(workdir: Path) -> None:
    path = workdir / "x.jsonl.gz"
    path.write_bytes(b"\x1f\x8b\x08")
    with pytest.raises(UnsupportedFormatError) as info:
        detect_format(path, forced=None, limits=_limits())
    assert ".gz" in str(info.value)


def test_detect_snapshot_dir(workdir: Path) -> None:
    path = workdir / "snap"
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0"}), encoding="utf-8"
    )
    assert detect_format(path, forced=None, limits=_limits()) == "snapshot"


def test_detect_chroma_before_faiss_precedence(workdir: Path) -> None:
    path = workdir / "store"
    path.mkdir()
    (path / "chroma.sqlite3").write_bytes(b"")
    (path / "index.faiss").write_bytes(b"")
    assert detect_format(path, forced=None, limits=_limits()) == "chroma"


def test_resolve_fields_basic_priority() -> None:
    mapping = resolve_fields(
        ["content", "text"],
        granularity_hint="AUTO",
        overrides=SideOptions(),
        location="side a",
    )
    assert mapping.text == "text"
    assert mapping.ignored["text"] == ["content"]


def test_resolve_fields_same_tier_collision() -> None:
    with pytest.raises(InvalidInputError) as info:
        resolve_fields(
            ["Text", "text"],
            granularity_hint="AUTO",
            overrides=SideOptions(),
            location="side a",
        )
    assert "Text" in str(info.value)
    assert "text" in str(info.value)


def test_resolve_fields_neither_text_nor_embedding() -> None:
    with pytest.raises(InvalidInputError) as info:
        resolve_fields(
            ["foo", "bar"],
            granularity_hint="AUTO",
            overrides=SideOptions(),
            location="side a",
        )
    assert "no text or embedding" in str(info.value)


def test_resolve_fields_override_forces_key() -> None:
    mapping = resolve_fields(
        ["foo", "content"],
        granularity_hint="AUTO",
        overrides=SideOptions(text_field="foo"),
        location="side a",
    )
    assert mapping.text == "foo"


def test_resolve_fields_override_missing_key_errors() -> None:
    with pytest.raises(InvalidInputError):
        resolve_fields(
            ["content"],
            granularity_hint="AUTO",
            overrides=SideOptions(text_field="missing"),
            location="side a",
        )


def test_source_identifier_copied_not_consumed() -> None:
    mapping = resolve_fields(
        ["source", "text"],
        granularity_hint="AUTO",
        overrides=SideOptions(),
        location="side a",
    )
    assert mapping.document_id == "source"
    assert "source" in mapping.copied_keys
    assert "source" not in mapping.consumed_keys


def test_document_id_high_priority_consumed() -> None:
    mapping = resolve_fields(
        ["document_id", "source", "text"],
        granularity_hint="AUTO",
        overrides=SideOptions(),
        location="side a",
    )
    assert mapping.document_id == "document_id"
    assert "document_id" in mapping.consumed_keys
