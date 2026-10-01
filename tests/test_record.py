"""Canonical record shape and value typing (Step 12)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from driftlens.errors import InvalidInputError
from driftlens.schema.record import (
    Record,
    canonicalize_timestamp,
    is_iso8601_datetime_string,
    type_csv_cell,
)


def test_csv_typing_order() -> None:
    assert type_csv_cell("") == ("null", None)
    assert type_csv_cell("True") == ("bool", True)
    assert type_csv_cell("007") == ("string", "007")
    assert type_csv_cell("0") == ("int", 0)
    assert type_csv_cell("3.14") == ("float", 3.14)
    assert type_csv_cell("2024-01-15") == ("datetime", "2024-01-15")
    assert type_csv_cell("2024-01-15T10:00:00Z")[0] == "datetime"
    assert type_csv_cell("[1,2,3]") == ("list", [1, 2, 3])
    assert type_csv_cell('{"a":1}') == ("json", {"a": 1})
    assert type_csv_cell("hello") == ("string", "hello")
    assert type_csv_cell("nan") == ("string", "nan")


def test_iso8601_validator() -> None:
    assert is_iso8601_datetime_string("2024-01-15") is True
    assert is_iso8601_datetime_string("2024-01-15T10:30:00.123Z") is True
    assert is_iso8601_datetime_string("2024-01-15T10:30:00+05:30") is True
    assert is_iso8601_datetime_string("not-a-date") is False
    assert is_iso8601_datetime_string("2024-13-40") is False


def test_canonicalize_iso_naive_is_utc() -> None:
    assert (
        canonicalize_timestamp("2024-01-15T10:00:00", unit="iso", location="x")
        == "2024-01-15T10:00:00Z"
    )


def test_canonicalize_iso_offset_converted() -> None:
    assert (
        canonicalize_timestamp("2024-01-15T10:00:00+05:30", unit="iso", location="x")
        == "2024-01-15T04:30:00Z"
    )


def test_canonicalize_fraction_trimmed() -> None:
    assert (
        canonicalize_timestamp("2024-01-15T10:00:00.500000Z", unit="iso", location="x")
        == "2024-01-15T10:00:00.5Z"
    )
    assert (
        canonicalize_timestamp("2024-01-15T10:00:00Z", unit="iso", location="x")
        == "2024-01-15T10:00:00Z"
    )
    assert (
        canonicalize_timestamp("2024-01-15T10:00:00.123400Z", unit="iso", location="x")
        == "2024-01-15T10:00:00.1234Z"
    )


def test_canonicalize_seconds_unit() -> None:
    epoch = 1705315200
    expected = (
        datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        + "Z"
    )
    assert canonicalize_timestamp(epoch, unit="seconds", location="x") == expected
    assert (
        canonicalize_timestamp(epoch * 1000, unit="milliseconds", location="x")
        == expected
    )


def test_canonicalize_bad_value_raises() -> None:
    with pytest.raises(InvalidInputError) as info:
        canonicalize_timestamp(
            "not a date", unit="iso", location="row 5 field created_at"
        )
    assert info.value.location == "row 5 field created_at"


def test_record_defaults() -> None:
    record = Record(record_id="r1")
    assert record.embedding.vector is None
    assert record.metadata == {}
