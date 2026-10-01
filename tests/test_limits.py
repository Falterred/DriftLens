"""Runtime limits and strict JSON parsing (Step 5)."""

from __future__ import annotations

import math

import pytest

from driftlens.config.schema import LimitsCfg
from driftlens.errors import InvalidInputError, LimitExceededError
from driftlens.security.limits import MAX_JSON_DEPTH, RuntimeLimits, loads_limited


def test_depth_65_rejected() -> None:
    with pytest.raises(LimitExceededError) as excinfo:
        loads_limited("[" * 65 + "]" * 65, location="x")
    assert excinfo.value.exit_code == 8
    assert MAX_JSON_DEPTH == 64
    assert isinstance(loads_limited("[" * 64 + "]" * 64, location="x"), list)


def test_duplicate_key() -> None:
    with pytest.raises(InvalidInputError):
        loads_limited('{"a":1,"a":2}', location="x")


def test_nan_tokens_accepted() -> None:
    values = loads_limited("[NaN, Infinity, -Infinity]", location="x")
    assert math.isnan(values[0])
    assert values[1] == math.inf
    assert values[2] == -math.inf


def test_records_limit() -> None:
    limits = RuntimeLimits(LimitsCfg(max_records=10))
    with pytest.raises(LimitExceededError) as excinfo:
        limits.check_records(11, location="records")
    assert "--max-records" in str(excinfo.value)
    limits.check_records(10, location="records")


def test_malformed_no_content() -> None:
    with pytest.raises(InvalidInputError) as excinfo:
        loads_limited('{"secret": tru}', location="x")
    assert "secret" not in str(excinfo.value)
