"""Error types, exit codes and diagnostics (Step 2)."""

from __future__ import annotations

import pytest

from driftlens.errors import (
    CODE_CLASS,
    EXIT_CODES,
    AnalysisFailure,
    Code,
    Diagnostic,
    DiagnosticSink,
    InvalidInputError,
    Severity,
    format_fatal,
    guarded,
)


def test_exit_codes_table() -> None:
    assert EXIT_CODES == {
        Code.INVALID_INPUT: 3,
        Code.RETRIEVER_UNAVAILABLE: 3,
        Code.NORMALIZATION_MISMATCH: 4,
        Code.IDENTITY_UNAVAILABLE: 4,
        Code.STRICT_COMPAT_VIOLATION: 4,
        Code.UNSUPPORTED_FORMAT: 5,
        Code.REFUSE_UNTRUSTED_PICKLE: 5,
        Code.INTERNAL_ERROR: 6,
        Code.UNDEFINED_CASE: 6,
        Code.DEPENDENCY_MISSING: 7,
        Code.LIMIT_EXCEEDED: 8,
    }


def test_class_table_complete() -> None:
    assert set(CODE_CLASS) == set(Code)
    assert CODE_CLASS[Code.RETRIEVER_UNAVAILABLE] is Severity.DEGRADED


def test_sink_order_and_merge() -> None:
    sink = DiagnosticSink()
    sink.add(Code.LOW_POWER, "low power")
    sink.add(Code.EMBEDDING_SPACE_INCOMPATIBLE, "different space")
    sink.add(Code.IDENTITY_WARNING, "shared message")
    sink.add(Code.IDENTITY_WARNING, "shared message")
    assert sink.items() == [
        Diagnostic(code=Code.IDENTITY_WARNING, message="shared message", count=2),
        Diagnostic(code=Code.LOW_POWER, message="low power", count=1),
        Diagnostic(
            code=Code.EMBEDDING_SPACE_INCOMPATIBLE,
            message="different space",
            count=1,
        ),
    ]


def test_format_fatal() -> None:
    error = InvalidInputError(
        Code.INVALID_INPUT, "boom", location="v2.jsonl line 3", hint="h"
    )
    assert (
        format_fatal([error])
        == "DRIFTLENS ERROR INVALID_INPUT (exit 3)\nv2.jsonl line 3: boom\nhint: h"
    )


def test_guarded_wraps() -> None:
    with pytest.raises(AnalysisFailure) as excinfo:
        with guarded("mmd"):
            raise ValueError("secret text")
    assert excinfo.value.exit_code == 6
    assert "secret text" not in str(excinfo.value)
