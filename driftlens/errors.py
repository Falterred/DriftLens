"""Error types, exit codes and diagnostics for DriftLens.

Fatal conditions are exceptions carrying an exit code. Warnings and degraded
conditions are collected in a shared :class:`DiagnosticSink`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar


class Severity(str, Enum):
    """Severity class of an error or diagnostic code."""

    FATAL = "FATAL"
    DEGRADED = "DEGRADED"
    WARNING = "WARNING"


class Code(str, Enum):
    """Every error and diagnostic code, in specification order."""

    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    REFUSE_UNTRUSTED_PICKLE = "REFUSE_UNTRUSTED_PICKLE"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    NORMALIZATION_MISMATCH = "NORMALIZATION_MISMATCH"
    IDENTITY_UNAVAILABLE = "IDENTITY_UNAVAILABLE"
    STRICT_COMPAT_VIOLATION = "STRICT_COMPAT_VIOLATION"
    RETRIEVER_UNAVAILABLE = "RETRIEVER_UNAVAILABLE"
    EMBEDDING_SPACE_INCOMPATIBLE = "EMBEDDING_SPACE_INCOMPATIBLE"
    EMBEDDINGS_UNAVAILABLE = "EMBEDDINGS_UNAVAILABLE"
    METRIC_MISMATCH = "METRIC_MISMATCH"
    SKIPPED_MODEL_CHANGED = "SKIPPED_MODEL_CHANGED"
    SKIPPED_SPACE_CHANGED = "SKIPPED_SPACE_CHANGED"
    SKIPPED_PARTIAL_MIGRATION = "SKIPPED_PARTIAL_MIGRATION"
    SKIPPED_MIXED_MODELS = "SKIPPED_MIXED_MODELS"
    GRANULARITY_MISMATCH = "GRANULARITY_MISMATCH"
    BASELINE_INCOMPATIBLE = "BASELINE_INCOMPATIBLE"
    CLUSTER_EXTRA_MISSING = "CLUSTER_EXTRA_MISSING"
    METRIC_UNKNOWN = "METRIC_UNKNOWN"
    SIGNIFICANCE_SKIPPED = "SIGNIFICANCE_SKIPPED"
    SPACE_UNTESTABLE = "SPACE_UNTESTABLE"
    EMPTY_SAMPLE = "EMPTY_SAMPLE"
    EMPTY_SNAPSHOT = "EMPTY_SNAPSHOT"
    DEGENERATE_KERNEL = "DEGENERATE_KERNEL"
    CONSTANT_SHIFT = "CONSTANT_SHIFT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    IDENTITY_WARNING = "IDENTITY_WARNING"
    MODEL_UNKNOWN = "MODEL_UNKNOWN"
    MODEL_CHANGED = "MODEL_CHANGED"
    SPACE_CHANGED = "SPACE_CHANGED"
    PARTIAL_MIGRATION = "PARTIAL_MIGRATION"
    MIXED_MODELS = "MIXED_MODELS"
    LABEL_CONFLICT = "LABEL_CONFLICT"
    NORMALIZATION_CLAIM_INVALID = "NORMALIZATION_CLAIM_INVALID"
    LOW_POWER = "LOW_POWER"
    DOC_IDENTITY_MISMATCH = "DOC_IDENTITY_MISMATCH"
    GRANULARITY_ASSUMED = "GRANULARITY_ASSUMED"
    MODEL_OVERRIDE_IGNORED = "MODEL_OVERRIDE_IGNORED"
    CHUNK_IDENTITY_MISMATCH = "CHUNK_IDENTITY_MISMATCH"
    DOC_ORDER_UNKNOWN = "DOC_ORDER_UNKNOWN"
    EMBEDDING_INVALID = "EMBEDDING_INVALID"
    ZERO_VECTOR = "ZERO_VECTOR"
    QUANTIZED = "QUANTIZED"
    FUTURE_TIMESTAMP = "FUTURE_TIMESTAMP"
    NO_QUERY_VECTOR = "NO_QUERY_VECTOR"
    RENORMALIZED = "RENORMALIZED"
    PICKLE_IGNORED = "PICKLE_IGNORED"
    FIELD_CANDIDATES_IGNORED = "FIELD_CANDIDATES_IGNORED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNDEFINED_CASE = "UNDEFINED_CASE"


_FATAL_CODES = (
    Code.INVALID_INPUT,
    Code.UNSUPPORTED_FORMAT,
    Code.REFUSE_UNTRUSTED_PICKLE,
    Code.DEPENDENCY_MISSING,
    Code.LIMIT_EXCEEDED,
    Code.NORMALIZATION_MISMATCH,
    Code.IDENTITY_UNAVAILABLE,
    Code.STRICT_COMPAT_VIOLATION,
    Code.INTERNAL_ERROR,
    Code.UNDEFINED_CASE,
)

_DEGRADED_CODES = (
    Code.RETRIEVER_UNAVAILABLE,
    Code.EMBEDDING_SPACE_INCOMPATIBLE,
    Code.EMBEDDINGS_UNAVAILABLE,
    Code.METRIC_MISMATCH,
    Code.SKIPPED_MODEL_CHANGED,
    Code.SKIPPED_SPACE_CHANGED,
    Code.SKIPPED_PARTIAL_MIGRATION,
    Code.SKIPPED_MIXED_MODELS,
    Code.GRANULARITY_MISMATCH,
    Code.BASELINE_INCOMPATIBLE,
    Code.CLUSTER_EXTRA_MISSING,
    Code.METRIC_UNKNOWN,
    Code.SIGNIFICANCE_SKIPPED,
    Code.SPACE_UNTESTABLE,
    Code.EMPTY_SAMPLE,
    Code.EMPTY_SNAPSHOT,
    Code.DEGENERATE_KERNEL,
    Code.CONSTANT_SHIFT,
    Code.INSUFFICIENT_DATA,
)

_WARNING_CODES = (
    Code.IDENTITY_WARNING,
    Code.MODEL_UNKNOWN,
    Code.MODEL_CHANGED,
    Code.SPACE_CHANGED,
    Code.PARTIAL_MIGRATION,
    Code.MIXED_MODELS,
    Code.LABEL_CONFLICT,
    Code.NORMALIZATION_CLAIM_INVALID,
    Code.LOW_POWER,
    Code.DOC_IDENTITY_MISMATCH,
    Code.GRANULARITY_ASSUMED,
    Code.MODEL_OVERRIDE_IGNORED,
    Code.CHUNK_IDENTITY_MISMATCH,
    Code.DOC_ORDER_UNKNOWN,
    Code.EMBEDDING_INVALID,
    Code.ZERO_VECTOR,
    Code.QUANTIZED,
    Code.FUTURE_TIMESTAMP,
    Code.NO_QUERY_VECTOR,
    Code.RENORMALIZED,
    Code.PICKLE_IGNORED,
    Code.FIELD_CANDIDATES_IGNORED,
)

CODE_CLASS: dict[Code, Severity] = {
    **{code: Severity.FATAL for code in _FATAL_CODES},
    **{code: Severity.DEGRADED for code in _DEGRADED_CODES},
    **{code: Severity.WARNING for code in _WARNING_CODES},
}

EXIT_CODES: dict[Code, int] = {
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

_EXIT_4_CODES = (
    Code.NORMALIZATION_MISMATCH,
    Code.IDENTITY_UNAVAILABLE,
    Code.STRICT_COMPAT_VIOLATION,
)


class DriftLensError(Exception):
    """A fatal condition that carries the process exit code."""

    def __init__(
        self,
        code: Code,
        message: str,
        *,
        location: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location
        self.hint = hint

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]

    def __str__(self) -> str:
        if self.location is None:
            return self.message
        return f"{self.location}: {self.message}"


class _FixedCodeError(DriftLensError):
    """Base for the fatal errors whose code is fixed by the subclass."""

    fixed_code: ClassVar[Code]

    def __init__(
        self,
        *args: object,
        message: str | None = None,
        location: str | None = None,
        hint: str | None = None,
    ) -> None:
        remaining = list(args)
        if remaining and isinstance(remaining[0], Code):
            remaining.pop(0)
        if remaining:
            if message is not None:
                raise TypeError(f"{type(self).__name__} received two messages")
            first = remaining.pop(0)
            if not isinstance(first, str):
                raise TypeError(f"{type(self).__name__} message must be a string")
            message = first
        if remaining:
            raise TypeError(f"{type(self).__name__} takes at most two positional arguments")
        super().__init__(
            type(self).fixed_code,
            "" if message is None else message,
            location=location,
            hint=hint,
        )


class InvalidInputError(_FixedCodeError):
    """Invalid input data or an invalid request."""

    fixed_code = Code.INVALID_INPUT


class UnsupportedFormatError(_FixedCodeError):
    """The input format cannot be read."""

    fixed_code = Code.UNSUPPORTED_FORMAT


class RefuseUntrustedPickleError(_FixedCodeError):
    """A pickle file was found and is refused."""

    fixed_code = Code.REFUSE_UNTRUSTED_PICKLE


class DependencyMissingError(_FixedCodeError):
    """An optional dependency is required but not installed."""

    fixed_code = Code.DEPENDENCY_MISSING


class LimitExceededError(_FixedCodeError):
    """A configured or fixed resource limit was exceeded."""

    fixed_code = Code.LIMIT_EXCEEDED

    def __init__(
        self,
        message: str = "limit exceeded",
        *,
        flag: str,
        location: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(f"{message} (override with {flag})", location=location, hint=hint)


class IncompatibleError(DriftLensError):
    """The two inputs are incompatible for the requested analysis."""

    def __init__(
        self,
        code: Code,
        message: str = "",
        *,
        location: str | None = None,
        hint: str | None = None,
    ) -> None:
        assert code in _EXIT_4_CODES, f"IncompatibleError does not accept {code.value}"
        super().__init__(code, message, location=location, hint=hint)


class RetrieverUnavailableError(_FixedCodeError):
    """A retriever is required but unavailable (fatal in single mode)."""

    fixed_code = Code.RETRIEVER_UNAVAILABLE


class UndefinedCaseError(_FixedCodeError):
    """A case the specification does not define was reached."""

    fixed_code = Code.UNDEFINED_CASE

    def __init__(
        self,
        case: str = "",
        *,
        location: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(f"undefined case: {case}", location=location, hint=hint)


class AnalysisFailure(_FixedCodeError):
    """An unexpected failure inside one analysis stage."""

    fixed_code = Code.INTERNAL_ERROR

    def __init__(
        self,
        analysis: str,
        *,
        location: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(f"analysis failed: {analysis}", location=location, hint=hint)


class FatalErrors(Exception):
    """One or more fatal conditions reached during a single pipeline run."""

    def __init__(self, errors: Sequence[DriftLensError]) -> None:
        self.errors: list[DriftLensError] = list(errors)
        if not self.errors:
            raise ValueError("FatalErrors requires at least one error")
        super().__init__(f"{len(self.errors)} fatal error(s)")

    @property
    def exit_code(self) -> int:
        return self.errors[0].exit_code


@dataclass(frozen=True)
class Diagnostic:
    """A warning or degraded condition, with how often it was seen."""

    code: Code
    message: str
    count: int = 1
    dimension: str | None = None


_SEVERITY_ORDER = {Severity.WARNING: 0, Severity.DEGRADED: 1}


class DiagnosticSink:
    """Collects warning and degraded diagnostics, merged by identity."""

    def __init__(self) -> None:
        self._items: list[Diagnostic] = []

    def add(
        self,
        code: Code,
        message: str,
        count: int = 1,
        dimension: str | None = None,
    ) -> None:
        severity = CODE_CLASS[code]
        if severity is not Severity.WARNING and severity is not Severity.DEGRADED:
            raise UndefinedCaseError(
                f"diagnostic code {code.value} is {severity.value}, not WARNING or DEGRADED"
            )
        for index, existing in enumerate(self._items):
            if (
                existing.code == code
                and existing.message == message
                and existing.dimension == dimension
            ):
                self._items[index] = replace(existing, count=existing.count + count)
                return
        self._items.append(
            Diagnostic(code=code, message=message, count=count, dimension=dimension)
        )

    def extend(self, items: Iterable[Diagnostic]) -> None:
        for item in items:
            self.add(item.code, item.message, item.count, item.dimension)

    def items(self) -> list[Diagnostic]:
        return sorted(
            self._items,
            key=lambda item: (
                _SEVERITY_ORDER[CODE_CLASS[item.code]],
                item.code.value,
                item.message,
            ),
        )

    def has(self, code: Code) -> bool:
        return any(item.code == code for item in self._items)


def format_fatal(errors: Sequence[DriftLensError]) -> str:
    """Render fatal errors for stderr, one block per error, no stack trace."""

    blocks: list[str] = []
    for error in errors:
        lines = [f"DRIFTLENS ERROR {error.code.value} (exit {error.exit_code})"]
        if error.location is None:
            lines.append(error.message)
        else:
            lines.append(f"{error.location}: {error.message}")
        if error.hint is not None:
            lines.append(f"hint: {error.hint}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


@contextmanager
def guarded(analysis: str) -> Iterator[None]:
    """Turn an unexpected exception into an :class:`AnalysisFailure`."""

    try:
        yield
    except DriftLensError:
        raise
    except Exception:
        raise AnalysisFailure(analysis) from None
