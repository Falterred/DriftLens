"""Runtime resource limits and strict JSON parsing."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable

from driftlens.config.schema import LimitsCfg
from driftlens.errors import InvalidInputError, LimitExceededError

try:  # pragma: no cover - the resource module is Unix only
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

MAX_JSON_DEPTH = 64


def detect_available_memory() -> int | None:
    """Best-effort physical memory available to this process, in bytes."""

    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    fields = line.split()
                    return int(fields[1]) * 1024
    except OSError:
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
    return int(page_size) * int(available_pages)


def _current_rss_bytes() -> int | None:
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def _scan_depth(text: str, location: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise LimitExceededError(
                    f"JSON depth {depth} exceeds the fixed maximum of {MAX_JSON_DEPTH} at {location}",
                    flag="none (fixed limit)",
                    location=location,
                    hint="no override",
                )
        elif char in "]}":
            depth -= 1


def loads_limited(text: str, *, location: str) -> Any:
    """Parse strict JSON after rejecting over-deep documents and duplicate keys."""

    _scan_depth(text, location)

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise InvalidInputError("duplicate key", location=location)
            seen.add(key)
        return dict(pairs)

    try:
        return json.loads(text, object_pairs_hook=hook)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(
            f"malformed JSON at column {exc.colno}", location=location
        ) from None


class RuntimeLimits:
    """Enforces the configured CLI-overridable resource limits."""

    def __init__(self, cfg: LimitsCfg, *, available_bytes: int | None = None) -> None:
        self._cfg = cfg
        if available_bytes is None:
            available_bytes = detect_available_memory()
        self._available_bytes = available_bytes
        self._input_bytes = 0

    def _check(self, value: int, limit: int, label: str, location: str, flag: str) -> None:
        if value > limit:
            raise LimitExceededError(
                f"{label} {value} exceeds the limit of {limit} at {location}",
                flag=flag,
                location=location,
            )

    def check_records(self, n: int, *, location: str) -> None:
        self._check(n, self._cfg.max_records, "records", location, "--max-records")

    def check_dimension(self, d: int, *, location: str) -> None:
        self._check(d, self._cfg.max_dimension, "dimension", location, "--max-dimension")

    def check_text_length(self, n_chars: int, *, location: str) -> None:
        self._check(
            n_chars,
            self._cfg.max_text_length,
            "text length",
            location,
            "--max-text-length",
        )

    def check_sample_size(self, n: int) -> None:
        self._check(
            n, self._cfg.max_sample_size, "sample size", "sample", "--max-sample-size"
        )

    def add_input_bytes(self, n: int, *, location: str) -> None:
        self._input_bytes += n
        self._check(
            self._input_bytes,
            self._cfg.max_input_bytes,
            "input bytes",
            location,
            "--max-input-bytes",
        )

    def check_memory(self, stage: str) -> None:
        available = self._available_bytes
        if available is None:
            return
        rss = _current_rss_bytes()
        if rss is None:
            return
        if rss > available * self._cfg.max_memory:
            raise LimitExceededError(
                f"memory {rss} exceeds {self._cfg.max_memory} of available {available} at {stage}",
                flag="--max-memory",
                location=stage,
            )

    def applied(self) -> dict[str, int | float | str]:
        result: dict[str, int | float | str] = {
            "max_records": self._cfg.max_records,
            "max_dimension": self._cfg.max_dimension,
            "max_text_length": self._cfg.max_text_length,
            "max_sample_size": self._cfg.max_sample_size,
            "max_input_bytes": self._cfg.max_input_bytes,
            "max_memory": self._cfg.max_memory,
        }
        if self._available_bytes is None:
            result["max_memory"] = "unenforced"
        return result
