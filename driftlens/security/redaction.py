"""Deterministic, snapshot-scoped identifier redaction (Step 41).

Redaction replaces identifier-valued metric values with a short pseudonym
derived from the snapshot id and the raw value (sha256(snapshot_id + US +
value)[:12]).  The same raw value therefore maps to the same pseudonym within
one snapshot, and to a different pseudonym in another snapshot, so reports
cannot be joined on displayed identifiers or labels.

The module is pure: it never reads files or configuration.
"""

from __future__ import annotations

from dataclasses import replace

from driftlens.analysis.health import Metric
from driftlens.identity.hashing import sha256_hex

#: Length of the hex pseudonym prefix (PROJECT_ARCHITECTURE.md privacy rules).
_REDACT_LENGTH = 12

#: Unit separator, matching the hash-input convention used by sample_hash.
_UNIT_SEPARATOR = chr(31)


def _pseudonym(value: str, *, snapshot_id: str) -> str:
    joined = snapshot_id + _UNIT_SEPARATOR + value
    return sha256_hex(joined.encode("utf-8"))[:_REDACT_LENGTH]


def redact_id(value: str, *, snapshot_id: str) -> str:
    """Pseudonymize a record/document identifier within one snapshot."""

    return _pseudonym(value, snapshot_id=snapshot_id)


def redact_label(value: str, *, snapshot_id: str) -> str:
    """Pseudonymize a categorical label within one snapshot.

    The construction is identical to redact_id; the distinct name marks the
    semantic difference (an id and a label with the same text still share a
    pseudonym, which is intentional so pairs stay joinable in one report).
    """

    return _pseudonym(value, snapshot_id=snapshot_id)


def apply_redaction_to_metric(m: Metric, *, snapshot_id: str, redact: bool) -> Metric:
    """Return *m* with identifier values pseudonymized when *redact* is set.

    Non-identifier metrics are returned unchanged, and numeric values are never
    touched.  Identifier metrics redact v1/v2/change; the redactor is chosen by
    metric.identifier_kind so callers can name the semantic role.
    """

    if not redact or not m.is_identifier_valued:
        return m
    pseudonymize = redact_label if m.identifier_kind == "label" else redact_id

    def _one(value: float | str | None) -> float | str | None:
        if isinstance(value, str):
            return pseudonymize(value, snapshot_id=snapshot_id)
        return value

    return replace(
        m,
        v1=_one(m.v1),
        v2=_one(m.v2),
        change=_one(m.change),
    )
