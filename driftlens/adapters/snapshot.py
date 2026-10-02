"""Snapshot directories as compare inputs (Step 36)."""

from __future__ import annotations

from pathlib import Path

from driftlens.adapters.detect import FieldMapping
from driftlens.adapters.jsonl import LoadedInput
from driftlens.schema.snapshot import Snapshot, read_snapshot
from driftlens.security.limits import RuntimeLimits


def load_snapshot_as_input(
    snapshot_dir: Path,
    *,
    limits: RuntimeLimits,
    verify_full: bool = False,
) -> LoadedInput:
    """Load a written snapshot directory without rebuilding identity.

    The returned LoadedInput carries the prebuilt Snapshot on the transient
    _prebuilt_snapshot attribute so the compare pipeline can skip
    build_snapshot (and therefore assign_identity) entirely.
    """
    snapshot: Snapshot = read_snapshot(
        Path(snapshot_dir), limits=limits, verify_full=verify_full
    )
    loaded = LoadedInput(
        records=snapshot.records,
        field_mapping=FieldMapping(),
        file_descriptors={},
        source_type="snapshot",
    )
    loaded._prebuilt_snapshot = snapshot
    return loaded
