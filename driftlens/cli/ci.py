"""CI-mode exit-code policy (Step 49, section 13.5).

The exit code is an orchestration outcome layered on a finished HealthReport:
it creates no new Metrics and only decides, per dimension, how much a status
contributes to the process exit code under the configured gate policy.
"""

from __future__ import annotations

from typing import Literal

from driftlens.analysis.health import Dimension, HealthReport
from driftlens.config.schema import DIMENSIONS, CiCfg

Level = Literal["warn", "fail", "ignore"]

#: Skip reasons caused by an embedding-space or model incompatibility rather
#: than by missing data. A SKIPPED dimension whose reason is in this set is
#: governed by ci.on_incompatible instead of ci.on_skipped.
COMPAT_SKIP_REASONS = frozenset(
    {
        "SKIPPED_MODEL_CHANGED",
        "SKIPPED_SPACE_CHANGED",
        "SKIPPED_PARTIAL_MIGRATION",
        "SKIPPED_MIXED_MODELS",
        "EMBEDDING_SPACE_INCOMPATIBLE",
        "METRIC_MISMATCH",
    }
)

_LEVEL_CONTRIBUTIONS: dict[str, int] = {"ignore": 0, "warn": 1, "fail": 2}


def _level_to_contribution(level: Level) -> int:
    """Map a CI policy level to its exit-code contribution."""
    return _LEVEL_CONTRIBUTIONS[level]


def dimension_gate_contribution(
    dim: Dimension,
    *,
    gate_level: Literal["warning", "critical"] | None,
    on_skipped: Level,
    on_incompatible: Level,
) -> int:
    """How much one dimension contributes to the CI exit code.

    An ungated dimension (gate_level is None) never contributes. For a gated
    dimension the contribution follows section 13.5: FAIL is 2 under any gate,
    WARN is 1 only under a warning gate, a SKIPPED dimension is governed by the
    on_incompatible/on_skipped policy, and every other status is 0.
    """
    if gate_level is None:
        return 0
    if dim.status == "SKIPPED":
        compat_caused = any(
            metric.reason in COMPAT_SKIP_REASONS
            for metric in dim.metrics
            if metric.status == "SKIPPED"
        )
        setting = on_incompatible if compat_caused else on_skipped
        return _level_to_contribution(setting)
    if dim.status == "WARN":
        return 1 if gate_level == "warning" else 0
    if dim.status == "FAIL":
        return 2
    return 0


def compute_ci_exit_code(health: HealthReport, *, cfg: CiCfg) -> int:
    """Return 0, 1 or 2: the maximum contribution over the gated dimensions."""
    contributions = [
        dimension_gate_contribution(
            health.dimensions[dim_name],
            gate_level=cfg.fail_on.get(dim_name),  # type: ignore[arg-type]
            on_skipped=cfg.on_skipped,  # type: ignore[arg-type]
            on_incompatible=cfg.on_incompatible,  # type: ignore[arg-type]
        )
        for dim_name in DIMENSIONS
    ]
    return max(contributions) if contributions else 0
