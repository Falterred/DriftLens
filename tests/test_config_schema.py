"""Configuration schema defaults and validation (Step 3)."""

from __future__ import annotations

import pytest

from driftlens.config.schema import (
    THRESHOLD_KEYS,
    default_config,
    validate_config,
)
from driftlens.errors import InvalidInputError, LimitExceededError


def test_defaults_match_spec() -> None:
    cfg = default_config()
    assert cfg.analysis.seed == 42
    assert cfg.embeddings.consistency_max_anchors == 5000
    assert cfg.limits.max_input_bytes == 68719476736
    assert cfg.ci.fail_on == {
        "embedding": "critical",
        "retrieval": "critical",
        "duplication": "warning",
    }
    assert set(cfg.drift) == set(THRESHOLD_KEYS)
    assert all(value is None for value in cfg.drift.values())


def test_primary_metric_validation() -> None:
    cfg = default_config()
    cfg.retrieval.k = [1, 5, 10]
    cfg.retrieval.primary_metric = "recall@3"
    with pytest.raises(InvalidInputError):
        validate_config(cfg)

    cfg.retrieval.primary_metric = "mrr"
    validate_config(cfg)

    cfg.retrieval.primary_metric = "hit_rate@10"
    validate_config(cfg)


def test_workers_ceiling() -> None:
    cfg = default_config()
    cfg.analysis.workers = 99
    cfg.limits.max_workers = "auto"
    with pytest.raises(LimitExceededError) as excinfo:
        validate_config(cfg)
    assert excinfo.value.exit_code == 8
