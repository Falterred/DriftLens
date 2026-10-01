"""Shared pytest fixtures for the DriftLens test suite."""

from __future__ import annotations

import random
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from driftlens.security.network_guard import network_guard

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _seed() -> Iterator[None]:
    """Seed the global random generators so every test is deterministic."""
    random.seed(42)
    np.random.seed(42)
    yield


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic NumPy generator."""
    return np.random.default_rng(42)


@pytest.fixture
def no_network() -> Iterator[None]:
    """Block outbound network access for the duration of a test."""
    with network_guard():
        yield
