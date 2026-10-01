"""Shingles (Step 10)."""

from __future__ import annotations

import pytest

from driftlens.text.shingles import coverage, jaccard, shingle_key, shingles


def test_shingles_basic_4gram() -> None:
    assert shingles(["a", "b", "c", "d", "e"], 4) == {
        ("a", "b", "c", "d"),
        ("b", "c", "d", "e"),
    }


def test_shingles_underflow_fallback() -> None:
    assert shingles(["a", "b"], 4) == {("a",), ("b",)}


def test_shingles_empty() -> None:
    assert shingles([], 5) == frozenset()


def test_casefold_applied() -> None:
    assert shingles(["Apple", "Pie"], 2, casefold=True) == shingles(
        ["apple", "pie"], 2, casefold=True
    )
    assert shingles(["Apple", "Pie"], 2, casefold=True) != shingles(
        ["Apple", "Pie"], 2, casefold=False
    )


def test_shingle_key_deterministic() -> None:
    # Literal comparison: the encoding contains no randomization.
    assert shingle_key(("a", "b")) == b"a\tb"
    assert shingle_key(("a", "b")) == shingle_key(("a", "b"))


def test_jaccard() -> None:
    assert jaccard(frozenset({(1, 2), (3, 4)}), frozenset({(3, 4), (5, 6)})) == pytest.approx(
        1 / 3
    )


def test_jaccard_both_empty() -> None:
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_coverage() -> None:
    a = frozenset({(1,), (2,), (3,)})
    group = frozenset({(2,), (3,), (9,)})
    assert coverage(a, group) == pytest.approx(2 / 3)


def test_coverage_empty_a() -> None:
    assert coverage(frozenset(), frozenset({(1,)})) == 0.0
