"""Shingle sets, stable shingle keys, and set-similarity helpers (Step 10)."""

from __future__ import annotations

from collections.abc import Sequence


def shingles(
    tokens: Sequence[str], n: int, *, casefold: bool = True
) -> frozenset[tuple[str, ...]]:
    """n-gram shingle set over tokens.

    Falls back to unigrams (n=1) when there are fewer than n tokens and n > 1.
    Returns an empty frozenset when there are no tokens, regardless of n.
    """

    if len(tokens) == 0:
        return frozenset()
    if casefold:
        working = [token.casefold() for token in tokens]
    else:
        working = list(tokens)
    n_eff = n if len(working) >= n else 1
    return frozenset(
        tuple(working[i : i + n_eff]) for i in range(len(working) - n_eff + 1)
    )


def shingle_key(shingle: tuple[str, ...]) -> bytes:
    """Canonical, PYTHONHASHSEED-independent byte encoding of a shingle."""

    return b"\t".join(token.encode("utf-8") for token in shingle)


def jaccard(a: frozenset, b: frozenset) -> float:
    """Set Jaccard similarity; two empty sets are identical (1.0)."""

    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def coverage(a: frozenset, group_union: frozenset) -> float:
    """Fraction of a covered by group_union; 0.0 when a has no shingles."""

    if not a:
        return 0.0
    return len(a & group_union) / len(a)
