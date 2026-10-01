"""Distribution distances and histogram binning (Step 24).

Every public function returns a (value, reason) pair where reason is a Code
string exactly when value is None, so analysis.health.make_metric can consume
the result directly.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import stats

from driftlens.analysis.health import round9

FIXED_BIN_EDGES: list[float] = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    math.inf,
]


def ks_statistic(
    v1: Sequence[float], v2: Sequence[float]
) -> tuple[float | None, str | None]:
    """Two-sample Kolmogorov-Smirnov D statistic."""
    if len(v1) == 0 or len(v2) == 0:
        return (None, "EMPTY_SAMPLE")
    result = stats.ks_2samp(list(v1), list(v2), method="asymp")
    return (round9(float(result.statistic)), None)


def wasserstein1_normalized(
    v1: Sequence[float], v2: Sequence[float]
) -> tuple[float | None, str | None]:
    """Wasserstein-1 divided by v1's standard deviation (pooled when zero)."""
    if len(v1) == 0 or len(v2) == 0:
        return (None, "EMPTY_SAMPLE")

    raw = float(stats.wasserstein_distance(list(v1), list(v2)))
    a1 = np.asarray(v1, dtype=np.float64)
    a2 = np.asarray(v2, dtype=np.float64)

    std1 = float(np.std(a1, ddof=0))
    if std1 == 0.0:
        var1 = float(np.var(a1, ddof=0))
        var2 = float(np.var(a2, ddof=0))
        n1 = int(a1.shape[0])
        n2 = int(a2.shape[0])
        pooled_std = math.sqrt((n1 * var1 + n2 * var2) / (n1 + n2))
        if pooled_std == 0.0:
            if float(a1[0]) == float(a2[0]):
                return (0.0, None)
            return (None, "CONSTANT_SHIFT")
        return (round9(raw / pooled_std), None)
    return (round9(raw / std1), None)


def _kl(a: np.ndarray, b: np.ndarray) -> float:
    total = 0.0
    for ai, bi in zip(a, b):
        if ai > 0:
            total += float(ai) * math.log2(float(ai) / float(bi))
    return total


def js_divergence_categorical(
    counts1: dict[str, int], counts2: dict[str, int]
) -> tuple[float | None, str | None]:
    """Jensen-Shannon divergence (base 2) over the union of category keys."""
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    if total1 == 0 and total2 == 0:
        return (None, "EMPTY_SAMPLE")
    if total1 == 0 or total2 == 0:
        return (1.0, None)

    keys = sorted(set(counts1) | set(counts2))
    p = np.asarray([counts1.get(key, 0) / total1 for key in keys], dtype=np.float64)
    q = np.asarray([counts2.get(key, 0) / total2 for key in keys], dtype=np.float64)
    m = (p + q) / 2
    divergence = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    clipped = min(max(divergence, 0.0), 1.0)
    return (round9(clipped), None)


def js_divergence_histogram(
    v1: Sequence[float],
    v2: Sequence[float],
    *,
    edges: Sequence[float],
) -> tuple[float | None, str | None]:
    """JS divergence between two numeric samples binned on the given edges."""
    boundaries = edges[1:-1]
    idx1 = np.digitize(np.asarray(v1, dtype=np.float64), boundaries)
    idx2 = np.digitize(np.asarray(v2, dtype=np.float64), boundaries)
    counts1: Counter[str] = Counter(str(int(index)) for index in idx1)
    counts2: Counter[str] = Counter(str(int(index)) for index in idx2)
    return js_divergence_categorical(counts1, counts2)


def numeric_js_bin_edges(
    pooled: Sequence[float], *, max_bins: int = 100
) -> list[float]:
    """Freedman-Diaconis bin edges, with Sturges and one-bin fallbacks."""
    if len(pooled) == 0:
        return [0.0, 1.0]

    arr = np.asarray(pooled, dtype=np.float64)
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    low = float(arr.min())
    high = float(arr.max())

    if iqr == 0.0:
        if bool(np.all(arr == arr[0])):
            return [low, low + 1e-9]
        n = int(arr.shape[0])
        n_bins = min(max_bins, math.ceil(math.log2(n) + 1))
        return [float(edge) for edge in np.linspace(low, high, n_bins + 1).tolist()]

    fd_width = 2 * iqr / (len(arr) ** (1 / 3))
    if fd_width <= 0:
        n_bins = 1
    else:
        n_bins = min(
            max_bins, max(1, math.ceil((high - low) / fd_width))
        )
    return [float(edge) for edge in np.linspace(low, high, n_bins + 1).tolist()]


def mean_median_variance_quantiles(
    v: Sequence[float],
) -> dict[str, float] | None:
    """Descriptive summary of a value set, or None when it is empty."""
    if len(v) == 0:
        return None
    arr = np.asarray(v, dtype=np.float64)
    q5, q25, q50, q75, q95 = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "mean": round9(float(np.mean(arr))),
        "median": round9(float(np.median(arr))),
        "variance": round9(float(np.var(arr, ddof=0))),
        "q5": round9(float(q5)),
        "q25": round9(float(q25)),
        "q50": round9(float(q50)),
        "q75": round9(float(q75)),
        "q95": round9(float(q95)),
    }
