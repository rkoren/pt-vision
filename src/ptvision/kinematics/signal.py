"""Small, NaN-aware 1-D signal helpers shared by segmenters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def derivative(x: np.ndarray, fs: float) -> np.ndarray:
    """Central-difference derivative (units per second). NaNs propagate to their neighbours."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        return np.zeros_like(x)
    return np.asarray(np.gradient(x) * fs, dtype=np.float64)


def robust_normalize(x: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 95.0) -> np.ndarray:
    """Scale so the lo/hi percentiles map to 0/1. Robust to a few outliers and NaNs."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanpercentile(x, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.asarray((x - lo) / (hi - lo), dtype=np.float64)


def walk_until(
    v: np.ndarray,
    start: int,
    step: int,
    predicate: Callable[[float], bool],
    stop: int | None = None,
) -> int:
    """Walk from `start` in direction `step` (+1/-1); return the first index satisfying `predicate`.

    If none does before `stop` (exclusive) or the array edge, return the last index visited."""
    n = len(v)
    if stop is None:
        stop = n if step > 0 else -1
    i = start
    last = start
    while (step > 0 and i < stop and i < n) or (step < 0 and i > stop and i >= 0):
        val = v[i]
        if not np.isnan(val) and predicate(float(val)):
            return i
        last = i
        i += step
    return last


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs for each run of True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def nan_runs(x: np.ndarray) -> list[tuple[int, int]]:
    return runs_of(np.isnan(np.asarray(x, dtype=np.float64)))
