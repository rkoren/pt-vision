"""Sit-to-stand events from force plates, the reference for our video-derived events.

The lab did not put the seat on its own plate (OpenCap's third plate reads zero), so seat-off
comes from the plates under the feet: the instant their summed
vertical force first exceeds body weight during a rise. The chair is fully unloaded by then, so
this lags the physical seat-off by a few tens of milliseconds; rep-to-rep intervals and the
first-to-last seat-off span cancel that offset. A rise is told apart from the braking peak of a
sit-down (which also exceeds body weight) by the lab's own pelvis trajectory: the pelvis must be
moving up. The force stays above 90 % of body weight for 0.25 s afterwards (a standing phase),
which rejects chair-landing spikes, and only the first crossing after the force was last below
half body weight counts, which rejects the wobble of an unsteady stand.

Signals are vertical force magnitudes in newtons, positive = load, on a time axis in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptvision.datasets.mocap_projection import ProjectedTrial

G = 9.80665


@dataclass
class ForceStsEvents:
    rule: str
    body_weight_n: float
    seat_off_s: list[float]
    seated_level_n: float | None = None

    @property
    def n_reps(self) -> int:
        return len(self.seat_off_s)

    @property
    def span_s(self) -> float | None:
        """First to last seat-off; None with fewer than two reps."""
        if len(self.seat_off_s) < 2:
            return None
        return self.seat_off_s[-1] - self.seat_off_s[0]


def pelvis_rising(projected: ProjectedTrial, t: np.ndarray, *, min_m_s: float = 0.05) -> np.ndarray:
    """(len(t),) bool: the lab pelvis moves up faster than `min_m_s` at each force sample."""
    hip = projected.track.keypoint("Hip")[:, 1]  # image y, down
    height_m = -hip / projected.px_per_m
    fps = projected.fps
    t_src = np.arange(len(height_m)) / fps
    ok = np.isfinite(height_m)
    if ok.sum() < 3:
        return np.ones_like(t, dtype=bool)
    h = np.interp(t_src, t_src[ok], height_m[ok])
    k = max(1, round(0.1 * fps))
    v = np.gradient(np.convolve(h, np.ones(k) / k, mode="same"), 1.0 / fps)
    return np.asarray(np.interp(t, t_src, v) > min_m_s, dtype=bool)


def events_from_feet_plates(
    t: np.ndarray,
    feet_fz: np.ndarray,
    body_weight_n: float,
    *,
    rising: np.ndarray | None = None,
    hold_frac: float = 0.9,
    hold_s: float = 0.25,
    min_rep_s: float = 0.6,
    rearm_frac: float = 0.5,
) -> ForceStsEvents:
    feet = np.abs(np.asarray(feet_fz, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    hold_n = max(1, round(hold_s / dt))
    seated_level = float(np.percentile(feet, 10))
    above = feet >= body_weight_n
    crossings = np.flatnonzero(~above[:-1] & above[1:]) + 1
    seat_off: list[float] = []
    last = -np.inf
    prev = 0
    for i in crossings:
        if not np.any(feet[prev:i] < rearm_frac * body_weight_n):
            continue  # the subject never sat back down since the last counted rise
        if rising is not None and not rising[i]:
            continue
        window = feet[i : i + hold_n]
        if window.size < hold_n or np.any(window < hold_frac * body_weight_n):
            continue
        if t[i] - last < min_rep_s:
            continue
        seat_off.append(float(t[i]))
        last = t[i]
        prev = int(i)
    return ForceStsEvents("feet_body_weight", body_weight_n, seat_off, seated_level_n=seated_level)


def match_reps(
    ours_s: list[float], ref_s: list[float], *, max_gap_s: float = 1.0
) -> list[tuple[int, int, float]]:
    """Greedy nearest-time matching of our seat-off instants to the reference ones.
    Returns (our index, reference index, ours − reference in seconds)."""
    pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(ours_s):
        for j, b in enumerate(ref_s):
            if abs(a - b) <= max_gap_s:
                pairs.append((abs(a - b), i, j))
    pairs.sort()
    used_i: set[int] = set()
    used_j: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for _, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j, ours_s[i] - ref_s[j]))
    out.sort()
    return out
