"""Sit-to-stand phase segmentation from the vertical pelvis trajectory.

Adapted from opencap-processing `ActivityAnalyses/sts_analysis.py::segment_sts` (Apache-2.0,
Stanford University), which works on 3D pelvis height in metres. Here the signal is the image-space
pelvis height of a single person, low-passed and robustly normalised so 0 ~ seated and 1 ~ standing,
which makes the velocity thresholds unit-free.

Per repetition:
  seat_off       velocity rises above `v_seated` before the rise-velocity peak
  stand_reached  velocity falls below `v_standing` after the rise-velocity peak
  descent_start  velocity falls below -`v_standing` before the descent-velocity trough
  seated_return  velocity rises above -`v_seated` after the descent-velocity trough
  lean_onset     (optional) trunk forward angular velocity exceeds `lean_velocity_deg_s`
                 before seat_off
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from ptvision.clinical.registry import register_segmenter
from ptvision.kinematics.preprocess import butterworth, interpolate_gaps
from ptvision.kinematics.signal import derivative, robust_normalize, walk_until


@dataclass(frozen=True)
class StsParams:
    n_reps_expected: int = 5  # -1 = unknown (count within window)
    start_rule: str = "first_seat_off"  # first_seat_off | lean_onset | manual
    end_rule: str = "fifth_stand"  # fifth_stand | fifth_sit | window
    window_s: float = 30.0
    partial_rule: str = "count_if_past_halfway"
    v_seated: float = 0.25
    v_standing: float = 0.25
    peak_height_min: float = 0.7
    peak_prominence_min: float = 0.5
    min_rep_s: float = 0.6
    lean_velocity_deg_s: float = 10.0
    segment_signal_cutoff_hz: float = 3.0
    manual_start_s: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StsParams:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StsRep:
    index: int
    seat_off: int
    peak: int
    stand_reached: int
    descent_start: int | None
    seated_return: int | None
    lean_onset: int | None
    rise_height: float  # normalised units

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StsEvents:
    reps: list[StsRep]
    test_start: int | None
    test_end: int | None
    fps: float
    height_norm: np.ndarray  # (T,)
    velocity: np.ndarray  # (T,) normalised units / s
    params: StsParams
    warnings: list[str] = field(default_factory=list)
    n_in_window: int | None = None

    @property
    def n_reps(self) -> int:
        return len(self.reps)

    @property
    def total_time_s(self) -> float | None:
        if self.test_start is None or self.test_end is None:
            return None
        return (self.test_end - self.test_start) / self.fps

    def to_dict(self) -> dict[str, Any]:
        return {
            "segmenter": "sts_velocity_threshold",
            "fps": self.fps,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "total_time_s": self.total_time_s,
            "n_reps": self.n_reps,
            "n_in_window": self.n_in_window,
            "reps": [r.to_dict() for r in self.reps],
            "warnings": list(self.warnings),
            "params": self.params.to_dict(),
        }


def pelvis_height_signal(
    coords: np.ndarray, layout: Any, fps: float, cutoff_hz: float
) -> np.ndarray:
    """Upward-positive, gap-filled, low-passed pelvis height (image px) from a (T, K, D) array."""
    if layout.has("Hip"):
        y = coords[:, layout.index("Hip"), 1].astype(np.float64)
        if np.isnan(y).mean() > 0.5 and layout.has("LHip") and layout.has("RHip"):
            y = np.nanmean(coords[:, layout.indices("LHip", "RHip"), 1], axis=1)
    else:
        y = np.nanmean(coords[:, layout.indices("LHip", "RHip"), 1], axis=1)
    h = -y
    h = interpolate_gaps(h, max_gap=round(2.0 * fps))
    # edge NaNs: hold nearest value so filtering/peak finding can run
    ok = ~np.isnan(h)
    if ok.any() and not ok.all():
        idx = np.arange(len(h))
        h = np.interp(idx, idx[ok], h[ok])
    return butterworth(h, fps, cutoff_hz=cutoff_hz, order=4)


def segment_sts_signal(
    h_norm: np.ndarray, fps: float, p: StsParams, trunk_lean: np.ndarray | None = None
) -> StsEvents:
    """Segment a normalised (0 seated, 1 standing) pelvis-height signal into repetitions."""
    t = len(h_norm)
    v = derivative(h_norm, fps)
    warnings: list[str] = []
    dist = max(1, round(p.min_rep_s * fps))
    # Pad both ends below the seated level so a recording that ends (or starts) standing still
    # yields a peak; indices are clipped back into range.
    padded = np.concatenate([[-1.0], h_norm, [-1.0]])
    peaks, _props = find_peaks(
        padded, height=p.peak_height_min, prominence=p.peak_prominence_min, distance=dist
    )
    peaks = np.clip(peaks - 1, 0, t - 1)

    lean_vel = derivative(trunk_lean, fps) if trunk_lean is not None else None

    reps: list[StsRep] = []
    prev_bound = 0
    for i, pk in enumerate(peaks):
        pk = int(pk)
        next_bound = int(peaks[i + 1]) if i + 1 < len(peaks) else t
        if pk <= prev_bound:
            continue
        seg = v[prev_bound:pk]
        if seg.size == 0 or np.all(np.isnan(seg)):
            continue
        i_vmax = int(np.nanargmax(seg)) + prev_bound
        seat_off = walk_until(v, i_vmax, -1, lambda x: x < p.v_seated, stop=prev_bound - 1)
        stand_reached = walk_until(v, i_vmax, +1, lambda x: x < p.v_standing, stop=next_bound)
        # descent (may be absent after the final stand)
        descent_start: int | None = None
        seated_return: int | None = None
        seg_down = v[pk:next_bound]
        if seg_down.size and np.nanmin(seg_down) < -p.v_standing:
            i_vmin = int(np.nanargmin(seg_down)) + pk
            descent_start = walk_until(v, i_vmin, -1, lambda x: x > -p.v_standing, stop=pk - 1)
            seated_return = walk_until(v, i_vmin, +1, lambda x: x > -p.v_seated, stop=next_bound)
        lean_onset: int | None = None
        if lean_vel is not None:
            lean_onset = walk_until(
                lean_vel,
                max(seat_off - 1, prev_bound),
                -1,
                lambda x: x < p.lean_velocity_deg_s,
                stop=prev_bound - 1,
            )
            lean_onset = min(lean_onset + 1, seat_off)

        rise_h = float(h_norm[pk] - h_norm[seat_off])
        valid = h_norm[seat_off] < 0.4 and h_norm[stand_reached] > 0.6 and seat_off < stand_reached
        if not valid:
            warnings.append(
                f"discarded candidate around frame {pk}: incomplete rise "
                f"(from {h_norm[seat_off]:.2f} to {h_norm[stand_reached]:.2f})"
            )
            prev_bound = seated_return if seated_return is not None else pk
            continue
        reps.append(
            StsRep(
                len(reps),
                seat_off,
                pk,
                stand_reached,
                descent_start,
                seated_return,
                lean_onset,
                rise_h,
            )
        )
        prev_bound = seated_return if seated_return is not None else stand_reached

    ev = StsEvents(reps, None, None, fps, h_norm, v, p, warnings)
    if not reps:
        ev.warnings.append("no sit-to-stand repetitions found")
        return ev

    # test start
    if p.start_rule == "manual" and p.manual_start_s is not None:
        start = round(p.manual_start_s * fps)
    elif p.start_rule == "lean_onset" and reps[0].lean_onset is not None:
        start = reps[0].lean_onset
    else:
        start = reps[0].seat_off

    if p.end_rule == "window":
        end = min(t - 1, start + round(p.window_s * fps))
        counted = [r for r in reps if r.stand_reached <= end]
        if p.partial_rule == "count_if_past_halfway":
            partial = [
                r for r in reps if r.seat_off <= end < r.stand_reached and h_norm[end] >= 0.5
            ]
            if partial:
                counted += partial
                ev.warnings.append(
                    "a repetition in progress at the end of the window was counted (past halfway)"
                )
        ev.n_in_window = len(counted)
    else:
        n = p.n_reps_expected if p.n_reps_expected > 0 else len(reps)
        if len(reps) < n:
            ev.warnings.append(f"expected {n} repetitions, found {len(reps)}")
            n = len(reps)
        elif len(reps) > n:
            ev.warnings.append(
                f"expected {n} repetitions, found {len(reps)}; timing uses the first {n}"
            )
        last = reps[n - 1]
        if p.end_rule == "fifth_sit":
            if last.seated_return is None:
                ev.warnings.append(
                    "end_rule fifth_sit but no final sit detected; using final stand"
                )
                end = last.stand_reached
            else:
                end = last.seated_return
        else:
            end = last.stand_reached
    ev.test_start, ev.test_end = int(start), int(end)
    return ev


@register_segmenter("sts_velocity_threshold", version="0.1.0")
def segment_sts(
    track: Any,
    fps: float,
    params: dict[str, Any] | StsParams,
    *,
    trunk_lean: np.ndarray | None = None,
    **_: Any,
) -> StsEvents:
    """Velocity-threshold sit-to-stand segmentation on a single person's pelvis height."""
    p = params if isinstance(params, StsParams) else StsParams.from_dict(params)
    h = pelvis_height_signal(track.coords[:, 0], track.layout, fps, p.segment_signal_cutoff_hz)
    h_norm = robust_normalize(h, 5, 95)
    return segment_sts_signal(h_norm, fps, p, trunk_lean=trunk_lean)
