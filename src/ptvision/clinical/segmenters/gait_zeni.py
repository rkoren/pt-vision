"""Gait event detection from a single sagittal camera, after Zeni et al. 2008.

Zeni JA, Richards JG, Higginson JS. Two simple methods for determining gait events during treadmill
and overground walking using kinematic data. Gait Posture 2008;27(4):710-714.
- Coordinate method: heel strike = local maximum of the heel's position relative to the pelvis along
  the walking direction; toe-off = local minimum of the toe's relative position.
- Velocity method: the zero crossings of the same relative signals' velocities.
Reported accuracy vs force plates: 94% of events within one frame (healthy), 89% within two frames
(impaired). Stenum et al. 2021 (PLoS Comput Biol) used the same signals from 2D pose with step,
stance, swing and double-support time MAE of 0.02 s.

Bouts: the person is "walking" when the pelvis moves faster than a fraction of body height per
second for at least `bout_min_s`; each bout has one walking direction. Events near the frame
edges (subject
partly out of view) are dropped, and the first/last cycle of each bout is excluded from steady-state
metrics because normative gait speed assumes steady walking (Bohannon & Williams Andrews 2011).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.signal import find_peaks

from ptvision.clinical.registry import register_segmenter
from ptvision.kinematics.preprocess import interpolate_gaps
from ptvision.kinematics.signal import derivative, runs_of
from ptvision.pose.track import PoseTrack

Side = Literal["left", "right"]
SIDES: tuple[Side, Side] = ("left", "right")


@dataclass(frozen=True)
class GaitParams:
    method: str = "coordinate"  # coordinate | velocity
    min_stride_s: float = 0.6
    max_stride_s: float = 2.5
    prominence_frac: float = 0.15  # of the relative-position signal range within the bout
    min_cycles: int = 3  # steady cycles needed for a valid analysis (quality check)
    drop_edge_cycles: int = 1  # cycles dropped at each end of every bout
    edge_margin_frac: float = 0.05  # events with the pelvis this close to a frame edge are dropped
    bout_min_s: float = 1.0
    bout_speed_frac: float = 0.15  # pelvis speed threshold, body heights per second (~0.25 m/s)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GaitParams:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Bout:
    index: int
    start: int
    end: int  # exclusive
    direction: int  # +1 toward image-right

    @property
    def n_frames(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class GaitEvent:
    frame: int
    side: Side
    kind: Literal["hs", "to"]
    bout: int


@dataclass(frozen=True)
class GaitCycle:
    side: Side
    bout: int
    hs: int
    to: int
    next_hs: int
    contra_hs: int
    contra_to: int
    steady: bool

    @property
    def stride(self) -> int:
        return self.next_hs - self.hs

    @property
    def stance(self) -> int:
        return self.to - self.hs

    @property
    def swing(self) -> int:
        return self.next_hs - self.to

    @property
    def step(self) -> int:
        """Contralateral step time: this heel strike to the other foot's heel strike."""
        return self.contra_hs - self.hs

    @property
    def ds_initial(self) -> int:
        return self.contra_to - self.hs

    @property
    def ds_terminal(self) -> int:
        return self.to - self.contra_hs

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            stride=self.stride,
            stance=self.stance,
            swing=self.swing,
            step=self.step,
            ds_initial=self.ds_initial,
            ds_terminal=self.ds_terminal,
        )
        return d


@dataclass
class GaitEvents:
    fps: float
    bouts: list[Bout]
    events: list[GaitEvent]
    cycles: list[GaitCycle]
    signals: dict[
        str, np.ndarray
    ]  # heel_left, toe_left, heel_right, toe_right (anterior px, NaN off-bout)
    params: GaitParams
    image_width: int | None = None
    warnings: list[str] = field(default_factory=list)

    def steady_cycles(self, side: Side | None = None) -> list[GaitCycle]:
        return [c for c in self.cycles if c.steady and (side is None or c.side == side)]

    @property
    def n_cycles(self) -> int:
        return len(self.steady_cycles())

    @property
    def direction(self) -> int:
        """Predominant walking direction (+1 image-right) over the bouts, weighted by length."""
        if not self.bouts:
            return 1
        s = sum(b.direction * b.n_frames for b in self.bouts)
        return 1 if s >= 0 else -1

    def steady_window(self) -> tuple[int, int] | None:
        sc = self.steady_cycles()
        if not sc:
            return None
        return min(c.hs for c in sc), max(c.next_hs for c in sc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segmenter": "gait_zeni",
            "fps": self.fps,
            "n_cycles": self.n_cycles,
            "direction": self.direction,
            "steady_window": self.steady_window(),
            "bouts": [asdict(b) for b in self.bouts],
            "events": [asdict(e) for e in self.events],
            "cycles": [c.to_dict() for c in self.cycles],
            "warnings": list(self.warnings),
            "params": self.params.to_dict(),
        }


# ---------------------------------------------------------------------------------------------
def _fill(x: np.ndarray, max_gap: int) -> np.ndarray:
    y = interpolate_gaps(np.asarray(x, dtype=np.float64), max_gap)
    ok = ~np.isnan(y)
    if ok.any() and not ok.all():
        idx = np.arange(len(y))
        y = np.interp(idx, idx[ok], y[ok])
    return np.asarray(y, dtype=np.float64)


def _body_height_px(track: PoseTrack) -> float:
    c = track.coords[:, 0]
    with np.errstate(all="ignore"):
        ext = np.nanmax(c[..., 1], axis=1) - np.nanmin(c[..., 1], axis=1)
    h = float(np.nanmedian(ext))
    return h if np.isfinite(h) and h > 0 else 400.0


def detect_bouts(hip_x: np.ndarray, fps: float, height_px: float, p: GaitParams) -> list[Bout]:
    """Walking bouts from pelvis horizontal velocity; one direction per bout."""
    v = derivative(hip_x, fps)
    thr = p.bout_speed_frac * height_px
    moving = np.abs(v) > thr
    bouts: list[Bout] = []
    min_len = max(2, round(p.bout_min_s * fps))
    for a, b in runs_of(moving):
        if b - a < min_len:
            continue
        # split on direction changes inside the run
        sign = np.sign(v[a:b])
        start = a
        for i in range(a + 1, b):
            if sign[i - a] != sign[i - 1 - a] and sign[i - a] != 0:
                if i - start >= min_len:
                    bouts.append(Bout(len(bouts), start, i, 1 if sign[i - 1 - a] > 0 else -1))
                start = i
        if b - start >= min_len:
            bouts.append(Bout(len(bouts), start, b, 1 if np.mean(v[start:b]) > 0 else -1))
    return bouts


def _peaks(sig: np.ndarray, fps: float, p: GaitParams, *, negate: bool) -> np.ndarray:
    s = -sig if negate else sig
    rng = float(np.nanmax(s) - np.nanmin(s)) if np.isfinite(s).any() else 0.0
    if rng <= 0:
        return np.array([], dtype=int)
    dist = max(2, round(0.7 * p.min_stride_s * fps))
    idx, _ = find_peaks(s, distance=dist, prominence=p.prominence_frac * rng)
    return np.asarray(idx, dtype=int)


def _zero_crossings(sig: np.ndarray, fps: float, p: GaitParams, *, falling: bool) -> np.ndarray:
    v = derivative(sig, fps)
    s = np.sign(v)
    cross = (
        np.flatnonzero((s[:-1] > 0) & (s[1:] <= 0))
        if falling
        else np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
    )
    cross = cross + 1
    # enforce minimum spacing
    keep: list[int] = []
    min_gap = round(0.7 * p.min_stride_s * fps)
    for c in cross:
        if not keep or c - keep[-1] >= min_gap:
            keep.append(int(c))
    return np.array(keep, dtype=int)


def _build_cycles(
    events: list[GaitEvent],
    bout: Bout,
    fps: float,
    p: GaitParams,
    hip_x: np.ndarray,
    width: int | None,
) -> tuple[list[GaitCycle], list[str]]:
    warnings: list[str] = []
    cycles: list[GaitCycle] = []
    by = {
        (side, kind): sorted(
            e.frame for e in events if e.side == side and e.kind == kind and e.bout == bout.index
        )
        for side in SIDES
        for kind in ("hs", "to")
    }
    margin = (p.edge_margin_frac * width) if width else 0.0
    for side in SIDES:
        contra: Side = "right" if side == "left" else "left"
        hs = by[(side, "hs")]
        per_side: list[GaitCycle] = []
        for i in range(len(hs) - 1):
            h0, h1 = hs[i], hs[i + 1]
            stride_s = (h1 - h0) / fps
            if not (p.min_stride_s <= stride_s <= p.max_stride_s):
                warnings.append(
                    f"{side}: stride of {stride_s:.2f} s at frame {h0} outside "
                    f"{p.min_stride_s}–{p.max_stride_s} s, skipped"
                )
                continue
            to = [f for f in by[(side, "to")] if h0 < f < h1]
            chs = [f for f in by[(contra, "hs")] if h0 < f < h1]
            if not to or not chs:
                warnings.append(
                    f"{side}: cycle at frame {h0} lacks a toe-off or contralateral heel strike, "
                    "skipped"
                )
                continue
            cto = [f for f in by[(contra, "to")] if h0 <= f <= chs[0]]
            if not cto:
                warnings.append(
                    f"{side}: cycle at frame {h0} lacks a contralateral toe-off, skipped"
                )
                continue
            if not (h0 < cto[0] <= chs[0] < to[0] < h1):
                warnings.append(f"{side}: events out of order around frame {h0}, skipped")
                continue
            near_edge = False
            if width and margin > 0:
                xs = hip_x[[h0, h1]]
                near_edge = bool(np.any(xs < margin) or np.any(xs > width - margin))
            per_side.append(
                GaitCycle(side, bout.index, h0, to[0], h1, chs[0], cto[0], steady=not near_edge)
            )
        # drop edge cycles of the bout
        k = p.drop_edge_cycles
        for j, c in enumerate(per_side):
            steady = c.steady and (k <= j < len(per_side) - k)
            cycles.append(
                GaitCycle(c.side, c.bout, c.hs, c.to, c.next_hs, c.contra_hs, c.contra_to, steady)
            )
    cycles.sort(key=lambda c: c.hs)
    return cycles, warnings


def segment_gait_track(
    track: PoseTrack, fps: float, p: GaitParams, *, image_width: int | None = None
) -> GaitEvents:
    if track.n_persons != 1:
        raise ValueError("segment_gait expects a single-person track")
    lay = track.layout
    n = track.n_frames
    hip_x = _fill(track.keypoint("Hip")[:, 0], max_gap=round(fps))
    height_px = _body_height_px(track)
    bouts = detect_bouts(hip_x, fps, height_px, p)
    warnings: list[str] = []
    if not bouts:
        warnings.append("no walking bout detected (pelvis never moved faster than the threshold)")

    signals = {f"{k}_{s}": np.full(n, np.nan) for s in SIDES for k in ("heel", "toe")}
    events: list[GaitEvent] = []
    for b in bouts:
        sl = slice(b.start, b.end)
        for side in SIDES:
            S = "L" if side == "left" else "R"
            heel_name = f"{S}Heel" if lay.has(f"{S}Heel") else f"{S}Ankle"
            toe_name = f"{S}BigToe" if lay.has(f"{S}BigToe") else f"{S}Ankle"
            heel = _fill(track.keypoint(heel_name)[:, 0], max_gap=round(fps))
            toe = _fill(track.keypoint(toe_name)[:, 0], max_gap=round(fps))
            heel_a = b.direction * (heel[sl] - hip_x[sl])
            toe_a = b.direction * (toe[sl] - hip_x[sl])
            signals[f"heel_{side}"][sl] = heel_a
            signals[f"toe_{side}"][sl] = toe_a
            if p.method == "velocity":
                hs_idx = _zero_crossings(heel_a, fps, p, falling=True)
                to_idx = _zero_crossings(toe_a, fps, p, falling=False)
            else:
                hs_idx = _peaks(heel_a, fps, p, negate=False)
                to_idx = _peaks(toe_a, fps, p, negate=True)
            events += [GaitEvent(int(i) + b.start, side, "hs", b.index) for i in hs_idx]
            events += [GaitEvent(int(i) + b.start, side, "to", b.index) for i in to_idx]
    events.sort(key=lambda e: e.frame)

    cycles: list[GaitCycle] = []
    for b in bouts:
        c, w = _build_cycles(events, b, fps, p, hip_x, image_width)
        cycles += c
        warnings += w
    ev = GaitEvents(fps, bouts, events, cycles, signals, p, image_width, warnings)
    if ev.n_cycles < p.min_cycles:
        ev.warnings.append(f"only {ev.n_cycles} steady gait cycles (need {p.min_cycles})")
    return ev


@register_segmenter("gait_zeni", version="0.1.0")
def segment_gait(
    track: PoseTrack,
    fps: float,
    params: dict[str, Any] | GaitParams,
    *,
    image_size: tuple[int, int] | None = None,
    **_: Any,
) -> GaitEvents:
    """Zeni 2008 heel-strike / toe-off detection from heel and toe positions vs the pelvis."""
    p = params if isinstance(params, GaitParams) else GaitParams.from_dict(params)
    width = image_size[0] if image_size else (track.image_size[0] if track.image_size else None)
    return segment_gait_track(track, fps, p, image_width=width)
