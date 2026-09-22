"""Walkway gait metrics. Tier 1 = timing, counting, speed; tier 2 = sagittal angles per cycle.

Distances need a `ScaleModel` (subject height); without one the metre-based metrics are reported as
unavailable rather than guessed. Step length is reported as a bout mean only: per-step values are
biased by the subject's position in the frame (Stenum 2021).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ptvision.clinical.metrics.base import Metric, MetricContext
from ptvision.clinical.registry import METRICS, register_metric
from ptvision.clinical.segmenters.gait_zeni import GaitCycle, GaitEvents

CITE_TIMING = (
    "Zeni et al. Gait Posture 2008 (events within 1–2 frames of force plates); "
    "Stenum et al. PLoS Comput Biol 2021 (single sagittal camera: step/stance/swing/double-support "
    "time MAE 0.02 s); Stenum et al. PLOS Digit Health 2024 (post-stroke and Parkinson's)."
)
CITE_SPEED = (
    "Stenum et al. 2021/2024 (gait speed error 0.02–0.04 m/s vs motion capture); "
    "Bohannon & Williams Andrews, Physiotherapy 2011 (reference values, n = 23,111)."
)
CITE_LENGTH = (
    "Stenum et al. PLoS Comput Biol 2021 (step length 0.049 m per step, <0.02 m as a bout mean; "
    "biased by position in the frame, so only the bout mean is reported)."
)
CITE_ANGLES = "Stenum et al. PLoS Comput Biol 2021 (sagittal hip MAE 4.0°, knee MAE 5.6°)."

TIMING_MAE_S = 0.02
SPEED_MAE_MS = 0.04
STEP_LENGTH_MAE_M = 0.02


def _events(ctx: MetricContext) -> GaitEvents:
    ev = ctx.events
    if not isinstance(ev, GaitEvents):
        raise TypeError("gait metrics need GaitEvents")
    return ev


def _make(name: str, **kw: Any) -> Metric:
    e = METRICS[name]
    return Metric(
        name=e.name,
        label=e.label,
        units=e.units,
        tier=e.tier,
        method_version=e.version,
        citation=e.citation,
        **kw,
    )


def _mean_sd(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else None)


def _no_scale(name: str, ev: GaitEvents) -> Metric:
    return _make(
        name,
        value=None,
        n_events=ev.n_cycles,
        flags=["subject height not recorded; metres unavailable"],
    )


def _symmetry_index(left: list[float], right: list[float]) -> tuple[float | None, float | None]:
    """(|L − R| / mean × 100, its expected error) from per-side timing lists.

    The error propagates the published single-camera timing MAE (0.02 s per event, Stenum 2021)
    through the difference of two means: at 30 fps a 0.55 s step is 16.5 frames, so one frame of
    event jitter alone is ~6 %; with n cycles per side the error shrinks by sqrt(n)."""
    if not left or not right:
        return None, None
    lm, rm = float(np.mean(left)), float(np.mean(right))
    denom = 0.5 * (lm + rm)
    if denom <= 0:
        return None, None
    si = abs(lm - rm) / denom * 100.0
    n = min(len(left), len(right))
    err = 100.0 * np.sqrt(2.0) * TIMING_MAE_S / denom / np.sqrt(n)
    return si, float(err)


def _heel_x_m(ctx: MetricContext, side: str, frame: int) -> float | None:
    """Along-floor position (m) of a heel at a frame."""
    if ctx.scale is None:
        return None
    name = f"{'L' if side == 'left' else 'R'}Heel"
    if not ctx.track.layout.has(name):
        name = f"{'L' if side == 'left' else 'R'}Ankle"
    p = ctx.track.keypoint(name)[frame]
    if np.isnan(p).any():
        return None
    return float(ctx.scale.to_meters(p)[0])


def _hip_x_m(ctx: MetricContext, frame: int) -> float | None:
    if ctx.scale is None:
        return None
    p = ctx.track.keypoint("Hip")[frame]
    if np.isnan(p).any():
        return None
    return float(ctx.scale.to_meters(p)[0])


# ---------------------------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------------------------
@register_metric(
    "gait_speed", version="0.1.0", tier=1, label="Gait speed", units="m/s", citation=CITE_SPEED
)
def gait_speed(ctx: MetricContext) -> Metric:
    """Pelvis displacement along the floor over each bout's steady-state cycles, divided by time."""
    ev = _events(ctx)
    if ctx.scale is None:
        return _no_scale("gait_speed", ev)
    speeds: list[float] = []
    weights: list[float] = []
    for b in ev.bouts:
        cyc = [c for c in ev.steady_cycles() if c.bout == b.index]
        if not cyc:
            continue
        f0, f1 = min(c.hs for c in cyc), max(c.next_hs for c in cyc)
        x0, x1 = _hip_x_m(ctx, f0), _hip_x_m(ctx, f1)
        if x0 is None or x1 is None or f1 <= f0:
            continue
        dt = (f1 - f0) / ctx.fps
        speeds.append(abs(x1 - x0) / dt)
        weights.append(dt)
    if not speeds:
        return _make("gait_speed", value=None, n_events=ev.n_cycles, flags=list(ev.warnings))
    v = float(np.average(speeds, weights=weights))
    return _make(
        "gait_speed",
        value=round(v, 2),
        error=SPEED_MAE_MS,
        error_kind="mae",
        n_events=ev.n_cycles,
        per_event=[round(s, 3) for s in speeds],
        note=f"{len(speeds)} bout(s); steady-state cycles only",
    )


@register_metric(
    "cadence", version="0.1.0", tier=1, label="Cadence", units="steps/min", citation=CITE_TIMING
)
def cadence(ctx: MetricContext) -> Metric:
    """120 / mean stride time over steady cycles (two steps per stride)."""
    ev = _events(ctx)
    strides = [c.stride / ctx.fps for c in ev.steady_cycles()]
    mean, _sd = _mean_sd(strides)
    if mean is None:
        return _make("cadence", value=None, n_events=0)
    per = [120.0 / s for s in strides]
    _m, psd = _mean_sd(per)
    return _make(
        "cadence",
        value=round(120.0 / mean, 1),
        error=None if psd is None else round(psd, 1),
        error_kind="sd",
        n_events=len(strides),
    )


def _timing(name: str, vals: list[float], digits: int = 2) -> Metric:
    mean, sd = _mean_sd(vals)
    return _make(
        name,
        value=None if mean is None else round(mean, digits),
        error=None if sd is None else round(sd, digits),
        error_kind="sd",
        n_events=len(vals),
        per_event=[round(v, 3) for v in vals],
    )


@register_metric(
    "stride_time", version="0.1.0", tier=1, label="Stride time", units="s", citation=CITE_TIMING
)
def stride_time(ctx: MetricContext) -> Metric:
    """Heel strike to the next ipsilateral heel strike, mean over steady cycles."""
    ev = _events(ctx)
    m = _timing("stride_time", [c.stride / ctx.fps for c in ev.steady_cycles()])
    m.error = max(m.error or 0.0, TIMING_MAE_S)
    return m


@register_metric(
    "stance_pct", version="0.1.0", tier=1, label="Stance phase", units="%", citation=CITE_TIMING
)
def stance_pct(ctx: MetricContext) -> Metric:
    """Stance as a percentage of the gait cycle, both sides, steady cycles."""
    ev = _events(ctx)
    return _timing(
        "stance_pct", [100.0 * c.stance / c.stride for c in ev.steady_cycles()], digits=1
    )


@register_metric(
    "swing_pct", version="0.1.0", tier=1, label="Swing phase", units="%", citation=CITE_TIMING
)
def swing_pct(ctx: MetricContext) -> Metric:
    """Swing as a percentage of the gait cycle."""
    ev = _events(ctx)
    return _timing("swing_pct", [100.0 * c.swing / c.stride for c in ev.steady_cycles()], digits=1)


@register_metric(
    "double_support_pct",
    version="0.1.0",
    tier=1,
    label="Double support",
    units="%",
    citation=CITE_TIMING,
)
def double_support_pct(ctx: MetricContext) -> Metric:
    """Initial plus terminal double support as a percentage of the gait cycle."""
    ev = _events(ctx)
    vals = [100.0 * (c.ds_initial + c.ds_terminal) / c.stride for c in ev.steady_cycles()]
    return _timing("double_support_pct", vals, digits=1)


def _side_vals(ev: GaitEvents, fn: Any) -> tuple[list[float], list[float]]:
    return [fn(c) for c in ev.steady_cycles("left")], [fn(c) for c in ev.steady_cycles("right")]


@register_metric(
    "step_time_symmetry",
    version="0.1.0",
    tier=1,
    label="Step-time asymmetry",
    units="%",
    citation=CITE_TIMING,
)
def step_time_symmetry(ctx: MetricContext) -> Metric:
    """|left − right| / mean × 100 of step times (0 = perfectly symmetric)."""
    ev = _events(ctx)
    left, right = _side_vals(ev, lambda c: c.step / ctx.fps)
    si, err = _symmetry_index(left, right)
    return _make(
        "step_time_symmetry",
        value=None if si is None else round(si, 1),
        error=None if err is None else round(err, 1),
        error_kind="mae",
        n_events=len(left) + len(right),
        side="bilateral",
        note=None if si is None else f"left {np.mean(left):.2f} s, right {np.mean(right):.2f} s",
    )


@register_metric(
    "stance_time_symmetry",
    version="0.1.0",
    tier=1,
    label="Stance-time asymmetry",
    units="%",
    citation=CITE_TIMING,
)
def stance_time_symmetry(ctx: MetricContext) -> Metric:
    """|left − right| / mean × 100 of stance times."""
    ev = _events(ctx)
    left, right = _side_vals(ev, lambda c: c.stance / ctx.fps)
    si, err = _symmetry_index(left, right)
    return _make(
        "stance_time_symmetry",
        value=None if si is None else round(si, 1),
        error=None if err is None else round(err, 1),
        error_kind="mae",
        n_events=len(left) + len(right),
        side="bilateral",
        note=None if si is None else f"left {np.mean(left):.2f} s, right {np.mean(right):.2f} s",
    )


@register_metric(
    "step_length",
    version="0.1.0",
    tier=2,
    label="Step length (bout mean)",
    units="m",
    citation=CITE_LENGTH,
)
def step_length(ctx: MetricContext) -> Metric:
    """Along-floor distance between consecutive contralateral heel strikes, mean only."""
    ev = _events(ctx)
    if ctx.scale is None:
        return _no_scale("step_length", ev)
    vals: list[float] = []
    for c in ev.steady_cycles():
        a = _heel_x_m(ctx, c.side, c.hs)
        b = _heel_x_m(ctx, "right" if c.side == "left" else "left", c.contra_hs)
        if a is not None and b is not None:
            vals.append(abs(b - a))
    mean, _sd = _mean_sd(vals)
    return _make(
        "step_length",
        value=None if mean is None else round(mean, 2),
        error=STEP_LENGTH_MAE_M,
        error_kind="mae",
        n_events=len(vals),
        per_event=None,
    )


@register_metric(
    "stride_length",
    version="0.1.0",
    tier=2,
    label="Stride length (bout mean)",
    units="m",
    citation=CITE_LENGTH,
)
def stride_length(ctx: MetricContext) -> Metric:
    """Along-floor distance of one heel between consecutive heel strikes, mean only."""
    ev = _events(ctx)
    if ctx.scale is None:
        return _no_scale("stride_length", ev)
    vals: list[float] = []
    for c in ev.steady_cycles():
        a = _heel_x_m(ctx, c.side, c.hs)
        b = _heel_x_m(ctx, c.side, c.next_hs)
        if a is not None and b is not None:
            vals.append(abs(b - a))
    mean, _sd = _mean_sd(vals)
    return _make(
        "stride_length",
        value=None if mean is None else round(mean, 2),
        error=2 * STEP_LENGTH_MAE_M,
        error_kind="mae",
        n_events=len(vals),
        per_event=None,
    )


@register_metric(
    "n_gait_cycles",
    version="0.1.0",
    tier=1,
    label="Steady gait cycles analysed",
    units="count",
    citation=CITE_TIMING,
)
def n_gait_cycles(ctx: MetricContext) -> Metric:
    """Number of steady-state gait cycles (both sides) after dropping bout edges."""
    ev = _events(ctx)
    return _make("n_gait_cycles", value=ev.n_cycles, n_events=ev.n_cycles, flags=list(ev.warnings))


# ---------------------------------------------------------------------------------------------
# Tier 2 angles
# ---------------------------------------------------------------------------------------------
def _per_cycle_extreme(
    ctx: MetricContext, angle: str, cycles: list[GaitCycle], start: str, end: str, fn: Any
) -> list[float]:
    if angle not in ctx.angles.names:
        return []
    a = ctx.angles[angle]
    out: list[float] = []
    for c in cycles:
        s, e = getattr(c, start), getattr(c, end)
        seg = a[s : e + 1]
        if seg.size and not np.all(np.isnan(seg)):
            out.append(float(fn(seg)))
    return out


def _near_side_cycles(ctx: MetricContext, ev: GaitEvents) -> list[GaitCycle]:
    return ev.steady_cycles(ctx.angles.side)


@register_metric(
    "peak_knee_flexion_swing",
    version="0.1.0",
    tier=2,
    label="Peak knee flexion in swing",
    units="deg",
    citation=CITE_ANGLES,
)
def peak_knee_flexion_swing(ctx: MetricContext) -> Metric:
    """Max knee flexion between toe-off and the next heel strike, near side, mean over cycles."""
    ev = _events(ctx)
    vals = _per_cycle_extreme(
        ctx, "knee_flexion", _near_side_cycles(ctx, ev), "to", "next_hs", np.nanmax
    )
    mean, _sd = _mean_sd(vals)
    return _make(
        "peak_knee_flexion_swing",
        value=None if mean is None else round(mean),
        error=5.6,
        error_kind="mae",
        n_events=len(vals),
        side=ctx.angles.side,
        per_event=[round(v, 1) for v in vals],
    )


@register_metric(
    "peak_hip_flexion",
    version="0.1.0",
    tier=2,
    label="Peak hip flexion",
    units="deg",
    citation=CITE_ANGLES,
)
def peak_hip_flexion(ctx: MetricContext) -> Metric:
    """Maximum hip flexion over the gait cycle, near side, mean over cycles."""
    ev = _events(ctx)
    vals = _per_cycle_extreme(
        ctx, "hip_flexion", _near_side_cycles(ctx, ev), "hs", "next_hs", np.nanmax
    )
    mean, _sd = _mean_sd(vals)
    return _make(
        "peak_hip_flexion",
        value=None if mean is None else round(mean),
        error=4.0,
        error_kind="mae",
        n_events=len(vals),
        side=ctx.angles.side,
        per_event=[round(v, 1) for v in vals],
    )
