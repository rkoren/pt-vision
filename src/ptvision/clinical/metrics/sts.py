"""Sit-to-stand metrics. Tier 1 = timing and counting; tier 2 = angles at events."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from ptvision.clinical.metrics.base import Metric, MetricContext
from ptvision.clinical.registry import METRICS, register_metric
from ptvision.clinical.segmenters.sts import StsEvents

CITE_5XSTS = (
    "Bohannon RW. Percept Mot Skills 2006;103:215-222 (reference values); "
    "Bertrand et al. PLOS Digit Health 2026 (video vs clinician ICC 0.995, n=228)."
)
CITE_30S = (
    "Bertrand et al. PLOS Digit Health 2026 (30-s chair stand video vs clinician ICC 0.928, n=228)."
)
CITE_ANGLES = (
    "Hwang et al. J Cachexia Sarcopenia Muscle 2026 (2D video STS: knee RMSE 5.9 deg, "
    "trunk RMSE 4.0 deg vs mocap)."
)


def _events(ctx: MetricContext) -> StsEvents:
    ev = ctx.events
    if not isinstance(ev, StsEvents):
        raise TypeError("STS metrics need StsEvents")
    return ev


def _make(name: str, **kw: Any) -> Metric:
    """Build a Metric with label/units/tier/version/citation taken from the registry entry."""
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
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else None)


def _timing(name: str, vals: list[float]) -> Metric:
    mean, sd = _mean_sd(vals)
    return _make(
        name,
        value=None if mean is None else round(mean, 2),
        error=None if sd is None else round(sd, 2),
        error_kind="sd",
        n_events=len(vals),
        per_event=[round(v, 3) for v in vals],
    )


@register_metric(
    "sts_total_time",
    version="0.1.0",
    tier=1,
    label="Five-times sit-to-stand time",
    units="s",
    citation=CITE_5XSTS,
)
def sts_total_time(ctx: MetricContext) -> Metric:
    """Time from first seat-off to the fifth full stand (or fifth sit, per protocol end_rule)."""
    ev = _events(ctx)
    value = ev.total_time_s
    return _make(
        "sts_total_time",
        value=None if value is None else round(value, 2),
        error=round(2.0 / ctx.fps, 3),
        error_kind="resolution",
        n_events=ev.n_reps,
        flags=list(ev.warnings),
        note=f"start: {ev.params.start_rule}, end: {ev.params.end_rule}",
    )


@register_metric(
    "sts_rep_count",
    version="0.1.0",
    tier=1,
    label="Repetitions detected",
    units="count",
    citation=CITE_5XSTS,
)
def sts_rep_count(ctx: MetricContext) -> Metric:
    """Number of complete sit-to-stand repetitions detected in the recording."""
    ev = _events(ctx)
    return _make("sts_rep_count", value=ev.n_reps, n_events=ev.n_reps)


@register_metric(
    "chair_stand_count_30s",
    version="0.1.0",
    tier=1,
    label="Chair stands in 30 s",
    units="count",
    citation=CITE_30S,
)
def chair_stand_count_30s(ctx: MetricContext) -> Metric:
    """Full stands completed within the 30-second window."""
    ev = _events(ctx)
    return _make(
        "chair_stand_count_30s", value=ev.n_in_window, n_events=ev.n_reps, flags=list(ev.warnings)
    )


@register_metric(
    "sts_rise_time",
    version="0.1.0",
    tier=1,
    label="Rise time (seat-off to stand)",
    units="s",
    citation=CITE_5XSTS,
)
def sts_rise_time(ctx: MetricContext) -> Metric:
    """Mean duration of the rising phase across repetitions."""
    ev = _events(ctx)
    return _timing("sts_rise_time", [(r.stand_reached - r.seat_off) / ctx.fps for r in ev.reps])


@register_metric(
    "sts_sit_time",
    version="0.1.0",
    tier=1,
    label="Sit time (descent start to seated)",
    units="s",
    citation=CITE_5XSTS,
)
def sts_sit_time(ctx: MetricContext) -> Metric:
    """Mean duration of the sitting-down phase across repetitions that end seated."""
    ev = _events(ctx)
    vals = [
        (r.seated_return - r.descent_start) / ctx.fps
        for r in ev.reps
        if r.descent_start is not None and r.seated_return is not None
    ]
    return _timing("sts_sit_time", vals)


@register_metric(
    "sts_cycle_time",
    version="0.1.0",
    tier=1,
    label="Cycle time (seat-off to next seat-off)",
    units="s",
    citation=CITE_5XSTS,
)
def sts_cycle_time(ctx: MetricContext) -> Metric:
    """Mean time between consecutive seat-offs."""
    ev = _events(ctx)
    vals = [(b - a) / ctx.fps for a, b in pairwise(r.seat_off for r in ev.reps)]
    return _timing("sts_cycle_time", vals)


def _angle_at(ctx: MetricContext, angle: str, frames: list[int]) -> list[float]:
    if angle not in ctx.angles.names:
        return []
    a = ctx.angles[angle]
    return [float(a[f]) for f in frames if 0 <= f < len(a) and not np.isnan(a[f])]


def _angle_metric(name: str, vals: list[float], rmse: float, side: str) -> Metric:
    mean, _sd = _mean_sd(vals)
    return _make(
        name,
        value=None if mean is None else round(mean),
        error=rmse,
        error_kind="rmse",
        n_events=len(vals),
        side=side,
        per_event=[round(v, 1) for v in vals],
    )


@register_metric(
    "trunk_lean_at_seat_off",
    version="0.1.0",
    tier=2,
    label="Trunk forward lean at seat-off",
    units="deg",
    citation=CITE_ANGLES,
)
def trunk_lean_at_seat_off(ctx: MetricContext) -> Metric:
    """Trunk angle from vertical when the pelvis starts rising, averaged over repetitions."""
    ev = _events(ctx)
    vals = _angle_at(ctx, "trunk_lean", [r.seat_off for r in ev.reps])
    return _angle_metric("trunk_lean_at_seat_off", vals, 4.0, "bilateral")


@register_metric(
    "peak_trunk_lean_rising",
    version="0.1.0",
    tier=2,
    label="Peak trunk forward lean during rise",
    units="deg",
    citation=CITE_ANGLES,
)
def peak_trunk_lean_rising(ctx: MetricContext) -> Metric:
    """Max trunk forward lean between lean onset (or seat-off) and full stand, per repetition."""
    ev = _events(ctx)
    vals: list[float] = []
    if "trunk_lean" in ctx.angles.names:
        a = ctx.angles["trunk_lean"]
        for r in ev.reps:
            s = r.lean_onset if r.lean_onset is not None else r.seat_off
            seg = a[s : r.stand_reached + 1]
            if seg.size and not np.all(np.isnan(seg)):
                vals.append(float(np.nanmax(seg)))
    return _angle_metric("peak_trunk_lean_rising", vals, 4.0, "bilateral")


@register_metric(
    "knee_flexion_at_seat_off",
    version="0.1.0",
    tier=2,
    label="Knee flexion at seat-off",
    units="deg",
    citation=CITE_ANGLES,
)
def knee_flexion_at_seat_off(ctx: MetricContext) -> Metric:
    """Knee flexion (near side) at seat-off, averaged over repetitions."""
    ev = _events(ctx)
    vals = _angle_at(ctx, "knee_flexion", [r.seat_off for r in ev.reps])
    return _angle_metric("knee_flexion_at_seat_off", vals, 5.9, ctx.angles.side)
