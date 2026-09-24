"""Matplotlib figures for reports (Agg backend, PNG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ptvision.clinical.segmenters.sts import StsEvents

PHASE_COLORS = {"rise": "#c8e6c9", "stand": "#e3f2fd", "descent": "#ffe0b2"}


def _shade_phases(ax: Axes, ev: StsEvents, fps: float) -> None:
    for r in ev.reps:
        ax.axvspan(r.seat_off / fps, r.stand_reached / fps, color=PHASE_COLORS["rise"], lw=0)
        if r.descent_start is not None and r.seated_return is not None:
            ax.axvspan(
                r.stand_reached / fps, r.descent_start / fps, color=PHASE_COLORS["stand"], lw=0
            )
            ax.axvspan(
                r.descent_start / fps, r.seated_return / fps, color=PHASE_COLORS["descent"], lw=0
            )


def fig_sts_trajectory(ev: StsEvents, out: Path) -> Path:
    fps = ev.fps
    t = np.arange(len(ev.height_norm)) / fps
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    _shade_phases(ax1, ev, fps)
    ax1.plot(t, ev.height_norm, color="#1f4e79", lw=1.6, label="pelvis height (normalised)")
    for r in ev.reps:
        ax1.plot(r.seat_off / fps, ev.height_norm[r.seat_off], "v", color="#2e7d32", ms=8)
        ax1.plot(r.stand_reached / fps, ev.height_norm[r.stand_reached], "^", color="#1565c0", ms=8)
        if r.seated_return is not None:
            ax1.plot(
                r.seated_return / fps, ev.height_norm[r.seated_return], "o", color="#ef6c00", ms=6
            )
        if r.lean_onset is not None:
            ax1.plot(
                r.lean_onset / fps, ev.height_norm[r.lean_onset], "|", color="#6a1b9a", ms=12, mew=2
            )
        ax1.annotate(
            str(r.index + 1),
            (r.peak / fps, ev.height_norm[r.peak]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )
    if ev.test_start is not None and ev.test_end is not None:
        ax1.axvline(ev.test_start / fps, color="k", ls="--", lw=1)
        ax1.axvline(ev.test_end / fps, color="k", ls="--", lw=1)
        ax1.text(
            (ev.test_start + ev.test_end) / 2 / fps,
            1.08,
            f"{ev.total_time_s:.2f} s",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_ylim(-0.15, 1.2)
    ax1.set_ylabel("0 = seated, 1 = standing")
    ax1.legend(loc="lower right", fontsize=8, frameon=False)
    ax1.set_title(
        "Sit-to-stand: pelvis height with detected phases "
        "(▼ seat-off, ▲ stand, ● seated, | lean onset)",
        fontsize=10,
    )
    ax2.plot(t, ev.velocity, color="#555", lw=1.2)
    ax2.axhline(ev.params.v_seated, color="#2e7d32", ls=":", lw=1)
    ax2.axhline(-ev.params.v_seated, color="#ef6c00", ls=":", lw=1)
    ax2.set_ylabel("velocity (1/s)")
    ax2.set_xlabel("time (s)")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_angles(
    raw: pd.DataFrame,
    filt: pd.DataFrame,
    fps: float,
    ev: StsEvents | None,
    out: Path,
    names: list[str] | None = None,
) -> Path:
    names = names or list(filt.columns)
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n + 0.6), sharex=True, squeeze=False)
    t = np.arange(len(filt)) / fps
    for ax, name in zip(axes[:, 0], names, strict=True):
        if ev is not None:
            _shade_phases(ax, ev, fps)
        if name in raw.columns:
            ax.plot(t, raw[name].to_numpy(), color="#bbb", lw=0.8, label="raw")
        ax.plot(t, filt[name].to_numpy(), color="#1f4e79", lw=1.5, label="filtered")
        ax.set_ylabel(f"{name}\n(deg)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("Joint angles, raw vs filtered (Hampel + zero-phase Butterworth)", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_per_rep(ev: StsEvents, out: Path) -> Path:
    fps = ev.fps
    idx = [r.index + 1 for r in ev.reps]
    rise = [(r.stand_reached - r.seat_off) / fps for r in ev.reps]
    sit = [
        ((r.seated_return - r.descent_start) / fps)
        if (r.descent_start is not None and r.seated_return is not None)
        else np.nan
        for r in ev.reps
    ]
    fig, ax = plt.subplots(figsize=(6, 3))
    w = 0.38
    x = np.arange(len(idx))
    ax.bar(x - w / 2, rise, w, color="#2e7d32", label="rise")
    ax.bar(x + w / 2, sit, w, color="#ef6c00", label="sit")
    ax.set_xticks(x, [str(i) for i in idx])
    ax.set_xlabel("repetition")
    ax.set_ylabel("seconds")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_gait_events(ev: Any, out: Path) -> Path:
    """Heel and toe positions relative to the pelvis (anterior +) with detected events."""
    fps = ev.fps
    n = len(next(iter(ev.signals.values())))
    t = np.arange(n) / fps
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    colors = {"left": "#1f77b4", "right": "#d62728"}
    for ax, kind, title in (
        (axes[0], "heel", "heel (heel strike = local maximum)"),
        (axes[1], "toe", "toe (toe-off = local minimum)"),
    ):
        for side in ("left", "right"):
            ax.plot(t, ev.signals[f"{kind}_{side}"], color=colors[side], lw=1.3, label=side)
        for e in ev.events:
            if e.kind == ("hs" if kind == "heel" else "to"):
                y = ev.signals[f"{kind}_{e.side}"][e.frame]
                ax.plot(
                    e.frame / fps, y, "v" if kind == "heel" else "^", color=colors[e.side], ms=7
                )
        for b in ev.bouts:
            ax.axvspan(b.start / fps, b.end / fps, color="#eeeeee", lw=0, zorder=-10)
        win = ev.steady_window()
        if win:
            ax.axvline(win[0] / fps, color="k", ls="--", lw=1)
            ax.axvline(win[1] / fps, color="k", ls="--", lw=1)
        ax.set_ylabel(f"{kind} − pelvis (px)")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_gait_cycles(
    angles: pd.DataFrame, ev: Any, side: str, out: Path, names: list[str] | None = None
) -> Path:
    """Angles normalised to % gait cycle (heel strike to heel strike), mean ± SD over cycles."""
    names = [n for n in (names or list(angles.columns)) if n in angles.columns]
    cycles = [c for c in ev.steady_cycles() if c.side == side]
    fig, axes = plt.subplots(
        1, max(1, len(names)), figsize=(3.4 * max(1, len(names)), 3.2), squeeze=False
    )
    grid = np.linspace(0, 100, 101)
    for ax, name in zip(axes[0], names, strict=False):
        curves = []
        for c in cycles:
            seg = angles[name].to_numpy()[c.hs : c.next_hs + 1]
            if seg.size < 3 or np.isnan(seg).all():
                continue
            x = np.linspace(0, 100, seg.size)
            ok = ~np.isnan(seg)
            curves.append(np.interp(grid, x[ok], seg[ok]))
        if curves:
            arr = np.vstack(curves)
            m, s = arr.mean(axis=0), arr.std(axis=0)
            ax.fill_between(grid, m - s, m + s, color="#7fb3ff", alpha=0.35, lw=0)
            ax.plot(grid, m, color="#1f4e79", lw=1.6)
            to_pct = np.mean([100.0 * c.stance / c.stride for c in cycles]) if cycles else None
            if to_pct is not None:
                ax.axvline(to_pct, color="#ef6c00", ls=":", lw=1)
        ax.set_title(f"{name.replace('_', ' ')} ({side}, n={len(curves)})", fontsize=9)
        ax.set_xlabel("% gait cycle")
        ax.set_ylabel("deg")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
