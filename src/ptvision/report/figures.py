"""Matplotlib figures for reports (Agg backend, PNG)."""

from __future__ import annotations

from pathlib import Path

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
