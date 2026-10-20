"""Turn marker-based mocap into what a single side camera would see.

An orthographic sagittal camera: image x runs along the walking direction (sign chosen so the
subject walks toward image-right), image y is the negated vertical axis. Both sides of the body are
projected onto the same plane, as a real side view would. Markers are mapped onto Halpe-26 keypoint
names per dataset; keypoints without a marker stay NaN with score 0, so downstream code treats them
exactly like undetected joints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ptvision.datasets.markers import MarkerTrial
from ptvision.pose.layout import HALPE26
from ptvision.pose.track import PoseTrack

# Halpe-26 keypoint -> candidate marker labels (first present wins). "{S}" = L/R. Label lookup
# ignores case (OpenCap mixes `r_knee` with `L_knee`).
# Sources: Vicon Plug-in-Gait (LHEE ...), UI-PRMD (LKNE ...), OpenCap LabValidation TRC
# (r_knee, R_HJC ...).
MARKER_MAP: dict[str, tuple[str, ...]] = {
    "{S}Heel": ("{S}HEE", "{S}_calc"),
    "{S}BigToe": ("{S}TOE", "{S}_toe"),
    "{S}SmallToe": ("{S}_5meta",),
    "{S}Ankle": ("{S}ANK", "{S}_ankle"),
    "{S}Knee": ("{S}KNE", "{S}_knee"),
    "{S}Hip": ("{S}_HJC", "{S}ASI", "{S}.ASIS"),
    "{S}Shoulder": ("{S}SHO", "{S}_Shoulder"),
    "{S}Elbow": ("{S}ELB", "{S}_elbow_lat"),
    "{S}Wrist": ("{S}WRA", "{S}WRB", "{S}_wrist_radius"),
}
# The pelvis point stands in for the Halpe-26 "Hip" keypoint, the midpoint of the two hips, so
# hip joint centres come first where a dataset has them; the ASIS/PSIS centre sits higher and
# tilts the trunk line by about 6 degrees against the video (OpenCap check, 2026-10-10).
PELVIS_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("R_HJC", "L_HJC"),
    ("L.ASIS", "R.ASIS", "L.PSIS", "R.PSIS"),
    ("LASI", "RASI", "LPSI", "RPSI"),
    ("LASI", "RASI", "SACR"),
)
NECK_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("C7", "CLAV"),
    ("C7", "R_Sternum", "L_Sternum"),
    ("C7",),
)
HEAD_CANDIDATES: tuple[tuple[str, ...], ...] = (("LFHD", "RFHD", "LBHD", "RBHD"),)


def side_names(template: str, s: str) -> str:
    """Expand the side placeholders in a marker template for side "L" or "R"."""
    return template.replace("{S}", s).replace("{SIDE}", "Left" if s == "L" else "Right")


def _first_present(trial: MarkerTrial, names: tuple[str, ...]) -> np.ndarray | None:
    for n in names:
        m = trial.marker(n)
        if m is not None and not np.isnan(m).all():
            return m
    return None


def _mean_present(trial: MarkerTrial, groups: tuple[tuple[str, ...], ...]) -> np.ndarray | None:
    for g in groups:
        ms = [trial.marker(n) for n in g]
        if all(m is not None for m in ms):
            arr = np.stack([m for m in ms if m is not None])  # (n, T, 3)
            with np.errstate(all="ignore"):
                return np.nanmean(arr, axis=0)
    return None


@dataclass
class LabFrame:
    vertical: int  # axis index
    forward: int  # axis index
    forward_sign: int  # +1 if increasing coordinate = walking direction
    lateral: int

    @property
    def description(self) -> str:
        ax = "xyz"
        sign = "+" if self.forward_sign > 0 else "-"
        return f"forward={ax[self.forward]}{sign} vertical={ax[self.vertical]}"


ForwardRule = Literal["auto", "travel", "lean"]


def infer_lab_frame(
    trial: MarkerTrial,
    pelvis: np.ndarray,
    heel: np.ndarray,
    neck: np.ndarray | None = None,
    forward_rule: ForwardRule = "auto",
) -> LabFrame:
    """Vertical = axis along which pelvis sits highest above the heel. Forward, rule "travel"
    (walking, treadmill included): the remaining axis with the largest heel excursion, signed by
    the pelvis trend or the heel's velocity skew. Rule "lean" (sit-to-stand, squats): the axis
    along which the trunk leans most, signed so the lean is forward; this also survives a walk
    to the chair at the start of a recording. "auto" picks "lean" when the heel barely moves."""
    with np.errstate(all="ignore"):
        diff = np.nanmean(pelvis - heel, axis=0)
    vertical = int(np.argmax(diff))
    rest = [a for a in range(3) if a != vertical]
    exc = [np.nanmax(heel[:, a]) - np.nanmin(heel[:, a]) for a in rest]
    heel_travel = float(np.nanmax(exc)) if np.isfinite(exc).any() else 0.0
    use_lean = forward_rule == "lean" or (forward_rule == "auto" and heel_travel < 0.3)
    if neck is not None and use_lean:
        with np.errstate(all="ignore"):
            lean = neck - pelvis
            spread = [np.nanmax(lean[:, a]) - np.nanmin(lean[:, a]) for a in rest]
        forward = rest[int(np.argmax(spread))]
        lateral = rest[1 - int(np.argmax(spread))]
        offs = lean[:, forward]
        offs = offs[~np.isnan(offs)]
        # the trunk leans forward (never backward) by a large amount during a rise
        sign = (
            1
            if offs.size == 0 or abs(np.percentile(offs, 95)) >= abs(np.percentile(offs, 5))
            else -1
        )
        return LabFrame(vertical, forward, sign, lateral)
    forward = rest[int(np.argmax(exc))]
    lateral = rest[1 - int(np.argmax(exc))]
    rel = heel[:, forward] - pelvis[:, forward]
    v = np.diff(rel)
    v = v[~np.isnan(v)]
    # forward direction: relative to the pelvis the heel swings forward about twice as fast as
    # it drifts back
    sign = 1 if v.size < 5 or (np.nanpercentile(v, 95) + np.nanpercentile(v, 5)) >= 0 else -1
    # if the pelvis translates (overground), trust its trend instead
    p = pelvis[:, forward]
    ok = ~np.isnan(p)
    if ok.sum() > 10:
        slope = np.polyfit(np.flatnonzero(ok), p[ok], 1)[0]
        travel = abs(np.nanmax(p) - np.nanmin(p))
        if travel > 0.5:  # metres
            sign = 1 if slope > 0 else -1
    return LabFrame(vertical, forward, sign, lateral)


@dataclass
class ProjectedTrial:
    track: PoseTrack
    px_per_m: float
    fps: float
    lab: LabFrame
    source: MarkerTrial

    @property
    def n_frames(self) -> int:
        return self.track.n_frames


def _resample(x: np.ndarray, src_rate: float, dst_fps: float) -> np.ndarray:
    """Linear resample along axis 0, NaN-aware per column."""
    t_src = np.arange(x.shape[0]) / src_rate
    n_dst = int(np.floor(t_src[-1] * dst_fps)) + 1
    t_dst = np.arange(n_dst) / dst_fps
    flat = x.reshape(x.shape[0], -1)
    out = np.full((n_dst, flat.shape[1]), np.nan)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = ~np.isnan(col)
        if ok.sum() >= 2:
            out[:, j] = np.interp(t_dst, t_src[ok], col[ok], left=np.nan, right=np.nan)
            # keep NaN inside long gaps: mark dst samples whose nearest src sample is NaN
            nearest = np.clip(np.round(t_dst * src_rate).astype(int), 0, x.shape[0] - 1)
            out[~ok[nearest], j] = np.nan
    return out.reshape(n_dst, *x.shape[1:])


def project_trial(
    trial: MarkerTrial,
    *,
    fps: float = 30.0,
    px_per_m: float = 500.0,
    margin_px: int = 100,
    score: float = 0.9,
    forward_rule: ForwardRule = "auto",
    azimuth_deg: float = 0.0,
) -> ProjectedTrial:
    """Project a marker trial onto a virtual camera. `azimuth_deg` turns the camera off the
    sagittal plane (0 = true side view, 90 = frontal) with the same orthographic model, which
    reproduces the foreshortening a misplaced phone produces."""
    pelvis = _mean_present(trial, PELVIS_CANDIDATES)
    if pelvis is None:
        lh, rh = (
            _first_present(trial, tuple(side_names(n, "L") for n in MARKER_MAP["{S}Hip"])),
            _first_present(trial, tuple(side_names(n, "R") for n in MARKER_MAP["{S}Hip"])),
        )
        if lh is None or rh is None:
            raise ValueError(f"{trial.path.name}: no pelvis markers found")
        pelvis = (lh + rh) / 2
    heel: np.ndarray | None = None
    for key in ("{S}Heel", "{S}Ankle"):  # a joint-centre set may have no heel
        for s in ("L", "R"):
            if heel is None:
                heel = _first_present(trial, tuple(side_names(n, s) for n in MARKER_MAP[key]))
    if heel is None:
        raise ValueError(f"{trial.path.name}: no heel or ankle marker found")

    named: dict[str, np.ndarray] = {"Hip": pelvis}
    for key, cands in MARKER_MAP.items():
        for s in ("L", "R"):
            m = _first_present(trial, tuple(side_names(c, s) for c in cands))
            if m is not None:
                named[side_names(key, s)] = m
    neck = _mean_present(trial, NECK_CANDIDATES)
    if neck is None and "LShoulder" in named and "RShoulder" in named:
        neck = (named["LShoulder"] + named["RShoulder"]) / 2
    if neck is not None:
        named["Neck"] = neck
    lab = infer_lab_frame(trial, pelvis, heel, neck, forward_rule)
    head = _mean_present(trial, HEAD_CANDIDATES)
    if head is not None:
        named["Head"] = head

    # 3D lab -> 2D image (metres): x forward, y up; an off-side camera mixes in the lateral axis
    c, s_ = float(np.cos(np.radians(azimuth_deg))), float(np.sin(np.radians(azimuth_deg)))

    def to_plane(m: np.ndarray) -> np.ndarray:
        x = lab.forward_sign * m[:, lab.forward] * c + m[:, lab.lateral] * s_
        return np.stack([x, m[:, lab.vertical]], axis=1)

    plane = {k: to_plane(v) for k, v in named.items()}
    allv = np.concatenate(list(plane.values()), axis=0)
    x0, y1 = np.nanmin(allv[:, 0]), np.nanmax(allv[:, 1])
    origin_x = x0 - margin_px / px_per_m

    def to_px(p: np.ndarray) -> np.ndarray:
        return np.stack(
            [(p[:, 0] - origin_x) * px_per_m, (y1 + margin_px / px_per_m - p[:, 1]) * px_per_m],
            axis=1,
        )

    px = {k: to_px(v) for k, v in plane.items()}
    px_rs = {k: _resample(v, trial.rate, fps) for k, v in px.items()}
    n = next(iter(px_rs.values())).shape[0]
    K = HALPE26.n
    coords = np.full((n, 1, K, 2), np.nan, dtype=np.float32)
    sc = np.zeros((n, 1, K), dtype=np.float32)
    for name, arr in px_rs.items():
        if HALPE26.has(name):
            i = HALPE26.index(name)
            coords[:, 0, i, :] = arr
            sc[:, 0, i] = np.where(np.isnan(arr).any(axis=1), 0.0, score)
    width = int(np.nanmax(coords[..., 0]) + margin_px)
    height = int(np.nanmax(coords[..., 1]) + margin_px)
    track = PoseTrack(
        HALPE26,
        fps,
        coords,
        sc,
        np.array([0], dtype=np.int16),
        image_size=(width, height),
        stage="mocap_projection",
    )
    return ProjectedTrial(track, px_per_m, fps, lab, trial)
