# [REVIEW] new file
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

from ptvision.datasets.c3d import C3DEvent, C3DTrial
from ptvision.kinematics.scale import ScaleModel
from ptvision.pose.layout import HALPE26
from ptvision.pose.track import PoseTrack

# Halpe-26 keypoint -> candidate marker labels (first present wins). "{S}" = L/R.
# Sources: Fukuchi 2018 (L.Heel ...), Van Criekinge 2023 Plug-in-Gait (LHEE ...),
# Schreiber 2019 ISB (L_FCC ...).
MARKER_MAP: dict[str, tuple[str, ...]] = {
    "{S}Heel": ("{S}.Heel", "{S}HEE", "{S}_FCC"),
    "{S}BigToe": ("{S}.MT1", "{S}TOE", "{S}_FM1", "{S}_FM2"),
    "{S}SmallToe": ("{S}.MT5", "{S}_FM5"),
    "{S}Ankle": ("{S}.Ankle", "{S}ANK", "{S}_FAL"),
    "{S}Knee": ("{S}.Knee", "{S}KNE", "{S}_FLE"),
    "{S}Hip": ("{S}.GTR", "{S}_FTC", "{S}.ASIS", "{S}ASI", "{S}_IAS"),
    "{S}Shoulder": ("{S}SHO", "{S}_SAE", "{S}_SAJ"),
    "{S}Elbow": ("{S}ELB", "{S}_HLE"),
    "{S}Wrist": ("{S}WRA", "{S}_UOA", "{S}WRB"),
}
PELVIS_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("L.ASIS", "R.ASIS", "L.PSIS", "R.PSIS"),
    ("LASI", "RASI", "LPSI", "RPSI"),
    ("LASI", "RASI", "SACR"),
    ("L_IAS", "R_IAS", "L_IPS", "R_IPS"),
)
NECK_CANDIDATES: tuple[tuple[str, ...], ...] = (("C7", "CLAV"), ("CV7", "SJN"), ("C7",), ("CV7",))
HEAD_CANDIDATES: tuple[tuple[str, ...], ...] = (("LFHD", "RFHD", "LBHD", "RBHD"), ("SGL",))


def _first_present(trial: C3DTrial, names: tuple[str, ...]) -> np.ndarray | None:
    for n in names:
        m = trial.marker(n)
        if m is not None and not np.isnan(m).all():
            return m
    return None


def _mean_present(trial: C3DTrial, groups: tuple[tuple[str, ...], ...]) -> np.ndarray | None:
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


def infer_lab_frame(trial: C3DTrial, pelvis: np.ndarray, heel: np.ndarray) -> LabFrame:
    """Vertical = axis along which pelvis sits highest above the heel; forward = remaining axis
    with the largest heel excursion (works on treadmills, where the pelvis barely translates); sign
    from the pelvis trend when it translates, else from the heel's velocity skew relative to the
    pelvis."""
    with np.errstate(all="ignore"):
        diff = np.nanmean(pelvis - heel, axis=0)
    vertical = int(np.argmax(diff))
    rest = [a for a in range(3) if a != vertical]
    exc = [np.nanmax(heel[:, a]) - np.nanmin(heel[:, a]) for a in rest]
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
    scale: ScaleModel
    fps: float
    lab: LabFrame
    events: list[tuple[str, str, int]]  # (side, kind hs|to, frame at fps)
    source: C3DTrial

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


EventKind = Literal["hs", "to"]


def classify_event(e: C3DEvent) -> tuple[str, EventKind] | None:
    """Map dataset event labels to (side, hs|to). Returns None for force-plate on/off etc."""
    lbl = e.label.upper().replace(" ", "")
    ctx = e.context.lower()
    side = "left" if ctx.startswith("l") else "right" if ctx.startswith("r") else None
    if side is None:
        if lbl.startswith("L"):
            side, lbl = "left", lbl[1:]
        elif lbl.startswith("R"):
            side, lbl = "right", lbl[1:]
        else:
            return None
    if lbl in ("HS", "FOOTSTRIKE", "FOOTSTRIKE1", "FOOTSTRIKE2", "IC", "HEELSTRIKE", "STRIKE"):
        return side, "hs"
    if lbl in ("TO", "FOOTOFF", "TOEOFF", "OFF"):
        return side, "to"
    return None


def project_trial(
    trial: C3DTrial,
    *,
    fps: float = 30.0,
    px_per_m: float = 500.0,
    margin_px: int = 100,
    score: float = 0.9,
) -> ProjectedTrial:
    pelvis = _mean_present(trial, PELVIS_CANDIDATES)
    if pelvis is None:
        lh, rh = (
            _first_present(trial, tuple(n.replace("{S}", "L") for n in MARKER_MAP["{S}Hip"])),
            _first_present(trial, tuple(n.replace("{S}", "R") for n in MARKER_MAP["{S}Hip"])),
        )
        if lh is None or rh is None:
            raise ValueError(f"{trial.path.name}: no pelvis markers found")
        pelvis = (lh + rh) / 2
    heel = _first_present(trial, tuple(n.replace("{S}", "L") for n in MARKER_MAP["{S}Heel"]))
    if heel is None:
        heel = _first_present(trial, tuple(n.replace("{S}", "R") for n in MARKER_MAP["{S}Heel"]))
    if heel is None:
        raise ValueError(f"{trial.path.name}: no heel marker found")
    lab = infer_lab_frame(trial, pelvis, heel)

    named: dict[str, np.ndarray] = {"Hip": pelvis}
    for key, cands in MARKER_MAP.items():
        for s in ("L", "R"):
            m = _first_present(trial, tuple(c.replace("{S}", s) for c in cands))
            if m is not None:
                named[key.replace("{S}", s)] = m
    neck = _mean_present(trial, NECK_CANDIDATES)
    if neck is None and "LShoulder" in named and "RShoulder" in named:
        neck = (named["LShoulder"] + named["RShoulder"]) / 2
    if neck is not None:
        named["Neck"] = neck
    head = _mean_present(trial, HEAD_CANDIDATES)
    if head is not None:
        named["Head"] = head

    # 3D lab -> 2D image (metres): x forward, y up
    def to_plane(m: np.ndarray) -> np.ndarray:
        return np.stack([lab.forward_sign * m[:, lab.forward], m[:, lab.vertical]], axis=1)

    plane = {k: to_plane(v) for k, v in named.items()}
    allv = np.concatenate(list(plane.values()), axis=0)
    x0, y1 = np.nanmin(allv[:, 0]), np.nanmax(allv[:, 1])
    y0 = np.nanmin(allv[:, 1])
    origin_x = x0 - margin_px / px_per_m
    floor_y = y0  # lowest point ~ floor

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
    floor_px = (y1 + margin_px / px_per_m - floor_y) * px_per_m
    scale = ScaleModel(
        px_per_m=px_per_m,
        floor_angle_rad=0.0,
        origin_xy=(0.0, float(floor_px)),
        method="mocap_known",
        notes=[f"orthographic projection, {lab.description}"],
    )

    events: list[tuple[str, str, int]] = []
    for ev in trial.events:
        c = classify_event(ev)
        if c is None:
            continue
        t_rel = ev.time_s - trial.first_frame / trial.rate
        f = round(t_rel * fps)
        if 0 <= f < n:
            events.append((c[0], c[1], f))
    events.sort(key=lambda x: x[2])
    # some datasets label the same event twice (Fukuchi 2018 repeats every toe-off); keep one per
    # side/kind within a frame
    deduped: list[tuple[str, str, int]] = []
    for item in events:
        if (
            deduped
            and deduped[-1][0] == item[0]
            and deduped[-1][1] == item[1]
            and abs(deduped[-1][2] - item[2]) <= 1
        ):
            continue
        deduped.append(item)
    return ProjectedTrial(track, scale, fps, lab, deduped, trial)
