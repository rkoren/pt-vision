"""Pixel-to-metre scaling for a single sagittal camera.

Method from Sports2D / Pose2Sim (BSD-3-Clause, David Pagnon):
- body height in pixels = sum of segment lengths (foot, shank, thigh, trunk, head) measured on
frames
  where the hips and knees are close to extended, combined with a trimmed mean; divided by the
  subject's height in metres this gives pixels per metre (`Pose2Sim.common.compute_height`).
- floor line fitted through the toes when they are stationary; its slope is the camera roll relative
  to the floor and its intercept the floor level (`Sports2D.process.compute_floor_line`).
- conversion rotates by the floor angle and flips y so Y points up (`convert_px_to_meters`, without
  the depth-perspective term; the subject is assumed to move in a plane at constant depth).

Limitations (from the validation literature): per-step distances are biased by the subject's
position
in the frame (Stenum 2021), so callers should report bout means, not per-step lengths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from ptvision.kinematics.angles import compute_angles
from ptvision.pose.track import PoseTrack

WINTER_LEG_TO_HEIGHT = 0.485  # leg length / stature (Winter 2009), used as a fallback


@dataclass
class ScaleModel:
    px_per_m: float
    floor_angle_rad: float = 0.0
    origin_xy: tuple[float, float] = (0.0, 0.0)  # image point mapped to (0, 0) m
    method: str = "subject_height"
    height_px: float | None = None
    height_m: float | None = None
    n_frames_used: int = 0
    notes: list[str] = field(default_factory=list)

    def to_meters(self, coords: np.ndarray) -> np.ndarray:
        """(..., 2) image px (y down) -> (..., 2) metres with X along the floor and Y up."""
        c = np.asarray(coords, dtype=np.float64)
        cx, cy = self.origin_xy
        a = (c[..., 0] - cx) / self.px_per_m
        b = -(c[..., 1] - cy) / self.px_per_m
        fa = self.floor_angle_rad
        x = a * np.cos(fa) - b * np.sin(fa)
        y = a * np.sin(fa) + b * np.cos(fa)
        return np.stack([x, y], axis=-1)

    def length_m(self, px: float | np.ndarray) -> Any:
        return np.asarray(px, dtype=np.float64) / self.px_per_m

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["floor_angle_deg"] = float(np.degrees(self.floor_angle_rad))
        return d


def _dist(track: PoseTrack, a: str, b: str) -> np.ndarray:
    pa = track.keypoint(a).astype(np.float64)
    pb = track.keypoint(b).astype(np.float64)
    return np.linalg.norm(pa - pb, axis=1)


def trimmed_mean(arr: np.ndarray, trimmed_extrema_percent: float = 50.0) -> float:
    """Mean after dropping the most extreme values (Pose2Sim `trimmed_mean`)."""
    a = np.sort(np.asarray(arr, dtype=np.float64)[~np.isnan(arr)])
    if a.size == 0:
        return float("nan")
    frac = trimmed_extrema_percent / 100.0
    lo = int(a.size * frac / 2)
    hi = int(a.size * (1 - frac / 2))
    core = a[lo:hi]
    return float(np.mean(core if core.size else a))


def estimate_height_px(
    track: PoseTrack,
    *,
    max_flexion_deg: float = 45.0,
    min_frames: int = 20,
    trimmed_extrema_percent: float = 50.0,
) -> tuple[float, int, list[str]]:
    """Body height in pixels from a single-person track.

    Frames are kept when mean hip/knee flexion (both sides) is below `max_flexion_deg`, i.e.
    the person
    is roughly upright; if fewer than `min_frames` qualify, the `min_frames` most extended
    frames are
    used. Height = foot + shank + thigh + trunk + head, averaged over sides, then a trimmed
    mean over
    frames. Returns (height_px, n_frames_used, notes).
    """
    if track.n_persons != 1:
        raise ValueError("estimate_height_px expects a single-person track")
    notes: list[str] = []
    lay = track.layout
    ang_l = compute_angles(track, ["knee_flexion", "hip_flexion"], side="left")
    ang_r = compute_angles(track, ["knee_flexion", "hip_flexion"], side="right")
    with np.errstate(invalid="ignore"):
        flex = np.nanmean(
            np.stack(
                [
                    np.abs(ang_l["knee_flexion"]),
                    np.abs(ang_l["hip_flexion"]),
                    np.abs(ang_r["knee_flexion"]),
                    np.abs(ang_r["hip_flexion"]),
                ]
            ),
            axis=0,
        )
    keep = flex < max_flexion_deg
    if keep.sum() < min_frames:
        order = np.argsort(np.nan_to_num(flex, nan=np.inf))
        keep = np.zeros_like(keep)
        keep[order[:min_frames]] = True
        notes.append(
            f"fewer than {min_frames} upright frames; used the {min_frames} most extended frames"
        )

    def seg(a: str, b: str) -> np.ndarray:
        return _dist(track, a, b)

    foot = (seg("RHeel", "RAnkle") + seg("LHeel", "LAnkle")) / 2
    shank = (seg("RAnkle", "RKnee") + seg("LAnkle", "LKnee")) / 2
    thigh = (seg("RKnee", "RHip") + seg("LKnee", "LHip")) / 2
    trunk = (seg("RHip", "RShoulder") + seg("LHip", "LShoulder")) / 2
    mid_sh = (track.keypoint("RShoulder") + track.keypoint("LShoulder")) / 2
    if lay.has("Head") and not np.isnan(track.keypoint("Head")).all():
        head = np.linalg.norm(track.keypoint("Head") - mid_sh, axis=1) * 1.008
    elif lay.has("Nose") and not np.isnan(track.keypoint("Nose")).all():
        head = np.linalg.norm(track.keypoint("Nose") - mid_sh, axis=1) * 1.5
        notes.append("Head keypoint missing; neck-to-nose × 1.5 used for the head segment")
    else:
        head = None
        notes.append(
            "no head/nose keypoints; height approximated as leg length / 0.485 (Winter 2009)"
        )

    if head is not None:
        heights = foot + shank + thigh + trunk + head
    else:
        heights = (foot + shank + thigh) / WINTER_LEG_TO_HEIGHT
    heights = np.where(keep, heights, np.nan)
    h = trimmed_mean(heights, trimmed_extrema_percent)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("could not estimate body height in pixels (keypoints missing)")
    return float(h), int(np.isfinite(heights).sum()), notes


def estimate_floor(
    track: PoseTrack,
    *,
    fps: float,
    px_per_m: float,
    toe_speed_below_m_s: float = 1.0,
    score_threshold: float = 0.5,
) -> tuple[float, tuple[float, float], int, list[str]]:
    """Calculate floor angle"""
    notes: list[str] = []
    lay = track.layout
    names = (
        ["LBigToe", "RBigToe"]
        if lay.has("LBigToe") and lay.has("RBigToe")
        else ["LAnkle", "RAnkle"]
    )
    offset_px = 0.0
    if names[0] == "LAnkle":
        offset_px = 0.13 * px_per_m
        notes.append("toe keypoints missing, so ankles + 13 cm used for the floor line")
    thr = toe_speed_below_m_s * px_per_m / fps
    xs: list[float] = []
    ys: list[float] = []
    for n in names:
        p = track.keypoint(n).astype(np.float64)
        s = track.keypoint_score(n)
        speed = np.full(len(p), np.nan)
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        speed[1:] = d
        ok = (speed < thr) & (s >= score_threshold) & ~np.isnan(p).any(axis=1)
        xs += p[ok, 0].tolist()
        ys += (p[ok, 1] + offset_px).tolist()
    if len(xs) < 10:
        notes.append("too few stationary foot samples; floor angle set to 0")
        return 0.0, (0.0, float(np.nanmax(track.coords[..., 1]))), len(xs), notes
    slope, intercept = np.polyfit(xs, ys, 1)
    angle = float(-np.arctan(slope))
    if abs(np.degrees(angle)) > 15:
        notes.append(f"implausible floor angle {np.degrees(angle):.1f}°; set to 0")
        return 0.0, (0.0, float(np.median(ys))), len(xs), notes
    return angle, (0.0, float(intercept)), len(xs), notes


def walking_direction(track: PoseTrack) -> Literal[1, -1]:
    """+1 if the pelvis moves toward image-right over the track, else -1."""
    hip = track.keypoint("Hip")[:, 0].astype(np.float64)
    ok = ~np.isnan(hip)
    if ok.sum() < 2:
        return 1
    idx = np.flatnonzero(ok)
    slope = np.polyfit(idx, hip[ok], 1)[0]
    return 1 if slope >= 0 else -1


def build_scale(
    track: PoseTrack,
    *,
    fps: float,
    height_m: float,
    floor_angle: Literal["auto"] | float = "auto",
) -> ScaleModel:
    """Full scale model given the subject's height"""
    if not (0.5 < height_m < 2.6):
        raise ValueError(f"implausible subject height {height_m} m")
    h_px, n_used, notes = estimate_height_px(track)
    ppm = h_px / height_m
    if floor_angle == "auto":
        angle, origin, n_pts, fnotes = estimate_floor(track, fps=fps, px_per_m=ppm)
        notes += fnotes
        notes.append(f"floor line from {n_pts} stationary foot samples")
    else:
        angle = float(np.radians(float(floor_angle)))
        origin = (0.0, float(np.nanmax(track.coords[..., 1])))
    return ScaleModel(
        px_per_m=ppm,
        floor_angle_rad=angle,
        origin_xy=origin,
        method="subject_height",
        height_px=h_px,
        height_m=height_m,
        n_frames_used=n_used,
        notes=notes,
    )
