"""Camera-view correction for a side recording

Two estimators:
- ``thigh_length``: the thigh is a rigid segment. Its apparent length is the true length when it
  is vertical (standing) and ``L · cos θ`` when it is horizontal (seated), so the ratio of the
  shortest to the longest apparent thigh length over a sit-to-stand is cos θ. Self-calibrating,
  no anthropometry, but it needs the thigh to pass through horizontal (sit-to-stand yes, gait no).
- ``shoulders``: the shoulders are separated across the body, so their apparent separation is
  ``W · sin θ``. Dividing by the trunk length (pelvis to neck) removes scale, and the ratio of
  shoulder breadth to trunk length comes from Drillis & Contini (1966) segment proportions:
  shoulder width 0.259 of stature, hip joint to shoulder joint 0.288 of stature.

Both are applied only when the recording is in the side or oblique band of the `camera_view`
quality check; a frontal recording fails that check and is left alone. A true side view gives
cos θ = 1 and the identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from ptvision.kinematics.angles import resolve_side
from ptvision.pose.track import PoseTrack

ViewMethod = Literal["thigh_length", "shoulders", "none"]

# Drillis & Contini (1966), body segment lengths
SHOULDER_WIDTH_FRAC = 0.259
HIP_TO_SHOULDER_FRAC = 0.288
SHOULDER_TO_TRUNK_RATIO = SHOULDER_WIDTH_FRAC / HIP_TO_SHOULDER_FRAC
MAX_VIEW_RATIO = 0.50
MIN_COS_THETA = 0.5  # 60° off the side


@dataclass(frozen=True)
class ViewEstimate:
    method: ViewMethod
    cos_theta: float  # 1.0 = true side view
    measured: float | None  # the ratio the estimate came from
    applied: bool
    note: str = ""

    @property
    def theta_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.cos_theta, -1.0, 1.0))))

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = asdict(self)
        d["theta_deg"] = round(self.theta_deg, 1)
        return d


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size < k:
        return x
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return x
    filled = np.interp(np.arange(x.size), np.flatnonzero(ok), x[ok])
    pad = k // 2
    padded = np.pad(filled, (pad, k - 1 - pad), mode="edge")
    out = np.convolve(padded, np.ones(k) / k, mode="valid")[: x.size]
    out[~ok] = np.nan
    return np.asarray(out, dtype=np.float64)


def view_ratio(track: PoseTrack, person: int = 0) -> float | None:
    """Median shoulder separation over trunk length, the `camera_view` check's measure."""
    lay = track.layout
    if not all(lay.has(n) for n in ("LShoulder", "RShoulder", "Neck", "Hip")):
        return None
    ls, rs = track.keypoint("LShoulder", person), track.keypoint("RShoulder", person)
    nk, hp = track.keypoint("Neck", person), track.keypoint("Hip", person)
    with np.errstate(invalid="ignore", divide="ignore"):
        sep = np.abs(ls[:, 0] - rs[:, 0])
        trunk = np.linalg.norm(nk - hp, axis=1)
        ratio = sep / trunk
    if not np.isfinite(ratio).any():
        return None
    return float(np.nanmedian(ratio))


def estimate_view_shoulders(track: PoseTrack, person: int = 0) -> ViewEstimate:
    """cos θ from the shoulder separation, via the Drillis & Contini breadth-to-trunk ratio."""
    ratio = view_ratio(track, person)
    if ratio is None:
        return ViewEstimate("shoulders", 1.0, None, False, "shoulders or trunk not tracked")
    sin_theta = min(1.0, ratio / SHOULDER_TO_TRUNK_RATIO)
    cos_theta = float(np.sqrt(max(0.0, 1.0 - sin_theta**2)))
    if ratio >= MAX_VIEW_RATIO or cos_theta < MIN_COS_THETA:
        return ViewEstimate("shoulders", cos_theta, ratio, False, "view is not side or oblique")
    return ViewEstimate("shoulders", cos_theta, ratio, True)


def estimate_view_thigh_length(
    track: PoseTrack, side: Literal["left", "right", "near"] = "near", person: int = 0
) -> ViewEstimate:
    """cos θ as the shortest over the longest apparent thigh length of the tracked side."""
    s = resolve_side(track, side, person)
    prefix = "L" if s == "left" else "R"
    hip, knee = track.keypoint(f"{prefix}Hip", person), track.keypoint(f"{prefix}Knee", person)
    with np.errstate(invalid="ignore"):
        length = np.linalg.norm(knee - hip, axis=1).astype(np.float64)
    length = _smooth(length, max(1, round(0.2 * track.fps)))
    ok = np.isfinite(length) & (length > 0)
    if ok.sum() < max(10, round(track.fps)):
        return ViewEstimate("thigh_length", 1.0, None, False, "thigh not tracked long enough")
    lo, hi = np.percentile(length[ok], [5, 95])
    ratio = float(lo / hi) if hi > 0 else 1.0
    cos_theta = float(min(1.0, ratio))
    vr = view_ratio(track, person)
    if (vr is not None and vr >= MAX_VIEW_RATIO) or cos_theta < MIN_COS_THETA:
        return ViewEstimate("thigh_length", cos_theta, ratio, False, "view is not side or oblique")
    return ViewEstimate("thigh_length", cos_theta, ratio, True)


def estimate_view(
    track: PoseTrack, method: ViewMethod, side: Literal["left", "right", "near"] = "near"
) -> ViewEstimate:
    if method == "thigh_length":
        return estimate_view_thigh_length(track, side)
    if method == "shoulders":
        return estimate_view_shoulders(track)
    return ViewEstimate("none", 1.0, None, False, "correction off")


def rectify_sagittal(track: PoseTrack, cos_theta: float, anchor: str = "Hip") -> PoseTrack:
    """Undo the horizontal shortening from side view: scale every keypoint's x about the
    anchor keypoint's x by 1 / cos θ, frame by frame and person by person."""
    coords = track.coords.copy()
    if cos_theta < 1.0:
        lay = track.layout
        ai = lay.index(anchor)
        ax = coords[:, :, ai, 0]  # (T, P)
        if lay.has("LHip") and lay.has("RHip"):
            with np.errstate(invalid="ignore"):
                mid = np.nanmean(coords[:, :, lay.indices("LHip", "RHip"), 0], axis=2)
            ax = np.where(np.isnan(ax), mid, ax)
        ax = np.where(np.isnan(ax), np.nanmean(coords[..., 0], axis=2), ax)
        coords[..., 0] = ax[:, :, None] + (coords[..., 0] - ax[:, :, None]) / cos_theta
    return PoseTrack(
        track.layout,
        track.fps,
        coords,
        track.score.copy(),
        track.person_ids.copy(),
        camera_id=track.camera_id,
        coord_space=track.coord_space,
        image_size=track.image_size,
        stage=f"{track.stage}+view_rectified",
        extra_metadata=dict(track.extra_metadata),
    )
