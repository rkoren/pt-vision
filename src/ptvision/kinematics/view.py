"""Camera-view correction for a side recording

A phone θ off the side view sees the sagittal plane at an angle: lengths along the direction the
person faces shrink by cos θ, and a segment that points partly toward a phone above or below it
looks tilted. The correction undoes this before the angles are computed, with one of two camera
models:

- ``orthographic`` (default): stretch every keypoint horizontally about the pelvis by 1 / cos θ.
  Right for a level camera far away compared with the body; it leaves the perspective tilt.
- ``perspective``: re-project every keypoint onto a virtual camera at the same place, level and
  square to the side (a pinhole camera turned about its own centre, the homography K R K⁻¹). It
  needs the focal length in pixels and whether the camera is toward the person's front or back,
  and takes the camera as level unless its tilt is given.

θ comes from the shoulders. Their apparent separation is ``W · sin θ``; dividing by the trunk length
(pelvis to neck) removes scale, and the ratio of shoulder breadth to trunk length comes from Drillis
& Contini (1966) segment proportions: shoulder width 0.259 of stature, hip joint to shoulder joint
0.288 of stature. The person's left shoulder appears to the right of their right shoulder when the
camera is toward their front, and to the left when it is toward their back.

The correction is applied only when the recording is in the side or oblique band of the
`camera_view` quality check; a frontal recording fails that check and is left alone. A true side
view gives cos θ = 1 and the identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np

from ptvision.kinematics.angles import facing_direction
from ptvision.pose.track import PoseTrack

ViewMethod = Literal["shoulders", "none"]
ViewModel = Literal["orthographic", "perspective"]

# Drillis & Contini (1966), body segment lengths
SHOULDER_WIDTH_FRAC = 0.259
HIP_TO_SHOULDER_FRAC = 0.288
SHOULDER_TO_TRUNK_RATIO = SHOULDER_WIDTH_FRAC / HIP_TO_SHOULDER_FRAC
# view_ratio bands of the camera_view quality check: a pure side view gives about 0.1-0.3
SAGITTAL_MAX_RATIO = 0.35
OBLIQUE_MAX_RATIO = 0.50
MIN_COS_THETA = 0.5  # 60° off the side


@dataclass(frozen=True)
class ViewEstimate:
    method: ViewMethod
    cos_theta: float  # 1.0 = true side view
    measured: float | None  # the ratio the estimate came from
    applied: bool
    note: str = ""
    toward_front: bool | None = None  # camera toward the person's front (True) or back (False)

    @property
    def theta_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.cos_theta, -1.0, 1.0))))

    @property
    def yaw_deg(self) -> float:
        """Signed angle off the side view, positive toward the person's front (0 if unknown)."""
        sign = 0 if self.toward_front is None else 1 if self.toward_front else -1
        return sign * self.theta_deg

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = asdict(self)
        d["theta_deg"] = round(self.theta_deg, 1)
        return d


def _signed_shoulder_ratio(track: PoseTrack, person: int) -> np.ndarray | None:
    """Per frame: left shoulder x minus right shoulder x, over the trunk length."""
    lay = track.layout
    if not all(lay.has(n) for n in ("LShoulder", "RShoulder", "Neck", "Hip")):
        return None
    ls, rs = track.keypoint("LShoulder", person), track.keypoint("RShoulder", person)
    nk, hp = track.keypoint("Neck", person), track.keypoint("Hip", person)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (ls[:, 0] - rs[:, 0]) / np.linalg.norm(nk - hp, axis=1)
    return ratio if np.isfinite(ratio).any() else None


def view_ratio(track: PoseTrack, person: int = 0) -> float | None:
    """Median shoulder separation over trunk length, the `camera_view` check's measure."""
    ratio = _signed_shoulder_ratio(track, person)
    return None if ratio is None else float(np.nanmedian(np.abs(ratio)))


def estimate_view_shoulders(track: PoseTrack, person: int = 0) -> ViewEstimate:
    """cos θ from the shoulder separation, via the Drillis & Contini breadth-to-trunk ratio, and
    the camera's side from which shoulder appears on the right."""
    signed = _signed_shoulder_ratio(track, person)
    if signed is None:
        return ViewEstimate("shoulders", 1.0, None, False, "shoulders or trunk not tracked")
    ratio = float(np.nanmedian(np.abs(signed)))
    lead = float(np.nanmedian(signed))
    front = None if lead == 0 else lead > 0
    sin_theta = min(1.0, ratio / SHOULDER_TO_TRUNK_RATIO)
    cos_theta = float(np.sqrt(max(0.0, 1.0 - sin_theta**2)))
    if ratio >= OBLIQUE_MAX_RATIO or cos_theta < MIN_COS_THETA:
        return ViewEstimate(
            "shoulders", cos_theta, ratio, False, "view is not side or oblique", front
        )
    return ViewEstimate("shoulders", cos_theta, ratio, True, toward_front=front)


def estimate_view(track: PoseTrack, method: ViewMethod) -> ViewEstimate:
    if method == "shoulders":
        return estimate_view_shoulders(track)
    return ViewEstimate("none", 1.0, None, False, "correction off")


def _with_coords(track: PoseTrack, coords: np.ndarray) -> PoseTrack:
    return replace(
        track,
        coords=coords,
        score=track.score.copy(),
        person_ids=track.person_ids.copy(),
        stage=f"{track.stage}+view_rectified",
        extra_metadata=dict(track.extra_metadata),
    )


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
    return _with_coords(track, coords)


# [REVIEW] perspective view correction: virtual camera rotation onto the sagittal plane
def plane_rotation(up: np.ndarray, lateral: np.ndarray) -> np.ndarray:
    """Rotation from camera coordinates (x right, y down, z forward) to a virtual camera at the
    same place that is level and looks along the person's left-right axis, from the real camera's
    side so the near side stays near and the facing direction is kept. ``up`` is the world
    vertical and ``lateral`` the left-right axis, both in camera coordinates."""
    y = -np.asarray(up, dtype=np.float64)
    y /= np.linalg.norm(y)
    z = np.asarray(lateral, dtype=np.float64)
    z = z - np.dot(z, y) * y
    z /= np.linalg.norm(z)
    if z[2] < 0:
        z = -z
    return np.stack([np.cross(y, z), y, z])


def level_camera_axes(
    yaw_deg: float, facing: int, pitch_deg: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """World vertical and the person's left-right axis in the coordinates of a camera
    ``yaw_deg`` off the side view (positive toward the person's front) and tilted down by
    ``pitch_deg``, filming a person who faces image-right (+1) or image-left (-1)."""
    g, p = np.radians(yaw_deg * facing), np.radians(pitch_deg)
    up = np.array([0.0, -np.cos(p), -np.sin(p)])
    lateral = np.array([np.sin(g), -np.cos(g) * np.sin(p), np.cos(g) * np.cos(p)])
    return up, lateral


def rectify_perspective(
    track: PoseTrack, rotation: np.ndarray, focal_px: float, center: tuple[float, float]
) -> PoseTrack:
    """Move every keypoint to where a pinhole camera with focal length ``focal_px`` and principal
    point ``center``, turned by ``rotation`` about its own centre, would have seen it (no lens
    distortion). The result is shifted so the principal point stays put; angles do not depend on
    the shift."""
    k = np.array([[focal_px, 0.0, center[0]], [0.0, focal_px, center[1]], [0.0, 0.0, 1.0]])
    h = k @ rotation @ np.linalg.inv(k)
    xy = track.coords.astype(np.float64)
    pts = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], axis=-1) @ h.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pts[..., :2] / pts[..., 2:]
    out[pts[..., 2] <= 0] = np.nan  # behind the virtual camera
    c = h @ np.array([center[0], center[1], 1.0])
    out += np.asarray(center) - c[:2] / c[2]
    return _with_coords(track, out.astype(track.coords.dtype))


def focal_px_from_35mm(focal_35mm: float, width: int, height: int) -> float:
    """Focal length in pixels from a 35 mm-equivalent focal length, taking the frame's long side
    as the 36 mm width of a 35 mm frame. Nominal: stabilisation and the video crop can change the
    real value by several percent, which a checkerboard clip from the phone would measure."""
    return focal_35mm * max(width, height) / 36.0


@dataclass(frozen=True)
class ViewCorrection:
    """What is done to the keypoints before the angles: the estimate and the camera model."""

    estimate: ViewEstimate
    model: ViewModel = "orthographic"
    focal_px: float | None = None  # perspective only
    center: tuple[float, float] | None = None
    facing: int = 1
    pitch_deg: float = 0.0
    note: str = ""

    @property
    def applied(self) -> bool:
        return self.estimate.applied

    def rotation(self) -> np.ndarray:
        up, lateral = level_camera_axes(self.estimate.yaw_deg, self.facing, self.pitch_deg)
        return plane_rotation(up, lateral)

    def apply(self, track: PoseTrack) -> PoseTrack:
        if not self.applied:
            return track
        if self.model == "perspective" and self.focal_px and self.center:
            return rectify_perspective(track, self.rotation(), self.focal_px, self.center)
        return rectify_sagittal(track, self.estimate.cos_theta)

    def to_dict(self) -> dict[str, object]:
        d = self.estimate.to_dict()
        d.update(
            model=self.model,
            yaw_deg=round(self.estimate.yaw_deg, 1),
            focal_px=None if self.focal_px is None else round(self.focal_px, 1),
            pitch_deg=self.pitch_deg,
        )
        if self.note:
            d["note"] = self.note
        return d


def view_correction(
    track: PoseTrack,
    method: ViewMethod,
    model: ViewModel = "orthographic",
    *,
    focal_px: float | None = None,
    center: tuple[float, float] | None = None,
    pitch_deg: float = 0.0,
) -> ViewCorrection:
    """The protocol's view correction for one person's filtered track. The perspective model
    falls back to the orthographic one when the focal length or the camera's side is unknown."""
    est = estimate_view(track, method)
    if model == "orthographic" or not est.applied:
        return ViewCorrection(est)
    if not focal_px or center is None:
        return ViewCorrection(est, note="focal length unknown, stretched horizontally instead")
    if est.toward_front is None:
        return ViewCorrection(est, note="camera side unknown, stretched horizontally instead")
    facing = facing_direction(track) or 1
    return ViewCorrection(est, "perspective", focal_px, center, facing, pitch_deg)
