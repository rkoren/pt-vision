"""Camera-view correction: geometry of the orthographic and perspective rectifications, the
shoulder estimator, the guards, and the record the pipeline keeps"""

from __future__ import annotations

import numpy as np
import pytest

from ptvision.kinematics.angles import compute_angles, facing_direction, resolve_side
from ptvision.kinematics.view import (
    ViewEstimate,
    estimate_view,
    estimate_view_shoulders,
    focal_px_from_35mm,
    level_camera_axes,
    plane_rotation,
    rectify_perspective,
    rectify_sagittal,
    view_correction,
    view_ratio,
)
from ptvision.pose.layout import HALPE26
from ptvision.pose.track import PoseTrack

FPS = 30.0
THIGH, SHANK, TRUNK = 0.42, 0.42, 0.52  # m
HIP_HALF_WIDTH, SHOULDER_HALF_WIDTH = 0.09, 0.19
PX_PER_M = 400.0


def rigid_sts_track(azimuth_deg: float, *, seconds: float = 6.0) -> tuple[PoseTrack, np.ndarray]:
    """A rigid-limbed sit-to-stand in the sagittal plane, filmed by an orthographic camera
    `azimuth_deg` off the side. Returns the track and the true thigh inclination (deg)."""
    n = int(seconds * FPS)
    t = np.linspace(0, 1, n)
    # thigh from horizontal (seated, 90°) to vertical (standing, 0°) and back, twice
    phi = np.radians(90.0 * (0.5 + 0.5 * np.cos(2 * np.pi * 2 * t)))
    lean = np.radians(25.0 * np.sin(phi))  # trunk leans forward as the thigh comes up
    c, s = np.cos(np.radians(azimuth_deg)), np.sin(np.radians(azimuth_deg))

    def img(x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
        # x forward, y up, z lateral (towards the camera when the camera is off the side)
        px = (x * c + z * s + 1.5) * PX_PER_M
        py = (2.2 - y) * PX_PER_M
        return np.stack([px, py], axis=1)

    knee = np.stack([np.zeros(n), np.full(n, SHANK)], axis=1)  # knee fixed above the ankle
    hip = knee + THIGH * np.stack([-np.sin(phi), np.cos(phi)], axis=1)
    ankle = np.zeros((n, 2))
    neck = hip + TRUNK * np.stack([np.sin(lean), np.cos(lean)], axis=1)
    coords = np.full((n, 1, HALPE26.n, 2), np.nan, dtype=np.float32)
    score = np.zeros((n, 1, HALPE26.n), dtype=np.float32)

    def put(name: str, p: np.ndarray, z: float, sc: float = 0.9) -> None:
        coords[:, 0, HALPE26.index(name)] = img(p[:, 0], p[:, 1], z)
        score[:, 0, HALPE26.index(name)] = sc

    for side, z in (("R", +1.0), ("L", -1.0)):
        put(f"{side}Hip", hip, z * HIP_HALF_WIDTH, 0.9 if side == "R" else 0.7)
        put(f"{side}Knee", knee, z * HIP_HALF_WIDTH, 0.9 if side == "R" else 0.7)
        put(f"{side}Ankle", ankle, z * HIP_HALF_WIDTH, 0.9 if side == "R" else 0.7)
        put(f"{side}Shoulder", neck, z * SHOULDER_HALF_WIDTH)
        put(f"{side}Heel", ankle + np.array([-0.05, -0.05]), z * HIP_HALF_WIDTH)
        put(f"{side}BigToe", ankle + np.array([0.15, -0.05]), z * HIP_HALF_WIDTH)
    put("Hip", hip, 0.0)
    put("Neck", neck, 0.0)
    track = PoseTrack(HALPE26, FPS, coords, score, np.array([0], dtype=np.int16))
    return track, np.degrees(phi)


@pytest.mark.parametrize("azimuth", [0.0, 20.0, 45.0])
def test_rectification_recovers_sagittal_angles(azimuth: float) -> None:
    side_view, _ = rigid_sts_track(0.0)
    oblique, _ = rigid_sts_track(azimuth)
    truth = compute_angles(side_view, side="right")
    cos_theta = float(np.cos(np.radians(azimuth)))
    rect = rectify_sagittal(oblique, cos_theta)
    got = compute_angles(rect, side="right")
    # segments on one side of the body (thigh, shank) and on the midline (trunk) come back exactly
    for name in ("knee_flexion", "trunk_lean"):
        assert np.nanmax(np.abs(got[name] - truth[name])) < 1.0, name
    # the hip angle joins the midline neck to a hip 9 cm off the midline; that lateral offset
    # projects to a horizontal shift the rectification cannot know about, bounded by
    # atan(w sin θ / (T cos θ)) with the trunk length T
    bound = np.degrees(
        np.arctan(HIP_HALF_WIDTH * np.sin(np.radians(azimuth)) / (TRUNK * cos_theta))
    )
    assert np.nanmax(np.abs(got["hip_flexion"] - truth["hip_flexion"])) < bound + 1.0
    if azimuth > 0:  # the uncorrected oblique view is wrong by more than that
        before = compute_angles(oblique, side="right")
        assert np.nanmax(np.abs(before["knee_flexion"] - truth["knee_flexion"])) > 1.5
        assert np.nanmax(np.abs(before["trunk_lean"] - truth["trunk_lean"])) > 1.0


def test_rectification_identity_and_copy() -> None:
    track, _ = rigid_sts_track(20.0)
    same = rectify_sagittal(track, 1.0)
    assert same is not track
    np.testing.assert_array_equal(same.coords, track.coords)
    assert same.stage.endswith("+view_rectified")


@pytest.mark.parametrize("azimuth", [20.0, 45.0])
def test_shoulder_estimator_is_monotonic_and_in_range(azimuth: float) -> None:
    # the synthetic body is not anthropometric, so only the direction and bounds are checked
    est = estimate_view_shoulders(rigid_sts_track(azimuth)[0])
    assert est.method == "shoulders" and est.measured is not None
    assert 0.0 < est.cos_theta <= 1.0
    side = estimate_view_shoulders(rigid_sts_track(0.0)[0])
    assert side.cos_theta == pytest.approx(1.0, abs=1e-6) and side.applied
    assert est.cos_theta < side.cos_theta


def test_frontal_view_is_left_alone() -> None:
    frontal, _ = rigid_sts_track(85.0)
    assert (view_ratio(frontal) or 0.0) > 0.5
    est = estimate_view_shoulders(frontal)
    assert not est.applied and est.note


def test_estimate_view_dispatch() -> None:
    track, _ = rigid_sts_track(20.0)
    off = estimate_view(track, "none")
    assert off == ViewEstimate("none", 1.0, None, False, "correction off")
    assert estimate_view(track, "shoulders").applied
    d = estimate_view(track, "shoulders").to_dict()
    assert set(d) >= {"method", "cos_theta", "applied", "theta_deg", "toward_front"}


def test_projection_azimuth_matches_the_model() -> None:
    """A marker trial projected by a virtual camera off the side shows the same foreshortening
    the correction assumes: stretching by the camera's 1 / cos θ brings the knee angle back to
    the side-view projection."""
    from pathlib import Path

    from ptvision.datasets.markers import MarkerTrial
    from ptvision.datasets.mocap_projection import project_trial

    track, _ = rigid_sts_track(0.0)
    names = {
        "R_HJC": "RHip",
        "L_HJC": "LHip",
        "r_knee": "RKnee",
        "L_knee": "LKnee",
        "r_ankle": "RAnkle",
        "L_ankle": "LAnkle",
        "r_calc": "RHeel",
        "L_calc": "LHeel",
        "R_Shoulder": "RShoulder",
        "L_Shoulder": "LShoulder",
        "C7": "Neck",
    }
    labels = list(names)
    pts = np.zeros((track.n_frames, len(labels), 3))
    for k, (lab, kp) in enumerate(names.items()):
        c = track.keypoint(kp).astype(np.float64)
        pts[:, k, 0] = c[:, 0] / PX_PER_M
        pts[:, k, 2] = -c[:, 1] / PX_PER_M
        pts[:, k, 1] = 0.1 if lab.lower().startswith(("r", "r_")) else -0.1
    for k, lab in enumerate(labels):
        if lab == "C7":
            pts[:, k, 1] = 0.0
    trial = MarkerTrial(Path("synthetic"), FPS, labels, pts, 0, "m")
    side = project_trial(trial, fps=FPS, forward_rule="lean")
    oblique = project_trial(trial, fps=FPS, forward_rule="lean", azimuth_deg=30.0)
    truth = compute_angles(side.track, ["knee_flexion"], side="right")["knee_flexion"]
    before = compute_angles(oblique.track, ["knee_flexion"], side="right")["knee_flexion"]
    after = compute_angles(
        rectify_sagittal(oblique.track, float(np.cos(np.radians(30.0)))),
        ["knee_flexion"],
        side="right",
    )["knee_flexion"]
    n = min(len(truth), len(after))
    assert np.nanmax(np.abs(before[:n] - truth[:n])) > 2.0
    assert np.nanmax(np.abs(after[:n] - truth[:n])) < 1.5


# ---------------------------------------------------------------------------------------------
# Perspective: a pinhole camera above the seated hip, off the side, tilted and rolled
# ---------------------------------------------------------------------------------------------
IMAGE = (1920, 1080)
FOCAL = 1000.0


def pinhole_sts(
    yaw_deg: float,
    *,
    pitch_deg: float = 8.0,
    roll_deg: float = 2.0,
    right_side: bool = True,
    height: float = 1.2,
    distance: float = 3.0,
) -> tuple[PoseTrack, PoseTrack, np.ndarray, np.ndarray, np.ndarray]:
    """The rigid sit-to-stand of `rigid_sts_track` in 3D (x forward, y to the person's left, z
    up), filmed by a pinhole camera on the person's right or left, `yaw_deg` off the side view
    (positive toward the front), `height` m up and `distance` m away, tilted down and rolled.
    Returns the camera's track, the orthographic side view (the truth), the world vertical and
    left-right axis in camera coordinates, and the true thigh inclination (deg)."""
    side_view, phi = rigid_sts_track(0.0)
    th, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    s = -1.0 if right_side else 1.0
    centre = np.array([distance * np.sin(th), s * distance * np.cos(th), height])
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.cos(p) * np.array([-np.sin(th), -s * np.cos(th), 0.0]) - np.sin(p) * up
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    right, down = np.cos(r) * right + np.sin(r) * down, -np.sin(r) * right + np.cos(r) * down
    axes = np.stack([right, down, fwd])  # camera axes in world coordinates

    coords = np.full_like(side_view.coords, np.nan)
    for j, name in enumerate(HALPE26.names):
        xy = side_view.coords[:, 0, j].astype(np.float64)
        if np.isnan(xy).all():
            continue
        lateral = {"R": -1.0, "L": 1.0}.get(name[0], 0.0)  # midline names start otherwise
        half = SHOULDER_HALF_WIDTH if name.endswith("Shoulder") else HIP_HALF_WIDTH
        world = np.stack(
            [
                xy[:, 0] / PX_PER_M - 1.5,
                np.full(len(xy), lateral * half),
                2.2 - xy[:, 1] / PX_PER_M,
            ],
            axis=1,
        )
        cam = (world - centre) @ axes.T
        coords[:, 0, j, 0] = FOCAL * cam[:, 0] / cam[:, 2] + IMAGE[0] / 2
        coords[:, 0, j, 1] = FOCAL * cam[:, 1] / cam[:, 2] + IMAGE[1] / 2
    track = PoseTrack(
        HALPE26, FPS, coords, side_view.score.copy(), side_view.person_ids, image_size=IMAGE
    )
    return track, side_view, axes @ up, axes @ np.array([0.0, 1.0, 0.0]), phi


def _max_err(got: object, truth: object, name: str, mask: np.ndarray | None = None) -> float:
    e = np.abs(got[name] - truth[name])  # type: ignore[index]
    return float(np.nanmax(e if mask is None else e[mask]))


def _perspective(track: PoseTrack, rotation: np.ndarray, focal: float = FOCAL) -> PoseTrack:
    return rectify_perspective(track, rotation, focal, (IMAGE[0] / 2, IMAGE[1] / 2))


@pytest.mark.parametrize("right_side", [True, False])
@pytest.mark.parametrize("yaw", [25.0, -25.0])
def test_perspective_rectification_recovers_the_side_view(yaw: float, right_side: bool) -> None:
    track, side_view, up, lateral, phi = pinhole_sts(yaw, right_side=right_side)
    truth = compute_angles(side_view, side="right")
    rect = _perspective(track, plane_rotation(up, lateral))
    got = compute_angles(rect, side="near")
    # the virtual camera is not mirrored: same leg, same facing
    assert got.side == ("right" if right_side else "left") == resolve_side(track, "near")
    assert facing_direction(rect) == facing_direction(track) == (1 if right_side else -1)
    for name in ("knee_flexion", "trunk_lean"):
        assert _max_err(got, truth, name) < 0.2, name
    # the hip joins the midline neck to a hip 9 cm off it, a depth offset no rotation removes,
    # bounded by atan(w tan θ / T) as for the orthographic stretch
    bound = np.degrees(np.arctan(HIP_HALF_WIDTH * np.tan(np.radians(abs(yaw))) / TRUNK))
    assert _max_err(got, truth, "hip_flexion") < bound + 0.5
    # a camera above the seated thigh tilts it: the uncorrected knee and the stretch are both off
    seated = phi > 60
    before = compute_angles(track, side="near")
    stretched = compute_angles(rectify_sagittal(track, float(np.cos(np.radians(yaw)))))
    assert _max_err(before, truth, "knee_flexion") > 5.0
    assert _max_err(stretched, truth, "knee_flexion", seated) > 4.0


@pytest.mark.parametrize("right_side", [True, False])
@pytest.mark.parametrize("yaw", [25.0, -25.0])
def test_two_segment_hip_has_no_midline_offset(yaw: float, right_side: bool) -> None:
    """The hip's remaining error under the perspective model is the neck-to-near-hip offset: a
    hip angle between the trunk from the pelvis midpoint and the near thigh is exact."""
    track, side_view, up, lateral, _ = pinhole_sts(yaw, right_side=right_side)
    rect = _perspective(track, plane_rotation(up, lateral))
    p = "R" if right_side else "L"
    trunk = rect.keypoint("Neck") - rect.keypoint("Hip")
    thigh = rect.keypoint(f"{p}Knee") - rect.keypoint(f"{p}Hip")
    cos = np.sum(trunk * thigh, axis=1) / (
        np.linalg.norm(trunk, axis=1) * np.linalg.norm(thigh, axis=1)
    )
    hip = 180.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    truth = compute_angles(side_view, ["hip_flexion"], side="right")["hip_flexion"]
    assert np.nanmax(np.abs(hip - truth)) < 0.2


@pytest.mark.parametrize("right_side", [True, False])
@pytest.mark.parametrize("yaw", [25.0, -25.0])
def test_level_camera_axes_match_the_camera(yaw: float, right_side: bool) -> None:
    track, _, up, lateral, _ = pinhole_sts(yaw, roll_deg=0.0, right_side=right_side)
    model = plane_rotation(*level_camera_axes(yaw, facing_direction(track), pitch_deg=8.0))
    np.testing.assert_allclose(model, plane_rotation(up, lateral), atol=1e-9)
    # and the shoulders say which way the camera is turned
    assert estimate_view_shoulders(track).toward_front == (yaw > 0)


def test_perspective_sensitivity_to_its_inputs() -> None:
    """How exact the inputs must be: the focal length matters little, the camera's side a lot."""
    track, side_view, _, _, _ = pinhole_sts(25.0, roll_deg=0.0)
    truth = compute_angles(side_view, side="right")
    facing = facing_direction(track)
    raw = _max_err(compute_angles(track), truth, "knee_flexion")
    right = plane_rotation(*level_camera_axes(25.0, facing, pitch_deg=8.0))
    for scale in (0.85, 1.15):
        got = compute_angles(_perspective(track, right, FOCAL * scale))
        assert _max_err(got, truth, "knee_flexion") < 1.0
        assert _max_err(got, truth, "trunk_lean") < 1.0
    wrong_side = plane_rotation(*level_camera_axes(-25.0, facing, pitch_deg=8.0))
    assert _max_err(compute_angles(_perspective(track, wrong_side)), truth, "knee_flexion") > raw
    # taking the 8° tilted camera as level still removes more than half of the knee error
    level = plane_rotation(*level_camera_axes(25.0, facing, pitch_deg=0.0))
    assert _max_err(compute_angles(_perspective(track, level)), truth, "knee_flexion") < raw / 2


def test_view_correction_model_and_fallbacks() -> None:
    track, _, _, _, _ = pinhole_sts(25.0, roll_deg=0.0)
    centre = (IMAGE[0] / 2, IMAGE[1] / 2)
    ortho = view_correction(track, "shoulders")
    assert ortho.model == "orthographic" and ortho.applied
    persp = view_correction(track, "shoulders", "perspective", focal_px=FOCAL, center=centre)
    assert persp.model == "perspective" and persp.facing == 1
    assert persp.estimate.yaw_deg > 0
    assert persp.apply(track).stage.endswith("+view_rectified")
    no_focal = view_correction(track, "shoulders", "perspective", center=centre)
    assert no_focal.model == "orthographic" and "focal" in no_focal.note
    d = persp.to_dict()
    assert d["model"] == "perspective" and d["focal_px"] == FOCAL and "theta_deg" in d
    off = view_correction(track, "none", "perspective", focal_px=FOCAL, center=centre)
    assert not off.applied and off.apply(track) is track


def test_focal_px_from_35mm() -> None:
    # the long side stands for the 36 mm frame width, whichever way the phone was held
    assert focal_px_from_35mm(24.0, 1920, 1080) == pytest.approx(1280.0)
    assert focal_px_from_35mm(24.0, 1080, 1920) == pytest.approx(1280.0)
