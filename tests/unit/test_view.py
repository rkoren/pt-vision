"""Camera-view correction: geometry of the rectification, the two cos θ estimators, the guards,
and the record the pipeline keeps"""

from __future__ import annotations

import numpy as np
import pytest

from ptvision.kinematics.angles import compute_angles
from ptvision.kinematics.view import (
    ViewEstimate,
    estimate_view,
    estimate_view_shoulders,
    estimate_view_thigh_length,
    rectify_sagittal,
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


@pytest.mark.parametrize("azimuth", [0.0, 20.0, 45.0])
def test_thigh_length_estimator_recovers_cos_theta(azimuth: float) -> None:
    track, _ = rigid_sts_track(azimuth)
    est = estimate_view_thigh_length(track, "right")
    assert est.method == "thigh_length"
    assert est.cos_theta == pytest.approx(np.cos(np.radians(azimuth)), abs=0.03)
    # applied only inside the side/oblique band of the camera-view check; this body's shoulders
    # read as frontal at 45°, exactly as the 45° OpenCap cameras do
    assert est.applied == (azimuth <= 20.0)


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
    for est in (estimate_view_thigh_length(frontal, "right"), estimate_view_shoulders(frontal)):
        assert not est.applied and est.note


def test_estimate_view_dispatch() -> None:
    track, _ = rigid_sts_track(20.0)
    off = estimate_view(track, "none")
    assert off == ViewEstimate("none", 1.0, None, False, "correction off")
    assert estimate_view(track, "thigh_length", "right").applied
    assert estimate_view(track, "shoulders").applied
    d = estimate_view(track, "thigh_length", "right").to_dict()
    assert set(d) >= {"method", "cos_theta", "applied", "theta_deg"}


def test_projection_azimuth_matches_the_model() -> None:
    """A marker trial projected by a virtual camera off the side shows the same foreshortening
    the correction assumes: the thigh-length estimator reads the azimuth back and the
    rectified knee angle matches the side-view projection."""
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
    est = estimate_view_thigh_length(oblique.track, "right")
    assert est.cos_theta == pytest.approx(np.cos(np.radians(30.0)), abs=0.03)
    truth = compute_angles(side.track, ["knee_flexion"], side="right")["knee_flexion"]
    before = compute_angles(oblique.track, ["knee_flexion"], side="right")["knee_flexion"]
    after = compute_angles(
        rectify_sagittal(oblique.track, est.cos_theta), ["knee_flexion"], side="right"
    )["knee_flexion"]
    n = min(len(truth), len(after))
    assert np.nanmax(np.abs(before[:n] - truth[:n])) > 2.0
    assert np.nanmax(np.abs(after[:n] - truth[:n])) < 1.5
