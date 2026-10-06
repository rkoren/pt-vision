import numpy as np
import pytest

from ptvision.kinematics.angles import (
    compute_angles,
    facing_direction,
    interior_angle,
    segment_angle_from_vertical,
)
from ptvision.pose.layout import HALPE26
from ptvision.pose.track import PoseTrack
from tests.synthetic import synthetic_sts_track


def test_interior_angle_basic() -> None:
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    c = np.array([[1.0, 1.0]])
    assert interior_angle(a, b, c)[0] == pytest.approx(90.0)
    c2 = np.array([[2.0, 0.0]])
    assert interior_angle(a, b, c2)[0] == pytest.approx(180.0)


def test_segment_angle_from_vertical_sign() -> None:
    base = np.array([[0.0, 100.0]])
    tip_up = np.array([[0.0, 0.0]])
    tip_right = np.array([[50.0, 13.4]])
    assert segment_angle_from_vertical(base, tip_up)[0] == pytest.approx(0.0)
    assert segment_angle_from_vertical(base, tip_right)[0] == pytest.approx(30.0, abs=0.5)


def _single(points: dict[str, tuple[float, float]]) -> PoseTrack:
    coords = np.full((1, 1, HALPE26.n, 2), np.nan, np.float32)
    score = np.zeros((1, 1, HALPE26.n), np.float32)
    for k, (x, y) in points.items():
        coords[0, 0, HALPE26.index(k)] = (x, y)
        score[0, 0, HALPE26.index(k)] = 0.9
    return PoseTrack(HALPE26, 30.0, coords, score, np.array([0]))


def test_knee_and_hip_flexion_conventions() -> None:
    # standing straight, facing right: hip above knee above ankle, neck above hip
    tr = _single(
        {
            "RHip": (100, 200),
            "RKnee": (100, 300),
            "RAnkle": (100, 400),
            "Neck": (100, 50),
            "Hip": (100, 200),
            "RBigToe": (130, 405),
            "RHeel": (90, 405),
        }
    )
    a = compute_angles(tr, ["knee_flexion", "hip_flexion", "trunk_lean"], side="right")
    assert a["knee_flexion"][0] == pytest.approx(0.0, abs=1e-4)
    assert a["hip_flexion"][0] == pytest.approx(0.0, abs=1e-4)
    assert a["trunk_lean"][0] == pytest.approx(0.0, abs=1e-4)
    # seated: thigh horizontal forward (+x), shank vertical -> knee 90, hip 90; trunk leaning forward 20 deg
    tr2 = _single(
        {
            "RHip": (100, 300),
            "RKnee": (200, 300),
            "RAnkle": (200, 400),
            "Neck": (100 + 150 * np.sin(np.radians(20)), 300 - 150 * np.cos(np.radians(20))),
            "Hip": (100, 300),
            "RBigToe": (230, 405),
            "RHeel": (190, 405),
        }
    )
    a2 = compute_angles(tr2, ["knee_flexion", "hip_flexion", "trunk_lean"], side="right")
    assert a2["knee_flexion"][0] == pytest.approx(90.0, abs=1e-3)
    assert a2["hip_flexion"][0] == pytest.approx(110.0, abs=1e-3)  # 90 plus 20 deg of trunk lean
    assert a2["trunk_lean"][0] == pytest.approx(20.0, abs=1e-3)
    assert a2.facing == 1


def test_trunk_lean_sign_follows_facing() -> None:
    # facing left: toes at smaller x than heels; neck displaced to -x => forward lean positive
    tr = _single(
        {
            "Hip": (100, 300),
            "Neck": (100 - 50, 300 - 140),
            "LBigToe": (60, 405),
            "LHeel": (110, 405),
            "LHip": (100, 300),
            "LKnee": (60, 300),
            "LAnkle": (60, 400),
        }
    )
    assert facing_direction(tr) == -1
    a = compute_angles(tr, ["trunk_lean"], side="left")
    assert a["trunk_lean"][0] > 0


def test_near_side_selection_and_synthetic_ranges() -> None:
    track, _tl = synthetic_sts_track()
    a = compute_angles(track, ["knee_flexion", "hip_flexion", "trunk_lean"], side="near")
    assert a.side in ("left", "right")
    k = a["knee_flexion"]
    assert k.max() > 60 and k.min() < 15  # seated ~ 90, standing ~ 0
    assert a["trunk_lean"].max() > 20


def test_angle_series_roundtrip_and_legacy(tmp_path) -> None:
    from ptvision.kinematics.angles import AngleSeries

    track, _tl = synthetic_sts_track()
    a = compute_angles(track, ["knee_flexion", "trunk_lean"], side="left")
    p = a.save(tmp_path / "angles.parquet")
    b = AngleSeries.load(p)
    assert b.side == "left" and b.facing == a.facing and b.fps == a.fps
    assert b.names == ["knee_flexion", "trunk_lean"]
    np.testing.assert_allclose(b["knee_flexion"], a["knee_flexion"])
    assert set(b.definitions) == {"knee_flexion", "trunk_lean"}
    # legacy file without metadata
    a.frame.to_parquet(tmp_path / "legacy.parquet")
    with pytest.raises(ValueError):
        AngleSeries.load(tmp_path / "legacy.parquet")
    c = AngleSeries.load(tmp_path / "legacy.parquet", side_fallback="right", fps_fallback=30.0)
    assert c.side == "right" and c.fps == 30.0
