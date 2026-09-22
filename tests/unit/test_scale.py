import numpy as np
import pytest

from ptvision.kinematics.scale import (
    build_scale,
    estimate_floor,
    estimate_height_px,
    trimmed_mean,
    walking_direction,
)
from ptvision.pose.track import PoseTrack
from tests.synthetic import GaitTimeline, synthetic_gait_track


def test_trimmed_mean() -> None:
    assert trimmed_mean(np.array([1, 2, 3, 4, 100.0]), 50) == pytest.approx(
        2.5
    )  # keeps the middle half
    assert np.isnan(trimmed_mean(np.array([np.nan])))


def test_height_and_scale_consistent() -> None:
    track, tl = synthetic_gait_track()
    h_px, n_used, _notes = estimate_height_px(track)
    # synthetic body: foot + shank + thigh + trunk + head, sum of the segment lengths used
    assert 400 < h_px < 560 and n_used > 20
    sm = build_scale(track, fps=tl.fps, height_m=1.75)
    assert sm.px_per_m == pytest.approx(h_px / 1.75)
    assert abs(np.degrees(sm.floor_angle_rad)) < 0.5
    # a known pixel distance converts through px_per_m
    assert sm.length_m(sm.px_per_m) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        build_scale(track, fps=tl.fps, height_m=3.0)


def test_floor_angle_recovered_after_rotation() -> None:
    track, tl = synthetic_gait_track()
    deg = 3.0
    w, h = tl.image_size
    theta = np.radians(deg)
    c = track.coords[:, 0].astype(np.float64)
    x = c[..., 0] - w / 2
    y = c[..., 1] - h / 2
    # rotate the image content (y down); a floor sloping down to the right in image coords
    xr = x * np.cos(theta) - y * np.sin(theta) + w / 2
    yr = x * np.sin(theta) + y * np.cos(theta) + h / 2
    rot = PoseTrack(
        track.layout,
        track.fps,
        np.stack([xr, yr], -1)[:, None].astype(np.float32),
        track.score,
        track.person_ids,
        image_size=tl.image_size,
    )
    angle, _origin, n_pts, _notes = estimate_floor(rot, fps=tl.fps, px_per_m=270.0)
    assert n_pts > 50
    assert np.degrees(angle) == pytest.approx(-deg, abs=0.4)
    sm = build_scale(rot, fps=tl.fps, height_m=1.75)
    # after correction the stationary heel keeps a constant floor height (Y up)
    heel = sm.to_meters(rot.keypoint("LHeel"))
    stance_frames = [f for f in range(tl.hs["left"][1], tl.to["left"][1])]
    ys = heel[stance_frames, 1]
    assert np.nanstd(ys) < 0.01


def test_walking_direction() -> None:
    track, _ = synthetic_gait_track()
    assert walking_direction(track) == 1
    track_l, _ = synthetic_gait_track(GaitTimeline(direction=-1))
    assert walking_direction(track_l) == -1
