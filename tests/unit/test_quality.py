import numpy as np
import pytest

from ptvision.clinical.protocol import load_protocol
from ptvision.pose.track import PoseTrack
from ptvision.quality import checks as Q
from ptvision.trials.models import CameraCapture
from tests.synthetic import synthetic_sts_track


def _cam(**kw) -> CameraCapture:  # type: ignore[no-untyped-def]
    base = dict(
        camera_id="cam0",
        source_file="x.mov",
        source_sha256="0" * 64,
        source_codec="hevc",
        source_pix_fmt="yuv420p10le",
        source_width=1920,
        source_height=1080,
        source_fps_avg=29.97,
        source_fps_nominal=30,
        source_is_vfr=False,
        rotation_applied_deg=0,
        normalized_file="video/cam0.mp4",
        width=1920,
        height=1080,
        fps=29.97,
        n_frames=400,
        duration_s=13.3,
    )
    base.update(kw)
    return CameraCapture(**base)  # type: ignore[arg-type]


def test_capture_checks() -> None:
    p = load_protocol("sts_5x")
    ok = Q.combine(Q.check_capture(_cam(), p))
    assert ok.status == "pass"
    low = Q.combine(Q.check_capture(_cam(fps=15.0, source_is_vfr=True), p))
    assert low.status == "fail"
    assert {c.name for c in low.failures} == {"frame_rate"}
    assert any(c.name == "variable_frame_rate" for c in low.warnings)


def test_track_checks_sagittal_pass_and_frontal_fail() -> None:
    p = load_protocol("sts_5x")
    track, _ = synthetic_sts_track()
    rep = Q.combine(Q.check_track(track, p))
    assert rep.status == "pass", [c.message for c in rep.checks if c.status != "pass"]
    # frontal: pull shoulders far apart
    frontal = track.coords.copy()
    frontal[:, 0, track.layout.index("LShoulder"), 0] -= 120
    frontal[:, 0, track.layout.index("RShoulder"), 0] += 120
    track.coords = frontal
    rep2 = Q.combine(Q.check_track(track, p))
    assert rep2.status == "fail"
    assert any(c.name == "camera_view" for c in rep2.failures)


def test_track_checks_oblique_view_warns_not_fails() -> None:
    p = load_protocol("sts_5x")
    track, _ = synthetic_sts_track()
    lay = track.layout
    trunk = np.nanmedian(
        np.linalg.norm(
            track.coords[:, 0, lay.index("Neck")] - track.coords[:, 0, lay.index("Hip")], axis=1
        )
    )
    mid = (
        track.coords[:, 0, lay.index("LShoulder"), 0]
        + track.coords[:, 0, lay.index("RShoulder"), 0]
    ) / 2
    half = 0.42 * trunk / 2
    track.coords[:, 0, lay.index("LShoulder"), 0] = mid - half
    track.coords[:, 0, lay.index("RShoulder"), 0] = mid + half
    rep = Q.combine(Q.check_track(track, p))
    view = next(c for c in rep.checks if c.name == "camera_view")
    assert view.status == "warn", view.message
    assert "angles" in view.message
    assert rep.status != "fail" or all(c.name != "camera_view" for c in rep.failures)


def test_track_checks_low_presence_fails() -> None:
    p = load_protocol("sts_5x")
    track, _ = synthetic_sts_track()
    track.coords[: track.n_frames // 2] = np.nan
    rep = Q.combine(Q.check_track(track, p))
    assert any(c.name == "subject_present" and c.status == "fail" for c in rep.checks)


def test_camera_motion_threshold_scales_with_frame_size() -> None:
    from ptvision.quality.camera_motion import motion_threshold_px

    assert motion_threshold_px(0) == 2.0
    assert motion_threshold_px(480) == 2.0
    assert abs(motion_threshold_px(1920) - 7.68) < 1e-6
    assert motion_threshold_px(3840) > motion_threshold_px(1920)


# [REVIEW] Synthetic coverage for the capture-quality warnings in issue #14.
def _quality_track() -> PoseTrack:
    track, _ = synthetic_sts_track()
    track.coords = track.coords[:100].copy()
    track.score = track.score[:100].copy()
    return track


@pytest.mark.parametrize("edge_frames,status", [(0, "pass"), (5, "pass"), (6, "warn")])
def test_subject_in_frame(edge_frames: int, status: str) -> None:
    track = _quality_track()
    # Move the subject to the left border in exactly the requested frames.
    track.coords[:edge_frames, 0, :, 0] -= track.bboxes()[:edge_frames, 0, 0, None]
    check = next(
        c for c in Q.check_track(track, load_protocol("sts_5x")) if c.name == "subject_in_frame"
    )
    assert check.status == status
    assert check.value == pytest.approx(edge_frames / 100)
    assert check.threshold == 0.05


@pytest.mark.parametrize("swap_frames,status", [(0, "pass"), (30, "pass"), (31, "warn")])
def test_left_right_stability(swap_frames: int, status: str) -> None:
    track = _quality_track()
    for joint in ("Hip", "Knee", "Ankle"):
        left, right = track.layout.indices(f"L{joint}", f"R{joint}")
        track.coords[:swap_frames, 0, [left, right], 0] = track.coords[
            :swap_frames, 0, [right, left], 0
        ]
    check = next(
        c for c in Q.check_track(track, load_protocol("sts_5x")) if c.name == "left_right_stability"
    )
    assert check.status == status
    assert check.value == pytest.approx(swap_frames / 100)
    assert check.threshold == 0.3


@pytest.mark.parametrize("other_frames,status", [(0, "pass"), (20, "pass"), (21, "warn")])
def test_other_people_in_frame(other_frames: int, status: str) -> None:
    track = _quality_track()
    assert Q.check_multi_person(track) == []
    coords = np.repeat(track.coords, 2, axis=1)
    score = np.repeat(track.score, 2, axis=1)
    coords[:, 1, :, 0] += 200
    coords[other_frames:, 1] = np.nan
    score[other_frames:, 1] = 0
    full_track = PoseTrack(
        track.layout, track.fps, coords, score, np.array([0, 1]), image_size=track.image_size
    )
    # Keep these quality tests independent of the identity heuristics.
    full_track.mark_spurious(np.array([False, False]))
    (check,) = Q.check_multi_person(full_track)
    assert check.name == "other_people_in_frame"
    assert check.status == status
    assert check.value == pytest.approx(other_frames / 100)
    assert check.threshold == 0.2
    full_track.mark_spurious(np.array([False, True]))
    assert Q.check_multi_person(full_track) == []


@pytest.mark.parametrize("step_px,status", [(0, "pass"), (4, "warn")])
def test_camera_motion_on_textured_frames(
    monkeypatch: pytest.MonkeyPatch, step_px: int, status: str
) -> None:
    from ptvision.quality.camera_motion import estimate_camera_motion

    texture = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    # Only replace the reader: resize, grayscale, blur and phase correlation run normally.
    monkeypatch.setattr(
        "ptvision.video.iter_frames",
        lambda _: ((i, np.roll(texture, i * step_px, axis=1)) for i in range(21)),
    )
    motion = estimate_camera_motion("synthetic")
    assert motion["unreliable"] == 0
    assert motion["n"] == 4
    assert motion["long_side"] == 640
    # Five-frame sampling gives 20 full-resolution pixels of motion in the moving case.
    expected_px = 5 * step_px
    assert motion["median_px"] == pytest.approx(expected_px, abs=1.0)
    assert motion["p95_px"] == pytest.approx(expected_px, abs=1.0)
    check = Q.check_camera_motion("synthetic")
    assert check.name == "camera_motion"
    assert check.status == status
    assert check.value == pytest.approx(expected_px, abs=1.0)
    assert check.threshold == pytest.approx(2.56)


@pytest.mark.parametrize("n_frames,textured", [(21, False), (6, True)])
def test_camera_motion_unassessable_frames_assume_steady(
    monkeypatch: pytest.MonkeyPatch, n_frames: int, textured: bool
) -> None:
    from ptvision.quality.camera_motion import estimate_camera_motion

    frame = (
        np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        if textured
        else np.zeros((480, 640, 3), dtype=np.uint8)
    )
    monkeypatch.setattr("ptvision.video.iter_frames", lambda _: enumerate([frame] * n_frames))
    motion = estimate_camera_motion("synthetic")
    assert motion["unreliable"] == 1
    assert motion["n"] == (1 if textured else 0)
    check = Q.check_camera_motion("synthetic")
    assert check.status == "pass"
    assert check.value is None
    assert check.threshold == pytest.approx(2.56)
