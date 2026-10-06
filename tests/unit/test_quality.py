import numpy as np

from ptvision.clinical.protocol import load_protocol
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
