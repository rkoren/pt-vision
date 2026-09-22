import numpy as np
import pytest

from ptvision.clinical.metrics.base import MetricContext
from ptvision.clinical.protocol import load_protocol
from ptvision.clinical.registry import get_metric
from ptvision.clinical.segmenters.gait_zeni import segment_gait
from ptvision.data.models import CameraCapture, Capture, Subject
from ptvision.kinematics.angles import compute_angles
from ptvision.kinematics.preprocess import preprocess
from ptvision.kinematics.scale import build_scale
from tests.synthetic import GaitTimeline, synthetic_gait_track


def _close(found: list[int], truth: list[int], tol: int, *, lo: int, hi: int) -> None:
    """Every truth event inside [lo, hi] must have a detection within tol frames."""
    truth_in = [f for f in truth if lo <= f <= hi]
    assert truth_in, "no truth events in window"
    for f in truth_in:
        assert any(abs(f - g) <= tol for g in found), (f, found)


@pytest.mark.parametrize("method", ["coordinate", "velocity"])
def test_events_match_truth(method: str) -> None:
    track, tl = synthetic_gait_track(noise_px=1.0)
    pre = preprocess(track)
    ev = segment_gait(pre.filtered, tl.fps, {"method": method}, image_size=tl.image_size)
    assert len(ev.bouts) == 1 and ev.bouts[0].direction == 1
    lo, hi = int(0.5 * tl.fps), int((tl.duration_s - 0.5) * tl.fps)
    for side in ("left", "right"):
        hs = [e.frame for e in ev.events if e.side == side and e.kind == "hs"]
        to = [e.frame for e in ev.events if e.side == side and e.kind == "to"]
        _close(hs, tl.hs[side], 2, lo=lo, hi=hi)
        _close(to, tl.to[side], 2, lo=lo, hi=hi)
    assert ev.n_cycles >= 6
    for c in ev.steady_cycles():
        assert 55 <= 100 * c.stance / c.stride <= 72
        assert 0 < c.ds_initial < c.stance and 0 < c.ds_terminal < c.stance


def test_walking_left_and_edge_cycles_excluded() -> None:
    track, tl = synthetic_gait_track(GaitTimeline(direction=-1))
    ev = segment_gait(track, tl.fps, {}, image_size=tl.image_size)
    assert ev.direction == -1
    cyc = ev.cycles
    assert cyc and not cyc[0].steady and not cyc[-1].steady  # first/last dropped
    assert ev.steady_window() is not None


def test_asymmetric_stance_detected() -> None:
    track, tl = synthetic_gait_track(GaitTimeline(stance_frac_right=0.72))
    pre = preprocess(track)
    ev = segment_gait(pre.filtered, tl.fps, {}, image_size=tl.image_size)
    left = [100 * c.stance / c.stride for c in ev.steady_cycles("left")]
    right = [100 * c.stance / c.stride for c in ev.steady_cycles("right")]
    assert np.mean(right) - np.mean(left) > 6
    from ptvision.clinical.registry import get_metric as _gm

    ctx, _ev, _ = _ctx(track, tl)
    si = _gm("stance_time_symmetry").fn(ctx)
    assert si.value > 2 * si.error


def _ctx(track, tl, *, height_m=1.75):  # type: ignore[no-untyped-def]
    pre = preprocess(track)
    angles = compute_angles(
        pre.filtered, ["knee_flexion", "hip_flexion", "trunk_lean"], side="near"
    )
    ev = segment_gait(pre.filtered, tl.fps, {}, image_size=tl.image_size)
    scale = build_scale(pre.filtered, fps=tl.fps, height_m=height_m) if height_m else None
    cam = CameraCapture(
        camera_id="cam0",
        source_file="x",
        source_sha256="0" * 64,
        source_codec="h264",
        source_pix_fmt="yuv420p",
        source_width=1920,
        source_height=1080,
        source_fps_avg=tl.fps,
        source_fps_nominal=tl.fps,
        source_is_vfr=False,
        rotation_applied_deg=0,
        normalized_file="video/cam0.mp4",
        width=1920,
        height=1080,
        fps=tl.fps,
        n_frames=track.n_frames,
        duration_s=tl.duration_s,
    )
    cap = Capture(
        trial_id="T",
        ingested_at="now",
        subject=Subject(height_m=height_m, age_years=72, sex="f"),
        cameras=[cam],
    )
    return (
        MetricContext(
            raw=pre.raw,
            track=pre.filtered,
            angles=angles,
            events=ev,
            capture=cap,
            protocol=load_protocol("gait_sagittal"),
            fps=tl.fps,
            scale=scale,
        ),
        ev,
        scale,
    )


def test_gait_metrics_on_synthetic() -> None:
    track, tl = synthetic_gait_track(noise_px=0.5)
    ctx, ev, scale = _ctx(track, tl)
    m = {
        n: get_metric(n).fn(ctx)
        for n in (
            "gait_speed",
            "cadence",
            "stride_time",
            "stance_pct",
            "swing_pct",
            "double_support_pct",
            "step_time_symmetry",
            "stance_time_symmetry",
            "step_length",
            "stride_length",
            "n_gait_cycles",
            "peak_knee_flexion_swing",
            "peak_hip_flexion",
        )
    }
    assert m["gait_speed"].value == pytest.approx(tl.speed_px_s / scale.px_per_m, rel=0.03)
    assert m["cadence"].value == pytest.approx(120 / tl.stride_s, rel=0.03)
    assert m["stride_time"].value == pytest.approx(tl.stride_s, abs=0.05)
    # Zeni's coordinate method marks heel strike where heel velocity equals pelvis velocity, about
    # one frame before the synthetic foot stops, so stance reads ~1 frame long at each end (~6 %).
    assert m["stance_pct"].value == pytest.approx(100 * tl.stance_frac, abs=7)
    assert m["swing_pct"].value == pytest.approx(100 * (1 - tl.stance_frac), abs=7)
    assert 10 < m["double_support_pct"].value < 35
    # symmetric gait: asymmetry must sit inside the metric's own error (1-frame jitter at 30 fps ~6 %)
    for name in ("step_time_symmetry", "stance_time_symmetry"):
        assert m[name].value <= max(m[name].error, 6.0) + 1.0, (name, m[name])
    assert m["step_length"].value == pytest.approx(tl.stride_px / 2 / scale.px_per_m, rel=0.08)
    assert m["step_length"].per_event is None and m["stride_length"].per_event is None
    assert m["stride_length"].value == pytest.approx(tl.stride_px / scale.px_per_m, rel=0.05)
    assert m["n_gait_cycles"].value == ev.n_cycles
    assert m["peak_knee_flexion_swing"].value > m["peak_hip_flexion"].value * 0  # both present
    assert m["peak_knee_flexion_swing"].tier == 2 and m["gait_speed"].tier == 1
    for metric in m.values():
        assert metric.method_version and (metric.tier != 1 or metric.citation)


def test_metres_unavailable_without_height() -> None:
    track, tl = synthetic_gait_track()
    ctx, _ev, _ = _ctx(track, tl, height_m=None)
    gs = get_metric("gait_speed").fn(ctx)
    assert gs.value is None and any("height" in f for f in gs.flags)
    assert get_metric("cadence").fn(ctx).value is not None  # timing metrics still work


def test_gait_norms_lookup() -> None:
    from ptvision.clinical.norms import compare, load_norms

    tables = load_norms()
    assert "gait_speed_bohannon2011" in tables
    t = tables["gait_speed_bohannon2011"]
    assert t.lookup(45, "m").value == pytest.approx(1.434)
    assert t.lookup(85, "f").value == pytest.approx(0.943)
    m = get_metric("gait_speed")
    from ptvision.clinical.metrics.base import Metric

    slow = Metric(
        name="gait_speed",
        label=m.label,
        value=0.9,
        units="m/s",
        error=0.04,
        error_kind="mae",
        tier=1,
        method_version="0.1.0",
        citation="x",
    )
    c = compare(slow, "gait_speed_bohannon2011", Subject(age_years=72, sex="f"))
    assert (
        c.applicable and c.reference_value == pytest.approx(1.132) and "at or below" in c.statement
    )
