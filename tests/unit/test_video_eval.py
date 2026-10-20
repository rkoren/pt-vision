"""Video-vs-mocap evaluation: readers, force events, agreement statistics and the harness on a
synthetic lab trial.  [REVIEW] new file"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptvision.datasets.lab_trial import LabTrial
from ptvision.datasets.markers import MarkerTrial
from ptvision.datasets.mocap_projection import ProjectedTrial, project_trial
from ptvision.datasets.opensim_io import read_mot, read_trc
from ptvision.datasets.sts_force import (
    ForceStsEvents,
    events_from_feet_plates,
    match_reps,
    pelvis_rising,
)
from ptvision.datasets.video_eval import Agreement, estimate_lag_s, evaluate_video, resample_to
from tests.synthetic import FakeBackend, StsTimeline, synth_video, synthetic_sts_track

FIXTURE_CLIP = Path(__file__).resolve().parents[1] / "fixtures" / "clip_2s_480p.mp4"


# ---------------------------------------------------------------------------------------------
# OpenSim text formats
# ---------------------------------------------------------------------------------------------
def _write_trc(path: Path, labels: list[str], pts_mm: np.ndarray, rate: float) -> None:
    T = pts_mm.shape[0]
    head = [
        "PathFileType\t4\t(X/Y/Z)\tsynthetic.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{rate}\t{rate}\t{T}\t{len(labels)}\tmm\t{rate}\t1\t{T}",
        "Frame#\tTime\t" + "\t\t\t".join(labels) + "\t\t",
        "\t\t" + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(labels))),
        "",
    ]
    rows = []
    for f in range(T):
        cells = [str(f + 1), f"{f / rate:.5f}"]
        for k in range(len(labels)):
            cells += ["" if np.isnan(pts_mm[f, k, 0]) else f"{v:.3f}" for v in pts_mm[f, k]]
        rows.append("\t".join(cells))
    path.write_text("\n".join(head + rows) + "\n")


def test_read_trc_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20, 3, 3)) * 1000
    pts[5, 1, :] = np.nan  # a missing marker
    _write_trc(tmp_path / "t.trc", ["r_knee", "L_knee", "C7"], pts, 100.0)
    tr = read_trc(tmp_path / "t.trc")
    assert tr.rate == 100.0 and tr.units == "m" and tr.labels == ["r_knee", "L_knee", "C7"]
    assert tr.points.shape == (20, 3, 3)
    assert np.allclose(tr.points[0], pts[0] / 1000)
    assert np.isnan(tr.points[5, 1]).all()
    assert tr.index("R_KNEE") == 0  # case-insensitive lookup


def test_read_mot_degrees_flag(tmp_path: Path) -> None:
    text = (
        "Coordinates\nversion=1\nnRows=3\nnColumns=3\ninDegrees=yes\n\nendheader\n"
        "time\tknee_angle_r\thip_flexion_r\n0.00\t90.0\t70.0\n0.01\t80.0\t60.0\n0.02\t70.0\t50.0\n"
    )
    (tmp_path / "ik.mot").write_text(text)
    df = read_mot(tmp_path / "ik.mot")
    assert df.attrs["in_degrees"] is True
    assert list(df.columns) == ["time", "knee_angle_r", "hip_flexion_r"]
    assert df["knee_angle_r"].tolist() == [90.0, 80.0, 70.0]


# ---------------------------------------------------------------------------------------------
# Synthetic lab trial built from the synthetic sit-to-stand skeleton
# ---------------------------------------------------------------------------------------------
PX_PER_M = 260.0
OPENCAP_NAMES = {
    "r.ASIS": "RHip",
    "L.ASIS": "LHip",
    "r.PSIS": "RHip",
    "L.PSIS": "LHip",
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


def _lab_points(track, names: dict[str, str]) -> tuple[list[str], np.ndarray]:  # type: ignore[no-untyped-def]
    """3D lab positions (m): x forward from image x, z up from image y, y lateral by side."""
    T = track.n_frames
    labels = list(names)
    pts = np.zeros((T, len(labels), 3))
    for k, (lab, kp) in enumerate(names.items()):
        c = track.keypoint(kp).astype(np.float64)
        pts[:, k, 0] = c[:, 0] / PX_PER_M
        pts[:, k, 2] = (900 - c[:, 1]) / PX_PER_M
        pts[:, k, 1] = -0.1 if lab.lower().startswith("l") else 0.1
    return labels, pts


def test_project_trial_from_lab_names() -> None:
    names = OPENCAP_NAMES
    from ptvision.kinematics.angles import compute_angles

    track, _tl = synthetic_sts_track(StsTimeline(fps=100.0, n_reps=2, lead_s=0.5, tail_s=0.5))
    labels, pts = _lab_points(track, names)
    trial = MarkerTrial(Path("synthetic"), 100.0, labels, pts, 0, "m")
    pt = project_trial(trial, fps=50.0, forward_rule="lean")
    assert pt.lab.vertical == 2 and pt.lab.forward == 0
    ours = compute_angles(track, ["knee_flexion"], side="right")["knee_flexion"]
    ref = compute_angles(pt.track, ["knee_flexion"], side="right")["knee_flexion"]
    # same skeleton, so the projected angles reproduce the source ones (resampled 100 -> 50)
    assert np.nanmax(np.abs(ref - ours[::2][: len(ref)])) < 1.0


# ---------------------------------------------------------------------------------------------
# Force-plate events
# ---------------------------------------------------------------------------------------------
def _synthetic_feet_force(t: np.ndarray, bw: float, rises: list[float]) -> np.ndarray:
    """Seated 0.2 BW; each rise ramps to 1.3 BW over 0.3 s, holds BW for 0.6 s, then a sit-down
    with a braking peak of 1.15 BW and back to seated."""
    f = np.full_like(t, 0.2 * bw)
    for r in rises:
        f[(t >= r) & (t < r + 0.3)] = (
            0.2 * bw + (1.3 - 0.2) * bw * (t[(t >= r) & (t < r + 0.3)] - r) / 0.3
        )
        f[(t >= r + 0.3) & (t < r + 0.9)] = bw
        f[(t >= r + 0.9) & (t < r + 1.0)] = 0.7 * bw  # unloading as the descent starts
        f[(t >= r + 1.0) & (t < r + 1.15)] = 1.15 * bw  # braking peak
    return f


def test_events_from_feet_plates_counts_rises_not_braking_peaks() -> None:
    t = np.arange(0, 10, 0.0005)
    bw = 700.0
    rises = [0.5, 2.5, 4.5, 6.5, 8.5]
    feet = _synthetic_feet_force(t, bw, rises)
    rising = np.zeros_like(t, dtype=bool)
    for r in rises:
        rising[(t >= r) & (t < r + 0.5)] = True
    ev = events_from_feet_plates(t, feet, bw, rising=rising)
    assert ev.n_reps == 5
    # the crossing of body weight happens 0.218 s into the 0.3 s ramp
    assert np.allclose(ev.seat_off_s, [r + 0.8 / 1.1 * 0.3 for r in rises], atol=0.002)
    assert ev.span_s == pytest.approx(8.0, abs=0.01)
    # without the pelvis mask the braking peaks are too short to pass the standing hold
    assert events_from_feet_plates(t, feet, bw).n_reps == 5


def test_pelvis_rising_from_projected_track() -> None:
    track, _tl = synthetic_sts_track(StsTimeline(fps=100.0, n_reps=1, lead_s=1.0, tail_s=1.0))
    labels, pts = _lab_points(track, OPENCAP_NAMES)
    pt = project_trial(MarkerTrial(Path("s"), 100.0, labels, pts, 0, "m"), fps=100.0)
    t = np.arange(0, track.duration_s, 0.01)
    rising = pelvis_rising(pt, t)
    assert rising.any() and not rising.all()
    # rising only while the pelvis goes up, i.e. inside the rise, not in the seated lead-in
    assert not rising[t < 0.9].any()


def test_match_reps_greedy_nearest() -> None:
    pairs = match_reps([1.0, 2.1, 5.0], [0.9, 2.0, 3.0, 4.95])
    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 1), (2, 3)]
    assert pairs[1][2] == pytest.approx(0.1)


# ---------------------------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------------------------
def test_agreement_bias_and_limits() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=500) * 20 + 40
    ours = ref + 3.0 + rng.normal(size=500) * 2.0
    ours[10] = np.nan
    a = Agreement.of(ours, ref)
    assert a.n == 499
    assert a.bias == pytest.approx(3.0, abs=0.3)
    assert a.sd == pytest.approx(2.0, abs=0.3)
    assert a.loa_lo < a.bias < a.loa_hi
    assert a.rmse == pytest.approx(np.sqrt(9 + 4), abs=0.4)
    assert a.r is not None and a.r > 0.99
    assert Agreement.of(np.array([1.0]), np.array([1.0])).rmse is None


def test_estimate_lag_and_resample() -> None:
    fps = 60.0
    t = np.arange(0, 6, 1 / fps)
    ref = np.sin(2 * np.pi * 0.5 * t) * 40
    ours = np.roll(ref, 6)  # ours lags by 6 frames = 0.1 s
    assert estimate_lag_s(ours, ref, fps) == pytest.approx(0.1, abs=1 / fps)
    y = resample_to(np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0, np.nan]), np.array([0.5, 1.5]))
    assert y[0] == pytest.approx(5.0) and np.isnan(y[1])


# ---------------------------------------------------------------------------------------------
# The harness end to end on a synthetic lab trial
# ---------------------------------------------------------------------------------------------
class SyntheticLabTrial(LabTrial):
    def __init__(self, video: Path, track, fps: float) -> None:  # type: ignore[no-untyped-def]
        super().__init__("synthetic", "s1", "STS1", {0: video}, {0: 90.0}, 1.8, 75.0, "m", 5)
        labels, pts = _lab_points(track, OPENCAP_NAMES)
        self._trial = MarkerTrial(Path("synthetic"), fps, labels, pts, 0, "m")
        self._fps = fps

    def reference_track(self, fps: float) -> ProjectedTrial:
        return project_trial(self._trial, fps=fps, forward_rule="lean")

    def reference_joint_angles(self, side):  # type: ignore[no-untyped-def]
        from ptvision.kinematics.angles import compute_angles

        a = compute_angles(self.reference_track(self._fps).track, side=side)
        df = pd.DataFrame({"time": np.arange(len(a.frame)) / self._fps})
        for n in ("knee_flexion", "hip_flexion", "trunk_lean"):
            df[n] = a[n]
        return df

    def force_sts_events(self) -> ForceStsEvents | None:
        return ForceStsEvents("synthetic", 75 * 9.80665, [1.0, 2.6, 4.2, 5.8, 7.4])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_evaluate_video_on_synthetic_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ptvision.datasets.video_eval as ve

    fps = 30.0
    tl = StsTimeline(fps=fps, n_reps=5, lead_s=1.0, tail_s=1.0)
    track, tl = synthetic_sts_track(tl)
    video = synth_video(tmp_path / "lab.mp4", track.duration_s, fps)
    trial = SyntheticLabTrial(video, track, fps)
    monkeypatch.setattr(ve, "_trials_for", lambda *_a, **_k: [trial])
    monkeypatch.setattr(ve, "load_manifest", lambda: {"synthetic": None})
    monkeypatch.setenv("PTV_HARD_EXIT", "0")
    results, out = evaluate_video(
        "synthetic",
        backend=FakeBackend(track, second_person=False),
        out_dir=tmp_path / "eval",
    )
    assert len(results) == 1 and results[0].status == "ok", results[0].note
    r = results[0]
    assert r.metrics["knee_flexion_mocap_rmse"] is not None
    assert r.metrics["knee_flexion_mocap_rmse"] < 3.0  # same skeleton on both sides
    assert r.metrics["knee_flexion_model_rmse"] < 3.0
    assert r.reps_video == 5 and r.reps_force == 5
    assert (out / "trials.csv").exists() and (out / "events.csv").exists()
    trials = pd.read_csv(out / "trials.csv")
    assert {"knee_flexion_mocap_rmse", "seat_off_err_ms_mean", "span_err_s"} <= set(trials.columns)
    frames = pd.read_csv(next((out / "frames").glob("*.csv")))
    assert {"knee_flexion_video", "knee_flexion_mocap", "in_window"} <= set(frames.columns)
    summary = (out / "summary.json").read_text()
    assert '"by_camera"' in summary and '"cam0"' in summary


def test_evaluate_video_records_errors_per_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ptvision.datasets.video_eval as ve

    class Broken(SyntheticLabTrial):
        def reference_track(self, fps: float) -> ProjectedTrial:
            raise ValueError("no markers")

    track, _ = synthetic_sts_track(StsTimeline(fps=30.0, n_reps=1, lead_s=0.5, tail_s=0.5))
    trial = Broken(FIXTURE_CLIP, track, 30.0)
    monkeypatch.setattr(ve, "_trials_for", lambda *_a, **_k: [trial])
    monkeypatch.setattr(ve, "load_manifest", lambda: {"synthetic": None})
    results, _out = evaluate_video(
        "synthetic", backend=FakeBackend(track, second_person=False), out_dir=tmp_path / "eval"
    )
    assert results[0].status == "error" and "no markers" in results[0].note


def test_cli_eval_video_help_and_manual_pull() -> None:
    from typer.testing import CliRunner

    from ptvision.cli.main import app

    runner = CliRunner()
    assert "eval-video" in runner.invoke(app, ["datasets", "--help"]).output
    res = runner.invoke(app, ["datasets", "eval-video", "--help"])
    assert res.exit_code == 0 and "--cameras" in res.output
    res = runner.invoke(app, ["datasets", "pull", "opencap"])
    assert res.exit_code == 0 and "manual" in res.output.lower()


def test_lag_correction_direction() -> None:
    """A video trace that runs early (negative lag) must be moved later to line up."""
    from ptvision.datasets.video_eval import _shift

    fps = 60.0
    t = np.arange(0, 6, 1 / fps)
    ref = np.sin(2 * np.pi * 0.5 * t) * 40
    ours = np.roll(ref, -8)  # ours is 8 frames early
    lag = estimate_lag_s(ours, ref, fps)
    assert lag == pytest.approx(-8 / fps, abs=1 / fps)
    shift = -round(lag * fps)
    corrected = _shift(ours, shift)
    inner = slice(20, -20)
    assert Agreement.of(corrected[inner], ref[inner]).rmse < 1.0
    assert Agreement.of(ours[inner], ref[inner]).rmse > 10.0
