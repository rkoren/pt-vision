import numpy as np
import pytest

from ptvision.clinical.segmenters.gait_zeni import GaitParams
from ptvision.datasets.c3d import C3DEvent, C3DTrial
from ptvision.datasets.eval_gait import score_trial
from ptvision.datasets.mocap_projection import classify_event, project_trial
from ptvision.datasets.registry import load_manifest
from tests.synthetic import GaitTimeline, synthetic_gait_track


def test_manifest_loads_with_licenses() -> None:
    specs = load_manifest()
    assert {"fukuchi2018", "schreiber2019", "vancriekinge2023", "uiprmd", "comfi"} <= set(specs)
    for s in specs.values():
        assert s.license and s.license_url.startswith("https://") and s.files
        for f in s.files:
            assert f.url.startswith("https://")
    assert specs["comfi-video"].dir_name == "comfi"


def test_classify_events() -> None:
    assert classify_event(C3DEvent("LHS", "", 1.0)) == ("left", "hs")
    assert classify_event(C3DEvent("RTO", "", 1.0)) == ("right", "to")
    assert classify_event(C3DEvent("Foot Strike", "Left", 1.0)) == ("left", "hs")
    assert classify_event(C3DEvent("Foot Off", "Right", 1.0)) == ("right", "to")
    assert classify_event(C3DEvent("Foot Strike1", "Right", 1.0)) == ("right", "hs")
    assert classify_event(C3DEvent("LON", "", 1.0)) is None
    assert classify_event(C3DEvent("General", "General", 1.0)) is None


def _trial_from_synthetic(
    vertical_axis: int = 2, forward_axis: int = 0, sign: int = 1, rate: float = 100.0
):  # type: ignore[no-untyped-def]
    """Lift the synthetic 2D gait (px, y down) to a 3D lab frame in metres with labelled events."""
    from pathlib import Path

    tl = GaitTimeline(fps=rate, duration_s=6.0)
    track, tl = synthetic_gait_track(tl)
    px_per_m = 260.0
    names = {
        "L.Heel": "LHeel",
        "R.Heel": "RHeel",
        "L.MT1": "LBigToe",
        "R.MT1": "RBigToe",
        "L.Ankle": "LAnkle",
        "R.Ankle": "RAnkle",
        "L.Knee": "LKnee",
        "R.Knee": "RKnee",
        "L.GTR": "LHip",
        "R.GTR": "RHip",
        "L.ASIS": "LHip",
        "R.ASIS": "RHip",
        "L.PSIS": "LHip",
        "R.PSIS": "RHip",
    }
    T = track.n_frames
    labels = list(names)
    pts = np.full((T, len(labels), 3), np.nan)
    floor_y = 900.0
    for i, (lbl, kp) in enumerate(names.items()):
        c = track.keypoint(kp).astype(np.float64)
        fwd = sign * c[:, 0] / px_per_m
        up = (floor_y - c[:, 1]) / px_per_m
        lat = -0.1 if lbl.startswith("L") else 0.1
        if lbl.endswith("PSIS"):
            fwd = fwd - 0.12
        pts[:, i, vertical_axis] = up
        pts[:, i, forward_axis] = fwd
        pts[:, i, 3 - vertical_axis - forward_axis] = lat
    events = []
    for side in ("left", "right"):
        S = side[0].upper()
        events += [C3DEvent(f"{S}HS", "", f / rate) for f in tl.hs[side]]
        events += [C3DEvent(f"{S}TO", "", f / rate) for f in tl.to[side]]
    events.sort(key=lambda e: e.time_s)
    return C3DTrial(Path("synthetic.c3d"), rate, labels, pts, events, 0, "m"), tl


@pytest.mark.parametrize("vertical,forward,sign", [(2, 0, 1), (1, 0, -1), (2, 1, 1)])
def test_lab_frame_and_projection_recover_direction(vertical: int, forward: int, sign: int) -> None:
    trial, tl = _trial_from_synthetic(vertical, forward, sign)
    pt = project_trial(trial, fps=30.0)
    assert pt.lab.vertical == vertical and pt.lab.forward == forward and pt.lab.forward_sign == sign
    hip = pt.track.keypoint("Hip")[:, 0]
    assert np.nanmean(np.diff(hip)) > 0  # walks toward image-right after projection
    assert pt.track.n_frames == pytest.approx(tl.duration_s * 30, abs=2)
    assert len(pt.events) > 10
    assert np.isnan(pt.track.keypoint("Neck")).all()  # lower-body-only set stays NaN


def test_score_trial_on_synthetic_mocap() -> None:
    trial, _tl = _trial_from_synthetic()
    pt = project_trial(trial, fps=30.0)
    s = score_trial(pt, GaitParams(min_cycles=1, drop_edge_cycles=0, edge_margin_frac=0.0))
    assert s.n_truth >= 20 and s.missed <= 2
    e = np.abs(np.asarray(s.errors_frames))
    assert (e <= 2).mean() >= 0.9, s.row()


def test_uiprmd_vicon_loader_and_sts(tmp_path) -> None:
    """A synthetic 39-marker UI-PRMD positions file: one sit-to-stand -> one rise."""
    from ptvision.clinical.segmenters.sts import StsParams, segment_sts
    from ptvision.datasets.uiprmd import VICON_39, read_vicon_positions
    from ptvision.kinematics.preprocess import preprocess
    from tests.synthetic import StsTimeline, synthetic_sts_track

    tl = StsTimeline(fps=100.0, n_reps=1, lead_s=1.0, tail_s=1.0)
    track, tl = synthetic_sts_track(tl)
    px_per_m = 260.0
    kp_for = {
        "LASI": "LHip",
        "RASI": "RHip",
        "LPSI": "LHip",
        "RPSI": "RHip",
        "LKNE": "LKnee",
        "RKNE": "RKnee",
        "LANK": "LAnkle",
        "RANK": "RAnkle",
        "LHEE": "LHeel",
        "RHEE": "RHeel",
        "LTOE": "LBigToe",
        "RTOE": "RBigToe",
        "LSHO": "LShoulder",
        "RSHO": "RShoulder",
        "C7": "Neck",
        "CLAV": "Neck",
        "LFHD": "Head",
        "RFHD": "Head",
    }
    T = track.n_frames
    data = np.zeros((T, len(VICON_39) * 3))
    for i, name in enumerate(VICON_39):
        kp = kp_for.get(name)
        if kp is None:
            continue
        c = track.keypoint(kp).astype(np.float64)
        x_mm = c[:, 0] / px_per_m * 1000
        z_mm = (900 - c[:, 1]) / px_per_m * 1000
        y_mm = -100.0 if name.startswith("L") else 100.0
        if name.endswith("PSI"):
            x_mm = x_mm - 120
        data[:, 3 * i : 3 * i + 3] = np.stack([x_mm, np.full(T, y_mm), z_mm], axis=1)
    f = tmp_path / "m05_s01_e01_positions.txt"
    np.savetxt(f, data, delimiter=",", fmt="%.4f")
    trial = read_vicon_positions(f)
    assert trial.rate == 100.0 and trial.points.shape == (T, 39, 3)
    pt = project_trial(trial, fps=30.0)
    pre = preprocess(pt.track)
    ev = segment_sts(pre.filtered, 30.0, StsParams(n_reps_expected=1))
    assert ev.n_reps == 1
    assert abs(ev.reps[0].seat_off - round(tl.seat_off[0] * 30 / 100)) <= 2


def test_side_swap_detection() -> None:
    from ptvision.datasets.eval_gait import score_trial

    trial, _tl = _trial_from_synthetic()
    swapped = [
        C3DEvent(("R" if e.label[0] == "L" else "L") + e.label[1:], "", e.time_s)
        for e in trial.events
    ]
    trial_sw = C3DTrial(trial.path, trial.rate, trial.labels, trial.points, swapped, 0, "m")
    s = score_trial(
        project_trial(trial_sw, fps=30.0),
        GaitParams(min_cycles=1, drop_edge_cycles=0, edge_margin_frac=0.0, bout_speed_frac=0.0),
    )
    assert s.side_swapped is True
    assert s.matched >= 0.8 * s.n_truth
