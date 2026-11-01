import numpy as np
import pytest

from ptvision.datasets.mocap_projection import project_trial
from ptvision.datasets.registry import load_manifest


def test_manifest_loads_with_licenses() -> None:
    specs = load_manifest()
    assert {"uiprmd", "opencap"} <= set(specs)
    for s in specs.values():
        assert s.license and s.license_url.startswith("https://")
        assert s.files or s.manual
        for f in s.files:
            assert f.url.startswith("https://")


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


def test_pull_keeps_verified_file_without_provenance_and_saves_per_file(
    tmp_path, monkeypatch
) -> None:
    import hashlib

    from ptvision.datasets import registry as R

    monkeypatch.setenv("PTV_DATASETS_DIR", str(tmp_path))
    good = b"already here"
    files = (
        R.DatasetFile(
            url="http://x/a.bin", name="a.bin", size=len(good), md5=hashlib.md5(good).hexdigest()
        ),
        R.DatasetFile(
            url="http://x/b.bin", name="b.bin", size=3, md5=hashlib.md5(b"new").hexdigest()
        ),
    )
    spec = R.DatasetSpec(
        name="t",
        title="t",
        citation="c",
        source="s",
        license="CC0",
        license_url="https://example.org",
        kind="mocap",
        notes="",
        files=files,
    )
    dirs = R.DatasetDirs.for_spec(spec)
    dirs.raw.mkdir(parents=True)
    (dirs.raw / "a.bin").write_bytes(good)  # on disk, but no PROVENANCE.json yet
    calls: list[str] = []

    def fake_download(url, dst, *, expected, label, progress):  # type: ignore[no-untyped-def]
        calls.append(label)
        if label == "b.bin" and len(calls) == 1:
            raise R.DownloadError("network down")
        dst.write_bytes(b"new")

    monkeypatch.setattr(R, "_download", fake_download)

    with pytest.raises(R.DownloadError):
        R.pull(spec)
    assert calls == ["b.bin"], "a.bin must be kept, not downloaded"
    assert dirs.provenance.exists(), "progress saved before the failure"
    res = R.pull(spec)
    assert res.skipped == ["a.bin"] and res.downloaded == ["b.bin"]


def test_numpy_pickle_loads_arrays_and_refuses_anything_else(tmp_path) -> None:
    import os
    import pickle

    from ptvision.datasets.opencap import load_numpy_pickle

    good = tmp_path / "cal.pickle"
    good.write_bytes(pickle.dumps({"K": np.eye(3), "size": np.array([[1280.0], [720.0]])}))
    loaded = load_numpy_pickle(good)
    assert np.array_equal(loaded["K"], np.eye(3))

    class Sneaky:
        def __reduce__(self):  # type: ignore[no-untyped-def]
            return (os.getcwd, ())  # harmless if it ran; it must not run at all

    bad = tmp_path / "bad.pickle"
    bad.write_bytes(pickle.dumps({"K": Sneaky()}))
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        load_numpy_pickle(bad)


def test_opencap_camera_calibration(tmp_path) -> None:
    import pickle

    from ptvision.datasets.opencap import BOARD_FROM_VIDEO, OpenCapTrial

    cam_dir = tmp_path / "VideoData" / "Session0" / "Cam0"
    (cam_dir / "STS1").mkdir(parents=True)
    video = cam_dir / "STS1" / "STS1_syncdWithMocap.avi"
    video.touch()
    rot = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    cal = {
        "intrinsicMat": np.array([[900.0, 0, 360], [0, 900.0, 640], [0, 0, 1]]),
        "distortion": np.zeros((1, 5)),
        "imageSize": np.array([[1280.0], [720.0]]),  # rows, cols
        "rotation": rot,
    }
    (cam_dir / "cameraIntrinsicsExtrinsics.pickle").write_bytes(pickle.dumps(cal))
    (cam_dir.parent / "mocapToVideoTransform.yaml").write_text(
        "R_fromMocap_toVideo:\n- [0, 0, -1]\n- [1, 0, 0]\n- [0, -1, 0]\n"
    )
    trial = OpenCapTrial("opencap", "s", "STS1", {0: video}, {0: -70.0})
    got = trial.camera_calibration(0)
    assert got is not None and got.image_size == (720, 1280)
    m2v = np.array([[0.0, 0, -1], [1, 0, 0], [0, -1, 0]])
    assert np.allclose(got.lab_to_camera, rot @ BOARD_FROM_VIDEO @ m2v)
    assert trial.camera_calibration(3) is None
