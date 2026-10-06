import numpy as np
import pytest

from ptvision.pose.layout import HALPE26
from ptvision.pose.track import PoseTrack


def make_track(t: int = 10, p: int = 2, d: int = 2, fps: float = 30.0) -> PoseTrack:
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 500, size=(t, p, HALPE26.n, d)).astype(np.float32)
    score = rng.uniform(0.3, 1.0, size=(t, p, HALPE26.n)).astype(np.float32)
    # person 1 absent in frames 3..5, and a NaN keypoint here and there
    coords[3:6, 1] = np.nan
    score[3:6, 1] = 0
    coords[0, 0, 5] = np.nan
    return PoseTrack(HALPE26, fps, coords, score, np.array([0, 7]), image_size=(854, 480))


def test_roundtrip_parquet(tmp_path) -> None:
    tr = make_track()
    path = tr.save(tmp_path / "pose.parquet")
    back = PoseTrack.load(path)
    assert back.layout is HALPE26
    assert back.fps == tr.fps
    assert back.n_frames == tr.n_frames
    assert back.image_size == (854, 480)
    np.testing.assert_array_equal(back.person_ids, tr.person_ids)
    np.testing.assert_allclose(back.coords, tr.coords, equal_nan=True)
    np.testing.assert_allclose(back.score, tr.score)
    assert back.coord_space == "image_px"
    assert back.stage == "raw"


def test_roundtrip_3d_and_metadata(tmp_path) -> None:
    tr = make_track(d=3)
    tr.coord_space = "world_m"
    tr.extra_metadata["note"] = "x"
    back = PoseTrack.load(tr.save(tmp_path / "p.parquet"))
    assert back.n_dims == 3
    assert back.coord_space == "world_m"
    assert back.extra_metadata["note"] == "x"


def test_long_frame_shape() -> None:
    tr = make_track()
    df = tr.to_frame()
    n_present = int(tr.present().sum())
    assert len(df) == n_present * HALPE26.n
    assert set(df.columns) == {
        "camera_id",
        "frame",
        "time_s",
        "person_id",
        "keypoint",
        "kp_index",
        "x",
        "y",
        "z",
        "score",
    }
    assert df["z"].isna().all()


def test_presence_and_selection() -> None:
    tr = make_track()
    pres = tr.presence_fraction()
    assert pres[0] == 1.0
    assert pres[1] == pytest.approx(0.7)
    one = tr.select_person(7)
    assert one.n_persons == 1
    assert one.person_ids.tolist() == [7]
    with pytest.raises(KeyError):
        tr.select_person(3)


def test_empty_track_roundtrip(tmp_path) -> None:
    tr = PoseTrack(
        HALPE26,
        25.0,
        np.full((5, 0, 26, 2), np.nan),
        np.zeros((5, 0, 26)),
        np.zeros(0, dtype=np.int16),
    )
    back = PoseTrack.load(tr.save(tmp_path / "e.parquet"))
    assert back.n_frames == 5 and back.n_persons == 0


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        PoseTrack(HALPE26, 30.0, np.zeros((3, 1, 17, 2)), np.zeros((3, 1, 17)), np.array([0]))


def test_spurious_mask_survives_parquet_roundtrip(tmp_path) -> None:
    # [REVIEW] B56
    import numpy as np

    from ptvision.pose.layout import HALPE26
    from ptvision.pose.track import PoseTrack

    coords = np.zeros((4, 3, HALPE26.n, 2), dtype=np.float32)
    score = np.full((4, 3, HALPE26.n), 0.9, dtype=np.float32)
    tr = PoseTrack(HALPE26, 30.0, coords, score, np.array([0, 1, 2]))
    tr.mark_spurious(np.array([False, False, False]))  # explicit: nothing spurious
    assert tr.spurious.tolist() == [False, False, False] and tr.n_persons_real == 3
    tr.mark_spurious(np.array([False, True, True]))
    p = tmp_path / "t.parquet"
    tr.save(p)
    back = PoseTrack.load(p)
    assert back.spurious.tolist() == [False, True, True]
    assert back.n_persons_real == 1 and back.n_persons == 3
