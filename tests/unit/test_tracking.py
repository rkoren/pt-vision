import numpy as np

from ptvision.pose.layout import HALPE26
from ptvision.pose.tracking import TrackerParams, assign_ids, select_primary_person

K = HALPE26.n


def skeleton(cx: float, cy: float, scale: float = 100.0, rng=None) -> np.ndarray:
    base = np.zeros((K, 2), dtype=np.float32)
    base[:, 0] = np.linspace(-0.3, 0.3, K) * scale + cx
    base[:, 1] = np.linspace(-1.0, 1.0, K) * scale + cy
    if rng is not None:
        base += rng.normal(0, 1.0, base.shape)
    return base


def test_two_people_crossing_keep_ids() -> None:
    rng = np.random.default_rng(1)
    t = 60
    kpts, scores = [], []
    for i in range(t):
        a = skeleton(100 + 8 * i, 300, rng=rng)  # moves right
        b = skeleton(600 - 8 * i, 320, scale=90, rng=rng)  # moves left, crosses ~frame 31
        order = [a, b] if i % 2 == 0 else [b, a]  # detector order shuffles
        kpts.append(np.stack(order))
        scores.append(np.full((2, K), 0.9, dtype=np.float32))
    coords, _score, ids = assign_ids(kpts, scores, fps=30, image_size=(1280, 720))
    assert coords.shape == (t, 2, K, 2)
    assert ids.tolist() == [0, 1]
    # slot 0 should move monotonically right, slot 1 left, with no swap
    x0 = np.nanmean(coords[:, 0, :, 0], axis=1)
    x1 = np.nanmean(coords[:, 1, :, 0], axis=1)
    assert np.all(np.diff(x0) > 0)
    assert np.all(np.diff(x1) < 0)


def test_dropout_and_reappearance_same_id() -> None:
    t = 40
    kpts, scores = [], []
    for i in range(t):
        if 15 <= i < 20:  # person disappears for 5 frames (< 1 s at 30 fps)
            kpts.append(np.empty((0, K, 2), np.float32))
            scores.append(np.empty((0, K), np.float32))
        else:
            kpts.append(skeleton(200 + 2 * i, 300)[None])
            scores.append(np.full((1, K), 0.8, np.float32))
    coords, _score, _ids = assign_ids(kpts, scores, fps=30, image_size=(1280, 720))
    assert coords.shape[1] == 1
    present = ~np.isnan(coords[:, 0]).all(axis=(1, 2))
    assert present.sum() == t - 5
    assert not present[15:20].any()


def test_low_confidence_keypoints_do_not_break_matching() -> None:
    t = 10
    kpts, scores = [], []
    for i in range(t):
        s = skeleton(300, 300 + i)
        sc = np.full((1, K), 0.9, np.float32)
        sc[0, :5] = 0.05  # face keypoints garbage
        s[:5] += 400 * (i % 2)  # wildly jumping low-confidence points
        kpts.append(s[None])
        scores.append(sc)
    coords, _score, _ids = assign_ids(
        kpts, scores, fps=30, image_size=(1280, 720), params=TrackerParams(min_keypoint_score=0.2)
    )
    assert coords.shape[1] == 1
    # full (unmasked) coordinates are restored, including the noisy ones
    assert not np.isnan(coords[:, 0, :5]).any()


def test_select_primary_prefers_present_and_large() -> None:
    t = 30
    coords = np.full((t, 2, K, 2), np.nan, np.float32)
    score = np.zeros((t, 2, K), np.float32)
    for i in range(t):
        coords[i, 0] = skeleton(300, 300, scale=150)  # big, always present
        score[i, 0] = 0.9
        if i < 10:
            coords[i, 1] = skeleton(800, 300, scale=200)  # bigger but present 1/3 of the time
            score[i, 1] = 0.9
    assert select_primary_person(coords, score) == 0
