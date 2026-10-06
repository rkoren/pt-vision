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


def test_spurious_slots_flags_short_and_low_score_tracks() -> None:
    # [REVIEW] B56: one real person for 3 s, a 6-frame ghost, and a low-confidence ghost
    from ptvision.pose.tracking import spurious_slots

    t, k = 90, 26
    coords = np.full((t, 3, k, 2), np.nan, dtype=np.float32)
    score = np.zeros((t, 3, k), dtype=np.float32)
    coords[:, 0] = 100.0 + np.arange(k)[None, :, None]
    score[:, 0] = 0.8
    coords[10:16, 1] = 400.0
    score[10:16, 1] = 0.7  # short-lived
    coords[:, 2] = 700.0
    score[:, 2] = 0.25  # present but never confident
    mask = spurious_slots(coords, score, fps=30)
    assert mask.tolist() == [False, True, True]
    # a clip whose only track is weak still keeps it
    only = spurious_slots(coords[:, 2:3], score[:, 2:3], fps=30)
    assert only.tolist() == [False]


def test_stitch_tracks_joins_fragments_across_a_detection_gap() -> None:
    # [REVIEW] demo MVP: one person seen as slot 0 (frames 0-39), lost for 12 frames, back as slot 1
    from ptvision.pose.tracking import stitch_tracks

    t, k = 100, 26
    coords = np.full((t, 2, k, 2), np.nan, dtype=np.float32)
    score = np.zeros((t, 2, k), dtype=np.float32)
    base = np.stack([np.linspace(300, 320, t), np.linspace(400, 420, t)], axis=1)[:, None, :]
    coords[:40, 0] = base[:40] + np.arange(k)[None, :, None]
    score[:40, 0] = 0.8
    coords[52:, 1] = base[52:] + np.arange(k)[None, :, None]
    score[52:, 1] = 0.8
    out_c, _ = stitch_tracks(coords, score, fps=30, image_size=(1280, 720))
    assert out_c.shape[1] == 1
    present = ~np.isnan(out_c[:, 0]).all(axis=(1, 2))
    assert present[:40].all() and present[52:].all() and not present[40:52].any()
    # a fragment far away (another person) is not merged
    far = coords.copy()
    far[52:, 1, :, 0] += 900
    out_c2, _ = stitch_tracks(far, score.copy(), fps=30, image_size=(1280, 720))
    assert out_c2.shape[1] == 2


def test_dedupe_persons_drops_overlapping_copy_keeps_distinct() -> None:
    from ptvision.pose.rtmlib_backend import dedupe_persons

    k = 26
    a = np.stack([np.linspace(100, 200, k), np.linspace(100, 400, k)], axis=1).astype(np.float32)
    b = a + 5  # near-identical copy
    c = a + np.array([600, 0], np.float32)  # another person far away
    kpts = np.stack([a, b, c])
    scores = np.array([[0.6] * k, [0.9] * k, [0.7] * k], np.float32)
    out_k, out_s = dedupe_persons(kpts, scores)
    assert out_k.shape[0] == 2
    assert np.allclose(out_s[:, 0], [0.9, 0.7])  # the better copy survives, input order kept
