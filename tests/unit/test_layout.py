import numpy as np
import pytest

from ptvision.pose.layout import HALPE26, KeypointLayout, remap


def test_halpe26_indices_match_rtmpose_order() -> None:
    assert HALPE26.n == 26
    assert HALPE26.index("Nose") == 0
    assert HALPE26.index("LHip") == 11
    assert HALPE26.index("RHip") == 12
    assert HALPE26.index("Head") == 17
    assert HALPE26.index("Neck") == 18
    assert HALPE26.index("Hip") == 19
    assert HALPE26.index("LBigToe") == 20
    assert HALPE26.index("RBigToe") == 21
    assert HALPE26.index("LSmallToe") == 22
    assert HALPE26.index("RSmallToe") == 23
    assert HALPE26.index("LHeel") == 24
    assert HALPE26.index("RHeel") == 25


def test_tree_parents() -> None:
    assert HALPE26.parents[HALPE26.index("Hip")] == -1
    assert HALPE26.names[HALPE26.parents[HALPE26.index("RKnee")]] == "RHip"
    assert HALPE26.names[HALPE26.parents[HALPE26.index("RHeel")]] == "RAnkle"
    assert HALPE26.names[HALPE26.parents[HALPE26.index("RSmallToe")]] == "RBigToe"
    assert HALPE26.names[HALPE26.parents[HALPE26.index("Neck")]] == "Hip"
    assert len(HALPE26.edges()) == 25  # every non-root keypoint has exactly one parent


def test_mirror() -> None:
    assert HALPE26.mirror("LKnee") == "RKnee"
    assert HALPE26.mirror("RHeel") == "LHeel"
    assert HALPE26.mirror("Hip") == "Hip"
    with pytest.raises(KeyError):
        HALPE26.index("Pelvis")


def test_remap_by_name_and_derived_midpoints() -> None:
    src = KeypointLayout("mini", ("LHip", "RHip", "LShoulder", "RShoulder", "Nose"), (-1,) * 5, ())
    coords = np.array([[[0, 0], [2, 0], [0, 4], [2, 4], [1, 6]]], dtype=np.float32)  # (1, 5, 2)
    score = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    out_c, out_s = remap(coords, score, src, HALPE26)
    assert out_c.shape == (1, 26, 2)
    np.testing.assert_allclose(out_c[0, HALPE26.index("Hip")], [1, 0])
    np.testing.assert_allclose(out_c[0, HALPE26.index("Neck")], [1, 4])
    assert out_s[0, HALPE26.index("Hip")] == pytest.approx(0.8)
    assert np.isnan(out_c[0, HALPE26.index("RHeel")]).all()
    assert out_s[0, HALPE26.index("RHeel")] == 0
