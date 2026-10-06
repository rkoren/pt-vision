import numpy as np
import pytest

from ptvision.pose.layout import HALPE26
from ptvision.viz import status as S
from ptvision.viz.skeleton import SEGMENT_NAMES, SEGMENTS, bone_segments, is_sided, segment_edges


def test_gradient_stops_and_nan() -> None:
    assert S.hex_of(S.status_rgb(0.0)) == "#D62828"
    assert S.hex_of(S.status_rgb(0.5)) == "#F2A900"
    assert S.hex_of(S.status_rgb(1.0)) == "#2BA84A"
    assert S.hex_of(S.status_rgb(np.nan)) == "#8C9BAB"
    arr = S.status_rgb(np.array([0.0, 0.25, 1.0]))
    assert arr.shape == (3, 3) and arr.dtype == np.uint8
    # red channel decreases and green increases monotonically along the ramp
    ramp = S.status_rgb(np.linspace(0, 1, 11)).astype(int)
    assert np.all(np.diff(ramp[:, 1]) >= -1)  # green non-decreasing (allow rounding)
    assert ramp[0, 0] > ramp[-1, 0]


def test_confidence_helpers() -> None:
    edges = [(0, 1), (1, 2)]
    score = np.array([[0.9, 0.4, np.nan]])
    conf = S.bone_confidence(score, edges)
    np.testing.assert_allclose(conf, [[0.4, 0.0]])
    np.testing.assert_allclose(
        S.confidence_status(np.array([0.3, 0.55, 0.8, 0.95])), [0, 0.5, 1, 1]
    )
    dimmed = S.dim_rgb(S.status_rgb(np.array([1.0, 1.0])), np.array([0.6, 0.3]))
    assert S.hex_of(dimmed[0]) == "#2BA84A"
    assert S.hex_of(dimmed[1]) == "#5A5F66"


def test_band_status_shape() -> None:
    band = S.Band(ok=(0, 40), warn=(-10, 55))
    x = np.array([20.0, 0.0, 40.0, -10.0, 55.0, -20.0, 70.0, 47.5, np.nan])
    s = S.band_status(x, band)
    assert s[0] == 1 and s[1] == 1 and s[2] == 1
    assert s[3] == pytest.approx(0.5) and s[4] == pytest.approx(0.5)
    assert s[5] == 0 and s[6] == 0
    assert s[7] == pytest.approx(0.75)
    assert np.isnan(s[8])
    with pytest.raises(ValueError):
        S.Band(ok=(10, 0), warn=(0, 10))
    with pytest.raises(ValueError):
        S.Band(ok=(0, 10), warn=(5, 10))


def test_segments_cover_all_edges_exactly() -> None:
    all_idx: list[int] = []
    for seg in SEGMENT_NAMES:
        all_idx += segment_edges(HALPE26, seg, None)
    assert sorted(all_idx) == list(range(len(HALPE26.edges())))  # every edge once
    assert len(segment_edges(HALPE26, "thigh", "left")) == 1
    assert len(segment_edges(HALPE26, "trunk", "left")) == 5
    assert is_sided("thigh") and not is_sided("trunk")
    with pytest.raises(KeyError):
        segment_edges(HALPE26, "wing", None)
    assert set(SEGMENTS) == set(SEGMENT_NAMES)


def test_bone_rule_status_colors_right_bones() -> None:
    t = 4
    values = {"trunk_lean": np.array([10.0, 50.0, 70.0, np.nan]), "knee_flexion": np.full(t, 20.0)}
    rules = [
        S.Rule("trunk_lean", S.Band((0, 40), (-10, 55)), ("trunk",)),
        S.Rule("knee_flexion", S.Band((0, 110), (0, 130)), ("thigh", "shank")),
        S.Rule(
            "hip_flexion", S.Band((0, 120), (0, 140)), ("trunk", "thigh")
        ),  # no series -> skipped
    ]
    rs = S.bone_rule_status(values, rules, HALPE26, "left")
    assert rs is not None and rs.shape == (t, len(HALPE26.edges()))
    trunk = segment_edges(HALPE26, "trunk", None)
    lthigh = segment_edges(HALPE26, "thigh", "left")[0]
    rthigh = segment_edges(HALPE26, "thigh", "right")[0]
    head = segment_edges(HALPE26, "head", None)
    assert np.all(rs[0, trunk] == 1.0)
    assert np.all(rs[1, trunk] == pytest.approx(2 / 3))  # 50 is 10 into a 15-wide warn band
    assert np.all(rs[2, trunk] == 0.0)
    assert np.all(np.isnan(rs[3, trunk]))
    assert rs[0, lthigh] == 1.0 and np.isnan(rs[0, rthigh])  # far side not assessed
    assert np.all(np.isnan(rs[:, head]))
    assert S.bone_rule_status({}, rules, HALPE26, "left") is None
    assert S.bone_rule_status(values, [], HALPE26, "left") is None


def test_min_over_overlapping_rules() -> None:
    values = {"knee_flexion": np.array([20.0]), "hip_flexion": np.array([200.0])}
    rules = [
        S.Rule("knee_flexion", S.Band((0, 110), (0, 130)), ("thigh",)),
        S.Rule("hip_flexion", S.Band((0, 120), (0, 140)), ("thigh",)),
    ]
    rs = S.bone_rule_status(values, rules, HALPE26, "right")
    assert rs is not None
    rthigh = segment_edges(HALPE26, "thigh", "right")[0]
    assert rs[0, rthigh] == 0.0


def test_bone_status_modes_and_colors() -> None:
    k = HALPE26.n
    score = np.full((3, k), 0.9, dtype=np.float32)
    score[1, HALPE26.index("LKnee")] = 0.1  # hides left thigh+shank in frame 1
    values = {"trunk_lean": np.array([10.0, 10.0, 70.0])}
    rules = [S.Rule("trunk_lean", S.Band((0, 40), (-10, 55)), ("trunk",))]
    bs = S.BoneStatus.build(score, HALPE26, values=values, rules=rules, side="left")
    assert bs.available_modes() == ["confidence", "rules"]
    rgb, vis = bs.colors(0, "rules")
    trunk = segment_edges(HALPE26, "trunk", None)
    assert all(S.hex_of(rgb[e]) == "#2BA84A" for e in trunk)
    rgb2, _vis2 = bs.colors(2, "rules")
    assert all(S.hex_of(rgb2[e]) == "#D62828" for e in trunk)
    lshank = segment_edges(HALPE26, "shank", "left")[0]
    assert vis[lshank] and not bs.colors(1, "rules")[1][lshank]
    assert S.hex_of(rgb[lshank]) == "#8C9BAB"  # not assessed
    rgb_c, _ = bs.colors(0, "confidence")
    assert S.hex_of(rgb_c[lshank]) == "#2BA84A"
    no_rules = S.BoneStatus.build(score, HALPE26)
    assert no_rules.available_modes() == ["confidence"]
    with pytest.raises(ValueError):
        no_rules.value("rules")
    assert len(S.legend("rules")) == 5 and len(S.legend("confidence")) == 3


def test_bone_segments_drops_nan_and_hidden() -> None:
    coords = np.full((HALPE26.n, 2), 10.0)
    coords[HALPE26.index("LWrist")] = np.nan
    edges = HALPE26.edges()
    colors = np.zeros((len(edges), 3), np.uint8)
    visible = np.ones(len(edges), bool)
    visible[0] = False
    segs = bone_segments(coords, edges, colors, visible)
    assert len(segs) == len(edges) - 2
    assert all(len(s) == 5 for s in segs)
