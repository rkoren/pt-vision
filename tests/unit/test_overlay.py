import numpy as np

from ptvision.pose.layout import HALPE26
from ptvision.pose.overlay import draw_legend, draw_pose
from ptvision.viz import status as vs
from ptvision.viz.skeleton import segment_edges


def _frame_with_horizontal_trunk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.zeros((200, 300, 3), np.uint8)
    coords = np.full((1, HALPE26.n, 2), np.nan, np.float32)
    score = np.zeros((1, HALPE26.n), np.float32)
    coords[0, HALPE26.index("Hip")] = (50, 100)
    coords[0, HALPE26.index("Neck")] = (250, 100)
    score[0, [HALPE26.index("Hip"), HALPE26.index("Neck")]] = 0.9
    return img, coords, score


def test_draw_pose_paints_status_color_on_bone() -> None:
    img, coords, score = _frame_with_horizontal_trunk()
    edges = HALPE26.edges()
    colors = np.zeros((len(edges), 3), np.uint8)
    visible = np.zeros(len(edges), bool)
    hip_neck = segment_edges(HALPE26, "trunk", None)[0]
    colors[hip_neck] = vs.status_rgb(0.0)  # red
    visible[hip_neck] = True
    draw_pose(
        img,
        coords,
        score,
        HALPE26,
        np.array([0]),
        primary=0,
        bone_colors=colors,
        bone_visible=visible,
        thickness=5,
        mode="rules",
    )
    b, g, r = (int(v) for v in img[100, 150])
    assert (r, g, b) == (0xD6, 0x28, 0x28)
    # joints are white dots in rules mode
    b2, g2, r2 = (int(v) for v in img[100, 50])
    assert (r2, g2, b2) == (255, 255, 255)


def test_draw_pose_default_confidence_coloring() -> None:
    img, coords, score = _frame_with_horizontal_trunk()
    draw_pose(img, coords, score, HALPE26, np.array([0]), primary=0, thickness=5)
    b, g, r = (int(v) for v in img[100, 150])
    assert (r, g, b) == (0x2B, 0xA8, 0x4A)  # 0.9 confidence -> green


def test_legend_draws_without_error() -> None:
    img = np.zeros((240, 640, 3), np.uint8)
    out = draw_legend(img, "rules")
    assert out.any()
