"""Skeleton overlay rendering (OpenCV) for QA videos and report thumbnails.

Colors come from `ptvision.viz.status` so the CLI overlay and the desktop app render the same:
Colors come from `ptvision.viz.status` so the CLI overlay and the desktop app render
the same:
assesses, dimmed for low confidence, hidden below the confidence floor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ptvision.io.video import FrameWriter, iter_frames
from ptvision.pose.base import ProgressFn
from ptvision.pose.layout import KeypointLayout
from ptvision.pose.track import PoseTrack
from ptvision.viz import status as vs
from ptvision.viz.skeleton import bone_segments


def score_color(s: float) -> tuple[int, int, int]:
    """BGR color for a joint confidence, on the shared gradient."""
    r, g, b = (int(v) for v in vs.status_rgb(vs.confidence_status(s)))
    return (b, g, r)


def _bgr(rgb: tuple[int, int, int] | np.ndarray) -> tuple[int, int, int]:
    r, g, b = (int(v) for v in rgb)
    return (b, g, r)


def draw_pose(
    img: np.ndarray,
    coords: np.ndarray,
    score: np.ndarray,
    layout: KeypointLayout,
    person_ids: np.ndarray,
    *,
    kpt_thr: float = vs.CONF_HIDE,
    primary: int | None = None,
    radius: int | None = None,
    thickness: int | None = None,
    bone_colors: np.ndarray | None = None,
    bone_visible: np.ndarray | None = None,
    mode: vs.Mode = "confidence",
) -> np.ndarray:
    """Draw all persons for one frame. coords (P, K, 2), score (P, K).

    `bone_colors` (B, 3) RGB and `bone_visible` (B,) apply to the primary person (from
    `BoneStatus.colors`); without them the primary's bones are colored by joint confidence.
    Other persons are drawn thin and grey.
    """
    import cv2

    h, w = img.shape[:2]
    scale = max(h, w) / 1080.0
    radius = radius or max(2, round(4 * scale))
    thickness = thickness or max(1, round(3 * scale))
    edges = layout.edges()
    for p in range(coords.shape[0]):
        c = coords[p]
        s = score[p]
        if np.isnan(c).all():
            continue
        is_primary = primary is None or int(person_ids[p]) == primary
        if is_primary:
            if bone_colors is None or bone_visible is None:
                conf = vs.bone_confidence(s, edges)
                colors = vs.status_rgb(vs.confidence_status(conf))
                visible = conf >= kpt_thr
            else:
                colors, visible = bone_colors, bone_visible
            for xa, ya, xb, yb, rgb in bone_segments(c, edges, colors, visible):
                cv2.line(
                    img,
                    (round(xa), round(ya)),
                    (round(xb), round(yb)),
                    _bgr(rgb),
                    thickness,
                    cv2.LINE_AA,
                )
            for k in range(c.shape[0]):
                if s[k] >= kpt_thr and not np.isnan(c[k]).any():
                    pt = (round(c[k, 0]), round(c[k, 1]))
                    col = _bgr(vs.JOINT_RULES_MODE) if mode == "rules" else score_color(float(s[k]))
                    cv2.circle(img, pt, radius, col, -1, cv2.LINE_AA)
            label_color = _bgr(vs.status_rgb(1.0))
        else:
            grey = _bgr(vs.OTHER_PERSON)
            for a, b in edges:
                if (
                    s[a] >= kpt_thr
                    and s[b] >= kpt_thr
                    and not (np.isnan(c[a]).any() or np.isnan(c[b]).any())
                ):
                    cv2.line(
                        img,
                        (round(c[a, 0]), round(c[a, 1])),
                        (round(c[b, 0]), round(c[b, 1])),
                        grey,
                        max(1, thickness // 2),
                        cv2.LINE_AA,
                    )
            label_color = grey
        good = s >= kpt_thr
        if good.any():
            x0 = int(np.nanmin(c[good, 0]))
            y0 = int(np.nanmin(c[good, 1]))
            label = f"id {int(person_ids[p])}" + (
                " *" if is_primary and primary is not None else ""
            )
            cv2.putText(
                img,
                label,
                (max(0, x0), max(12, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * max(1.0, scale),
                label_color,
                max(1, thickness // 2),
                cv2.LINE_AA,
            )
    return img


def draw_legend(img: np.ndarray, mode: vs.Mode) -> np.ndarray:
    """Color swatches with labels in the top-right corner."""
    import cv2

    h, w = img.shape[:2]
    scale = max(h, w) / 1080.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45 * max(1.0, scale)
    line_h = round(18 * max(1.0, scale))
    entries = vs.legend(mode)
    widths = [cv2.getTextSize(t, font, fs, 1)[0][0] for t, _ in entries]
    box_w = max(widths) + round(34 * max(1.0, scale))
    box_h = line_h * len(entries) + round(10 * scale)
    x1, y1 = w - box_w - 8, 8
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, (text, hx) in enumerate(entries):
        rgb = tuple(int(hx[j : j + 2], 16) for j in (1, 3, 5))
        y = y1 + round(5 * scale) + line_h * i
        cv2.rectangle(
            img,
            (x1 + 6, y + 3),
            (x1 + 6 + round(16 * max(1.0, scale)), y + line_h - 4),
            _bgr((rgb[0], rgb[1], rgb[2])),
            -1,
        )
        cv2.putText(
            img,
            text,
            (x1 + round(28 * max(1.0, scale)), y + line_h - 6),
            font,
            fs,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return img


def render_overlay(
    video_path: Path | str,
    track: PoseTrack,
    out_path: Path | str,
    *,
    kpt_thr: float = vs.CONF_HIDE,
    primary: int | None = None,
    status: vs.BoneStatus | None = None,
    mode: vs.Mode = "confidence",
    legend: bool = True,
    progress: ProgressFn | None = None,
) -> Path:
    """Write an overlay video. `status` (for the primary person) selects bone colors per frame."""
    import cv2

    out_path = Path(out_path)
    if status is not None and mode == "rules" and status.rules is None:
        mode = "confidence"
    writer: FrameWriter | None = None
    try:
        for idx, frame in iter_frames(video_path):
            if writer is None:
                h, w = frame.shape[:2]
                writer = FrameWriter(out_path, track.fps, (w, h))
            if idx < track.n_frames:
                colors = visible = None
                if status is not None and idx < status.n_frames:
                    colors, visible = status.colors(idx, mode)
                draw_pose(
                    frame,
                    track.coords[idx],
                    track.score[idx],
                    track.layout,
                    track.person_ids,
                    kpt_thr=kpt_thr,
                    primary=primary,
                    bone_colors=colors,
                    bone_visible=visible,
                    mode=mode,
                )
            if legend:
                draw_legend(frame, mode)
            cv2.putText(
                frame,
                f"{idx}  t={idx / track.fps:6.2f}s",
                (8, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            if progress is not None:
                progress(idx + 1, track.n_frames)
    finally:
        if writer is not None:
            writer.close()
    return out_path


def thumbnail(
    video_path: Path | str,
    track: PoseTrack,
    frame_index: int,
    out_path: Path | str,
    *,
    max_width: int = 480,
) -> Path:
    import cv2

    from ptvision.io.video import read_frame

    frame = read_frame(video_path, frame_index)
    draw_pose(
        frame,
        track.coords[frame_index],
        track.score[frame_index],
        track.layout,
        track.person_ids,
    )
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return out_path
