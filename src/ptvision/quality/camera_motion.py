"""Estimate camera motion between frames with phase correlation on downsampled grayscale images."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def estimate_camera_motion(
    video_path: Path | str,
    *,
    step: int = 5,
    max_samples: int = 60,
    width: int = 320,
    min_response: float = 0.2,
) -> dict[str, float]:
    """Median and 95th-percentile translation (px at full resolution) between sampled frames.

    Featureless or heavily changed frame pairs give a flat correlation surface (low response) and a
    meaningless peak, so only well-supported estimates count; with fewer than three of them the
    result is flagged `unreliable`.
    """
    import cv2

    from ptvision.io.video import iter_frames

    prev: np.ndarray | None = None
    shifts: list[float] = []
    scale = 1.0
    samples = 0
    for idx, frame in iter_frames(video_path):
        if idx % step:
            continue
        h, w = frame.shape[:2]
        scale = w / width
        small = cv2.resize(frame, (width, int(h / scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = np.asarray(cv2.GaussianBlur(gray, (5, 5), 0), dtype=np.float32)
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, gray)
            if resp >= min_response:
                shifts.append(float(np.hypot(dx, dy)) * scale)
            samples += 1
            if samples >= max_samples:
                break
        prev = gray
    if len(shifts) < 3:
        return {"median_px": 0.0, "p95_px": 0.0, "n": float(len(shifts)), "unreliable": 1.0}
    return {
        "median_px": float(np.median(shifts)),
        "p95_px": float(np.percentile(shifts, 95)),
        "n": float(len(shifts)),
        "unreliable": 0.0,
    }
