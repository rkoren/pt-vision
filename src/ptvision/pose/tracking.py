"""Frame-to-frame person association and primary-subject selection.

Association uses Pose2Sim's `sort_people_sports2d` (Hungarian matching on mean per-keypoint
distance, with a grace period for short dropouts). Selection of the subject is the person 
present most often with the largest median bounding box. CLI `--person` flag overrides it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptvision._vendor.pose2sim.tracking import pad_shape, sort_people_sports2d


@dataclass(frozen=True)
class TrackerParams:
    match_by: str = "keypoints"  # keypoints | centroid | bbox
    max_dist_frac: float = 0.10  # fraction of the image's larger side, per frame
    max_unseen_s: float = 1.0
    min_keypoint_score: float = 0.2  # keypoints below this are ignored in matching

    def to_dict(self) -> dict[str, object]:
        return {
            "name": "pose2sim.sort_people_sports2d",
            "match_by": self.match_by,
            "max_dist_frac": self.max_dist_frac,
            "max_unseen_s": self.max_unseen_s,
            "min_keypoint_score": self.min_keypoint_score,
        }


def assign_ids(
    per_frame_kpts: list[np.ndarray],
    per_frame_scores: list[np.ndarray],
    *,
    fps: float,
    image_size: tuple[int, int],
    params: TrackerParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn ragged per-frame detections into dense (T, P, K, D) arrays with stable ids.

    per_frame_kpts[t]: (N_t, K, D); per_frame_scores[t]: (N_t, K). N_t may be 0.
    Returns coords (T, P, K, D) float32 with NaN where absent, score (T, P, K), person_ids (P,).
    """
    params = params or TrackerParams()
    t_total = len(per_frame_kpts)
    if t_total == 0:
        raise ValueError("no frames")
    k = next((a.shape[1] for a in per_frame_kpts if a.ndim == 3), None)
    d = next((a.shape[2] for a in per_frame_kpts if a.ndim == 3), None)
    if k is None or d is None:
        raise ValueError("no detections in any frame")

    max_dist = params.max_dist_frac * max(image_size)
    max_unseen = max(1, round(params.max_unseen_s * fps))

    keyptpre = np.empty((0, k, d))
    fsls: np.ndarray | None = None
    sorted_frames: list[np.ndarray] = []
    sorted_scores: list[np.ndarray] = []

    for kp, sc in zip(per_frame_kpts, per_frame_scores, strict=True):
        kp = np.asarray(kp, dtype=np.float64).reshape(-1, k, d)
        sc = np.asarray(sc, dtype=np.float64).reshape(-1, k)
        # Low-confidence keypoints should not drive matching; NaN them out for the tracker only.
        kp_match = kp.copy()
        kp_match[sc < params.min_keypoint_score] = np.nan
        tracked, sorted_kp, sorted_sc, fsls = sort_people_sports2d(  # type: ignore[no-untyped-call]
            keyptpre,
            kp_match,
            scores=sc,
            match_by=params.match_by,
            max_dist=max_dist,
            max_unseen_frames=max_unseen,
            frames_since_last_seen=fsls,
        )
        # Restore the full (unmasked) coordinates for matched slots.
        full = np.full_like(sorted_kp, np.nan)
        if sorted_kp.size:
            for slot in range(sorted_kp.shape[0]):
                row = sorted_kp[slot]
                if np.isnan(row).all():
                    continue
                # find the source detection whose masked coords equal this slot
                matches = [
                    i
                    for i in range(kp_match.shape[0])
                    if np.array_equal(kp_match[i], row, equal_nan=True)
                ]
                full[slot] = kp[matches[0]] if matches else row
        keyptpre = tracked
        sorted_frames.append(full)
        sorted_scores.append(sorted_sc)

    p = max((a.shape[0] for a in sorted_frames), default=0)
    coords = np.full((t_total, p, k, d), np.nan, dtype=np.float32)
    score = np.zeros((t_total, p, k), dtype=np.float32)
    for t, (a, s) in enumerate(zip(sorted_frames, sorted_scores, strict=True)):
        if a.shape[0]:
            coords[t] = pad_shape(a, p, fill_value=np.nan)  # type: ignore[no-untyped-call]
            s_p = pad_shape(s, p, fill_value=np.nan)  # type: ignore[no-untyped-call]
            score[t] = np.nan_to_num(s_p, nan=0.0)
    person_ids = np.arange(p, dtype=np.int16)
    return coords, score, person_ids


def select_primary_person(coords: np.ndarray, score: np.ndarray, *, min_score: float = 0.3) -> int:
    """Index (slot) of the most likely subject: presence-weighted median bbox area."""
    p = coords.shape[1]
    if p == 0:
        raise ValueError("no persons in track")
    best, best_val = 0, -1.0
    for slot in range(p):
        c = coords[:, slot]  # (T, K, D)
        good = score[:, slot] >= min_score  # (T, K)
        c = np.where(good[..., None], c, np.nan)
        present = ~np.isnan(c).all(axis=(1, 2))
        if not present.any():
            continue
        cp = c[present]
        w = np.nanmax(cp[..., 0], axis=1) - np.nanmin(cp[..., 0], axis=1)
        h = np.nanmax(cp[..., 1], axis=1) - np.nanmin(cp[..., 1], axis=1)
        area = float(np.nanmedian(w * h))
        val = float(present.mean() * area)
        if val > best_val:
            best, best_val = slot, val
    return best
