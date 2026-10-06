"""Frame-to-frame person association and primary-subject selection.

Uses Pose2Sim's `sort_people_sports2d`
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
    """Turn per-frame detections into (T, P, K, D) arrays

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
        full = np.full_like(sorted_kp, np.nan)
        if sorted_kp.size:
            for slot in range(sorted_kp.shape[0]):
                row = sorted_kp[slot]
                if np.isnan(row).all():
                    continue
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
            coords[t] = pad_shape(a, p, fill_value=np.nan)
            s_p = pad_shape(s, p, fill_value=np.nan)
            score[t] = np.nan_to_num(s_p, nan=0.0)
    person_ids = np.arange(p, dtype=np.int16)
    return coords, score, person_ids


def stitch_tracks(
    coords: np.ndarray,
    score: np.ndarray,
    *,
    fps: float,
    image_size: tuple[int, int],
    max_gap_s: float = 1.5,
    max_dist_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """merge track fragments of the same person"""
    p = coords.shape[1]
    if p < 2:
        return coords, score
    present = ~np.isnan(coords).all(axis=(2, 3))  # (T, P)
    max_gap = max(1, round(max_gap_s * fps))
    max_dist = max_dist_frac * max(image_size)

    def centroid(fr: int, slot: int) -> np.ndarray:
        good = score[fr, slot] >= 0.3
        pts = coords[fr, slot][good] if good.any() else coords[fr, slot]
        return np.nanmean(pts, axis=0)

    spans = []
    for slot in range(p):
        fr = np.flatnonzero(present[:, slot])
        if fr.size:
            spans.append([slot, int(fr[0]), int(fr[-1])])
    spans.sort(key=lambda s: s[1])
    merged_into: dict[int, int] = {}
    for i, (slot, _start, end) in enumerate(spans):
        if slot in merged_into:
            continue
        cur_end = end
        for j in range(i + 1, len(spans)):
            other, ostart, oend = spans[j]
            if other in merged_into or ostart <= cur_end:
                continue
            if ostart - cur_end > max_gap:
                break
            if np.linalg.norm(centroid(cur_end, slot) - centroid(ostart, other)) > max_dist:
                continue
            merged_into[other] = slot
            coords[:, slot] = np.where(
                present[:, other, None, None], coords[:, other], coords[:, slot]
            )
            score[:, slot] = np.where(present[:, other, None], score[:, other], score[:, slot])
            present[:, slot] |= present[:, other]
            cur_end = oend
    keep = [s for s in range(p) if s not in merged_into]
    return coords[:, keep], score[:, keep]


def spurious_slots(
    coords: np.ndarray,
    score: np.ndarray,
    *,
    fps: float,
    min_track_s: float = 0.5,
    min_mean_score: float = 0.4,
    min_score: float = 0.3,
) -> np.ndarray:
    """(P,) bool marking tracks that are probably not a person"""
    p = coords.shape[1]
    out = np.zeros(p, dtype=bool)
    if p == 0:
        return out
    present = ~np.isnan(coords).all(axis=(2, 3))  # (T, P)
    min_frames = max(1, round(min_track_s * fps))
    for slot in range(p):
        n = int(present[:, slot].sum())
        if n == 0:
            out[slot] = True
            continue
        sc = score[present[:, slot], slot]
        mean_sc = float(np.nanmean(np.where(sc > 0, sc, np.nan))) if np.any(sc > 0) else 0.0
        out[slot] = n < min_frames or mean_sc < min_mean_score
    keep = select_primary_person(coords, score, min_score=min_score)
    out[keep] = False
    return out


def select_primary_person(coords: np.ndarray, score: np.ndarray, *, min_score: float = 0.3) -> int:
    """Index of most likely subject: presence-weighted median bbox area."""
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
