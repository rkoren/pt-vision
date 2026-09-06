"""Confidence gating, gap interpolation, outlier rejection and zero-phase low-pass filtering.

Defaults follow the biomechanics convention used by Pose2Sim/Sports2D: Hampel outlier rejection
(window 7, 2 sigma) then a zero-phase Butterworth low-pass, "4th order 6 Hz" meaning a 2nd-order
design applied forward and backward (`scipy.signal.filtfilt`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal as sps

from ptvision._vendor.pose2sim.filtering import hampel_filter
from ptvision.kinematics.signal import nan_runs, runs_of
from ptvision.pose.track import PoseTrack


@dataclass(frozen=True)
class PreprocessConfig:
    score_threshold: float = 0.3
    interp_max_gap_s: float = 0.3
    hampel_window: int = 7
    hampel_n_sigma: float = 2.0
    filter_type: str = "butterworth"  # butterworth | none
    filter_order: int = 4
    filter_cutoff_hz: float = 6.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def gate_by_score(coords: np.ndarray, score: np.ndarray, threshold: float) -> np.ndarray:
    """NaN out coordinates whose confidence is below `threshold`.

    coords (..., K, D), score (..., K)."""
    out = coords.copy()
    out[score < threshold] = np.nan
    return out


def interpolate_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of length <= max_gap (frames) along axis 0.

    Longer runs and runs touching either edge stay NaN."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return _interp_1d(x, max_gap)
    flat = x.reshape(x.shape[0], -1)
    out = np.empty_like(flat)
    for j in range(flat.shape[1]):
        out[:, j] = _interp_1d(flat[:, j], max_gap)
    return out.reshape(x.shape)


def _interp_1d(x: np.ndarray, max_gap: int) -> np.ndarray:
    out = x.copy()
    n = len(x)
    for a, b in nan_runs(x):
        if a == 0 or b == n or (b - a) > max_gap:
            continue
        out[a:b] = np.interp(np.arange(a, b), [a - 1, b], [x[a - 1], x[b]])
    return out


def hampel(x: np.ndarray, window: int = 7, n_sigma: float = 2.0) -> np.ndarray:
    """NaN-aware Hampel outlier rejection, applied per contiguous non-NaN run."""
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    for a, b in runs_of(~np.isnan(x)):
        if b - a >= window:
            out[a:b] = hampel_filter(x[a:b].copy(), window_size=window, n_sigma=n_sigma)  # type: ignore[no-untyped-call]
    return out


def butterworth(x: np.ndarray, fs: float, cutoff_hz: float = 6.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass per contiguous non-NaN run.

    "Order 4" means a 2nd-order design applied forward and backward (filtfilt)."""
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        return x.copy()
    b, a = sps.butter(max(1, order // 2), cutoff_hz / nyq, btype="low", analog=False)
    padlen = 3 * max(len(a), len(b))
    out = x.copy()
    for s, e in runs_of(~np.isnan(x)):
        if e - s > padlen:
            out[s:e] = sps.filtfilt(b, a, x[s:e])
    return out


def filter_series(x: np.ndarray, fs: float, cfg: PreprocessConfig) -> np.ndarray:
    y = (
        hampel(x, cfg.hampel_window, cfg.hampel_n_sigma)
        if cfg.hampel_window > 0
        else np.asarray(x, float)
    )
    if cfg.filter_type == "butterworth":
        y = butterworth(y, fs, cfg.filter_cutoff_hz, cfg.filter_order)
    elif cfg.filter_type != "none":
        raise ValueError(f"unknown filter_type {cfg.filter_type!r}")
    return y


@dataclass
class PreprocessedTrack:
    raw: PoseTrack
    filtered: PoseTrack
    config: PreprocessConfig
    stats: dict[str, float]


def preprocess(track: PoseTrack, cfg: PreprocessConfig | None = None) -> PreprocessedTrack:
    """Gate, interpolate, and filter every keypoint coordinate of a single-person track."""
    cfg = cfg or PreprocessConfig()
    if track.n_persons != 1:
        raise ValueError("preprocess expects a single-person track; use select_person first")
    coords = gate_by_score(track.coords[:, 0], track.score[:, 0], cfg.score_threshold)  # (T, K, D)
    gated_missing = float(np.isnan(coords[..., 0]).mean())
    max_gap = round(cfg.interp_max_gap_s * track.fps)
    coords = interpolate_gaps(coords, max_gap)
    after_interp_missing = float(np.isnan(coords[..., 0]).mean())
    _t, k, d = coords.shape
    out = np.empty_like(coords)
    for kk in range(k):
        for dd in range(d):
            out[:, kk, dd] = filter_series(coords[:, kk, dd], track.fps, cfg)
    filtered = PoseTrack(
        layout=track.layout,
        fps=track.fps,
        coords=out[:, None],
        score=track.score,
        person_ids=track.person_ids,
        camera_id=track.camera_id,
        coord_space=track.coord_space,
        image_size=track.image_size,
        stage="filtered",
        extra_metadata=dict(track.extra_metadata),
    )
    stats = {
        "missing_after_gating": gated_missing,
        "missing_after_interpolation": after_interp_missing,
    }
    return PreprocessedTrack(raw=track, filtered=filtered, config=cfg, stats=stats)
