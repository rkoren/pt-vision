import numpy as np
import pytest

from ptvision.kinematics.preprocess import (
    PreprocessConfig,
    butterworth,
    hampel,
    interpolate_gaps,
    preprocess,
)
from ptvision.kinematics.signal import derivative, robust_normalize, runs_of, walk_until
from tests.synthetic import synthetic_sts_track


def test_runs_of() -> None:
    assert runs_of(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert runs_of(np.zeros(3, bool)) == []


def test_interpolate_gaps_fills_short_only() -> None:
    x = np.array([0, 1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, 9, np.nan])
    out = interpolate_gaps(x, max_gap=2)
    np.testing.assert_allclose(out[:5], [0, 1, 2, 3, 4])
    assert np.isnan(out[5:9]).all()  # too long
    assert np.isnan(out[10])  # edge


def test_butterworth_reduces_noise_and_handles_nan_chunks() -> None:
    fs = 30.0
    t = np.arange(0, 6, 1 / fs)
    clean = np.sin(2 * np.pi * 0.5 * t)
    rng = np.random.default_rng(0)
    noisy = clean + rng.normal(0, 0.2, t.size)
    noisy[60:70] = np.nan
    out = butterworth(noisy, fs, cutoff_hz=3.0, order=4)
    assert np.isnan(out[60:70]).all()
    ok = ~np.isnan(out)
    assert (
        np.sqrt(np.mean((out[ok] - clean[ok]) ** 2))
        < np.sqrt(np.mean((noisy[ok] - clean[ok]) ** 2)) / 2
    )


def test_hampel_removes_spike() -> None:
    x = np.linspace(0, 1, 50)
    x[25] += 5
    out = hampel(x, window=7, n_sigma=2)
    assert abs(out[25] - np.linspace(0, 1, 50)[25]) < 0.1


def test_derivative_and_normalize() -> None:
    x = np.arange(10, dtype=float)
    np.testing.assert_allclose(derivative(x, 2.0), 2.0)
    n = robust_normalize(np.array([0.0, 5.0, 10.0]), 0, 100)
    np.testing.assert_allclose(n, [0, 0.5, 1])
    assert walk_until(np.array([3, 2, 1, 0.1, 0.05]), 0, +1, lambda v: v < 0.5) == 3
    assert walk_until(np.array([3, 2, 1]), 2, -1, lambda v: v > 10) == 0  # never satisfied -> edge


def test_preprocess_pipeline_keeps_shape_and_marks_stage() -> None:
    track, _ = synthetic_sts_track(noise_px=2.0, dropout=0.03)
    pre = preprocess(track, PreprocessConfig())
    assert pre.filtered.coords.shape == track.coords.shape
    assert pre.filtered.stage == "filtered"
    assert pre.stats["missing_after_interpolation"] <= pre.stats["missing_after_gating"]
    with pytest.raises(ValueError):
        preprocess(_two_person(track))


def _two_person(track):  # type: ignore[no-untyped-def]
    from ptvision.pose.track import PoseTrack

    return PoseTrack(
        track.layout,
        track.fps,
        np.concatenate([track.coords] * 2, axis=1),
        np.concatenate([track.score] * 2, axis=1),
        np.array([0, 1]),
    )
