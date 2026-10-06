import numpy as np
import pytest

from ptvision.clinical.segmenters.sts import StsParams, segment_sts, segment_sts_signal
from ptvision.kinematics.angles import compute_angles
from ptvision.kinematics.preprocess import preprocess
from tests.synthetic import StsTimeline, synthetic_sts_track


def _events_close(found: list[int], expected: list[int], tol: int) -> None:
    assert len(found) == len(expected), (found, expected)
    for f, e in zip(found, expected, strict=True):
        assert abs(f - e) <= tol, (found, expected)


def test_clean_signal_events_within_one_frame() -> None:
    tl = StsTimeline()
    h, _lean = tl.build()
    ev = segment_sts_signal(h, tl.fps, StsParams())
    assert ev.n_reps == 5
    _events_close([r.seat_off for r in ev.reps], tl.seat_off, 1)
    _events_close([r.stand_reached for r in ev.reps], tl.stand_reached, 1)
    _events_close([r.descent_start for r in ev.reps], tl.descent_start, 1)
    _events_close([r.seated_return for r in ev.reps], tl.seated_return, 1)
    assert ev.total_time_s == pytest.approx(tl.total_time_s, abs=2 / tl.fps)
    assert not ev.warnings


def test_noisy_signal_with_dropout_within_two_frames() -> None:
    tl = StsTimeline(fps=30.0)
    h, _ = tl.build()
    rng = np.random.default_rng(3)
    noisy = h + rng.normal(0, 0.02, h.size)
    noisy[rng.random(h.size) < 0.05] = np.nan
    from ptvision.kinematics.preprocess import butterworth, interpolate_gaps

    filled = interpolate_gaps(noisy, max_gap=10)
    ok = ~np.isnan(filled)
    filled = np.interp(np.arange(h.size), np.flatnonzero(ok), filled[ok])
    sm = butterworth(filled, tl.fps, 3.0, 4)
    ev = segment_sts_signal(sm, tl.fps, StsParams())
    assert ev.n_reps == 5
    _events_close([r.seat_off for r in ev.reps], tl.seat_off, 2)
    _events_close([r.stand_reached for r in ev.reps], tl.stand_reached, 2)


def test_fidget_not_counted() -> None:
    tl = StsTimeline()
    h, _ = tl.build()
    # small bump before the test (leaning/shifting in the chair): 0.3 of the rise height
    n = int(0.6 * tl.fps)
    h[5 : 5 + n] = 0.3 * np.sin(np.pi * np.arange(n) / n)
    ev = segment_sts_signal(h, tl.fps, StsParams())
    assert ev.n_reps == 5
    assert ev.reps[0].seat_off == pytest.approx(tl.seat_off[0], abs=1)


def test_four_reps_warns() -> None:
    tl = StsTimeline(n_reps=4)
    h, _ = tl.build()
    ev = segment_sts_signal(h, tl.fps, StsParams(n_reps_expected=5))
    assert ev.n_reps == 4
    assert any("expected 5" in w for w in ev.warnings)
    assert ev.test_end == ev.reps[3].stand_reached


def test_end_rule_fifth_sit_and_no_final_sit_fallback() -> None:
    tl = StsTimeline(final_sit=True)
    h, _ = tl.build()
    ev = segment_sts_signal(h, tl.fps, StsParams(end_rule="fifth_sit"))
    assert ev.test_end == pytest.approx(tl.seated_return[4], abs=1)
    tl2 = StsTimeline(final_sit=False)
    h2, _ = tl2.build()
    ev2 = segment_sts_signal(h2, tl2.fps, StsParams(end_rule="fifth_sit"))
    assert ev2.reps[4].seated_return is None
    assert ev2.test_end == ev2.reps[4].stand_reached
    assert any("no final sit" in w for w in ev2.warnings)


def test_window_count_with_partial_rule() -> None:
    tl = StsTimeline(n_reps=12, lead_s=1.0, rise_s=1.0, stand_s=0.3, descent_s=1.0, seated_s=0.4)
    h, _ = tl.build()
    # 12 reps of 2.7 s each; window 30 s from first seat-off covers 11 complete rises (11*2.7=29.7) and
    # the 12th rise is in progress (started at 29.7 s, halfway at 30.2 s) -> not past halfway.
    ev = segment_sts_signal(
        h, tl.fps, StsParams(n_reps_expected=-1, end_rule="window", window_s=30.0)
    )
    assert ev.n_in_window == 11
    ev2 = segment_sts_signal(
        h, tl.fps, StsParams(n_reps_expected=-1, end_rule="window", window_s=30.4)
    )
    assert ev2.n_in_window == 12  # past halfway -> counted


def test_manual_start_rule() -> None:
    tl = StsTimeline()
    h, _ = tl.build()
    ev = segment_sts_signal(h, tl.fps, StsParams(start_rule="manual", manual_start_s=1.0))
    assert ev.test_start == 30


def test_segmenter_on_track_with_lean_onset() -> None:
    track, tl = synthetic_sts_track(noise_px=1.5)
    pre = preprocess(track)
    angles = compute_angles(pre.filtered, ["trunk_lean"], side="near")
    ev = segment_sts(pre.filtered, tl.fps, {"n_reps_expected": 5}, trunk_lean=angles["trunk_lean"])
    assert ev.n_reps == 5
    _events_close([r.seat_off for r in ev.reps], tl.seat_off, 2)
    for r in ev.reps:
        assert r.lean_onset is not None and r.lean_onset <= r.seat_off
        assert r.seat_off - r.lean_onset <= int(0.6 * tl.fps)
    ev_lean = segment_sts(
        pre.filtered,
        tl.fps,
        {"n_reps_expected": 5, "start_rule": "lean_onset"},
        trunk_lean=angles["trunk_lean"],
    )
    assert ev_lean.test_start <= ev.test_start


def test_truncated_clip_flags_and_warns() -> None:
    tl = StsTimeline()
    h, _ = tl.build()
    cut = h[: tl.stand_reached[-1] + 2]
    ev = segment_sts_signal(cut, tl.fps, StsParams())
    assert ev.n_reps == 5
    assert ev.truncated
    assert any("clip end" in w for w in ev.warnings)
    assert ev.to_dict()["truncated"] is True
    full = segment_sts_signal(h, tl.fps, StsParams())
    assert not full.truncated
