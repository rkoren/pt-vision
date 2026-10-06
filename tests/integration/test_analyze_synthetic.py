"""Full `analyze` pipeline on a synthetic video + synthetic pose backend (no model weights needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ptvision.files import read_json
from ptvision.pipeline import AnalyzeOptions, analyze
from tests.synthetic import FakeBackend, StsTimeline, synth_video, synthetic_sts_track

pytestmark = pytest.mark.ffmpeg


def test_analyze_sts_5x_synthetic(tmp_path: Path) -> None:
    track, tl = synthetic_sts_track(noise_px=1.5, dropout=0.02)
    video = tmp_path / "sts.mp4"
    synth_video(video, track.n_frames / tl.fps, tl.fps)
    out = tmp_path / "trial"
    res = analyze(
        video,
        AnalyzeOptions(protocol="sts_5x", out_dir=out, overlay=False),
        backend=FakeBackend(track),
    )

    assert res.report_json.exists() and res.report_html.exists()
    html = res.report_html.read_text()
    assert "Not a medical device" in html and "Five Times Sit-to-Stand" in html
    by_name = {m.name: m for m in res.metrics}
    assert by_name["sts_rep_count"].value == 5
    assert by_name["sts_total_time"].value == pytest.approx(tl.total_time_s, abs=3 / tl.fps)
    assert by_name["sts_total_time"].tier == 1 and by_name["sts_total_time"].citation
    assert (
        by_name["knee_flexion_at_seat_off"].value is not None
        and 60 <= by_name["knee_flexion_at_seat_off"].value <= 110
    )
    assert (
        by_name["trunk_lean_at_seat_off"].value is not None
        and by_name["trunk_lean_at_seat_off"].value > 10
    )
    assert not any(c.name == "repetition_count" and c.status == "fail" for c in res.quality.checks)
    assert res.primary_person == 0

    # artifacts
    run = res.run.path
    for f in (
        "provenance.json",
        "quality.json",
        "events.json",
        "metrics.json",
        "angles.parquet",
        "angles.mot",
        "keypoints_filtered_px.trc",
        "figures/sts_trajectory.png",
        "figures/angles.png",
        "figures/per_rep.png",
    ):
        assert (run / f).exists(), f
    prov = read_json(run / "provenance.json")
    assert prov["protocol"]["id"] == "sts_5x" and prov["metrics"]["sts_total_time"] == "0.1.0"
    assert prov["pose"]["pose_sha256"] == "1" * 64
    events = read_json(run / "events.json")
    assert len(events["reps"]) == 5

    # second run reuses cached pose when the backend matches
    res2 = analyze(
        out, AnalyzeOptions(protocol="sts_5x", overlay=False), backend=FakeBackend(track)
    )
    # the reuse flag, not wall time: a cold first run on a busy laptop took 1.2 s (handoff eval 2)
    assert read_json(res2.run.path / "provenance.json")["pose_reused"] is True
    assert res2.trial.latest_run().run_id == res2.run.run_id


def test_analyze_flags_wrong_rep_count(tmp_path: Path) -> None:
    track, tl = synthetic_sts_track(StsTimeline(n_reps=4))
    video = tmp_path / "sts4.mp4"
    synth_video(video, track.n_frames / tl.fps, tl.fps)
    res = analyze(
        video,
        AnalyzeOptions(protocol="sts_5x", out_dir=tmp_path / "t", overlay=False),
        backend=FakeBackend(track),
    )
    assert res.quality.status == "fail"
    assert any(c.name == "repetition_count" for c in res.quality.failures)
    assert any("expected 5" in w for w in res.warnings)


def test_analyze_30s_variant(tmp_path: Path) -> None:
    tl = StsTimeline(
        n_reps=8, lead_s=1.0, rise_s=1.2, stand_s=0.4, descent_s=1.2, seated_s=0.6, tail_s=2.0
    )
    track, tl = synthetic_sts_track(tl)
    video = tmp_path / "sts30.mp4"
    synth_video(video, track.n_frames / tl.fps, tl.fps)
    res = analyze(
        video,
        AnalyzeOptions(protocol="sts_30s", out_dir=tmp_path / "t", overlay=False),
        backend=FakeBackend(track),
    )
    by_name = {m.name: m for m in res.metrics}
    assert by_name["chair_stand_count_30s"].value == 8
