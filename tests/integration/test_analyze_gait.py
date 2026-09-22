"""Full analyze() run of the gait protocol on synthetic walking (no model weights needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ptvision.data.models import Subject
from ptvision.io.jsonio import read_json
from ptvision.pipeline import AnalyzeOptions, analyze
from tests.synthetic import FakeBackend, synth_video, synthetic_gait_track

pytestmark = pytest.mark.ffmpeg


def test_analyze_gait_synthetic(tmp_path: Path) -> None:
    track, tl = synthetic_gait_track(noise_px=1.0)
    video = synth_video(tmp_path / "walk.mp4", track.n_frames / tl.fps, tl.fps, size="1920x1080")
    res = analyze(
        video,
        AnalyzeOptions(
            protocol="gait_sagittal",
            out_dir=tmp_path / "trial",
            overlay=False,
            subject=Subject(height_m=1.75, age_years=72, sex="f"),
        ),
        backend=FakeBackend(track, second_person=False),
    )
    by = {m.name: m for m in res.metrics}
    assert by["gait_speed"].value is not None and by["gait_speed"].value > 0.3
    assert by["n_gait_cycles"].value >= 6
    assert not any(c.status == "fail" for c in res.quality.checks), [
        c.message for c in res.quality.failures
    ]
    html = res.report_html.read_text()
    assert "Walkway gait" in html and "Not a medical device" in html
    run = res.run.path
    for f in (
        "events.json",
        "metrics.json",
        "figures/gait_events.png",
        "figures/gait_cycles.png",
        "angles.parquet",
    ):
        assert (run / f).exists(), f
    prov = read_json(run / "provenance.json")
    assert prov["preprocess"]["scale"]["method"] == "subject_height"
    assert prov["protocol"]["id"] == "gait_sagittal"
    events = read_json(run / "events.json")
    assert events["segmenter"] == "gait_zeni" and events["n_cycles"] >= 6
    norms = read_json(run / "metrics.json")["norms"]
    assert norms and norms[0]["applicable"]


def test_analyze_gait_without_height_flags(tmp_path: Path) -> None:
    track, tl = synthetic_gait_track()
    video = synth_video(tmp_path / "walk2.mp4", track.n_frames / tl.fps, tl.fps, size="1920x1080")
    res = analyze(
        video,
        AnalyzeOptions(protocol="gait_sagittal", out_dir=tmp_path / "t2", overlay=False),
        backend=FakeBackend(track, second_person=False),
    )
    by = {m.name: m for m in res.metrics}
    assert by["gait_speed"].value is None
    assert any("height" in w for w in res.warnings)
