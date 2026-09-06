from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from ptvision.pipeline import AnalyzeOptions, analyze, pose_only
from ptvision.pose.track import PoseTrack
from tests.synthetic import FakeBackend, StsTimeline, synth_video, synthetic_sts_track

pytestmark = pytest.mark.qt

TIGHT_RULES = """
[[rules]]
angle = "trunk_lean"
label = "Trunk forward lean"
ok = [0, 10]
warn = [-5, 20]

[[rules]]
angle = "knee_flexion"
label = "Knee flexion"
ok = [0, 110]
warn = [0, 130]
"""


@dataclass
class SynthTrial:
    trial_dir: Path
    run_id: str
    track: PoseTrack
    tl: StsTimeline
    video: Path
    protocol_path: Path | None


def _tight_protocol(tmp: Path) -> Path:
    text = resources.files("ptvision.clinical.protocols").joinpath("sts_5x.toml").read_text()
    head, _, tail = text.partition("[[rules]]")
    # drop the built-in rules block(s) and keep the [report] section
    report = "[report]" + tail.split("[report]", 1)[1]
    out = tmp / "sts_5x_tight.toml"
    out.write_text(head + TIGHT_RULES + "\n" + report)
    return out


@pytest.fixture(scope="session")
def synthetic_trial(tmp_path_factory: pytest.TempPathFactory) -> SynthTrial:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    tmp = tmp_path_factory.mktemp("app_trial")
    track, tl = synthetic_sts_track(noise_px=1.5, dropout=0.02)
    video = synth_video(tmp / "sts.mp4", track.n_frames / tl.fps, tl.fps)
    proto = _tight_protocol(tmp)
    res = analyze(
        video,
        AnalyzeOptions(protocol=str(proto), out_dir=tmp / "trial", overlay=False),
        backend=FakeBackend(track),
    )
    return SynthTrial(res.trial.path, res.run.run_id, track, tl, video, proto)


@pytest.fixture(scope="session")
def pose_only_trial(tmp_path_factory: pytest.TempPathFactory) -> SynthTrial:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    tmp = tmp_path_factory.mktemp("app_pose")
    track, tl = synthetic_sts_track()
    video = synth_video(tmp / "pose.mp4", track.n_frames / tl.fps, tl.fps)
    res = pose_only(
        video, AnalyzeOptions(out_dir=tmp / "trial", overlay=False), backend=FakeBackend(track)
    )
    return SynthTrial(res.run.path.parent.parent, res.run.run_id, track, tl, video, None)
