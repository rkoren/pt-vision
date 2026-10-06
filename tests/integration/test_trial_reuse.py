"""[REVIEW] Same-name clips must not share a trial (two phones' IMG_0001.MOV used to reuse the first
clip's capture and cached keypoints)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ptvision.pipeline import (
    AnalyzeOptions,
    TrialMismatchError,
    default_trial_dir,
    ingest,
    pose_only,
)
from ptvision.trials.store import TrialDir
from tests.synthetic import FakeBackend, synth_video, synthetic_sts_track

pytestmark = pytest.mark.ffmpeg


def _two_clips(tmp_path: Path) -> tuple[Path, Path]:
    for phone in ("phone_a", "phone_b"):
        (tmp_path / phone).mkdir()
    a = synth_video(tmp_path / "phone_a" / "IMG_0001.mp4", 1.0, 30.0)
    b = synth_video(tmp_path / "phone_b" / "IMG_0001.mp4", 1.5, 30.0)
    return a, b


def test_default_trial_dir_separates_same_name_clips(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    a, b = _two_clips(tmp_path)

    dir_a = default_trial_dir(a)
    assert dir_a == Path("ptv_out") / "IMG_0001"
    ingest(a, TrialDir(dir_a))

    assert default_trial_dir(a) == dir_a  # same clip reuses its trial
    dir_b = default_trial_dir(b)
    assert dir_b == Path("ptv_out") / "IMG_0001-2"
    ingest(b, TrialDir(dir_b))
    assert default_trial_dir(b) == dir_b


def test_ingest_refuses_a_different_video(tmp_path: Path) -> None:
    a, b = _two_clips(tmp_path)
    trial = TrialDir(tmp_path / "trial")
    ingest(a, trial)
    with pytest.raises(TrialMismatchError, match="different video"):
        ingest(b, trial)


def test_explicit_out_dir_refuses_a_different_video(tmp_path: Path) -> None:
    a, b = _two_clips(tmp_path)
    track, _ = synthetic_sts_track()
    out = tmp_path / "trial"
    pose_only(a, AnalyzeOptions(out_dir=out, overlay=False), backend=FakeBackend(track))
    with pytest.raises(TrialMismatchError, match="different video"):
        pose_only(b, AnalyzeOptions(out_dir=out, overlay=False), backend=FakeBackend(track))
