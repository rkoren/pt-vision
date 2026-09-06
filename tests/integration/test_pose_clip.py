"""End-to-end pose extraction on the 2-second fixture. Needs downloaded weights."""

import os

import numpy as np
import pytest

from ptvision.io.video import iter_frames, probe
from ptvision.pose.layout import HALPE26
from ptvision.pose.models import ModelManager
from ptvision.pose.rtmlib_backend import RtmlibBackend
from ptvision.pose.tracking import select_primary_person

pytestmark = [pytest.mark.model, pytest.mark.ffmpeg]


def _available_mode() -> str:
    mm = ModelManager()
    preferred = os.environ.get("PTV_TEST_MODE")
    order = [preferred] if preferred else []
    order += ["lightweight", "balanced", "performance"]
    for mode in order:
        if (
            mode
            and mm.is_available(mm.modes[mode]["det"])
            and mm.is_available(mm.modes[mode]["pose"])
        ):
            return mode
    pytest.skip("no model weights cached; run `ptv models pull --mode lightweight`")


def test_pose_on_fixture_clip(clip_path, tmp_path) -> None:
    mode = _available_mode()
    info = probe(clip_path)
    be = RtmlibBackend(mode=mode, device="cpu", det_frequency=4)
    track = be.estimate(
        iter_frames(clip_path),
        fps=info.fps,
        image_size=(info.width, info.height),
        n_frames=info.n_frames,
    )

    assert track.n_frames == info.n_frames
    assert track.layout is HALPE26
    assert track.n_persons >= 1
    slot = select_primary_person(track.coords, track.score)
    present = track.present()[:, slot]
    assert present.mean() >= 0.9, f"primary person present in only {present.mean():.0%} of frames"
    body = HALPE26.indices(
        "LShoulder", "RShoulder", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"
    )
    mean_body_score = float(np.nanmean(track.score[present][:, slot][:, body]))
    assert mean_body_score > 0.5
    feet = HALPE26.indices("LHeel", "RHeel", "LBigToe", "RBigToe")
    assert float(np.nanmean(track.score[present][:, slot][:, feet])) > 0.3

    path = track.save(tmp_path / "pose.parquet")
    assert path.stat().st_size < 2_000_000
    assert be.info.pose_sha256 and be.info.det_sha256
    assert be.info.providers  # onnxruntime providers recorded for provenance
