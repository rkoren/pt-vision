import subprocess
from pathlib import Path

import numpy as np
import pytest

from ptvision.video import FrameWriter, iter_frames, normalize, probe

pytestmark = pytest.mark.ffmpeg


def synth(
    path: Path,
    *,
    rotation: int | None = None,
    fps: int = 30,
    size: str = "320x240",
    seconds: float = 1.0,
) -> Path:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if rotation is not None:
        # Input-side option: attach a display matrix and keep it (no autorotate), like a phone would.
        cmd += ["-display_rotation", str(rotation), "-noautorotate"]
    cmd += [
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={seconds}:size={size}:rate={fps}",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(path),
    ]
    subprocess.run(cmd, check=True)
    return path


def test_probe_basic(tmp_path) -> None:
    p = synth(tmp_path / "a.mp4")
    info = probe(p)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(30.0)
    assert info.n_frames == 30
    assert info.rotation_deg == 0
    assert info.is_vfr is False
    assert info.codec == "h264"


def test_rotation_metadata_is_baked_in_by_normalize(tmp_path) -> None:
    p = synth(tmp_path / "rot.mp4", rotation=90)
    info = probe(p)
    if info.rotation_deg == 0:
        pytest.skip("this ffmpeg build does not write a display matrix via -display_rotation")
    assert info.rotation_deg in (90, 270)
    assert (info.display_width, info.display_height) == (240, 320)
    out = normalize(p, tmp_path / "norm.mp4")
    assert out.rotation_deg == 0
    assert (out.width, out.height) == (240, 320)


def test_normalize_resamples_and_scales(tmp_path) -> None:
    p = synth(tmp_path / "b.mp4", fps=60, size="640x480")
    out = normalize(p, tmp_path / "n.mp4", fps=30, max_height=240)
    assert out.fps == pytest.approx(30.0)
    assert out.height == 240
    assert out.width == 320
    assert out.pix_fmt == "yuv420p"


def test_max_height_caps_the_shorter_side_for_portrait(tmp_path) -> None:
    # [REVIEW] B65: a portrait clip keeps its long side; only the short side is capped
    p = synth(tmp_path / "p.mp4", size="480x640")
    out = normalize(p, tmp_path / "n.mp4", max_height=240)
    assert (out.width, out.height) == (240, 320)
    q = synth(tmp_path / "small.mp4", size="200x300")
    out2 = normalize(q, tmp_path / "n2.mp4", max_height=240)
    assert (out2.width, out2.height) == (200, 300)  # already within the cap: untouched


def test_iter_frames_and_writer_roundtrip(tmp_path) -> None:
    p = synth(tmp_path / "c.mp4", seconds=0.5)
    frames = list(iter_frames(p))
    assert len(frames) == 15
    idx, f = frames[0]
    assert idx == 0 and f.shape == (240, 320, 3) and f.dtype == np.uint8
    out = tmp_path / "w.mp4"
    with FrameWriter(out, 30.0, (320, 240)) as w:
        for _, fr in frames:
            w.write(fr)
    info = probe(out)
    assert info.n_frames == 15
    assert (info.width, info.height) == (320, 240)


def test_fixture_clip_probe(clip_path) -> None:
    info = probe(clip_path)
    assert info.height == 480
    assert info.n_frames == 60
    assert info.fps == pytest.approx(30.0)


def _iframe_count(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pict_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sum(1 for line in out.splitlines() if line.strip().startswith("I"))


def test_normalize_uses_short_gop(tmp_path) -> None:
    p = synth(tmp_path / "long.mp4", seconds=3.0)  # 90 frames
    out = normalize(p, tmp_path / "n.mp4")
    assert out.n_frames == 90
    assert _iframe_count(out.path) >= 90 // 15
    out2 = normalize(p, tmp_path / "n2.mp4", gop=None)
    assert _iframe_count(out2.path) <= 2


def test_frame_source_random_access_matches_sequential(tmp_path) -> None:
    from ptvision.video import FrameSource

    p = synth(tmp_path / "src.mp4", seconds=3.0, size="320x240")
    norm = normalize(p, tmp_path / "n.mp4").path
    sequential = [f for _, f in iter_frames(norm)]
    rng = np.random.default_rng(0)
    with FrameSource(norm, cache_bytes=4 << 20) as fs:
        assert fs.n_frames == len(sequential)
        assert fs.size == (320, 240)
        for i in rng.integers(0, len(sequential), size=30):
            np.testing.assert_array_equal(fs.read(int(i)), sequential[int(i)])
        # forward-within-skip path and cache hit path
        fs.read(10)
        seeks = fs.seeks
        fs.read(12)
        assert fs.seeks == seeks
        fs.read(12)
        assert fs.seeks == seeks
        with pytest.raises(IndexError):
            fs.read(len(sequential))
