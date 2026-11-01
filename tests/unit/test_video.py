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
    p = synth(tmp_path / "p.mp4", size="480x640")
    out = normalize(p, tmp_path / "n.mp4", max_height=240)
    assert (out.width, out.height) == (240, 320)
    q = synth(tmp_path / "small.mp4", size="200x300")
    out2 = normalize(q, tmp_path / "n2.mp4", max_height=240)
    assert (out2.width, out2.height) == (200, 300)  # already within the cap:


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


def _box(typ: bytes, payload: bytes) -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + typ + payload


def _meta(items: dict[str, bytes]) -> bytes:
    """A QuickTime `meta` box: hdlr, keys, and an ilst of UTF-8 `data` entries."""
    hdlr = _box(b"hdlr", bytes(8) + b"mdta" + bytes(13))
    keys = b"".join(_box(b"mdta", k.encode()) for k in items)
    keys = _box(b"keys", bytes(4) + len(items).to_bytes(4, "big") + keys)
    entries = b"".join(
        _box(i.to_bytes(4, "big"), _box(b"data", (1).to_bytes(4, "big") + bytes(4) + v))
        for i, v in enumerate(items.values(), start=1)
    )
    return _box(b"meta", hdlr + keys + _box(b"ilst", entries))


def test_read_lens_info_reads_whitelisted_keys_only(tmp_path) -> None:
    from ptvision.video import LensInfo, read_lens_info

    movie = _meta({"com.apple.quicktime.make": b"Apple", "com.apple.quicktime.model": b"iPhone"})
    track = _meta(
        {
            "com.apple.quicktime.location.ISO6709": b"+40.0000-075.0000/",
            "com.apple.quicktime.camera.lens_model": b"iPhone back camera 6.765mm f/1.78",
            "com.apple.quicktime.camera.focal_length.35mm_equivalent": b"24",
        }
    )
    # a 64-bit-size mdat before the moov, as long recordings have; it must be skipped, not read
    mdat = (1).to_bytes(4, "big") + b"mdat" + (16 + 1000).to_bytes(8, "big") + bytes(1000)
    path = tmp_path / "phone.mov"
    path.write_bytes(
        _box(b"ftyp", b"qt  " + bytes(4)) + mdat + _box(b"moov", movie + _box(b"trak", track))
    )
    info = read_lens_info(path)
    assert info == LensInfo("Apple", "iPhone", "iPhone back camera 6.765mm f/1.78", 24.0)
    assert "40.0000" not in repr(info)
    (tmp_path / "junk.mov").write_bytes(b"not a movie at all")
    assert read_lens_info(tmp_path / "junk.mov") == LensInfo()


@pytest.mark.ffmpeg
def test_ingest_keeps_the_lens_metadata(tmp_path) -> None:
    from ptvision.pipeline import ingest
    from ptvision.trials.store import TrialDir

    clip = tmp_path / "IMG_0001.mov"
    tags = {
        "com.apple.quicktime.model": "iPhone Test",
        "com.apple.quicktime.camera.lens_model": "Test back camera 6.765mm f/1.78",
        "com.apple.quicktime.camera.focal_length.35mm_equivalent": "24",
        "com.apple.quicktime.location.ISO6709": "+40.0000-075.0000/",
    }
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:s=320x240:d=1:r=30"]
    for k, v in tags.items():
        cmd += ["-metadata", f"{k}={v}"]
    subprocess.run(
        [*cmd, "-c:v", "libx264", "-movflags", "use_metadata_tags", str(clip)], check=True
    )
    cam = ingest(clip, TrialDir(tmp_path / "trial")).cameras[0]
    assert (cam.device_model, cam.focal_length_35mm) == ("iPhone Test", 24.0)
    assert cam.lens_model == "Test back camera 6.765mm f/1.78"
    assert "40.0000" not in (tmp_path / "trial" / "capture.json").read_text()
