"""Video probing, normalization, frame iteration and writing.

Converts from HEVC (often 10-bit), handling frame rate and rotation"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


class FFmpegNotFoundError(RuntimeError):
    pass


def ffmpeg_install_hint() -> str:
    """One copy-pasteable install line for this platform (shared by errors and `ptv doctor`)."""
    if sys.platform == "win32":
        return "winget install Gyan.FFmpeg   (then open a new terminal so PATH is refreshed)"
    if sys.platform == "darwin":
        return "brew install ffmpeg"
    return "sudo apt install ffmpeg"


def require_ffmpeg() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise FFmpegNotFoundError(
            f"{', '.join(missing)} not found on PATH. Install ffmpeg: {ffmpeg_install_hint()}"
        )


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "N/A"):
        return 0.0
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 0.0


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int  # stored (pre-rotation) width
    height: int
    fps: float  # average frame rate
    fps_fraction: str  # avg_frame_rate as "num/den", suitable for ffmpeg's fps filter
    nominal_fps: float  # r_frame_rate
    is_vfr: bool
    rotation_deg: int  # display rotation metadata, 0 if none
    codec: str
    pix_fmt: str
    n_frames: int | None
    duration_s: float | None
    creation_time: str | None

    @property
    def display_width(self) -> int:
        return self.height if self.rotation_deg in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation_deg in (90, 270) else self.height


def probe(path: Path | str) -> VideoInfo:
    """Read stream metadata with ffprobe."""
    require_ffmpeg()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-count_packets",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError(f"No video stream in {path}")
    s = streams[0]

    rotation = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = round(float(sd["rotation"])) % 360
            break
    if rotation == 0 and "rotate" in (s.get("tags") or {}):
        rotation = round(float(s["tags"]["rotate"])) % 360

    avg = _parse_rate(s.get("avg_frame_rate"))
    nominal = _parse_rate(s.get("r_frame_rate"))
    is_vfr = nominal > 0 and abs(nominal - avg) / nominal > 0.005

    n_frames: int | None = None
    for key in ("nb_frames", "nb_read_packets"):
        if s.get(key) not in (None, "N/A"):
            n_frames = int(s[key])
            break

    duration = s.get("duration") or data.get("format", {}).get("duration")
    fmt_tags = data.get("format", {}).get("tags", {}) or {}
    creation = (s.get("tags") or {}).get("creation_time") or fmt_tags.get("creation_time")

    return VideoInfo(
        path=path,
        width=int(s["width"]),
        height=int(s["height"]),
        fps=avg,
        fps_fraction=s.get("avg_frame_rate", "0/0"),
        nominal_fps=nominal,
        is_vfr=is_vfr,
        rotation_deg=rotation,
        codec=s.get("codec_name", "unknown"),
        pix_fmt=s.get("pix_fmt", "unknown"),
        n_frames=n_frames,
        duration_s=float(duration) if duration not in (None, "N/A") else None,
        creation_time=creation,
    )


# lens metadata reader for iOS devices
LENS_KEYS = {
    "com.apple.quicktime.make": "make",
    "com.apple.quicktime.model": "model",
    "com.apple.quicktime.camera.lens_model": "lens_model",
    "com.apple.quicktime.camera.focal_length.35mm_equivalent": "focal_length_35mm",
}
_MAX_MOOV_BYTES = 64 * 2**20


@dataclass(frozen=True)
class LensInfo:
    make: str | None = None
    model: str | None = None
    lens_model: str | None = None  # e.g. "iPhone 17 Pro back camera 6.765mm f/1.78"
    focal_length_35mm: float | None = None


def _boxes(buf: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """(type, payload start, box end) of the ISO-BMFF boxes in buf[start:end]."""
    i = start
    while i + 8 <= end:
        size, typ = int.from_bytes(buf[i : i + 4], "big"), buf[i + 4 : i + 8]
        head = 8
        if size == 1:
            size, head = int.from_bytes(buf[i + 8 : i + 16], "big"), 16
        elif size == 0:
            size = end - i
        if size < head or i + size > end:
            return
        yield typ, i + head, i + size
        i += size


def _meta_items(buf: bytes, start: int, end: int) -> dict[str, str | float]:
    """The whitelisted entries of one `meta` box (QuickTime layout, or ISO with version/flags)."""
    if buf[start + 4 : start + 8] != b"hdlr" and buf[start + 8 : start + 12] == b"hdlr":
        start += 4
    keys: dict[int, str] = {}
    out: dict[str, str | float] = {}
    for typ, s, e in _boxes(buf, start, end):
        if typ == b"keys":
            j = s + 8  # version/flags, entry count
            for index in range(1, int.from_bytes(buf[s + 4 : s + 8], "big") + 1):
                size = int.from_bytes(buf[j : j + 4], "big")
                if size < 8:
                    break
                keys[index] = buf[j + 8 : j + size].decode("utf-8", "replace")
                j += size
        elif typ == b"ilst":
            for item, s2, e2 in _boxes(buf, s, e):
                field = LENS_KEYS.get(keys.get(int.from_bytes(item, "big"), ""))
                if field is None:
                    continue  # only the keys we use are decoded (never location or faces)
                for dtyp, s3, e3 in _boxes(buf, s2, e2):
                    if dtyp != b"data":
                        continue
                    kind, raw = int.from_bytes(buf[s3 : s3 + 4], "big") & 0xFFFFFF, buf[s3 + 8 : e3]
                    if kind == 1:
                        out[field] = raw.decode("utf-8", "replace").strip()
                    elif kind in (23, 24) and len(raw) in (4, 8):
                        out[field] = struct.unpack(">f" if len(raw) == 4 else ">d", raw)[0]
                    elif kind in (21, 22) and 0 < len(raw) <= 8:
                        out[field] = float(int.from_bytes(raw, "big", signed=kind == 21))
    return out


def read_lens_info(path: Path | str) -> LensInfo:
    """The phone and lens a QuickTime/MP4 video was filmed with, from the movie's and its
    tracks' `keys`/`ilst` metadata. Only `LENS_KEYS` are decoded. Empty when there is none
    (Android phones, re-encoded files) or the file cannot be parsed."""
    found: dict[str, str | float] = {}
    try:
        with Path(path).open("rb") as f:
            size = f.seek(0, 2)
            pos = 0
            while pos + 8 <= size:
                f.seek(pos)
                head = f.read(16)
                box, typ, head_len = int.from_bytes(head[:4], "big"), head[4:8], 8
                if box == 1:
                    box, head_len = int.from_bytes(head[8:16], "big"), 16
                elif box == 0:
                    box = size - pos
                if box < head_len:
                    break
                if typ == b"moov":
                    if box <= _MAX_MOOV_BYTES:
                        f.seek(pos)
                        moov = f.read(box)
                        _collect_meta(moov, head_len, len(moov), found)
                    break
                pos += box  # mdat and the rest are skipped, not read
    except (OSError, ValueError, IndexError, struct.error):
        return LensInfo()

    def text(key: str) -> str | None:
        v = found.get(key)
        return str(v) if v not in (None, "") else None

    try:
        focal = float(found["focal_length_35mm"]) if "focal_length_35mm" in found else None
    except ValueError:
        focal = None
    return LensInfo(text("make"), text("model"), text("lens_model"), focal)


def _collect_meta(buf: bytes, start: int, end: int, found: dict[str, str | float]) -> None:
    """Whitelisted metadata of the movie (`moov/meta`, `moov/udta/meta`) and its tracks."""
    for typ, s, e in _boxes(buf, start, end):
        if typ == b"meta":
            for k, v in _meta_items(buf, s, e).items():
                found.setdefault(k, v)
        elif typ in (b"trak", b"udta"):
            _collect_meta(buf, s, e, found)


def normalize(
    src: Path | str,
    dst: Path | str,
    *,
    fps: float | str | None = None,
    max_height: int | None = None,
    crf: int = 18,
    gop: int | None = 15,
) -> VideoInfo:
    """Transcode to constant-frame-rate 8-bit H.264 with rotation baked into the pixels.

    `fps` defaults to the source's average frame rate (as an exact fraction), which turns a
    variable-frame-rate clip into a constant one at the same nominal rate without resampling
    a CFR clip. Pass a number to resample.

    `gop` caps the keyframe interval (default 15 frames). libx264's default of 250 makes frame
    seeking slow and, with OpenCV, sometimes inexact; a short GOP keeps every seek cheap and exact
    at a few percent extra bitrate.
    """
    require_ffmpeg()
    src, dst = Path(src), Path(dst)
    info = probe(src)
    fps_arg = info.fps_fraction if fps is None else str(fps)
    filters = [f"fps={fps_arg}"]
    if max_height is not None:
        w, h = info.display_width, info.display_height
        if h <= w and h > max_height:
            filters.append(f"scale=-2:{max_height}")
        elif w < h and w > max_height:
            filters.append(f"scale={max_height}:-2")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        ",".join(filters),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "fast",
    ]
    if gop is not None:
        cmd += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
    cmd += ["-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    return probe(dst)


def iter_frames(path: Path | str) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR uint8 image) for every frame."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video {path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


class FrameSource:
    """Random access to the frames of a (normalized, short-GOP) video with a small LRU cache.

    Sequential reads are cheapest; short forward jumps read-and-discard; anything else seeks.
    Frame indices are tracked here and never read back from OpenCV.
    """

    def __init__(
        self, path: Path | str, *, cache_bytes: int = 256 << 20, max_forward_skip: int = 15
    ):
        import cv2

        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise OSError(f"Could not open video {self.path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._pos = 0  # index of the next frame cap.read() will return
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = cache_bytes
        self._max_forward_skip = max_forward_skip
        self.seeks = 0  # instrumentation for tests/benchmarks

    def _store(self, index: int, frame: np.ndarray) -> None:
        self._cache[index] = frame
        self._cache_bytes += frame.nbytes
        while self._cache_bytes > self._cache_limit and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes

    def _read_next(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            raise IndexError(f"frame {self._pos} not readable from {self.path}")
        self._pos += 1
        return frame

    def read(self, index: int) -> np.ndarray:
        """BGR uint8 frame `index` (a cached array; do not modify in place)."""
        import cv2

        if index < 0 or (self.n_frames and index >= self.n_frames):
            raise IndexError(f"frame {index} out of range 0..{self.n_frames - 1}")
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        delta = index - self._pos
        if delta < 0 or delta > self._max_forward_skip:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._pos = index
            self.seeks += 1
        while self._pos < index:
            self._read_next()
        frame = self._read_next()
        self._store(index, frame)
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None  # type: ignore[assignment]

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_frame(path: Path | str, index: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise IndexError(f"Frame {index} not readable from {path}")
        return frame
    finally:
        cap.release()


class FrameWriter:
    """Write BGR frames to an H.264 mp4 through an ffmpeg pipe."""

    def __init__(self, path: Path | str, fps: float, size: tuple[int, int], crf: int = 23):
        require_ffmpeg()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        self._size = (w, h)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-preset",
            "fast",
            "-an",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            raise ValueError(f"Frame size {(w, h)} does not match writer size {self._size}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with status {rc} while writing {self.path}")

    def __enter__(self) -> FrameWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
