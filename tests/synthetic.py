"""Synthetic sit-to-stand pose data with known event frames, for tests without real video."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ptvision.pose.layout import HALPE26
from ptvision.pose.track import PoseTrack


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class StsTimeline:
    fps: float = 30.0
    lead_s: float = 1.5
    rise_s: float = 0.8
    stand_s: float = 0.4
    descent_s: float = 0.9
    seated_s: float = 0.6
    tail_s: float = 1.5
    n_reps: int = 5
    final_sit: bool = True
    lean_s: float = 0.35  # forward lean starts this long before seat-off
    # filled by build()
    seat_off: list[int] = field(default_factory=list)
    stand_reached: list[int] = field(default_factory=list)
    descent_start: list[int] = field(default_factory=list)
    seated_return: list[int] = field(default_factory=list)

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (h in [0,1] up-positive normalised pelvis height, lean in degrees) per frame."""
        fps = self.fps
        segments: list[tuple[str, float]] = [("seated", self.lead_s)]
        for i in range(self.n_reps):
            segments += [("rise", self.rise_s), ("stand", self.stand_s)]
            if i < self.n_reps - 1 or self.final_sit:
                segments += [("descent", self.descent_s), ("seated", self.seated_s)]
        segments.append(("tail", self.tail_s))
        total = round(sum(d for _, d in segments) * fps)
        h = np.zeros(total)
        lean = np.zeros(total)
        t = 0
        level = 0.0
        for kind, dur in segments:
            n = round(dur * fps)
            if kind == "rise":
                self.seat_off.append(t)
                h[t : t + n] = smoothstep(np.arange(n) / n)
                self.stand_reached.append(t + n)
                level = 1.0
                # lean peaks around seat-off then recovers during the rise
                lean[t : t + n] = 30 * (1 - np.arange(n) / n)
                pre = round(self.lean_s * fps)
                lean[max(0, t - pre) : t] = 30 * smoothstep(np.arange(min(pre, t)) / max(pre, 1))
            elif kind == "descent":
                self.descent_start.append(t)
                h[t : t + n] = 1 - smoothstep(np.arange(n) / n)
                self.seated_return.append(t + n)
                level = 0.0
                lean[t : t + n] = 15 * np.sin(np.pi * np.arange(n) / n)
            else:
                h[t : t + n] = level
            t += n
        h[t:] = level
        return h, lean

    @property
    def total_time_s(self) -> float:
        return (self.stand_reached[self.n_reps - 1] - self.seat_off[0]) / self.fps


def synthetic_sts_track(
    tl: StsTimeline | None = None,
    *,
    image_size: tuple[int, int] = (854, 480),
    noise_px: float = 0.0,
    dropout: float = 0.0,
    seed: int = 0,
    facing: int = 1,
) -> tuple[PoseTrack, StsTimeline]:
    """A single-person Halpe-26 track of a sagittal sit-to-stand with plausible limb geometry."""
    tl = tl or StsTimeline()
    h, lean_deg = tl.build()
    rng = np.random.default_rng(seed)
    T = len(h)
    K = HALPE26.n
    coords = np.full((T, 1, K, 2), np.nan, dtype=np.float32)
    score = np.full((T, 1, K), 0.9, dtype=np.float32)

    # image geometry (y down). seated hip y=340, standing hip y=220; knee fixed y=340; ankle y=440.
    hip_y = 340 - 120 * h
    hip_x = 420 + 0 * h
    knee_y = np.full(T, 340.0)
    knee_x = hip_x + facing * (70 * (1 - h) + 6)  # knee forward of hip when seated
    ankle_y = np.full(T, 440.0)
    ankle_x = knee_x + facing * 4 - facing * 60 * (1 - h) * 0  # roughly under the knee
    trunk_len = 150.0
    lean = np.radians(lean_deg)
    neck_x = hip_x + facing * trunk_len * np.sin(lean)
    neck_y = hip_y - trunk_len * np.cos(lean)
    head_x, head_y = neck_x + facing * 10, neck_y - 30
    nose_x, nose_y = head_x + facing * 12, head_y + 8
    sh_off = 5.0

    def put(name: str, x: np.ndarray, y: np.ndarray) -> None:
        coords[:, 0, HALPE26.index(name), 0] = x
        coords[:, 0, HALPE26.index(name), 1] = y

    for side, dz in (("L", -sh_off), ("R", sh_off)):
        put(f"{side}Hip", hip_x + dz, hip_y)
        put(f"{side}Knee", knee_x + dz, knee_y)
        put(f"{side}Ankle", ankle_x + dz, ankle_y)
        put(f"{side}Heel", ankle_x + dz - facing * 12, ankle_y + 8)
        put(f"{side}BigToe", ankle_x + dz + facing * 40, ankle_y + 10)
        put(f"{side}SmallToe", ankle_x + dz + facing * 34, ankle_y + 12)
        put(f"{side}Shoulder", neck_x + dz, neck_y + 10)
        put(f"{side}Elbow", neck_x + dz + facing * 30, neck_y + 60)
        put(f"{side}Wrist", neck_x + dz + facing * 55, neck_y + 40)
        put(f"{side}Eye", nose_x + dz - facing * 6, nose_y - 6)
        put(f"{side}Ear", nose_x + dz - facing * 18, nose_y - 2)
    put("Hip", hip_x, hip_y)
    put("Neck", neck_x, neck_y)
    put("Head", head_x, head_y)
    put("Nose", nose_x, nose_y)

    if noise_px > 0:
        coords += rng.normal(0, noise_px, coords.shape).astype(np.float32)
    if dropout > 0:
        mask = rng.random((T, 1, K)) < dropout
        coords[mask] = np.nan
        score[mask] = 0.0
    track = PoseTrack(
        HALPE26, tl.fps, coords, score, np.array([0], dtype=np.int16), image_size=image_size
    )
    return track, tl


# ---------------------------------------------------------------------------------------------
# Test doubles for the pipeline
# ---------------------------------------------------------------------------------------------
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from ptvision.pose.base import PoseModelInfo  # noqa: E402


class FakeBackend:
    """PoseBackend that ignores pixels and returns a prebuilt track (plus a second faint person for
    the first 15 frames). `delay_s` sleeps per frame and calls `progress`, for cancel tests."""

    def __init__(self, track: PoseTrack, *, delay_s: float = 0.0, second_person: bool = True):
        self._track = track
        self.delay_s = delay_s
        self.second_person = second_person
        self.layout = HALPE26
        self.info = PoseModelInfo(
            backend="fake",
            model_class="Synthetic",
            mode="test",
            layout="halpe26",
            det_sha256="0" * 64,
            pose_sha256="1" * 64,
            det_frequency=1,
            runtime="none",
        )

    def estimate(self, frames, *, fps, image_size, n_frames=None, progress=None):  # type: ignore[no-untyped-def]
        count = 0
        for idx, _frame in frames:
            count = idx + 1
            if self.delay_s:
                time.sleep(self.delay_s)
            if progress is not None:
                progress(count, n_frames)
        n = count
        tr = self._track
        t = min(n, tr.n_frames)
        p = 2 if self.second_person else 1
        coords = np.full((n, p, HALPE26.n, 2), np.nan, np.float32)
        score = np.zeros((n, p, HALPE26.n), np.float32)
        coords[:t, 0] = tr.coords[:t, 0]
        score[:t, 0] = tr.score[:t, 0]
        if self.second_person:
            coords[:15, 1] = tr.coords[:15, 0] * 0.4 + 300
            score[:15, 1] = 0.6
        ids = np.arange(p, dtype=np.int16)
        return PoseTrack(HALPE26, fps, coords, score, ids, image_size=image_size)


def synth_video(path: Path, seconds: float, fps: float, size: str = "854x480") -> Path:
    """A plain grey clip of the given length (needs ffmpeg)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:s={size}:r={fps}:d={seconds}",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(path),
        ],
        check=True,
    )
    return path
