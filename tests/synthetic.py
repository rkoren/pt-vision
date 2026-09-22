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

@dataclass
class GaitTimeline:
    fps: float = 30.0
    duration_s: float = 7.5
    speed_px_s: float = 230.0  # pelvis forward speed (~0.9 m/s at ~260 px/m)
    stride_s: float = 1.1  # one gait cycle
    stance_frac: float = 0.62
    stance_frac_right: float | None = None  # set to make the right side asymmetric
    direction: int = 1  # +1 walks toward image-right
    x0: float = 60.0
    floor_y: float = 900.0
    thigh_px: float = 130.0
    shank_px: float = 130.0
    foot_lift_px: float = 30.0
    image_size: tuple[int, int] = (1920, 1080)
    near_side: str = "left"
    hs: dict[str, list[int]] = field(default_factory=dict)
    to: dict[str, list[int]] = field(default_factory=dict)

    @property
    def stride_px(self) -> float:
        return self.speed_px_s * self.stride_s

    @property
    def n_frames(self) -> int:
        return round(self.duration_s * self.fps)

    def stance_frac_for(self, side: str) -> float:
        if side == "right" and self.stance_frac_right is not None:
            return self.stance_frac_right
        return self.stance_frac


def _two_link_knee(
    hip: np.ndarray, ankle: np.ndarray, l1: float, l2: float, forward: int
) -> np.ndarray:
    """Knee position for a planar two-link leg, choosing the anterior (forward-bending) solution."""
    d_vec = ankle - hip
    d = np.linalg.norm(d_vec, axis=1)
    d = np.clip(d, 1e-6, l1 + l2 - 1e-3)
    unit = d_vec / d[:, None]
    a = (l1**2 - l2**2 + d**2) / (2 * d)
    h = np.sqrt(np.clip(l1**2 - a**2, 0, None))
    p = hip + a[:, None] * unit
    perp = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
    k1 = p + h[:, None] * perp
    k2 = p - h[:, None] * perp
    pick = np.where((k1[:, 0] - k2[:, 0]) * forward >= 0, 1, 0)
    return np.where(pick[:, None] == 1, k1, k2)


def synthetic_gait_track(
    tl: GaitTimeline | None = None,
    *,
    noise_px: float = 0.0,
    seed: int = 0,
) -> tuple[PoseTrack, GaitTimeline]:
    """A single person walking across the frame in the sagittal plane.

    Each foot alternates a stationary stance and a smooth swing of one stride; knees follow from a
    two-link inverse kinematics so knee flexion rises during swing. `tl.hs`/`tl.to` are filled with the
    ground-truth event frames per side.
    """
    tl = tl or GaitTimeline()
    rng = np.random.default_rng(seed)
    fps, T = tl.fps, tl.stride_s
    n = tl.n_frames
    t = np.arange(n) / fps
    hip_x = tl.x0 + tl.speed_px_s * t
    # ankle sits 18 px above the heel; leg ~98.5% extended at midstance -> ~20 deg knee flexion
    hip_y = np.full(n, tl.floor_y - 18 - (tl.thigh_px + tl.shank_px) * 0.985)
    lead = 0.25 * tl.stride_px  # heel lands this far ahead of the pelvis
    feet: dict[str, dict[str, np.ndarray]] = {}
    tl.hs, tl.to = {}, {}
    for side, phase in (("left", 0.0), ("right", 0.5)):
        sf = tl.stance_frac_for(side)
        tau_abs = t / T - phase  # cycles since this side's first (virtual) heel strike
        k = np.floor(tau_abs)
        tau = tau_abs - k
        # landing position of the current cycle's stance foot
        t_land = (k + phase) * T
        x_land = tl.x0 + tl.speed_px_s * t_land + lead
        in_stance = tau < sf
        p = np.clip((tau - sf) / (1 - sf), 0, 1)
        swing_x = x_land + tl.stride_px * smoothstep(p)
        heel_x = np.where(in_stance, x_land, swing_x)
        heel_y = np.where(in_stance, tl.floor_y, tl.floor_y - tl.foot_lift_px * np.sin(np.pi * p))
        feet[side] = {"heel_x": heel_x, "heel_y": heel_y}
        hs_frames = [
            round(((kk + phase) * T) * fps) for kk in range(-1, int(tl.duration_s / T) + 2)
        ]
        to_frames = [
            round(((kk + phase + sf) * T) * fps) for kk in range(-1, int(tl.duration_s / T) + 2)
        ]
        tl.hs[side] = [f for f in hs_frames if 0 <= f < n]
        tl.to[side] = [f for f in to_frames if 0 <= f < n]

    K = HALPE26.n
    coords = np.full((n, 1, K, 2), np.nan, dtype=np.float32)
    score = np.zeros((n, 1, K), dtype=np.float32)

    def put(name: str, x: np.ndarray, y: np.ndarray, sc: float) -> None:
        coords[:, 0, HALPE26.index(name), 0] = x
        coords[:, 0, HALPE26.index(name), 1] = y
        score[:, 0, HALPE26.index(name)] = sc

    lean = np.radians(5.0)
    neck_x = hip_x + 150 * np.sin(lean)
    neck_y = hip_y - 150 * np.cos(lean)
    hip = np.stack([hip_x, hip_y], axis=1)
    for side, dz in (("left", -6.0), ("right", 6.0)):
        near = side == tl.near_side
        sc = 0.9 if near else 0.72
        S = "L" if side == "left" else "R"
        hx, hy = feet[side]["heel_x"], feet[side]["heel_y"]
        ankle = np.stack([hx + 14, hy - 18], axis=1)
        knee = _two_link_knee(hip, ankle, tl.thigh_px, tl.shank_px, forward=1)
        put(f"{S}Hip", hip_x + dz, hip_y, sc)
        put(f"{S}Knee", knee[:, 0] + dz, knee[:, 1], sc)
        put(f"{S}Ankle", ankle[:, 0] + dz, ankle[:, 1], sc)
        put(f"{S}Heel", hx + dz, hy, sc)
        put(f"{S}BigToe", hx + 68 + dz, hy + 2, sc)
        put(f"{S}SmallToe", hx + 60 + dz, hy + 4, sc)
        arm = np.sin(2 * np.pi * (t / T - (0.5 if side == "left" else 0.0)))
        sh_x, sh_y = neck_x + dz, neck_y + 12
        put(f"{S}Shoulder", sh_x, sh_y, sc)
        put(f"{S}Elbow", sh_x + 30 * arm, sh_y + 60, sc)
        put(f"{S}Wrist", sh_x + 55 * arm, sh_y + 95, sc)
        put(f"{S}Eye", neck_x + 20 + dz, neck_y - 38, sc)
        put(f"{S}Ear", neck_x + 6 + dz, neck_y - 34, sc)
    put("Hip", hip_x, hip_y, 0.9)
    put("Neck", neck_x, neck_y, 0.9)
    put("Head", neck_x + 6, neck_y - 32, 0.9)
    put("Nose", neck_x + 24, neck_y - 26, 0.9)

    if tl.direction < 0:
        w = tl.image_size[0]
        coords[..., 0] = w - coords[..., 0]
    if noise_px > 0:
        coords += rng.normal(0, noise_px, coords.shape).astype(np.float32)
    track = PoseTrack(
        HALPE26, fps, coords, score, np.array([0], dtype=np.int16), image_size=tl.image_size
    )
    return track, tl
