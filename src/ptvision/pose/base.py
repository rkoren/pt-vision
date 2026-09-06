"""PoseBackend protocol and model provenance record."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from ptvision.pose.layout import KeypointLayout
from ptvision.pose.track import PoseTrack


class PoseModelInfo(BaseModel):
    """Everything needed to reproduce a pose run, written into run provenance."""

    backend: str  # "rtmlib", "mediapipe"
    model_class: str  # "BodyWithFeet"
    mode: str | None = None
    layout: str
    det_name: str | None = None
    det_url: str | None = None
    det_sha256: str | None = None
    det_input_size: tuple[int, int] | None = None
    pose_name: str | None = None
    pose_url: str | None = None
    pose_sha256: str | None = None
    pose_input_size: tuple[int, int] | None = None
    runtime: str | None = None  # e.g. "onnxruntime 1.29.0"
    providers: list[str] = []
    device: str | None = None
    det_frequency: int | None = None
    tracker: dict[str, object] = {}


ProgressFn = Callable[[int, int | None], None]


@runtime_checkable
class PoseBackend(Protocol):
    info: PoseModelInfo
    layout: KeypointLayout

    def estimate(
        self,
        frames: Iterable[tuple[int, np.ndarray]],
        *,
        fps: float,
        image_size: tuple[int, int],
        n_frames: int | None = None,
        progress: ProgressFn | None = None,
    ) -> PoseTrack: ...
