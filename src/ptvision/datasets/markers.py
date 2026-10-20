"""Marker trajectories from a lab motion-capture file"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MarkerTrial:
    path: Path
    rate: float
    labels: list[str]
    points: np.ndarray  # (T, K, 3) metres, NaN where missing
    first_frame: int = 0
    units: str = "m"

    @property
    def n_frames(self) -> int:
        return int(self.points.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.rate

    def index(self, label: str) -> int | None:
        if label in self.labels:
            return self.labels.index(label)
        # tolerate "Subject:LHEE" style prefixes, then case differences (OpenCap mixes
        # `r_knee` with `L_knee`)
        for i, lbl in enumerate(self.labels):
            if lbl.split(":")[-1] == label:
                return i
        want = label.lower()
        for i, lbl in enumerate(self.labels):
            if lbl.split(":")[-1].lower() == want:
                return i
        return None

    def marker(self, label: str) -> np.ndarray | None:
        i = self.index(label)
        return None if i is None else self.points[:, i, :]

    def has(self, *labels: str) -> bool:
        return all(self.index(lbl) is not None for lbl in labels)
