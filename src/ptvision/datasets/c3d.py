# [REVIEW] new file (BACKLOG B29)
"""Minimal c3d access via ezc3d (MIT): marker trajectories in metres plus labelled events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class C3DEvent:
    label: str
    context: str  # "Left" | "Right" | "General" | ""
    time_s: float

    @property
    def frame_at(self) -> None:
        return None


@dataclass
class C3DTrial:
    path: Path
    rate: float
    labels: list[str]
    points: np.ndarray  # (T, K, 3) metres, NaN where missing
    events: list[C3DEvent]
    first_frame: int
    units: str

    @property
    def n_frames(self) -> int:
        return int(self.points.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.rate

    def index(self, label: str) -> int | None:
        if label in self.labels:
            return self.labels.index(label)
        # tolerate "Subject:LHEE" style prefixes
        for i, lbl in enumerate(self.labels):
            if lbl.split(":")[-1] == label:
                return i
        return None

    def marker(self, label: str) -> np.ndarray | None:
        i = self.index(label)
        return None if i is None else self.points[:, i, :]

    def has(self, *labels: str) -> bool:
        return all(self.index(lbl) is not None for lbl in labels)


def read_c3d(path: Path | str) -> C3DTrial:
    import ezc3d

    path = Path(path)
    c = ezc3d.c3d(str(path))
    params = c["parameters"]
    header = c["header"]["points"]
    rate = float(header["frame_rate"])
    first = int(header["first_frame"])
    labels = [str(lbl) for lbl in params["POINT"]["LABELS"]["value"]]
    units = str(params["POINT"]["UNITS"]["value"][0]) if "UNITS" in params["POINT"] else "mm"
    pts = np.asarray(c["data"]["points"], dtype=np.float64)  # (4, K, T)
    xyz = np.transpose(pts[:3], (2, 1, 0))  # (T, K, 3)
    # missing markers: residual < 0 or all-zero
    resid = pts[3] if pts.shape[0] > 3 else None
    missing = np.all(xyz == 0, axis=2)
    if resid is not None:
        missing |= np.transpose(resid) < 0
    xyz = xyz.copy()
    xyz[missing] = np.nan
    if units.lower().startswith("mm"):
        xyz /= 1000.0

    events: list[C3DEvent] = []
    if "EVENT" in params and "TIMES" in params["EVENT"]:
        ev = params["EVENT"]
        times = np.asarray(ev["TIMES"]["value"], dtype=np.float64)  # (2, N): minutes, seconds
        n = times.shape[1] if times.ndim == 2 else 0
        labels_ev = [str(x) for x in ev["LABELS"]["value"]] if "LABELS" in ev else [""] * n
        contexts = [str(x) for x in ev["CONTEXTS"]["value"]] if "CONTEXTS" in ev else [""] * n
        for i in range(n):
            t = float(times[0, i]) * 60.0 + float(times[1, i])
            events.append(
                C3DEvent(labels_ev[i].strip(), contexts[i].strip() if i < len(contexts) else "", t)
            )
        events.sort(key=lambda e: e.time_s)
    return C3DTrial(path, rate, labels, xyz, events, first, units)
