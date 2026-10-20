"""OpenCap LabValidation (Uhlrich et al. 2023): iPhone videos synchronised with marker mocap,
OpenSim inverse kinematics and force plates.

Layout under `raw/subject<N>/`:
  VideoData/Session<k>/Cam<c>/<trial>/<trial>_syncdWithMocap.avi   (t=0 = mocap t=0)
  MarkerData/Mocap/<trial>.trc                                       (100 Hz, mm)
  OpenSimData/Mocap/IK/<trial>.mot                                   (100 Hz, degrees)
  ForceData/<trial>_forces.mot                                       (2 kHz; plates R_, L_, 3_)
  sessionMetadata.yaml                                               (height_m, mass_kg, sex)

Cameras stand on an arc in front of the subject (checked on frame grabs of subject2 STS1 and
on the extrinsics): Cam2 faces the subject, Cam1/Cam3 sit about 45 degrees to either side,
Cam0/Cam4 about 70 degrees. Cam0 is the most side-on view (subject nearly in profile, facing
image-left); nothing is a true 90-degree side view.

Sign conventions of the IK coordinates were fixed by comparing with the marker-projected angles on
subject2 STS1 (see notebooks/validation.ipynb), not against our video output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ptvision.datasets.lab_trial import JOINT_ANGLE_COLUMNS, LabTrial, Side
from ptvision.datasets.mocap_projection import ProjectedTrial, project_trial
from ptvision.datasets.opensim_io import read_mot, read_trc
from ptvision.datasets.registry import DatasetDirs, load_manifest
from ptvision.datasets.sts_force import (
    ForceStsEvents,
    G,
    events_from_feet_plates,
    pelvis_rising,
)

STS_TASKS: tuple[str, ...] = ("STS1", "STSweakLegs1")
CAMERAS: tuple[int, ...] = (0, 1, 2, 3, 4)

# nominal azimuth (just means horizontal angle) of each camera
# based on subject's facing direction (0 = frontal, ±90 = side)
CAMERA_VIEW_DEG: dict[int, float] = {0: -70.0, 1: -45.0, 2: 0.0, 3: 45.0, 4: 70.0}
SUBJECT_RE = re.compile(r"^subject(\d+)$")


def root() -> Path:
    return DatasetDirs.for_spec(load_manifest()["opencap"]).raw


def subjects(raw: Path | None = None) -> list[str]:
    raw = raw or root()
    if not raw.exists():
        return []
    found = [p.name for p in raw.iterdir() if p.is_dir() and SUBJECT_RE.match(p.name)]
    return sorted(found, key=lambda n: int(SUBJECT_RE.match(n).group(1)))  # type: ignore[union-attr]


@lru_cache(maxsize=32)
def metadata(subject_dir: Path) -> dict[str, Any]:
    import yaml

    p = subject_dir / "sessionMetadata.yaml"
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return dict(data) if isinstance(data, dict) else {}


@dataclass
class OpenCapTrial(LabTrial):
    trc: Path = Path()
    ik: Path | None = None
    forces: Path | None = None

    def reference_track(self, fps: float) -> ProjectedTrial:
        return project_trial(read_trc(self.trc), fps=fps, forward_rule="lean")

    def reference_joint_angles(self, side: Side) -> pd.DataFrame | None:
        if self.ik is None or not self.ik.exists():
            return None
        df = read_mot(self.ik)
        if df.attrs.get("in_degrees") is False:
            for c in df.columns:
                if c != "time":
                    df[c] = np.degrees(df[c])
        s = "r" if side == "right" else "l"
        out = pd.DataFrame({"time": df["time"].to_numpy(dtype=np.float64)})
        out["knee_flexion"] = df[f"knee_angle_{s}"].to_numpy(dtype=np.float64)
        # our hip angle is thigh vs trunk (neck-hip-knee); the model's hip_flexion is thigh vs
        # pelvis, so the lumbar flexion (negative lumbar_extension) is added
        out["hip_flexion"] = (df[f"hip_flexion_{s}"] - df["lumbar_extension"]).to_numpy(
            dtype=np.float64
        )
        # torso angle from vertical: pelvis tilt plus lumbar extension, forward positive
        out["trunk_lean"] = -(df["pelvis_tilt"] + df["lumbar_extension"]).to_numpy(dtype=np.float64)
        return out[["time", *JOINT_ANGLE_COLUMNS]]

    def force_sts_events(self) -> ForceStsEvents | None:
        if self.forces is None or not self.forces.exists() or not self.mass_kg:
            return None
        df = read_mot(self.forces)
        t = df["time"].to_numpy(dtype=np.float64)
        feet = np.zeros_like(t)
        for pre in ("R_", "L_"):
            col = f"{pre}ground_force_vy"
            if col in df:
                feet += np.abs(df[col].to_numpy(dtype=np.float64))
        rising = pelvis_rising(self.reference_track(100.0), t)
        return events_from_feet_plates(t, feet, self.mass_kg * G, rising=rising)


def trials(
    *,
    tasks: tuple[str, ...] = STS_TASKS,
    subjects_wanted: tuple[str, ...] | None = None,
    raw: Path | None = None,
) -> list[OpenCapTrial]:
    """Every (subject, task) that has at least one synced video; cameras found per trial."""
    raw = raw or root()
    out: list[OpenCapTrial] = []
    for subj in subjects(raw):
        if subjects_wanted and subj not in subjects_wanted:
            continue
        sdir = raw / subj
        meta = metadata(sdir)
        for task in tasks:
            videos: dict[int, Path] = {}
            for avi in sorted(
                sdir.glob(f"VideoData/Session*/Cam*/{task}/{task}_syncdWithMocap.avi")
            ):
                m = re.search(r"Cam(\d+)", str(avi.parent.parent))
                if m:
                    videos[int(m.group(1))] = avi
            if not videos:
                continue
            trc = sdir / "MarkerData" / "Mocap" / f"{task}.trc"
            if not trc.exists():
                continue
            ik = sdir / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
            forces = sdir / "ForceData" / f"{task}_forces.mot"
            out.append(
                OpenCapTrial(
                    dataset="opencap",
                    subject=subj,
                    task=task,
                    videos=videos,
                    view_deg={c: CAMERA_VIEW_DEG.get(c, float("nan")) for c in videos},
                    height_m=_float_or_none(meta.get("height_m")),
                    mass_kg=_float_or_none(meta.get("mass_kg")),
                    sex=str(meta["sex"]) if meta.get("sex") else None,
                    trc=trc,
                    ik=ik if ik.exists() else None,
                    forces=forces if forces.exists() else None,
                )
            )
    return out


def _float_or_none(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
