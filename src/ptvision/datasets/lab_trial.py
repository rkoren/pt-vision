"""Validating whether we can compare a lab trial to our tooling"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from ptvision.datasets.mocap_projection import ProjectedTrial
from ptvision.datasets.sts_force import ForceStsEvents

Side = Literal["left", "right"]

# columns `reference_joint_angles` may return, in degrees, in our conventions
JOINT_ANGLE_COLUMNS = ("knee_flexion", "hip_flexion", "trunk_lean")


@dataclass
class LabTrial(ABC):
    dataset: str
    subject: str
    task: str
    videos: dict[int, Path]  # camera index -> video whose t=0 is the reference's t=0
    view_deg: dict[int, float]  # nominal camera azimuth: 0 = facing the subject, ±90 = side
    height_m: float | None = None
    mass_kg: float | None = None
    sex: str | None = None
    n_reps_expected: int = 5  # -1 = count whatever the recording holds
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.subject}_{self.task}"

    @abstractmethod
    def reference_track(self, fps: float) -> ProjectedTrial:
        """Lab joint centres projected onto the sagittal plane as a Halpe-26 track at `fps`,
        frame 0 at t=0 of the videos."""

    @abstractmethod
    def reference_joint_angles(self, side: Side) -> pd.DataFrame | None:
        """The lab's own model coordinates mapped to our angle names (degrees) with a `time`
        column, or None when the dataset has none."""

    @abstractmethod
    def force_sts_events(self) -> ForceStsEvents | None:
        """Sit-to-stand events from the force plates, or None when forces are missing."""
