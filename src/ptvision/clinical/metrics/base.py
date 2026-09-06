"""Metric record and the context handed to metric functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ptvision.data.models import Capture
from ptvision.kinematics.angles import AngleSeries
from ptvision.pose.track import PoseTrack

ErrorKind = Literal["resolution", "mdd", "rmse", "ci95", "sd", "unknown"]


class Metric(BaseModel):
    name: str
    label: str
    value: float | int | None
    units: str
    error: float | None = None  # half-width in `units`
    error_kind: ErrorKind = "unknown"
    n_events: int | None = None
    side: Literal["left", "right", "near", "bilateral"] | None = None
    tier: Literal[1, 2, 3]
    method_version: str
    citation: str = ""
    flags: list[str] = Field(default_factory=list)
    per_event: list[float] | None = None
    note: str | None = None

    def formatted(self) -> str:
        if self.value is None:
            return "n/a"
        if self.units == "count":
            return f"{int(self.value)}"
        if self.units == "deg":
            v = f"{round(self.value):d}°"
            return f"{v} ± {round(self.error):d}°" if self.error is not None else v
        digits = 2 if self.units == "s" else 1
        v = f"{self.value:.{digits}f} {self.units}"
        return f"{v} ± {self.error:.{digits}f}" if self.error is not None else v


class NormComparison(BaseModel):
    metric: str
    norm_id: str
    population: str
    statistic: str
    reference_value: float | None
    units: str
    direction: Literal["higher_is_worse", "lower_is_worse"]
    statement: str
    citation: str
    applicable: bool = True


@dataclass
class MetricContext:
    raw: PoseTrack  # single person, unfiltered
    track: PoseTrack  # single person, filtered
    angles: AngleSeries
    events: Any
    capture: Capture
    protocol: Any
    fps: float
    params: dict[str, Any] = field(default_factory=dict)
