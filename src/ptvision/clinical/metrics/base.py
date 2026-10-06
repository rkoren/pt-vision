"""Metric record and the context handed to metric functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ptvision.kinematics.angles import AngleSeries
from ptvision.pose.track import PoseTrack
from ptvision.trials.models import Capture

ErrorKind = Literal["resolution", "mdd", "rmse", "mae", "ci95", "sd", "unknown"]


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

    _DIGITS: dict[str, int] = {"s": 2, "m/s": 2, "m": 2, "%": 1, "steps/min": 1}

    def formatted(self) -> str:
        if self.value is None:
            return "n/a"
        if self.units == "count":
            return f"{int(self.value)}"
        if self.units == "deg":
            v = f"{round(self.value):d}°"
            return f"{v} ± {round(self.error):d}°" if self.error is not None else v
        digits = self._DIGITS.get(self.units, 1)
        v = f"{self.value:.{digits}f} {self.units}"
        if self.error is None:
            return v
        err_digits = digits
        while err_digits < 3 and 0 < self.error < 0.5 * 10**-err_digits:
            err_digits += 1  # never print a non-zero error as 0.0
        return f"{v} ± {self.error:.{err_digits}f}"


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
