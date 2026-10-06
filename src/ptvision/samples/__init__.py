"""demo MVP: sample clips for `ptv demo` and the app's sample button
The clip is me doing sit to stands, feel free to use it
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    name: str
    file: str
    protocol: str
    description: str
    height_m: float | None = None
    age_years: int | None = None
    sex: str | None = None
    stopwatch_s: float | None = None # hand-time

    @property
    def path(self) -> Path:
        with resources.as_file(resources.files("ptvision.samples").joinpath(self.file)) as p:
            return Path(p)


SAMPLES: dict[str, Sample] = {
    "sit_to_stand": Sample(
        name="sit_to_stand",
        file="sts_side.mp4",
        protocol="sts_5x",
        description="Five-Times Sit-to-Stand, side view, phone on a tripod, 13 s, 720p at 30 fps.",
        height_m=1.80,
        age_years=26,
        sex="m",
        stopwatch_s=10.75,
    ),
}

DEFAULT_SAMPLE = "sit_to_stand"


def sample(name: str = DEFAULT_SAMPLE) -> Sample:
    try:
        return SAMPLES[name]
    except KeyError as e:
        raise KeyError(f"unknown sample {name!r}; choose from {sorted(SAMPLES)}") from e
