"""Protocol definitions"""

from __future__ import annotations

import hashlib
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ptvision.data.models import View
from ptvision.kinematics.angles import ANGLE_DEFS
from ptvision.viz.skeleton import SEGMENT_NAMES
from ptvision.viz.status import Band, Rule


class ProtocolMeta(BaseModel):
    id: str
    version: str
    name: str
    family: str
    summary: str


class CaptureSpec(BaseModel):
    view: View = "sagittal"
    distance_m: float | None = None
    camera_height_m: float | None = None
    min_fps: float = 24
    duration_s: tuple[float, float] = (2, 600)
    required_keypoints: list[str] = Field(default_factory=list)
    instructions: str = ""


class PoseSpec(BaseModel):
    backend: str = "rtmlib"
    model: str = "body_with_feet"
    mode: str | None = None
    det_frequency: int | None = None


class PreprocessSpec(BaseModel):
    score_threshold: float = 0.3
    interp_max_gap_s: float = 0.3
    filter_type: str = "butterworth"
    filter_order: int = 4
    filter_cutoff_hz: float = 6.0
    segment_signal_cutoff_hz: float = 3.0


class ScaleSpec(BaseModel):
    method: Literal["none", "subject_height"] = "none"
    floor_angle: Literal["auto"] | float = "auto"


class SegmenterSpec(BaseModel):
    name: str
    version: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class MetricSpec(BaseModel):
    name: str
    primary: bool = False
    norm: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RuleSpec(BaseModel):
    """Angle-band rule used to color bones: 1 inside `ok`, falling to 0.5 at the `warn` edge."""

    angle: str
    label: str | None = None
    ok: tuple[float, float]
    warn: tuple[float, float] | None = None
    margin: float = 10.0
    segments: list[str] | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> RuleSpec:
        if self.angle not in ANGLE_DEFS:
            raise ValueError(f"unknown angle {self.angle!r}; known: {sorted(ANGLE_DEFS)}")
        lo, hi = self.ok
        if not lo < hi:
            raise ValueError(f"ok band must have lo < hi, got {self.ok}")
        if self.warn is not None:
            wlo, whi = self.warn
            if not (wlo <= lo and whi >= hi):
                raise ValueError(f"warn band {self.warn} must contain ok band {self.ok}")
        if self.margin < 0:
            raise ValueError("margin must be >= 0")
        for seg in self.segments or []:
            if seg not in SEGMENT_NAMES:
                raise ValueError(f"unknown segment {seg!r}; known: {SEGMENT_NAMES}")
        return self

    def band(self) -> Band:
        lo, hi = self.ok
        warn = self.warn if self.warn is not None else (lo - self.margin, hi + self.margin)
        return Band(ok=(lo, hi), warn=warn)

    def resolved_segments(self) -> tuple[str, ...]:
        if self.segments:
            return tuple(self.segments)
        return ANGLE_DEFS[self.angle].segments

    def to_rule(self) -> Rule:
        return Rule(
            angle=self.angle,
            band=self.band(),
            segments=self.resolved_segments(),
            label=self.label or self.angle,
        )


class ReportSpec(BaseModel):
    template: str = "generic.html.j2"
    angles: list[str] = Field(default_factory=lambda: ["knee_flexion", "hip_flexion", "trunk_lean"])


class Protocol(BaseModel):
    protocol: ProtocolMeta
    capture: CaptureSpec = Field(default_factory=CaptureSpec)
    pose: PoseSpec = Field(default_factory=PoseSpec)
    preprocess: PreprocessSpec = Field(default_factory=PreprocessSpec)
    scale: ScaleSpec = Field(default_factory=ScaleSpec)
    segmenter: SegmenterSpec
    metrics: list[MetricSpec]
    rules: list[RuleSpec] = Field(default_factory=list)
    report: ReportSpec = Field(default_factory=ReportSpec)
    source_sha256: str = ""
    source_path: str = ""

    @property
    def id(self) -> str:
        return self.protocol.id

    @property
    def version(self) -> str:
        return self.protocol.version

    def rule_list(self) -> list[Rule]:
        return [r.to_rule() for r in self.rules]

    def primary_metric(self) -> MetricSpec | None:
        for m in self.metrics:
            if m.primary:
                return m
        return self.metrics[0] if self.metrics else None


def _parse(text: str, source: str) -> Protocol:
    data = tomllib.loads(text)
    # TOML tables named "segmenter" carry params inline; lift non-reserved keys into params.
    seg = dict(data.get("segmenter", {}))
    params = {k: v for k, v in seg.items() if k not in ("name", "version")}
    data["segmenter"] = {"name": seg["name"], "version": seg.get("version"), "params": params}
    proto = Protocol.model_validate(data)
    proto.source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    proto.source_path = source
    return proto


def builtin_protocol_ids() -> list[str]:
    files = resources.files("ptvision.clinical.protocols")
    return sorted(p.name[:-5] for p in files.iterdir() if p.name.endswith(".toml"))


def load_protocol(id_or_path: str | Path) -> Protocol:
    """Load a built-in protocol by id or a TOML file by path."""
    p = Path(id_or_path)
    if p.suffix == ".toml" and p.exists():
        return _parse(p.read_text(encoding="utf-8"), str(p))
    name = str(id_or_path)
    res = resources.files("ptvision.clinical.protocols").joinpath(f"{name}.toml")
    if not res.is_file():
        raise KeyError(f"unknown protocol {name!r}; built-in: {builtin_protocol_ids()}")
    return _parse(res.read_text(encoding="utf-8"), f"builtin:{name}")
