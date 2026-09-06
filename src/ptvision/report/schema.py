from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ptvision.clinical.metrics.base import Metric, NormComparison
from ptvision.quality.checks import QualityReport

DISCLAIMER = (
    "ptvision reports measurements from video against published reference values with their known "
    "measurement error. It does not diagnose, classify, or recommend treatment; all "
    "interpretation is the "
    "responsibility of a licensed clinician. Not a medical device. Video was processed locally."
)


class Figure(BaseModel):
    name: str
    path: str  # relative to the run directory
    caption: str = ""


class ReportBundle(BaseModel):
    schema_version: int = 1
    title: str
    protocol: dict[str, Any]
    trial_id: str
    run_id: str
    created_at: str
    ptvision_version: str
    capture: dict[str, Any]
    subject: dict[str, Any]
    quality: QualityReport
    primary: Metric | None = None
    primary_norm: NormComparison | None = None
    metrics: list[Metric] = Field(default_factory=list)
    norms: list[NormComparison] = Field(default_factory=list)
    events: dict[str, Any] = Field(default_factory=dict)
    figures: list[Figure] = Field(default_factory=list)
    thumbnails: list[Figure] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    angle_definitions: list[dict[str, str]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
