"""Pydantic records written to disk"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ptvision.pose.base import PoseModelInfo

SCHEMA_VERSION = 1

View = Literal["sagittal_left", "sagittal_right", "sagittal", "frontal", "posterior", "unknown"]
Side = Literal["left", "right", "bilateral", "none", "unknown"]


class Subject(BaseModel):
    height_m: float | None = Field(default=None, gt=0.5, lt=2.6)
    mass_kg: float | None = Field(default=None, gt=10, lt=400)
    age_years: int | None = Field(default=None, ge=0, le=120)  # never store a birth date
    sex: Literal["f", "m", "other", "unknown"] | None = None
    affected_side: Side | None = None


class CameraCapture(BaseModel):
    camera_id: str = "cam0"
    source_file: str
    source_sha256: str
    source_codec: str
    source_pix_fmt: str
    source_width: int
    source_height: int
    source_fps_avg: float
    source_fps_nominal: float
    source_is_vfr: bool
    rotation_applied_deg: int
    normalized_file: str
    width: int
    height: int
    fps: float
    n_frames: int | None
    duration_s: float | None
    view: View = "unknown"
    distance_m: float | None = None
    camera_height_m: float | None = None
    device_model: str | None = None


class Capture(BaseModel):
    schema_version: int = SCHEMA_VERSION
    trial_id: str
    patient_id: str | None = None
    visit_id: str | None = None
    episode_id: str | None = None
    protocol_id: str | None = None
    protocol_version: str | None = None
    recorded_at: str | None = None
    ingested_at: str
    subject: Subject = Field(default_factory=Subject)
    cameras: list[CameraCapture]
    notes: str | None = None


class RunProvenance(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    created_at: str
    ptvision_version: str
    git_sha: str | None
    python: str
    platform: str
    pose: PoseModelInfo | None = None
    preprocess: dict[str, object] = Field(default_factory=dict)
    protocol: dict[str, object] = Field(default_factory=dict)
    segmenter: dict[str, object] = Field(default_factory=dict)
    metrics: dict[str, str] = Field(default_factory=dict)  # metric name -> method_version
    quality: dict[str, object] = Field(default_factory=dict)
    pose_parquet: str | None = None
    pose_parquet_sha256: str | None = None
    pose_reused: bool = False  # keypoints loaded from a previous run with matching model
    timing_s: dict[str, float] = Field(default_factory=dict)


class Patient(BaseModel):
    schema_version: int = SCHEMA_VERSION
    patient_id: str
    label: str  # don't use real names
    created_at: str
    subject: Subject = Field(default_factory=Subject)
    notes: str | None = None


class Episode(BaseModel):
    episode_id: str
    label: str
    started: str | None = None
    ended: str | None = None
    notes: str | None = None


class Visit(BaseModel):
    schema_version: int = SCHEMA_VERSION
    visit_id: str
    patient_id: str
    date: str
    episode_id: str | None = None
    notes: str | None = None
