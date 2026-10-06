"""The report bundle: the JSON-first record of one run and its HTML rendering"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import BaseModel, Field

from ptvision.clinical.metrics.base import Metric, NormComparison
from ptvision.files import write_json
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
    template: str = "sts.html.j2"  # [REVIEW] recorded so `ptv report` re-renders with the same one


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("ptvision.report", "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(bundle: ReportBundle, template: str = "sts.html.j2") -> str:
    return _env().get_template(template).render(b=bundle)


def build_report(
    bundle: ReportBundle, run_dir: Path, template: str = "sts.html.j2"
) -> tuple[Path, Path]:
    """Write report.json (the primary artifact) and report.html rendered from it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(run_dir / "report.json", bundle)
    html_path = run_dir / "report.html"
    html_path.write_text(render_html(bundle, template), encoding="utf-8")
    return json_path, html_path
