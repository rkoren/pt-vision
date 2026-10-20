"""The core package must never import research-licensed code or data."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "ptvision"

# Research-only or copyleft projects that must never be imported by the core package: the
# SMPL/SMPL-X body models and everything built on them (WHAM, GVHMR, TRAM, HMR2.0, HSMR/SKEL), the
# BEDLAM, AMASS and Human3.6M datasets, Sapiens (CC-BY-NC), opencap-monocular (PolyForm-NC), and
# AGPL/GPL code (Ultralytics YOLO, OpenPose, FreeMoCap, PosePipeline, MonocularBiomechanics, PyAV
# with its GPL FFmpeg build).
FORBIDDEN = [
    r"\bsmplx?\b",
    r"\bbedlam\b",
    r"\bgvhmr\b",
    r"\btram\b",
    r"\bhmr2?\b",
    r"\bskel\b",
    r"\bhsmr\b",
    r"\bwham\b",
    r"\bamass\b",
    r"\bh36m\b",
    r"\bsapiens\b",
    r"\bopencap_monocular\b",
    r"\bultralytics\b",
    r"\bopenpose\b",
    r"\bfreemocap\b",
    r"\bposepipeline\b",
    r"\bmonocularbiomechanics\b",
    r"\bav\b",
]


def test_no_research_licensed_imports() -> None:
    pattern = re.compile("|".join(FORBIDDEN), re.IGNORECASE)
    hits: list[str] = []
    for py in SRC.rglob("*.py"):
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith(('"', "'")):
                continue  # comments and docstrings may mention them
            if stripped.startswith(("import ", "from ")) and pattern.search(stripped):
                hits.append(f"{py.relative_to(SRC)}:{lineno}: {stripped}")
    assert not hits, "research-licensed imports in core package:\n" + "\n".join(hits)
