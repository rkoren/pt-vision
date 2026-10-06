"""The core package must never import research-licensed code or data."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "ptvision"

FORBIDDEN = [
    r"\bsmplx?\b",
    r"\bbedlam\b",
    r"\bgvhmr\b",
    r"\bskel\b",
    r"\bhsmr\b",
    r"\bwham\b",
    r"\bamass\b",
    r"\bultralytics\b",
    r"\bopenpose\b",
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
