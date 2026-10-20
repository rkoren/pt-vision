"""Filesystem layout of a trial directory (`ptv_out/<clip>/` by default).

    <trial>/capture.json  source/<original>  video/cam0.mp4  pose/cam0.parquet
    <trial>/runs/<run_id>/{provenance.json, overlay.mp4, ...}  latest.json

`capture.json` is written once per trial; every processing run gets its own `runs/<run_id>/`.
"""

from __future__ import annotations

import platform
import secrets
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ptvision._version import __version__
from ptvision.files import read_json, write_json
from ptvision.trials.models import Capture, RunProvenance

# ---- ids ---------------------------------------------------------------------------------


def new_id(prefix: str, when: datetime | None = None) -> str:
    """Sortable, human-readable id like `T-20260901-140322-a3f9`."""
    when = when or datetime.now(UTC)
    return f"{prefix}-{when.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_sha(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd or Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass(frozen=True)
class RunDir:
    path: Path

    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def provenance_path(self) -> Path:
        return self.path / "provenance.json"

    def write_provenance(self, prov: RunProvenance) -> Path:
        return write_json(self.provenance_path, prov)

    def read_provenance(self) -> RunProvenance:
        return RunProvenance.model_validate(read_json(self.provenance_path))


@dataclass(frozen=True)
class TrialDir:
    path: Path

    @property
    def trial_id(self) -> str:
        return self.path.name

    @property
    def capture_path(self) -> Path:
        return self.path / "capture.json"

    @property
    def source_dir(self) -> Path:
        return self.path / "source"

    @property
    def video_dir(self) -> Path:
        return self.path / "video"

    @property
    def pose_dir(self) -> Path:
        return self.path / "pose"

    @property
    def runs_dir(self) -> Path:
        return self.path / "runs"

    @property
    def latest_path(self) -> Path:
        return self.path / "latest.json"

    def video_path(self, camera_id: str = "cam0") -> Path:
        return self.video_dir / f"{camera_id}.mp4"

    def pose_path(self, camera_id: str = "cam0") -> Path:
        return self.pose_dir / f"{camera_id}.parquet"

    def has_capture(self) -> bool:
        return self.capture_path.exists()

    def read_capture(self) -> Capture:
        return Capture.model_validate(read_json(self.capture_path))

    def write_capture(self, capture: Capture) -> Path:
        return write_json(self.capture_path, capture)

    def new_run(self) -> RunDir:
        run = RunDir(self.runs_dir / new_id("R"))
        run.path.mkdir(parents=True, exist_ok=False)
        return run

    def runs(self) -> list[RunDir]:
        if not self.runs_dir.exists():
            return []
        return [RunDir(p) for p in sorted(self.runs_dir.iterdir()) if p.is_dir()]

    def set_latest(self, run: RunDir) -> None:
        write_json(self.latest_path, {"run_id": run.run_id, "updated_at": now_iso()})

    def latest_run(self) -> RunDir | None:
        if not self.latest_path.exists():
            return None
        return RunDir(self.runs_dir / read_json(self.latest_path)["run_id"])


def base_provenance(run_id: str) -> RunProvenance:
    return RunProvenance(
        run_id=run_id,
        created_at=now_iso(),
        ptvision_version=__version__,
        git_sha=git_sha(),
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
    )


def anonymous_trial(out_dir: Path) -> TrialDir:
    """A trial directory at `out_dir`, created if needed."""
    trial = TrialDir(Path(out_dir))
    trial.path.mkdir(parents=True, exist_ok=True)
    return trial
