"""Filesystem layout of the data store.

    <root>/patients/<patient_id>/patient.json
    <root>/patients/<patient_id>/episodes.json
    <root>/patients/<patient_id>/visits/<visit_id>/visit.json
    <root>/patients/<patient_id>/visits/<visit_id>/trials/<trial_id>/
        capture.json  source/<original>  video/cam0.mp4  pose/cam0.parquet
        runs/<run_id>/{provenance.json, overlay.mp4, ...}  latest.json

Anonymous trials (no patient) use the same trial layout under an arbitrary output directory.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ptvision._version import __version__
from ptvision.data.ids import new_id, now_iso
from ptvision.data.models import Capture, Episode, Patient, RunProvenance, Visit
from ptvision.io.jsonio import read_json, write_json


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


class DataStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    # ---- anonymous trials ----------------------------------------------------------------
    @staticmethod
    def anonymous_trial(out_dir: Path, trial_id: str | None = None) -> TrialDir:
        """A trial directory outside the patient tree (e.g. ./ptv_out/<clip>/)."""
        trial = TrialDir(Path(out_dir))
        trial.path.mkdir(parents=True, exist_ok=True)
        return trial

    # ---- patients ---------------------------------------------------------------------
    def patients_dir(self) -> Path:
        return self.root / "patients"

    def patient_dir(self, patient_id: str) -> Path:
        return self.patients_dir() / patient_id

    def list_patients(self) -> list[Patient]:
        d = self.patients_dir()
        if not d.exists():
            return []
        out = []
        for p in sorted(d.iterdir()):
            f = p / "patient.json"
            if f.exists():
                out.append(Patient.model_validate(read_json(f)))
        return out

    def get_patient(self, patient_id: str) -> Patient:
        f = self.patient_dir(patient_id) / "patient.json"
        if not f.exists():
            raise KeyError(f"patient {patient_id!r} not found in {self.root}")
        return Patient.model_validate(read_json(f))

    def create_patient(self, patient: Patient) -> Path:
        d = self.patient_dir(patient.patient_id)
        if d.exists():
            raise FileExistsError(d)
        d.mkdir(parents=True)
        write_json(d / "episodes.json", [])
        return write_json(d / "patient.json", patient)

    def episodes(self, patient_id: str) -> list[Episode]:
        f = self.patient_dir(patient_id) / "episodes.json"
        return [Episode.model_validate(e) for e in read_json(f)] if f.exists() else []

    def add_episode(self, patient_id: str, episode: Episode) -> None:
        eps = self.episodes(patient_id)
        if any(e.episode_id == episode.episode_id for e in eps):
            raise FileExistsError(episode.episode_id)
        eps.append(episode)
        write_json(self.patient_dir(patient_id) / "episodes.json", eps)

    # ---- visits -----------------------------------------------------------------------
    def visit_dir(self, patient_id: str, visit_id: str) -> Path:
        return self.patient_dir(patient_id) / "visits" / visit_id

    def list_visits(self, patient_id: str) -> list[Visit]:
        d = self.patient_dir(patient_id) / "visits"
        if not d.exists():
            return []
        return [
            Visit.model_validate(read_json(p / "visit.json"))
            for p in sorted(d.iterdir())
            if (p / "visit.json").exists()
        ]

    def get_visit(self, patient_id: str, visit_id: str) -> Visit:
        f = self.visit_dir(patient_id, visit_id) / "visit.json"
        if not f.exists():
            raise KeyError(f"visit {visit_id!r} not found for patient {patient_id!r}")
        return Visit.model_validate(read_json(f))

    def create_visit(self, visit: Visit) -> Path:
        self.get_patient(visit.patient_id)
        d = self.visit_dir(visit.patient_id, visit.visit_id)
        if d.exists():
            raise FileExistsError(d)
        d.mkdir(parents=True)
        return write_json(d / "visit.json", visit)

    # ---- trials -----------------------------------------------------------------------
    def list_trials(self, patient_id: str, visit_id: str) -> list[TrialDir]:
        d = self.visit_dir(patient_id, visit_id) / "trials"
        if not d.exists():
            return []
        return [TrialDir(p) for p in sorted(d.iterdir()) if p.is_dir()]

    def new_trial(self, patient_id: str, visit_id: str) -> TrialDir:
        self.get_visit(patient_id, visit_id)
        trial = TrialDir(self.visit_dir(patient_id, visit_id) / "trials" / new_id("T"))
        trial.path.mkdir(parents=True, exist_ok=False)
        return trial

    def trial(self, patient_id: str, visit_id: str, trial_id: str) -> TrialDir:
        t = TrialDir(self.visit_dir(patient_id, visit_id) / "trials" / trial_id)
        if not t.path.exists():
            raise KeyError(f"trial {trial_id!r} not found")
        return t
