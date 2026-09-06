"""Background pipeline execution with cooperative cancellation."""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ptvision.pipeline import AnalyzeOptions, analyze, pose_only
from ptvision.pose.base import PoseBackend


class PipelineCancelled(Exception):
    pass


@dataclass
class JobSpec:
    source: Path
    protocol: str | None = None  # None / "pose" = pose only
    out_dir: Path | None = None
    mode: str | None = None
    device: str | None = None
    det_frequency: int | None = None

    @property
    def is_pose_only(self) -> bool:
        return self.protocol in (None, "", "pose")

    def trial_dir(self) -> Path:
        if self.source.is_dir():
            return self.source
        return self.out_dir or Path("ptv_out") / self.source.stem


@dataclass
class WorkerResult:
    trial_dir: Path
    run_id: str


class PipelineWorker(QThread):
    progress = Signal(int, int)  # done, total (-1 when unknown)
    stage = Signal(str)
    finished_ok = Signal(object)  # WorkerResult
    failed = Signal(str, str)  # message, traceback
    cancelled = Signal()

    def __init__(self, spec: JobSpec, *, backend: PoseBackend | None = None, parent=None):  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.spec = spec
        self._backend = backend
        self._cancel = threading.Event()
        self._last_emit = 0.0
        self._started_at = 0.0

    def cancel(self) -> None:
        self._cancel.set()

    def _check(self) -> None:
        if self._cancel.is_set():
            raise PipelineCancelled()

    def _on_progress(self, done: int, total: int | None) -> None:
        self._check()
        now = time.monotonic()
        if now - self._last_emit >= 0.05 or (total is not None and done >= total):
            self._last_emit = now
            self.progress.emit(int(done), int(total) if total else -1)

    def _on_stage(self, name: str) -> None:
        self._check()
        self.stage.emit(name)

    def _cleanup_orphans(self) -> None:
        runs = self.spec.trial_dir() / "runs"
        if not runs.exists():
            return
        for d in runs.iterdir():
            if (
                d.is_dir()
                and not (d / "provenance.json").exists()
                and d.stat().st_mtime >= self._started_at - 1
            ):
                for f in sorted(d.rglob("*"), reverse=True):
                    if f.is_file():
                        f.unlink()
                    else:
                        f.rmdir()
                d.rmdir()

    def run(self) -> None:
        self._started_at = time.time()
        spec = self.spec
        opts = AnalyzeOptions(
            protocol=spec.protocol or "sts_5x",
            out_dir=spec.out_dir,
            mode=spec.mode,
            device=spec.device,
            det_frequency=spec.det_frequency,
            overlay=False,
        )
        try:
            if spec.is_pose_only:
                res = pose_only(
                    spec.source,
                    opts,
                    backend=self._backend,
                    progress=self._on_progress,
                    on_stage=self._on_stage,
                )
                result = WorkerResult(res.run.path.parent.parent, res.run.run_id)
            else:
                ares = analyze(
                    spec.source,
                    opts,
                    backend=self._backend,
                    progress=self._on_progress,
                    on_stage=self._on_stage,
                )
                result = WorkerResult(ares.trial.path, ares.run.run_id)
            self._check()
            self.finished_ok.emit(result)
        except PipelineCancelled:
            self._cleanup_orphans()
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e), traceback.format_exc())
