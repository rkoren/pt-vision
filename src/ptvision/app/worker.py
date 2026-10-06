"""Background pipeline execution, has cooperative cancellation"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from ptvision.pipeline import AnalyzeOptions, analyze, default_trial_dir, pose_only
from ptvision.pose.base import PoseBackend
from ptvision.trials.models import Subject


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
    subject: Subject | None = None
    max_height: int | None = None

    @property
    def is_pose_only(self) -> bool:
        return self.protocol in (None, "", "pose")

    def trial_dir(self) -> Path:
        if self.source.is_dir():
            return self.source
        return self.out_dir or default_trial_dir(self.source)


@dataclass
class WorkerResult:
    trial_dir: Path
    run_id: str


class PipelineWorker(QThread):
    progress = Signal(int, int)  # done, total (-1 when unknown)
    stage = Signal(str)
    finished_ok = Signal(object)  # WorkerResult
    failed = Signal(str, str)  # message + traceback
    cancelled = Signal()
    preview = Signal(object)  # preview frame

    def __init__(self, spec: JobSpec, *, backend: PoseBackend | None = None, parent=None):  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.spec = spec
        self._backend = backend
        self._cancel = threading.Event()
        self._last_emit = 0.0
        self._last_preview = 0.0
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

    def _on_preview(self, idx: int, frame: Any, kpts: Any, scores: Any) -> None:
        """at most ~6 preview frames a second, skeleton coloured by confidence"""
        now = time.monotonic()
        if now - self._last_preview < 0.15:
            return
        self._last_preview = now
        import cv2
        import numpy as np

        from ptvision.pose.layout import HALPE26
        from ptvision.pose.overlay import draw_pose

        img = np.ascontiguousarray(frame).copy()
        k = np.asarray(kpts, np.float32)
        s = np.asarray(scores, np.float32)
        if k.ndim == 3 and k.shape[0]:
            draw_pose(img, k, s, HALPE26, np.arange(k.shape[0]), mode="confidence")
        h, w = img.shape[:2]
        scale = 720 / max(h, w)
        if scale < 1:
            img = cv2.resize(
                img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
        self.preview.emit(img)

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
            subject=spec.subject,
            max_height=spec.max_height,
        )
        try:
            if spec.is_pose_only:
                res = pose_only(
                    spec.source,
                    opts,
                    backend=self._backend,
                    progress=self._on_progress,
                    on_stage=self._on_stage,
                    preview=self._on_preview,
                )
                result = WorkerResult(res.run.path.parent.parent, res.run.run_id)
            else:
                ares = analyze(
                    spec.source,
                    opts,
                    backend=self._backend,
                    progress=self._on_progress,
                    on_stage=self._on_stage,
                    preview=self._on_preview,
                )
                result = WorkerResult(ares.trial.path, ares.run.run_id)
            self._check()
            self.finished_ok.emit(result)
        except PipelineCancelled:
            self._cleanup_orphans()
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e), traceback.format_exc())
