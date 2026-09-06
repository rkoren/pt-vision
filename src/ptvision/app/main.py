"""Main window: drop a video → progress → viewer."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ptvision._version import __version__
from ptvision.app.panel import SidePanel
from ptvision.app.player import Player
from ptvision.app.session import TrialSession
from ptvision.app.timeline import AngleTimeline
from ptvision.app.view import VideoView
from ptvision.app.worker import JobSpec, PipelineWorker, WorkerResult
from ptvision.clinical.protocol import builtin_protocol_ids
from ptvision.io.video import FrameSource
from ptvision.pose.base import PoseBackend
from ptvision.viz import status as vs

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".3gp"}
BackendFactory = Callable[[], PoseBackend | None]


def _is_trial_dir(p: Path) -> bool:
    return p.is_dir() and (p / "capture.json").exists()


class DropPage(QWidget):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint = QLabel(
            "<h2>Drop a video here</h2>"
            "<p>or a trial folder from a previous run.<br>"
            "The video is processed on this computer and never uploaded.</p>"
        )
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setStyleSheet("border: 2px dashed #6b7280; border-radius: 14px; padding: 48px;")
        lay.addWidget(self.hint)
        row = QHBoxLayout()
        row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(QLabel("Protocol:"))
        self.protocol = QComboBox()
        self.protocol.addItem("Pose only", "pose")
        for pid in builtin_protocol_ids():
            self.protocol.addItem(pid, pid)
        row.addWidget(self.protocol)
        self.open_video = QPushButton("Open video…")
        self.open_trial = QPushButton("Open trial…")
        row.addWidget(self.open_video)
        row.addWidget(self.open_trial)
        lay.addLayout(row)
        self.note = QLabel(
            f"<span style='color:#9aa4b2'>ptvision {__version__} · "
            "measurement tool, not a medical device</span>"
        )
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.note)

    def selected_protocol(self) -> str:
        return str(self.protocol.currentData())

    def select_protocol(self, pid: str) -> None:
        idx = self.protocol.findData(pid)
        if idx >= 0:
            self.protocol.setCurrentIndex(idx)


class ProgressPage(QWidget):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file = QLabel("")
        self.file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stage = QLabel("starting")
        self.stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stage.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setFixedWidth(420)
        self.cancel = QPushButton("Cancel")
        lay.addWidget(self.file)
        lay.addWidget(self.stage)
        lay.addWidget(self.bar, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.cancel, alignment=Qt.AlignmentFlag.AlignCenter)

    def set_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(done)
        else:
            self.bar.setRange(0, 0)

    def set_stage(self, name: str) -> None:
        self.stage.setText(name)
        self.bar.setRange(0, 0)


class MainWindow(QMainWindow):
    def __init__(self, *, backend_factory: BackendFactory | None = None) -> None:
        super().__init__()
        self.setWindowTitle("ptvision")
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self._backend_factory = backend_factory
        self._worker: PipelineWorker | None = None
        self._session: TrialSession | None = None
        self._player: Player | None = None
        self._job_out: Path | None = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.drop_page = DropPage()
        self.progress_page = ProgressPage()
        self.viewer = QWidget()
        self.stack.addWidget(self.drop_page)
        self.stack.addWidget(self.progress_page)
        self.stack.addWidget(self.viewer)

        # viewer layout
        self.view = VideoView()
        self.timeline = AngleTimeline()
        self.panel = SidePanel()
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.view)
        left.addWidget(self.timeline)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.panel)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        vl = QVBoxLayout(self.viewer)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(split)

        self.drop_page.open_video.clicked.connect(self._choose_video)
        self.drop_page.open_trial.clicked.connect(self._choose_trial)
        self.progress_page.cancel.clicked.connect(self.cancel_job)
        self.timeline.seekRequested.connect(self._seek)
        self.panel.eventActivated.connect(self._seek)
        self.panel.modeChanged.connect(self._set_mode)
        self.panel.angleToggled.connect(self._angle_toggled)
        self.panel.showOthersToggled.connect(self.view.skeleton.set_show_others)

        self._build_actions()
        self.statusBar().showMessage("drop a video to begin")
        # Fallback key handling: QAction shortcuts only fire for the active window, which never
        # happens on headless platforms and can lag behind focus changes; the filter dispatches the
        # same playback commands from any widget inside this window (except text inputs).
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    _KEY_COMMANDS: dict[tuple[int, bool], str] = {
        (int(Qt.Key.Key_Space), False): "toggle",
        (int(Qt.Key.Key_Left), False): "prev",
        (int(Qt.Key.Key_Right), False): "next",
        (int(Qt.Key.Key_Left), True): "prev10",
        (int(Qt.Key.Key_Right), True): "next10",
        (int(Qt.Key.Key_Home), False): "first",
        (int(Qt.Key.Key_End), False): "last",
    }

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.KeyPress
            and self._player is not None
            and self.stack.currentWidget() is self.viewer
            and isinstance(obj, QWidget)
            and obj.window() is self
            and not isinstance(QApplication.focusWidget(), QLineEdit | QAbstractSpinBox)
        ):
            ke = event
            assert isinstance(ke, QKeyEvent)
            mods = ke.modifiers()
            if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
                if ke.key() == Qt.Key.Key_Left:
                    self._seek(0)
                    return True
                if ke.key() == Qt.Key.Key_Right:
                    self._seek_last()
                    return True
                return False
            cmd = self._KEY_COMMANDS.get((ke.key(), bool(mods & Qt.KeyboardModifier.ShiftModifier)))
            if cmd is None:
                return False
            {
                "toggle": self._toggle_play,
                "prev": lambda: self._step(-1),
                "next": lambda: self._step(1),
                "prev10": lambda: self._step(-10),
                "next10": lambda: self._step(10),
                "first": lambda: self._seek(0),
                "last": self._seek_last,
            }[cmd]()
            return True
        return super().eventFilter(obj, event)

    # ---- actions ----------------------------------------------------------------------
    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        act_open = QAction("Open video…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._choose_video)
        act_trial = QAction("Open trial folder…", self)
        act_trial.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_trial.triggered.connect(self._choose_trial)
        act_close = QAction("Close trial", self)
        act_close.setShortcut(QKeySequence("Ctrl+W"))
        act_close.triggered.connect(self.close_trial)
        file_menu.addActions([act_open, act_trial, act_close])

        pb = self.menuBar().addMenu("&Playback")

        def add(text: str, keys: list[str], fn: Callable[[], None]) -> QAction:
            a = QAction(text, self)
            a.setShortcuts([QKeySequence(k) for k in keys])
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            a.triggered.connect(fn)
            pb.addAction(a)
            return a

        self.act_play = add("Play / pause", ["Space"], self._toggle_play)
        self.act_prev = add("Previous frame", ["Left"], lambda: self._step(-1))
        self.act_next = add("Next frame", ["Right"], lambda: self._step(1))
        self.act_prev10 = add("Back 10 frames", ["Shift+Left"], lambda: self._step(-10))
        self.act_next10 = add("Forward 10 frames", ["Shift+Right"], lambda: self._step(10))
        self.act_first = add("First frame", ["Home", "Ctrl+Left"], lambda: self._seek(0))
        self.act_last = add("Last frame", ["End", "Ctrl+Right"], self._seek_last)
        pb.addSeparator()
        self.act_fit = add("Fit to window", ["F"], self.view.fit)
        self.act_mode = add("Toggle colour mode", ["C"], self._toggle_mode)

    # ---- drag and drop ---------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self._handle_paths(paths)

    def _handle_paths(self, paths: list[Path], protocol: str | None = None) -> None:
        for p in paths:
            if _is_trial_dir(p):
                self.open_path(p, protocol=protocol)
                return
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
                self.open_path(p, protocol=protocol)
                return
        QMessageBox.warning(
            self, "ptvision", "Drop a video file or a trial folder (with capture.json)."
        )

    def open_path(
        self,
        p: Path,
        *,
        protocol: str | None = None,
        run_id: str | None = None,
        out_dir: Path | None = None,
    ) -> None:
        protocol = protocol or self.drop_page.selected_protocol()
        if _is_trial_dir(p):
            from ptvision.data.store import TrialDir

            trial = TrialDir(p)
            if trial.latest_run() is not None and trial.pose_path().exists():
                self.open_trial(p, run_id=run_id)
                return
            self.start_job(JobSpec(p, protocol, out_dir))
            return
        self.start_job(JobSpec(p, protocol, out_dir))

    def _choose_video(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "Open video", "", "Videos (*.mp4 *.mov *.m4v *.avi *.mkv *.webm)"
        )
        if f:
            self.open_path(Path(f))

    def _choose_trial(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open trial folder")
        if d:
            self.open_path(Path(d))

    # ---- jobs -------------------------------------------------------------------------
    def start_job(self, spec: JobSpec) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "ptvision", "A job is already running.")
            return
        self.close_trial()
        backend = self._backend_factory() if self._backend_factory else None
        self._worker = PipelineWorker(spec, backend=backend, parent=self)
        self._worker.progress.connect(self.progress_page.set_progress)
        self._worker.stage.connect(self.progress_page.set_stage)
        self._worker.finished_ok.connect(self._job_done)
        self._worker.failed.connect(self._job_failed)
        self._worker.cancelled.connect(self._job_cancelled)
        self._worker.finished.connect(self._worker.deleteLater)
        self.progress_page.file.setText(str(spec.source))
        self.progress_page.set_stage("starting")
        self.stack.setCurrentWidget(self.progress_page)
        self.statusBar().showMessage(f"processing {spec.source.name}")
        self._worker.start()

    def cancel_job(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.progress_page.set_stage("cancelling…")

    def _job_done(self, result: WorkerResult) -> None:
        self._worker = None
        self.open_trial(result.trial_dir, run_id=result.run_id)

    def _job_failed(self, message: str, tb: str) -> None:
        self._worker = None
        self.stack.setCurrentWidget(self.drop_page)
        self.statusBar().showMessage("failed")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("ptvision")
        box.setText(f"Processing failed:\n{message}")
        box.setDetailedText(tb)
        box.exec()

    def _job_cancelled(self) -> None:
        self._worker = None
        self.stack.setCurrentWidget(self.drop_page)
        self.statusBar().showMessage("cancelled")

    # ---- viewer -----------------------------------------------------------------------
    def open_trial(self, trial_dir: Path, *, run_id: str | None = None) -> None:
        try:
            session = TrialSession.load(trial_dir, run_id)
            source = FrameSource(session.video_path)
        except Exception as e:
            QMessageBox.critical(self, "ptvision", f"Could not open trial:\n{e}")
            self.stack.setCurrentWidget(self.drop_page)
            return
        self.close_trial()
        self._session = session
        self._player = Player(session.n_frames, session.fps, self)
        self._player.frameChanged.connect(self._on_frame)
        self._player.playingChanged.connect(self._on_playing)
        self.view.set_session(session, source)
        self.timeline.set_session(session)
        self.panel.set_session(session)
        self.setWindowTitle(f"ptvision — {session.title}")
        self.stack.setCurrentWidget(self.viewer)
        self.view.setFocus()
        self._on_frame(0)
        QTimer.singleShot(0, self.view.fit)

    def close_trial(self) -> None:
        if self._player is not None:
            self._player.pause()
            self._player.deleteLater()
            self._player = None
        self.view.set_session(None, None)
        self.timeline.set_session(None)
        self.panel.set_session(None)
        self._session = None
        self.setWindowTitle("ptvision")
        if self.stack.currentWidget() is self.viewer:
            self.stack.setCurrentWidget(self.drop_page)

    @property
    def session(self) -> TrialSession | None:
        return self._session

    @property
    def player(self) -> Player | None:
        return self._player

    def _on_frame(self, t: int) -> None:
        s = self._session
        if s is None:
            return
        self.view.show_frame(t)
        self.timeline.set_frame(t)
        readout = "  ".join(
            f"{k.replace('_', ' ')} {v:.0f}°" for k, v in s.angle_readout(t).items()
        )
        self.statusBar().showMessage(f"frame {t}/{s.n_frames - 1}   {t / s.fps:.2f} s   {readout}")

    def _on_playing(self, playing: bool) -> None:
        self.act_play.setText("Pause" if playing else "Play")

    def _seek(self, t: int) -> None:
        if self._player is not None:
            self._player.seek(int(t))

    def _seek_last(self) -> None:
        if self._player is not None:
            self._player.seek(self._player.n_frames - 1)

    def _step(self, d: int) -> None:
        if self._player is not None:
            self._player.step(d)

    def _toggle_play(self) -> None:
        if self._player is not None:
            self._player.toggle()

    def _set_mode(self, mode: str) -> None:
        m: vs.Mode = "rules" if mode == "rules" else "confidence"
        self.view.skeleton.set_mode(m)
        self.panel.legend.set_mode(self.view.skeleton.mode)

    def _toggle_mode(self) -> None:
        if self._session is None:
            return
        modes = self._session.available_modes()
        cur = self.view.skeleton.mode
        nxt = modes[(modes.index(cur) + 1) % len(modes)] if cur in modes else modes[0]
        self.panel.set_mode(nxt)

    def _angle_toggled(self, name: str, on: bool) -> None:
        self.timeline.set_angles(self.panel.selected_angles())
        if self._player is not None:
            self.timeline.set_frame(self._player.frame)

    # ---- lifecycle ---------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        self.close_trial()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().closeEvent(event)

    def sizeHint(self) -> QSize:
        return QSize(1280, 820)


def main(
    source: Path | None = None,
    *,
    protocol: str = "pose",
    out_dir: Path | None = None,
    run_id: str | None = None,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("ptvision")
    win = MainWindow()
    win.show()
    if source is not None:
        win.drop_page.select_protocol(protocol)
        win.open_path(Path(source), protocol=protocol, run_id=run_id, out_dir=out_dir)
    shot = os.environ.get("PTV_APP_SCREENSHOT")
    if shot:
        delay = int(os.environ.get("PTV_APP_SCREENSHOT_DELAY_MS", "2500"))

        def grab() -> None:
            win.grab().save(shot)
            app.quit()

        QTimer.singleShot(delay, grab)
    return app.exec()
