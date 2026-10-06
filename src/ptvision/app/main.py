"""Main window"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSize, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QDragEnterEvent,
    QDropEvent,
    QKeyEvent,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QFileDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ptvision.app.pages import DropPage, ProgressPage
from ptvision.app.panel import SidePanel
from ptvision.app.player import Player
from ptvision.app.session import TrialSession
from ptvision.app.timeline import AngleTimeline
from ptvision.app.view import VideoView
from ptvision.app.worker import JobSpec, PipelineWorker, WorkerResult
from ptvision.pose.base import PoseBackend
from ptvision.trials.models import Subject
from ptvision.video import FrameSource
from ptvision.viz import status as vs

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".3gp"}
BackendFactory = Callable[[], PoseBackend | None]


def _is_trial_dir(p: Path) -> bool:
    return p.is_dir() and (p / "capture.json").exists()


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
        self.screenshot_mode = False  # headless capture: never block on modal dialogs

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
        self.drop_page.try_sample.clicked.connect(self.open_sample)
        self.drop_page.recent.itemActivated.connect(
            lambda item: self.open_trial(Path(str(item.data(Qt.ItemDataRole.UserRole))))
        )
        self.progress_page.cancel.clicked.connect(self.cancel_job)
        self.timeline.seekRequested.connect(self._seek)
        self.panel.eventActivated.connect(self._seek)
        self.panel.modeChanged.connect(self._set_mode)
        self.panel.angleToggled.connect(self._angle_toggled)
        self.panel.showOthersToggled.connect(self.view.skeleton.set_show_others)
        self.panel.smoothToggled.connect(self.view.skeleton.set_smooth)

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
            from ptvision.trials.store import TrialDir

            trial = TrialDir(p)
            if trial.latest_run() is not None and trial.pose_path().exists():
                self.open_trial(p, run_id=run_id)
                return
            self.start_job(self._job(p, protocol, out_dir))
            return
        self.start_job(self._job(p, protocol, out_dir))

    def _job(self, p: Path, protocol: str | None, out_dir: Path | None) -> JobSpec:
        return JobSpec(
            p,
            protocol,
            out_dir,
            subject=self.drop_page.subject(),
            max_height=self.drop_page.selected_max_height(),
        )

    def open_sample(self, name: str | None = None) -> None:
        """Run the bundled sample clip with its protocol and subject details (demo MVP)."""
        from ptvision.samples import DEFAULT_SAMPLE, sample

        smp = sample(name or DEFAULT_SAMPLE)
        self.drop_page.select_protocol(smp.protocol)
        self.drop_page.set_subject(height_m=smp.height_m, age=smp.age_years, sex=smp.sex)
        self.open_path(smp.path, protocol=smp.protocol, out_dir=Path("ptv_out") / "sample")

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
        self._worker.preview.connect(self.progress_page.set_preview)
        self.progress_page.clear_preview()
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
        self.statusBar().showMessage(f"failed: {message}")
        if self.screenshot_mode:
            return
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
        start = session.first_present_frame
        if start:
            self._player.seek(start)  # emits frameChanged -> _on_frame
        else:
            self._on_frame(0)
        QTimer.singleShot(0, self.view.fit)
        self.drop_page.remember(trial_dir)

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
    def job_running(self) -> bool:
        return self._worker is not None

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
        values = s.angle_readout(t)
        self.panel.set_readout(values)
        rep = s.rep_at(t)
        self.view.set_badge(f"Rep {rep[0]} / {rep[1]}" if rep else None)
        readout = "  ".join(f"{k.replace('_', ' ')} {v:.0f}°" for k, v in values.items())
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
    subject: Subject | None = None,
    max_height: int | None = None,
    hard_exit: bool = True,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("ptvision")
    win = MainWindow()
    if subject is not None:
        win.drop_page.set_subject(height_m=subject.height_m, age=subject.age_years, sex=subject.sex)
    if max_height:
        win.drop_page.max_height_box.setCurrentIndex(
            max(0, win.drop_page.max_height_box.findData(int(max_height)))
        )
    win.show()
    if source is not None:
        win.drop_page.select_protocol(protocol)
        win.open_path(Path(source), protocol=protocol, run_id=run_id, out_dir=out_dir)
    shot = os.environ.get("PTV_APP_SCREENSHOT")
    if shot:
        win.screenshot_mode = True
        arm_screenshot(
            app,
            win,
            Path(shot),
            expect_viewer=source is not None,
            delay_ms=int(os.environ.get("PTV_APP_SCREENSHOT_DELAY_MS", "1500")),
            timeout_ms=int(os.environ.get("PTV_APP_SCREENSHOT_TIMEOUT_MS", "900000")),
        )
    code = app.exec()
    if hard_exit:
        # [REVIEW] in the app: after the event loop ends, interpreter teardown races ONNX
        # Runtime's worker threads (created inside the QThread job) and aborts with
        # "recursive_mutex lock failed" once the report is already written. Everything is flushed
        # and saved by now, so skip teardown. Tests call main() with hard_exit=False.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


def arm_screenshot(
    app: QCoreApplication,
    win: MainWindow,
    path: Path,
    *,
    expect_viewer: bool,
    delay_ms: int = 1500,
    timeout_ms: int = 900_000,
    poll_ms: int = 200,
    on_done: Callable[[int], None] | None = None,
) -> None:
    """[REVIEW]. Save a PNG of the window and quit with a meaningful exit code.

    With a video source the pipeline runs first, so the old fixed 2.5 s timer captured the
    progress page. Now: wait until the viewer page is showing (or the job has ended without it),
    then ``delay_ms`` later grab the window. Exit 0 on success, 2 if the PNG could not be written,
    3 if the viewer never appeared (the current page is still saved so the failure is visible).
    """
    elapsed = 0
    finish = on_done or app.exit  # tests pass a recorder: app.exit() poisons later event loops

    def grab(code: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = win.grab().save(str(path))
        finish(code if ok else 2)

    def poll() -> None:
        nonlocal elapsed
        elapsed += poll_ms
        if win.stack.currentWidget() is win.viewer:
            QTimer.singleShot(delay_ms, lambda: grab(0))
        elif elapsed >= timeout_ms or (
            win.stack.currentWidget() is win.drop_page and not win.job_running
        ):
            grab(3)
        else:
            QTimer.singleShot(poll_ms, poll)

    if expect_viewer:
        QTimer.singleShot(poll_ms, poll)
    else:
        QTimer.singleShot(delay_ms, lambda: grab(0))
