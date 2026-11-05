"""Main window"""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QDesktopServices,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDropEvent,
    QKeyEvent,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ptvision.app import theme
from ptvision.app.cards import ResultColumn
from ptvision.app.pages import TEST_NAMES, DropPage, ProgressPage
from ptvision.app.panel import SidePanel
from ptvision.app.player import Player
from ptvision.app.session import TrialSession
from ptvision.app.theme import apply_theme
from ptvision.app.timeline import AngleTimeline
from ptvision.app.transport import TransportBar
from ptvision.app.view import VideoView
from ptvision.app.worker import JobSpec, PipelineWorker, WorkerResult
from ptvision.pose.base import PoseBackend
from ptvision.trials.models import Subject
from ptvision.video import FrameSource
from ptvision.viz import status as vs

SIDE_WIDTH = 440  # the right-hand column of the result view
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".3gp"}
BackendFactory = Callable[[], PoseBackend | None]


def _is_trial_dir(p: Path) -> bool:
    return p.is_dir() and (p / "capture.json").exists()


def friendly_failure(message: str) -> tuple[str, str]:
    """A title and a next step for a failed job, in words for whoever is at the laptop."""
    if "no person detected" in message.lower():
        return (
            "No one was found in the video",
            "Film again with the whole body, both feet and the chair in view.",
        )
    return "The video could not be processed", message


def result_title(session: TrialSession) -> str:
    if session.protocol is None:
        return TEST_NAMES["pose"]
    meta = session.protocol.protocol
    return TEST_NAMES.get(meta.id, meta.name)


def result_subtitle(session: TrialSession) -> str:
    parts = [session.trial.path.name]
    age = session.capture.subject.age_years
    if age is not None:
        parts.append(f"Age {age}")
    return " · ".join(parts)


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
        self._job_source: Path | None = None
        self._show_clock = False
        self.screenshot_mode = False  # headless capture

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.drop_page = DropPage()
        self.progress_page = ProgressPage()
        self.viewer = QWidget()
        self.stack.addWidget(self.drop_page)
        self.stack.addWidget(self.progress_page)
        self.stack.addWidget(self.viewer)

        # viewer: a header, the video with its overlays and transport, the panel on the right;
        # Details adds the angle timeline under the video
        self.view = VideoView()
        self.transport = TransportBar()
        self.timeline = AngleTimeline()
        self.timeline.hide()
        self.panel = SidePanel()
        self.cards = ResultColumn()
        self.act_details = QAction("Details", self)
        self.act_details.setCheckable(True)
        self.act_details.setShortcut(QKeySequence("Ctrl+D"))
        self.act_details.setToolTip("Angle timeline and display settings (⌘D)")
        self.act_details.toggled.connect(self.set_details)
        self._build_viewer()

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

    def _build_viewer(self) -> None:
        self.new_test_button = QToolButton()
        self.new_test_button.setText("‹  New test")
        self.new_test_button.setToolTip("Back to the start page (⌘W)")
        self.new_test_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.new_test_button.clicked.connect(self.close_trial)
        self.details_button = QToolButton()
        self.details_button.setDefaultAction(self.act_details)
        self.details_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.result_title = QLabel()
        theme.set_role(self.result_title, "headline")
        self.result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_subtitle = QLabel()
        theme.set_role(self.result_subtitle, "caption")
        self.result_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(self.result_title)
        titles.addWidget(self.result_subtitle)
        header = QHBoxLayout()
        header.addWidget(self.new_test_button)
        header.addStretch(1)
        header.addLayout(titles)
        header.addStretch(1)
        header.addWidget(self.details_button)
        # the two ends share a width, so the title centres over the video
        width = max(self.new_test_button.sizeHint().width(), 110)
        self.new_test_button.setMinimumWidth(width)
        self.details_button.setMinimumWidth(width)

        video = QWidget()
        vb = QVBoxLayout(video)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(10)
        vb.addWidget(self.view, 1)
        vb.addWidget(self.transport)
        self.left_split = QSplitter(Qt.Orientation.Vertical)
        self.left_split.setChildrenCollapsible(False)
        self.left_split.addWidget(video)
        self.left_split.addWidget(self.timeline)
        self.left_split.setStretchFactor(0, 3)
        self.left_split.setStretchFactor(1, 2)
        left = QVBoxLayout()
        left.setSpacing(10)
        left.addLayout(header)
        left.addWidget(self.left_split, 1)

        self.side = QStackedWidget()
        self.side.addWidget(self.cards)
        self.side.addWidget(self.panel)
        self.side.setFixedWidth(SIDE_WIDTH)
        row = QHBoxLayout(self.viewer)
        row.setContentsMargins(16, 10, 16, 16)
        row.setSpacing(16)
        row.addLayout(left, 1)
        row.addWidget(self.side)

        self.transport.playToggled.connect(self._toggle_play)
        self.transport.stepRequested.connect(self._step)
        self.transport.scrubbed.connect(self._scrub)
        self.transport.speedRequested.connect(self._cycle_speed)
        self.cards.seekRequested.connect(self._scrub)
        self.cards.report_button.clicked.connect(self.open_report)

    def set_details(self, on: bool) -> None:
        """Details: the angle timeline under the video, and the settings panel in place of the
        result cards."""
        if self.act_details.isChecked() != on:
            self.act_details.setChecked(on)  # re-enters through toggled
            return
        self.timeline.setVisible(on)
        self.side.setCurrentWidget(self.panel if on else self.cards)
        if on:
            total = max(1, self.left_split.height())
            self.left_split.setSizes([round(total * 0.62), round(total * 0.38)])
            if self._player is not None:
                self.timeline.set_frame(self._player.frame)

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
        self.act_report = QAction("Open report", self)
        self.act_report.setShortcut(QKeySequence("Ctrl+R"))
        self.act_report.triggered.connect(self.open_report)
        self.act_report.setEnabled(False)
        file_menu.addActions([act_open, act_trial, act_close])
        file_menu.addSeparator()
        file_menu.addAction(self.act_report)

        # "Window", not "View": macOS adds its own Enter Full Screen item to a menu titled View
        window_menu = self.menuBar().addMenu("&Window")
        self.act_fullscreen = QAction("Full screen", self)
        self.act_fullscreen.setShortcuts(
            [*QKeySequence.keyBindings(QKeySequence.StandardKey.FullScreen), QKeySequence("F11")]
        )
        self.act_fullscreen.triggered.connect(self.toggle_full_screen)
        window_menu.addAction(self.act_fullscreen)
        window_menu.addAction(self.act_details)

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
        self.act_speed = add("Playback speed (1×, ½×, ¼×)", ["S"], self._cycle_speed)
        pb.addSeparator()
        self.act_fit = add("Fit to window", ["F"], self.view.fit)
        self.act_mode = add("Toggle colour mode", ["C"], self._toggle_mode)

    # ---- drag and drop ---------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_page.set_drag_active(True)

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self.drop_page.set_drag_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self.drop_page.set_drag_active(False)
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
        self.notify(
            "That isn't a video",
            "Drop a video file (.mov, .mp4) or a folder from a previous run.",
            level="warn",
        )

    def notify(self, title: str, body: str = "", *, details: str = "", level: str = "fail") -> None:
        """Inline, never modal: a card on the drop page, else the status bar (a card written while
        the progress page or the viewer shows would surface later, stale)."""
        if self.stack.currentWidget() is not self.drop_page:
            self.statusBar().showMessage(f"{title}. {body}".strip(), 10000)
            return
        self.drop_page.notice.show_message(title, body, details=details, level=level)

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
        # Downloads first: AirDrop puts the clip there
        start = Path.home() / "Downloads"
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            str(start) if start.is_dir() else "",
            "Videos (*.mp4 *.mov *.m4v *.avi *.mkv *.webm)",
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
            if self._job_source is not None and self._job_source.resolve() == spec.source.resolve():
                return  # the same video again (dropped twice, or dropped and auto-detected)
            self.statusBar().showMessage(
                f"Still processing {self._job_source.name if self._job_source else 'a video'}; "
                f"drop {spec.source.name} again when it finishes.",
                10000,
            )
            return
        self.close_trial()
        self.drop_page.notice.clear()
        self._job_source = spec.source
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
        self.progress_page.set_job(spec.source, spec.protocol)
        self.stack.setCurrentWidget(self.progress_page)
        self.statusBar().showMessage(f"processing {spec.source.name}")
        self._worker.start()

    def cancel_job(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.progress_page.set_stage("cancelling…")

    def _job_done(self, result: WorkerResult) -> None:
        self._worker = None
        self._job_source = None
        self.open_trial(result.trial_dir, run_id=result.run_id)

    def _job_failed(self, message: str, tb: str) -> None:
        self._worker = None
        self._job_source = None
        self.stack.setCurrentWidget(self.drop_page)
        self.statusBar().showMessage(f"failed: {message}")
        title, body = friendly_failure(message)
        self.notify(title, body, details=tb)

    def _job_cancelled(self) -> None:
        self._worker = None
        self._job_source = None
        self.stack.setCurrentWidget(self.drop_page)
        self.statusBar().showMessage("cancelled")

    # ---- viewer -----------------------------------------------------------------------
    def open_trial(self, trial_dir: Path, *, run_id: str | None = None) -> None:
        try:
            session = TrialSession.load(trial_dir, run_id)
            source = FrameSource(session.video_path)
        except Exception as e:
            self.stack.setCurrentWidget(self.drop_page)
            self.notify("Couldn't open that trial", str(e), details=traceback.format_exc())
            return
        self.close_trial()
        self._session = session
        self._player = Player(session.n_frames, session.fps, self)
        self._player.frameChanged.connect(self._on_frame)
        self._player.playingChanged.connect(self._on_playing)
        self._player.rateChanged.connect(self.transport.set_rate)
        self.view.set_session(session, source)
        self.view.set_angle_names(session.default_angle_names())
        self._show_clock = session.test_window is not None and (
            session.primary_metric is not None and session.primary_metric.units == "s"
        )
        self.timeline.set_session(session)
        self.panel.set_session(session)
        self.cards.set_session(session)
        self.transport.set_session(session)
        self.result_title.setText(result_title(session))
        self.result_subtitle.setText(result_subtitle(session))
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
        has_report = self.report_path() is not None
        self.act_report.setEnabled(has_report)
        self.cards.report_button.setVisible(has_report)

    def close_trial(self) -> None:
        if self._player is not None:
            self._player.pause()
            self._player.deleteLater()
            self._player = None
        self.view.set_session(None, None)
        self.timeline.set_session(None)
        self.panel.set_session(None)
        self.cards.set_session(None)
        self.transport.set_session(None)
        self._session = None
        self.act_report.setEnabled(False)
        self.setWindowTitle("ptvision")
        if self.stack.currentWidget() is self.viewer:
            self.stack.setCurrentWidget(self.drop_page)

    def report_path(self) -> Path | None:
        s = self._session
        if s is None or s.run is None:
            return None
        path = s.run.path / "report.html"
        return path.resolve() if path.exists() else None

    def open_report(self) -> bool:
        path = self.report_path()
        if path is None:
            self.statusBar().showMessage("No report for this trial (skeleton-only runs have none)")
            return False
        return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

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
        if self.timeline.isVisible():
            self.timeline.set_frame(t)
        self.transport.set_frame(t)
        self.cards.set_frame(t)
        values = s.angle_readout(t)
        self.panel.set_readout(values)
        self.view.set_angles(values)
        self.view.set_badge(s.rep_label(t))
        clock = s.test_clock(t) if self._show_clock else None
        done = s.test_window is not None and t >= s.test_window[1]
        self.view.set_clock(clock, done=done)
        readout = "  ".join(f"{k.replace('_', ' ')} {v:.0f}°" for k, v in values.items())
        self.statusBar().showMessage(f"frame {t}/{s.n_frames - 1}   {t / s.fps:.2f} s   {readout}")

    def _on_playing(self, playing: bool) -> None:
        self.act_play.setText("Pause" if playing else "Play")
        self.transport.set_playing(playing)

    def _scrub(self, t: int) -> None:
        if self._player is not None:
            self._player.pause()
            self._player.seek(int(t))

    def _cycle_speed(self) -> None:
        if self._player is not None:
            self._player.cycle_rate()

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
    protocol: str | None = None,
    out_dir: Path | None = None,
    run_id: str | None = None,
    subject: Subject | None = None,
    max_height: int | None = None,
    hard_exit: bool = True,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("ptvision")
    if isinstance(app, QApplication):
        apply_theme(app)
    win = MainWindow()
    if subject is not None:
        win.drop_page.set_subject(height_m=subject.height_m, age=subject.age_years, sex=subject.sex)
    if max_height:
        win.drop_page.max_height_box.setCurrentIndex(
            max(0, win.drop_page.max_height_box.findData(int(max_height)))
        )
    if protocol:
        win.drop_page.select_protocol(protocol)  # a built-in id; a TOML path still runs below
    win.show()
    if source is not None:
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
