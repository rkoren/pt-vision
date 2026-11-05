"""[REVIEW] The live-demo path: a dropped clip is measured, nothing modal blocks the window, the
report opens, and the Recent list survives a different working directory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

import ptvision.app.main as app_main
from ptvision.app.main import MainWindow, friendly_failure
from ptvision.app.worker import JobSpec
from tests.synthetic import FakeBackend


@pytest.fixture
def no_modals(monkeypatch):
    """Fail the test if anything opens a QMessageBox."""

    def boom(*_args, **_kwargs):
        raise AssertionError("a modal dialog was opened")

    for name in ("exec", "warning", "critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, boom)


def test_protocol_defaults_to_sts_5x_and_is_remembered(qtbot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.drop_page.selected_protocol() == "sts_5x"
    w.drop_page.select_protocol("pose")
    w2 = MainWindow()
    qtbot.addWidget(w2)
    assert w2.drop_page.selected_protocol() == "pose"


def test_cli_protocol_applies_without_a_source(qtbot, monkeypatch) -> None:
    made: list[MainWindow] = []

    class Recording(MainWindow):
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            super().__init__(**kwargs)
            made.append(self)

    monkeypatch.setattr(app_main, "MainWindow", Recording)
    monkeypatch.setattr(QApplication, "exec", lambda *_a: 0)
    assert app_main.main(protocol="pose", hard_exit=False) == 0
    qtbot.addWidget(made[0])
    assert made[0].drop_page.selected_protocol() == "pose"


def test_bad_drop_failure_and_unopenable_trial_are_inline(qtbot, tmp_path, no_modals) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    notes = tmp_path / "notes.txt"
    notes.write_text("not a video")
    w._handle_paths([notes])
    card = w.drop_page.notice
    assert card.isVisible() and card.title.text() == "That isn't a video"

    w._job_failed("No person detected; see quality.json", "Traceback: …")
    assert w.stack.currentWidget() is w.drop_page
    assert card.title.text() == "No one was found in the video"
    assert card.toggle.isVisible() and not card.details.isVisible()
    card.toggle.setChecked(True)
    assert card.details.isVisible() and card.details.toPlainText() == "Traceback: …"

    w.open_trial(tmp_path / "not-a-trial")
    assert card.isVisible() and card.title.text() == "Couldn't open that trial"
    card.dismiss.click()
    assert not card.isVisible()


def test_friendly_failure_keeps_unknown_messages() -> None:
    assert friendly_failure("ffmpeg exited with 1") == (
        "The video could not be processed",
        "ffmpeg exited with 1",
    )


def test_second_start_while_running_is_not_modal(qtbot, synthetic_trial, tmp_path, no_modals):
    w = MainWindow(backend_factory=lambda: FakeBackend(synthetic_trial.track))
    qtbot.addWidget(w)
    w.show()
    w.drop_page.select_protocol("pose")
    w.start_job(JobSpec(synthetic_trial.video, None, tmp_path / "t1"))
    worker = w._worker
    w.start_job(JobSpec(synthetic_trial.video, None, tmp_path / "t2"))  # same clip: ignored
    assert w._worker is worker
    other = tmp_path / "other.mp4"
    other.write_bytes(synthetic_trial.video.read_bytes())
    w.start_job(JobSpec(other, None, tmp_path / "t3"))
    assert w._worker is worker and "Still processing" in w.statusBar().currentMessage()
    qtbot.waitUntil(lambda: w.stack.currentWidget() is w.viewer, timeout=60000)
    assert w.session is not None and w.session.trial.path == tmp_path / "t1"


def test_open_report_button_opens_the_run_report(qtbot, synthetic_trial, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        app_main.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True
    )
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(synthetic_trial.trial_dir)
    assert w.act_report.isEnabled() and w.panel.report_button.isVisible()
    w.panel.report_button.click()
    assert opened and opened[0].endswith("report.html") and Path(opened[0]).exists()
    w.close_trial()
    assert not w.act_report.isEnabled()


def test_pose_only_trial_has_no_report(qtbot, pose_only_trial) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(pose_only_trial.trial_dir)
    assert not w.act_report.isEnabled() and not w.panel.report_button.isVisible()
    assert w.open_report() is False


def test_recent_entries_survive_another_working_directory(
    qtbot, synthetic_trial, tmp_path, monkeypatch
) -> None:
    from ptvision.app.pages import DropPage

    monkeypatch.setattr(DropPage, "hide_temp_dirs", False)
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(Path(os.path.relpath(synthetic_trial.trial_dir)))
    assert w.drop_page.recent_paths()[0] == synthetic_trial.trial_dir.resolve()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    w2 = MainWindow()
    qtbot.addWidget(w2)
    w2.show()
    w2.drop_page.recent.itemActivated.emit(w2.drop_page.recent.item(0))
    assert w2.stack.currentWidget() is w2.viewer


def test_full_screen_toggles(qtbot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.act_fullscreen.trigger()
    assert w.windowState() & Qt.WindowState.WindowFullScreen
    w.act_fullscreen.trigger()
    assert not w.windowState() & Qt.WindowState.WindowFullScreen
