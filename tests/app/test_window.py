from pathlib import Path

from PySide6.QtCore import Qt

from ptvision.app.main import MainWindow
from ptvision.app.worker import JobSpec, PipelineWorker
from ptvision.pose.layout import HALPE26
from ptvision.viz.skeleton import segment_edges
from tests.synthetic import FakeBackend


def test_window_constructs(qtbot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    assert w.stack.currentWidget() is w.drop_page
    titles = [a.text() for a in w.menuBar().actions()]
    assert "&Playback" in titles
    assert w.act_next.shortcut().toString() == "Right"


def test_open_trial_shows_viewer_and_steps(qtbot, synthetic_trial) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(synthetic_trial.trial_dir)
    assert w.stack.currentWidget() is w.viewer
    assert w.session is not None and w.player is not None
    assert w.view.current_frame is not None
    assert w.view.current_frame.shape[1::-1] == w.session.image_size
    assert "frame 0/" in w.statusBar().currentMessage()

    with qtbot.waitSignal(w.player.frameChanged, timeout=1000):
        w.act_next.trigger()
    assert w.player.frame == 1
    w.act_next10.trigger()
    assert w.player.frame == 11
    w.act_prev.trigger()
    assert w.player.frame == 10
    w.act_last.trigger()
    assert w.player.frame == w.session.n_frames - 1
    w.act_first.trigger()
    assert w.player.frame == 0

    # timeline and event list seek
    with qtbot.waitSignal(w.timeline.seekRequested, timeout=1000):
        w.timeline.seek_to_x(2.0)
    assert w.player.frame == round(2.0 * w.session.fps)
    first_seat_off = next(e.frame for e in w.session.events if e.label == "seat-off")
    w.panel.eventActivated.emit(first_seat_off)
    assert w.player.frame == first_seat_off
    assert w.timeline.angle_names == ["knee_flexion", "hip_flexion", "trunk_lean"]


def test_arrow_keys_reach_player_regardless_of_focus(qtbot, synthetic_trial) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(synthetic_trial.trial_dir)
    w.view.setFocus()
    qtbot.keyClick(w.view, Qt.Key.Key_Right)
    assert w.player.frame == 1
    w.panel.events.setFocus()
    qtbot.keyClick(w.panel.events, Qt.Key.Key_Right)
    assert w.player.frame == 2
    qtbot.keyClick(w, Qt.Key.Key_Space)
    assert w.player.playing
    qtbot.keyClick(w, Qt.Key.Key_Space)
    assert not w.player.playing


def test_mode_switch_changes_colors_and_paints(qtbot, synthetic_trial) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(synthetic_trial.trial_dir)
    assert w.view.skeleton.mode == "rules"
    t = synthetic_trial.tl.seat_off[0]
    w.player.seek(t)
    trunk = segment_edges(HALPE26, "trunk", None)[0]
    rgb_rules, _ = w.session.bone_colors(t, "rules")
    rgb_conf, _ = w.session.bone_colors(t, "confidence")
    assert tuple(rgb_rules[trunk]) != tuple(rgb_conf[trunk])
    w.panel.radio_conf.setChecked(True)
    assert w.view.skeleton.mode == "confidence"
    w.act_mode.trigger()
    assert w.view.skeleton.mode == "rules"
    img = w.view.grab().toImage()
    assert not img.isNull() and img.width() > 0
    # angle checkbox toggling rebuilds the timeline
    w.panel._angle_checks["hip_flexion"].setChecked(False)
    assert w.timeline.angle_names == ["knee_flexion", "trunk_lean"]


def test_handle_paths_video_starts_job_and_completes(qtbot, synthetic_trial, tmp_path) -> None:
    w = MainWindow(backend_factory=lambda: FakeBackend(synthetic_trial.track))
    qtbot.addWidget(w)
    w.show()
    w.drop_page.select_protocol("pose")
    w._handle_paths([synthetic_trial.video])
    assert w.stack.currentWidget() is w.progress_page
    qtbot.waitUntil(lambda: w.stack.currentWidget() is w.viewer, timeout=60000)
    assert w.session is not None and w.session.protocol is None


def test_handle_paths_trial_dir_opens_viewer(qtbot, synthetic_trial) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w._handle_paths([synthetic_trial.trial_dir])
    assert w.stack.currentWidget() is w.viewer


def test_worker_completes_and_cancels(qtbot, synthetic_trial, tmp_path) -> None:
    spec = JobSpec(synthetic_trial.video, str(synthetic_trial.protocol_path), tmp_path / "t1")
    worker = PipelineWorker(spec, backend=FakeBackend(synthetic_trial.track))
    stages: list[str] = []
    worker.stage.connect(stages.append)
    with qtbot.waitSignal(worker.finished_ok, timeout=60000) as blocker:
        worker.start()
    result = blocker.args[0]
    assert (Path(result.trial_dir) / "runs" / result.run_id / "provenance.json").exists()
    assert "pose" in stages
    worker.wait(5000)

    slow = PipelineWorker(
        JobSpec(synthetic_trial.video, "pose", tmp_path / "t2"),
        backend=FakeBackend(synthetic_trial.track, delay_s=0.02),
    )
    with qtbot.waitSignal(slow.cancelled, timeout=30000):
        slow.start()
        qtbot.waitUntil(lambda: slow.isRunning(), timeout=5000)
        qtbot.wait(300)
        slow.cancel()
    slow.wait(5000)
    runs = tmp_path / "t2" / "runs"
    assert not runs.exists() or all((d / "provenance.json").exists() for d in runs.iterdir())


def test_close_while_running(qtbot, synthetic_trial, tmp_path) -> None:
    w = MainWindow(backend_factory=lambda: FakeBackend(synthetic_trial.track, delay_s=0.02))
    qtbot.addWidget(w)
    w.show()
    w.start_job(JobSpec(synthetic_trial.video, "pose", tmp_path / "t3"))
    qtbot.waitUntil(lambda: w._worker is not None and w._worker.isRunning(), timeout=5000)
    worker = w._worker
    w.close()
    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=10000)
