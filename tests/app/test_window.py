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
    assert w.player.frame == w.session.first_present_frame  # opens on the test start
    w.player.seek(0)
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
    assert w.player is not None
    w.player.seek(0)
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
    w.drop_page.set_subject(height_m=1.7, age=40, sex="f")
    w._handle_paths([synthetic_trial.video])
    assert w.stack.currentWidget() is w.progress_page
    qtbot.waitUntil(lambda: w.stack.currentWidget() is w.viewer, timeout=60000)
    assert w.session is not None and w.session.protocol is None
    subj = w.session.capture.subject
    assert (subj.height_m, subj.age_years, subj.sex) == (1.7, 40, "f")


def test_drop_page_subject_blank_is_none(qtbot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.drop_page.set_subject()
    assert w.drop_page.subject() is None
    w.drop_page.set_subject(age=71)
    s = w.drop_page.subject()
    assert s is not None and s.age_years == 71 and s.height_m is None
    assert w.drop_page.selected_max_height() is None
    # subject details never survive a relaunch (only the ingest preference does)
    w2 = MainWindow()
    qtbot.addWidget(w2)
    assert w2.drop_page.subject() is None


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


def test_screenshot_waits_for_viewer(qtbot, synthetic_trial, tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    from ptvision.app.main import arm_screenshot

    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    png = tmp_path / "shots" / "viewer.png"
    codes: list[int] = []
    arm_screenshot(
        QApplication.instance(),
        w,
        png,
        expect_viewer=True,
        delay_ms=50,
        poll_ms=20,
        on_done=codes.append,
    )
    w.open_trial(synthetic_trial.trial_dir)
    qtbot.waitUntil(lambda: bool(codes), timeout=5000)
    assert codes == [0] and png.stat().st_size > 1000


def test_screenshot_gives_up_when_viewer_never_appears(qtbot, tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    from ptvision.app.main import arm_screenshot

    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    png = tmp_path / "drop.png"
    codes: list[int] = []
    arm_screenshot(
        QApplication.instance(),
        w,
        png,
        expect_viewer=True,
        delay_ms=10,
        timeout_ms=100,
        poll_ms=20,
        on_done=codes.append,
    )
    qtbot.waitUntil(lambda: bool(codes), timeout=5000)
    assert codes == [3] and png.exists()


def test_sample_button_fills_subject_and_starts_job(
    qtbot, synthetic_trial, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)  # ptv_out/sample lands here
    w = MainWindow(backend_factory=lambda: FakeBackend(synthetic_trial.track))
    qtbot.addWidget(w)
    w.show()
    w.drop_page.try_sample.click()
    assert w.stack.currentWidget() is w.progress_page
    assert w.drop_page.selected_protocol() == "sts_5x"
    s = w.drop_page.subject()
    assert s is not None and s.height_m == 1.8 and s.age_years == 26
    w.cancel_job()
    qtbot.waitUntil(lambda: w.stack.currentWidget() is w.drop_page, timeout=30000)


def test_worker_emits_live_preview_and_page_shows_it(
    qtbot, synthetic_trial, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)  # fresh ptv_out, so keypoints are extracted rather than reused
    w = MainWindow(backend_factory=lambda: FakeBackend(synthetic_trial.track, delay_s=0.01))
    qtbot.addWidget(w)
    w.show()
    w.drop_page.select_protocol("pose")
    frames: list[object] = []
    w._handle_paths([synthetic_trial.video])
    assert w._worker is not None
    w._worker.preview.connect(frames.append)
    qtbot.waitUntil(lambda: bool(frames), timeout=30000)
    qtbot.waitUntil(lambda: w.progress_page.preview.isVisible(), timeout=5000)
    assert w.progress_page.preview.pixmap() is not None
    qtbot.waitUntil(lambda: w.stack.currentWidget() is w.viewer, timeout=60000)


def test_pose_only_hides_measurement_and_event_groups(
    qtbot, pose_only_trial, synthetic_trial
) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(pose_only_trial.trial_dir)
    assert w.session is not None and w.session.protocol is None
    assert not w.panel.metrics_box.isVisible() and not w.panel.events_box.isVisible()
    assert "pose only" in w.panel.quality.text()
    w.open_trial(synthetic_trial.trial_dir)
    assert w.panel.metrics_box.isVisible() and w.panel.events_box.isVisible()


def test_smooth_toggle_switches_between_filtered_and_raw_coords(qtbot, synthetic_trial) -> None:
    import numpy as np

    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(synthetic_trial.trial_dir)
    s = w.session
    assert s is not None and s.smoothed is not None and s.smoothed.shape[0] == s.n_frames
    sk = w.view.skeleton
    t = 10
    assert np.allclose(sk.primary_coords(t), s.smoothed[t], equal_nan=True)
    w.panel.smooth.setChecked(False)
    assert np.allclose(sk.primary_coords(t), s.track.coords[t, s.primary_slot], equal_nan=True)


def test_recent_trials_list_remembers_opened_trial(qtbot, synthetic_trial, monkeypatch) -> None:
    from ptvision.app.pages import DropPage

    monkeypatch.setattr(DropPage, "hide_temp_dirs", False)
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(synthetic_trial.trial_dir)
    assert w.drop_page.recent_paths()[0] == synthetic_trial.trial_dir
    w2 = MainWindow()  # a new window sees it (same isolated settings store)
    qtbot.addWidget(w2)
    w2.show()
    assert w2.drop_page.recent.count() >= 1  # the settings store is shared across tests
    assert w2.drop_page.recent_paths()[0] == synthetic_trial.trial_dir
    w2.drop_page.recent.itemActivated.emit(w2.drop_page.recent.item(0))
    assert w2.stack.currentWidget() is w2.viewer


def test_rep_badge_and_angle_readout_follow_the_playhead(qtbot, synthetic_trial) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(synthetic_trial.trial_dir)
    s = w.session
    assert s is not None
    starts = sorted(m.frame for m in s.events if m.label == "seat-off")
    w.player.seek(max(0, starts[0] - 3))
    assert not w.view.badge.isVisible()
    w.player.seek(starts[1] + 1)
    assert w.view.badge.text() == f"Rep 2 / {len(starts)}"
    assert "knee_flexion" in w.panel._readout_values
    assert w.panel._readout_values["knee_flexion"].text().endswith("°")
