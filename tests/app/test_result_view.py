"""[REVIEW] The result view: playback speed, the transport bar, the video overlays and the Details
toggle."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt

from ptvision.app.main import MainWindow
from ptvision.app.player import Player


class FakeClock:
    def __init__(self) -> None:
        self.ms = 0

    def restart(self) -> int:
        self.ms = 0
        return 0

    def elapsed(self) -> int:
        return self.ms


def test_slow_motion_advances_a_quarter_as_fast_and_never_jumps(qtbot) -> None:
    p = Player(600, 30.0)
    clock = FakeClock()
    p._clock = clock  # type: ignore[assignment]
    p.play()
    clock.ms = 1000
    p._tick()
    assert p.frame == 30
    p.set_rate(0.25)  # mid-play: carries on from frame 30
    assert p.rate == 0.25
    p._tick()
    assert p.frame == 30
    clock.ms = 1000
    p._tick()
    assert p.frame == 37  # 30 + 7.5 frames
    p.pause()


def test_speed_cycles_one_half_quarter(qtbot) -> None:
    p = Player(100, 30.0)
    seen: list[float] = []
    p.rateChanged.connect(seen.append)
    assert [p.cycle_rate() for _ in range(3)] == [0.5, 0.25, 1.0]
    assert seen == [0.5, 0.25, 1.0]


def _open(qtbot, trial) -> MainWindow:  # type: ignore[no-untyped-def]
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.open_trial(trial.trial_dir)
    assert w.session is not None and w.player is not None
    return w


def test_transport_controls_drive_the_player(qtbot, synthetic_trial) -> None:
    w = _open(qtbot, synthetic_trial)
    tr, player = w.transport, w.player
    assert player is not None
    seat_offs = sorted(e.frame for e in w.session.events if e.label == "seat-off")
    assert len(tr.scrubber.seat_offs) == len(seat_offs) == 5

    tr.speed.click()
    assert player.rate == 0.5 and tr.speed.text() == "½×"
    w.act_speed.trigger()
    assert player.rate == 0.25 and tr.speed.text() == "¼×"

    tr.play.click()
    assert player.playing
    qtbot.mouseClick(
        tr.scrubber, Qt.MouseButton.LeftButton, pos=tr.scrubber.chip_rect(2).center().toPoint()
    )
    assert not player.playing and player.frame == seat_offs[2]  # a rep chip pauses and jumps
    assert tr.scrubber.current_rep == 2
    tr.forward.click()
    assert player.frame == seat_offs[2] + 1
    tr.back.click()
    tr.back.click()
    assert player.frame == seat_offs[2] - 1
    assert tr.time.text().startswith(f"<b>{player.frame / w.session.fps:.2f}</b>")

    on_track = QPoint(tr.scrubber.width() // 2, int(tr.scrubber._track_y()))  # below the chips
    qtbot.mouseClick(tr.scrubber, Qt.MouseButton.LeftButton, pos=on_track)
    assert player.frame == tr.scrubber.frame_at(on_track.x())


def test_overlays_follow_the_playhead_and_the_clock_stops_on_the_total(
    qtbot, synthetic_trial
) -> None:
    w = _open(qtbot, synthetic_trial)
    s, player, view = w.session, w.player, w.view
    assert s is not None and player is not None and s.test_window is not None
    start, end = s.test_window
    total = next(m for m in s.metrics if m.name == "sts_total_time")
    assert s.primary_metric is not None and s.primary_metric.name == "sts_total_time"

    player.seek(max(0, start - 5))
    assert view.badge.text() == "Before the test"
    assert view.clock.isVisible() and view.clock.text() == "0.00 s"
    rise = next(ph for ph in s.phases if ph.kind == "rise")
    player.seek(rise.start + 1)
    assert view.badge.text() == "Rep 1 of 5 · rising"
    descent = next(ph for ph in s.phases if ph.kind == "descent")
    player.seek(descent.start + 1)
    assert view.badge.text().endswith("· sitting")
    player.seek(end)
    assert view.clock.text() == f"{total.value:.2f} s" and view.clock.property("done")
    player.seek(s.n_frames - 1)
    assert view.clock.text() == f"{total.value:.2f} s"  # frozen after the test

    assert list(view.chips.values) == ["knee_flexion", "hip_flexion", "trunk_lean"]
    player.seek(rise.start)
    knee = s.angle_readout(rise.start)["knee_flexion"]
    assert view.chips.values["knee_flexion"].text() == f"{knee:.0f}°"


def test_pose_only_has_no_rep_badge_or_clock(qtbot, pose_only_trial) -> None:
    w = _open(qtbot, pose_only_trial)
    assert not w.view.badge.isVisible() and not w.view.clock.isVisible()
    assert w.transport.scrubber.seat_offs == []
    assert w.result_title.text() == "Skeleton only"


def test_details_shows_the_timeline_and_new_test_goes_back(qtbot, synthetic_trial) -> None:
    w = _open(qtbot, synthetic_trial)
    assert not w.timeline.isVisible()
    w.details_button.click()
    assert w.act_details.isChecked() and w.timeline.isVisible()
    w.act_details.trigger()
    assert not w.timeline.isVisible()
    w.new_test_button.click()
    assert w.stack.currentWidget() is w.drop_page and w.session is None
