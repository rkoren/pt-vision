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


# ---- result cards -------------------------------------------------------------------------
def _total(value: float | None = 10.43, **kw):  # type: ignore[no-untyped-def]
    from ptvision.clinical.metrics.base import Metric

    return Metric(
        name="sts_total_time",
        label="Five-times sit-to-stand time",
        value=value,
        units="s",
        error=0.067,
        error_kind="resolution",
        tier=1,
        method_version="0.1.0",
        **kw,
    )


def _norm(metric, age):  # type: ignore[no-untyped-def]
    from ptvision.clinical.norms import compare
    from ptvision.trials.models import Subject

    return compare(metric, "sts_5x_bohannon2006", Subject(age_years=age))


def test_reference_sentence_comes_from_the_norm_table() -> None:
    from ptvision.app.cards import band_name, band_rows, reference_for
    from ptvision.clinical.norms import load_norms

    table = load_norms()["sts_5x_bohannon2006"]
    rows = band_rows(table, None)
    m = _total()

    row = table.lookup(65, None)
    assert row is not None
    ref = reference_for(m, _norm(m, 65), 65, None)
    assert ref is not None and ref.match == row
    diff = abs(10.43 - row.value)
    assert ref.sentence == (
        f"10.43 s is <b>{diff:.1f} s faster</b> than the reference mean of {row.value:g} s "
        f"for ages {row.age_lo}–{row.age_hi}."
    )
    assert ref.caption == "Bohannon 2006 · shorter is faster"

    listing = ", ".join(f"{r.value:g} s ({band_name(r)})" for r in rows)
    lo, hi = rows[0].age_lo, rows[-1].age_hi
    young = reference_for(m, _norm(m, 26), 26, None)
    assert young is not None and young.match is None
    assert young.sentence == (
        f"No reference values for age 26. The table covers ages {lo}–{hi}: {listing}."
    )
    no_age = reference_for(m, _norm(m, None), None, None)
    assert no_age is not None and no_age.sentence.startswith("Age not entered.")


def test_reference_bar_range_includes_a_fast_young_adult(qtbot) -> None:
    from ptvision.app.cards import ReferenceBar, band_rows
    from ptvision.clinical.norms import load_norms

    rows = band_rows(load_norms()["sts_5x_bohannon2006"], None)
    bar = ReferenceBar()
    qtbot.addWidget(bar)
    bar.set_data(6.8, rows, None, "s")
    lo, hi = bar.axis_range()
    assert lo < 6.8 and hi > max(r.value for r in rows)


def test_error_labels_say_what_kind_of_error() -> None:
    from ptvision.app.cards import error_text, value_text

    m = _total()
    assert value_text(m) == "10.43 s" and error_text(m) == "± 0.07 s"
    sd = _total(1.2).model_copy(update={"error": 0.05, "error_kind": "sd"})
    assert error_text(sd) == "SD 0.05 s"
    deg = _total(54).model_copy(update={"units": "deg", "error": 6.2, "error_kind": "rmse"})
    assert value_text(deg) == "54°" and error_text(deg) == "RMSE 6°"
    assert error_text(_total().model_copy(update={"error_kind": "unknown"})) == ""


def test_cards_show_the_result_reps_and_measurements(qtbot, synthetic_trial) -> None:
    from ptvision.app.cards import timing_trusted

    w = _open(qtbot, synthetic_trial)
    s, cards, player = w.session, w.cards, w.player
    assert s is not None and player is not None and s.primary_metric is not None
    assert cards.isVisible() and not w.panel.isVisible()
    head = cards.headline
    assert head.isVisible() and head.title.text() == s.primary_metric.label
    assert f"{head.value.text()} {head.unit.text()}" == f"{s.primary_metric.value:.2f} s"
    trusted = timing_trusted(s.quality)
    assert head.tag.text() == ("Validated" if trusted else "Check filming")
    assert cards.report_button.isVisible()

    seat_offs = sorted(e.frame for e in s.events if e.label == "seat-off")
    assert cards.reps.count() == len(seat_offs) == 5
    player.play()
    item = cards.reps.item(3)
    qtbot.mouseClick(
        cards.reps.viewport(),
        Qt.MouseButton.LeftButton,
        pos=cards.reps.visualItemRect(item).center(),
    )
    assert not player.playing and player.frame == seat_offs[3]
    assert cards.reps.currentRow() == 3
    player.seek(seat_offs[1] + 2)
    assert cards.reps.currentRow() == 1  # the highlight follows the playhead

    names = {r.name.text() for r in cards.measurement_rows()}
    assert s.primary_metric.label not in names  # the headline isn't repeated
    assert len(cards.measurement_rows()) == len(s.metrics) - 1

    w.set_details(True)
    assert w.panel.isVisible() and not cards.isVisible()


def test_a_rejected_timing_is_not_shown_as_validated(qtbot, synthetic_trial) -> None:
    from ptvision.quality.checks import QualityCheck, combine

    w = _open(qtbot, synthetic_trial)
    s = w.session
    assert s is not None and s.quality is not None
    original = s.quality
    try:
        s.quality = combine(
            [
                *[c for c in original.checks if c.name != "test_truncated"],
                QualityCheck(
                    name="test_truncated",
                    status="fail",
                    value=None,
                    threshold=None,
                    message="the test starts or ends at the clip boundary",
                ),
            ]
        )
        w.cards.set_session(s)
        head = w.cards.headline
        assert head.tag.text() == "Check filming"
        text = head.sentence.text()  # a rejected time is listed, not called faster or slower
        assert "faster" not in text and "slower" not in text and "Not compared" in text
        assert head.bar.muted
        assert w.cards.filming.level == "fail"
        assert "The video starts or ends mid-test" in w.cards.filming.item_titles()
        tier1 = [r for r in w.cards.measurement_rows() if r.tag.text() != "Estimate"]
        assert tier1 and all(r.tag.text() == "Check filming" for r in tier1)
    finally:
        s.quality = original


def test_skeleton_only_card_for_a_pose_only_run(qtbot, pose_only_trial) -> None:
    w = _open(qtbot, pose_only_trial)
    c = w.cards
    assert c.skeleton_only.isVisible()
    assert not c.headline.isVisible() and not c.reps.isVisible() and not c.measures.isVisible()
    assert not c.report_button.isVisible()
