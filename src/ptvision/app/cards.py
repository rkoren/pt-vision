"""[REVIEW] The result cards beside the video, after the iPhone result screen.

The headline measurement against the reference values, the filming checks in plain words, the
repetitions (click one to jump to it) and the other measurements. Reference numbers are read from
the norm tables, never typed here. A timing the quality checks reject loses its "Validated" tag."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ptvision.app import theme
from ptvision.app.pages import card, label
from ptvision.app.session import TrialSession
from ptvision.clinical.metrics.base import Metric, NormComparison
from ptvision.clinical.norms import NormRow, NormTable, load_norms
from ptvision.quality.checks import QualityCheck, QualityReport

# the checks whose failure, in the pipeline's own words, makes the headline timing untrustworthy
TIMING_CHECKS = {"test_truncated", "repetition_count"}

ROW_HEIGHT = 44  # a repetition row
_ERROR_LABELS = {"sd": "SD", "rmse": "RMSE", "mae": "MAE", "mdd": "MDD", "ci95": "95% CI ±"}


# ---- wording -----------------------------------------------------------------------------
def timing_trusted(quality: QualityReport | None) -> bool:
    if quality is None:
        return True
    return not any(c.name in TIMING_CHECKS and c.status == "fail" for c in quality.checks)


def value_text(m: Metric) -> str:
    """The value alone, without its error ("10.43 s", "5", "54°")."""
    return m.model_copy(update={"error": None}).formatted()


def error_text(m: Metric) -> str:
    """The error with what kind it is: "± 0.07 s", "SD 0.05 s", "RMSE 6°"; empty if unknown."""
    if m.error is None or m.error_kind == "unknown":
        return ""
    num = m.model_copy(update={"value": m.error, "error": None}).formatted()
    if m.error_kind == "resolution":
        return f"± {num}"
    return f"{_ERROR_LABELS.get(m.error_kind, m.error_kind.upper())} {num}"


def band_name(row: NormRow) -> str:
    """ "60s" for a 60–69 decade, else "60–74"."""
    if row.age_lo % 10 == 0 and row.age_hi == row.age_lo + 9:
        return f"{row.age_lo}s"
    return f"{row.age_lo}–{row.age_hi}"


def band_rows(table: NormTable, sex: str | None) -> list[NormRow]:
    """The table's reference means for this person's sex (or for everyone), youngest first."""
    means = [r for r in table.rows if r.statistic == "mean"]
    mine = [r for r in means if sex and r.sex == sex]
    rows = mine or [r for r in means if r.sex == "all"] or means
    return sorted(rows, key=lambda r: r.age_lo)


def short_citation(citation: str) -> str:
    """ "Bohannon 2006" from a full reference."""
    year = re.search(r"\b(19|20)\d{2}\b", citation)
    author = citation.split()[0].rstrip(",.") if citation.split() else ""
    return f"{author} {year.group(0)}" if author and year else citation


@dataclass(frozen=True)
class Reference:
    """Where a value sits against a norm table, ready to draw and to say."""

    rows: list[NormRow]
    match: NormRow | None  # the band for this person's age, if the table covers it
    sentence: str
    caption: str


def reference_for(
    m: Metric,
    norm: NormComparison | None,
    age: int | None,
    sex: str | None,
    tables: dict[str, NormTable] | None = None,
    *,
    trusted: bool = True,
) -> Reference | None:
    """The comparison to show. A time the filming checks reject is not compared, only listed."""
    if norm is None:
        return None
    tables = tables if tables is not None else load_norms()
    table = tables.get(norm.norm_id)
    if table is None:
        return None
    rows = band_rows(table, sex)
    if not rows:
        return None
    unit = f" {m.units}" if m.units not in ("count", "") else ""
    match = table.lookup(age, sex) if age is not None else None
    lo, hi = min(r.age_lo for r in rows), max(r.age_hi for r in rows)
    listing = ", ".join(f"{r.value:g}{unit} ({band_name(r)})" for r in rows)
    if m.value is None:
        sentence = f"Reference values cover ages {lo}–{hi}: {listing}."
    elif not trusted:
        sentence = (
            f"Not compared until the filming is fixed. Reference values cover ages {lo}–{hi}: "
            f"{listing}."
        )
    elif match is not None and norm.applicable:
        diff = float(m.value) - match.value
        ages = f"ages {match.age_lo}–{match.age_hi}"
        if m.units == "s":
            if round(abs(diff), 1) == 0:
                rel = "<b>the same as</b>"
            else:
                rel = f"<b>{abs(diff):.1f} s {'slower' if diff > 0 else 'faster'}</b> than"
        else:
            rel = f"<b>{abs(diff):g}{unit} {'above' if diff > 0 else 'below'}</b>"
        sentence = (
            f"{value_text(m)} is {rel} the reference mean of {match.value:g}{unit} for {ages}."
        )
    else:
        why = f"No reference values for age {age}." if age is not None else "Age not entered."
        sentence = f"{why} The table covers ages {lo}–{hi}: {listing}."
    caption = short_citation(table.citation)
    if m.units == "s" and table.direction == "higher_is_worse":
        caption += " · shorter is faster"
    return Reference(rows=rows, match=match, sentence=sentence, caption=caption)


def plain_check(c: QualityCheck, view: dict[str, object] | None = None) -> tuple[str, str]:
    """A title and a sentence for a check that did not pass, for whoever is watching."""
    if c.name == "camera_view":
        if c.status == "warn":
            body = (
                "The phone was at an angle to their side, so joint angles are less exact. "
                "<b>Timings are not affected.</b>"
            )
            if view and view.get("applied") and view.get("theta_deg") is not None:
                body += f" Angles were corrected for about {float(str(view['theta_deg'])):.0f}°."
            return "Joint angles: read with care", body + " Next time, film straight from the side."
        return (
            "Filmed from the front",
            "This test needs a side view, so the joint angles are not reliable. "
            "Film from the person's side.",
        )
    if c.name == "repetition_count":
        n = int(c.value or 0)
        want = int(c.threshold or 0)
        found = "No stands found" if n == 0 else f"{n} stand{'s' if n != 1 else ''} found"
        return (
            f"{found}, the test needs {want}",
            "The time may not cover the whole test. Check the video, then film again with "
            f"exactly {want} full stands.",
        )
    if c.name == "test_truncated":
        return (
            "The video starts or ends mid-test",
            "The start or end of the test is cut off, so the time can't be trusted. "
            "Film two seconds of sitting still before and after.",
        )
    if c.name == "subject_present":
        seen = f" They were seen in only {c.value:.0%} of the video." if c.value is not None else ""
        return "The person was out of view", seen.strip() + " Keep the whole body in view."
    if c.name == "subject_in_frame":
        return "Close to the edge of the frame", "Part of the body touched the edge. Move back."
    if c.name == "required_keypoint_confidence":
        return (
            "Hips or knees were hard to see",
            "Chair arms, loose clothing or another person may hide them.",
        )
    if c.name == "other_people_in_frame":
        return (
            "Someone else was in view",
            "The person tested was picked automatically. Check the skeleton follows them.",
        )
    if c.name == "camera_motion":
        return "The phone moved", "Use a tripod or prop the phone so it stays still."
    title = c.name.replace("_", " ").capitalize()
    return title, c.message[:1].upper() + c.message[1:]


# ---- widgets ---------------------------------------------------------------------------
def tag(text: str, kind: str) -> QLabel:
    """A small capsule: validated (accent), estimate (grey) or fail (red)."""
    t = QLabel(text)
    t.setProperty("tag", kind)
    t.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return t


def set_tag(t: QLabel, text: str, kind: str) -> None:
    t.setText(text)
    if t.property("tag") != kind:
        t.setProperty("tag", kind)
        theme.repolish(t)


class ReferenceBar(QWidget):
    """The reference means as ticks on a line, this person's value as an accent dot. The range
    always includes the value: a young adult's 7 s is off the left of the phone's fixed bar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(54)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.value: float | None = None
        self.rows: list[NormRow] = []
        self.match: NormRow | None = None
        self.unit = ""
        self.muted = False

    def set_data(
        self,
        value: float | None,
        rows: list[NormRow],
        match: NormRow | None,
        unit: str,
        *,
        muted: bool = False,
    ) -> None:
        self.value, self.rows, self.match, self.unit = value, rows, match, unit
        self.muted = muted  # grey, not accent, for a time the filming checks reject
        self.update()

    def axis_range(self) -> tuple[float, float]:
        vals = [r.value for r in self.rows] + ([self.value] if self.value is not None else [])
        if not vals:
            return 0.0, 1.0
        return math.floor(min(vals) - 1), math.ceil(max(vals) + 1)

    def x_of(self, v: float) -> float:
        lo, hi = self.axis_range()
        pad = 10.0
        return pad + (self.width() - 2 * pad) * (v - lo) / max(1e-9, hi - lo)

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = 16.0
        lo, hi = self.axis_range()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(118, 118, 128, 61))
        p.drawRoundedRect(QRectF(self.x_of(lo), y - 3, self.x_of(hi) - self.x_of(lo), 6), 3, 3)
        font = self.font()
        font.setPixelSize(12)
        p.setFont(font)
        for r in self.rows:
            x = self.x_of(r.value)
            on = self.match is not None and r == self.match
            h = 22.0 if on else 16.0
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.TEXT if on else theme.TEXT_2))
            p.drawRoundedRect(QRectF(x - 1.5, y - h / 2, 3, h), 1.5, 1.5)
            p.setPen(QColor(theme.TEXT if on else theme.TEXT_2))
            p.drawText(QRectF(x - 30, y + 13, 60, 16), Qt.AlignmentFlag.AlignCenter, band_name(r))
        p.setPen(QColor(theme.TEXT_3))
        unit = f" {self.unit}" if self.unit else ""
        p.drawText(QRectF(0, y + 13, 40, 16), Qt.AlignmentFlag.AlignLeft, f"{lo:g}{unit}")
        p.drawText(
            QRectF(self.width() - 40, y + 13, 40, 16), Qt.AlignmentFlag.AlignRight, f"{hi:g}{unit}"
        )
        if self.value is not None:
            c = QPointF(self.x_of(self.value), y)
            p.setPen(QPen(QColor(theme.CARD), 3))
            p.setBrush(QColor(theme.TEXT_2 if self.muted else theme.ACCENT))
            p.drawEllipse(c, 9, 9)
        p.end()


class HeadlineCard(QFrame):
    """The protocol's primary measurement, large, with its reference comparison."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(8)
        top = QHBoxLayout()
        self.title = label("", "secondary", wrap=True)
        self.tag = tag("Validated", "validated")
        top.addWidget(self.title, 1)
        top.addWidget(self.tag, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(top)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.value = QLabel("–")
        self.value.setFont(theme.number_font(self.value.font(), 64))
        self.unit = QLabel("")
        self.unit.setFont(theme.number_font(self.unit.font(), 24))
        self.error = label("", "secondary")
        self.error.setFont(theme.number_font(self.error.font(), 16, bold=False))
        row.addWidget(self.value, 0, Qt.AlignmentFlag.AlignBaseline)
        row.addWidget(self.unit, 0, Qt.AlignmentFlag.AlignBaseline)
        row.addWidget(self.error, 0, Qt.AlignmentFlag.AlignBaseline)
        row.addStretch(1)
        lay.addLayout(row)
        self.bar = ReferenceBar()
        lay.addWidget(self.bar)
        self.sentence = label("", wrap=True)
        self.sentence.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.sentence)
        self.caption = label("", "caption", wrap=True)
        lay.addWidget(self.caption)

    def set_metric(self, m: Metric, ref: Reference | None, *, trusted: bool) -> None:
        self.title.setText(m.label)
        if not trusted:
            set_tag(self.tag, "Check filming", "fail")
        elif m.tier == 1:
            set_tag(self.tag, "Validated", "validated")
        else:
            set_tag(self.tag, "Estimate", "estimate")
        if m.value is None:
            self.value.setText("–")
            self.unit.setText("")
            self.error.setText("Not measured. See the filming notes.")
        else:
            text = value_text(m)
            num, _, unit = text.partition(" ")
            self.value.setText(num)
            self.unit.setText(unit)
            self.error.setText(error_text(m))
        muted = not trusted or m.value is None
        self.value.setStyleSheet(f"color: {theme.TEXT_2};" if muted else "")
        self.bar.setVisible(ref is not None)
        self.sentence.setVisible(ref is not None)
        self.caption.setVisible(ref is not None)
        if ref is not None:
            value = float(m.value) if m.value is not None else None
            unit = m.units if m.units == "s" else ""
            self.bar.set_data(value, ref.rows, ref.match, unit, muted=not trusted)
            self.sentence.setText(ref.sentence)
            self.caption.setText(ref.caption)


class FilmingCard(QFrame):
    """Green when every check passed; otherwise an amber or red note listing what to fix."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("filming")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)
        self.title = QLabel("")
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)
        self.title.setWordWrap(True)
        lay.addWidget(self.title)
        self.items = QVBoxLayout()
        self.items.setSpacing(8)
        lay.addLayout(self.items)
        self.level = "pass"

    def _clear(self) -> None:
        while self.items.count():
            item = self.items.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.setParent(None)  # off screen now, not at the next event loop
                w.deleteLater()

    def item_titles(self) -> list[str]:
        out = []
        for i in range(self.items.count()):
            item = self.items.itemAt(i)
            w = item.widget() if item is not None else None
            if isinstance(w, QLabel):
                out.append(w.property("check_title"))
        return out

    def set_quality(self, quality: QualityReport, view: dict[str, object] | None) -> None:
        self._clear()
        bad = [c for c in quality.checks if c.status == "fail"]
        bad += [c for c in quality.checks if c.status == "warn"]
        self.level = "fail" if any(c.status == "fail" for c in bad) else "warn" if bad else "pass"
        if not bad:
            self.title.setText("✓  All filming checks passed")
            ink, fill = theme.GREEN, "rgba(43, 168, 74, 0.14)"
        elif self.level == "fail":
            self.title.setText("Check the filming")
            ink, fill = "#FF8A80", "rgba(214, 40, 40, 0.18)"
        else:
            self.title.setText("Filming notes")
            ink, fill = theme.NOTE_INK, theme.NOTE_FILL
        self.setStyleSheet(f"QFrame#filming {{ background: {fill}; border-radius: 16px; }}")
        self.title.setStyleSheet(f"color: {ink};")
        for c in bad:
            title, body = plain_check(c, view)
            item = QLabel(f"<b>{title}</b><br>{body}")
            item.setTextFormat(Qt.TextFormat.RichText)
            item.setWordWrap(True)
            item.setProperty("check_title", title)
            self.items.addWidget(item)


class _RepRow(QWidget):
    def __init__(self, n: int, at_s: float, rise_s: float | None) -> None:
        super().__init__()
        hl = QHBoxLayout(self)
        hl.setContentsMargins(10, 6, 10, 6)
        name = QLabel(f"Rep {n}")
        name.setFont(theme.number_font(name.font(), 15))
        name.setMinimumWidth(64)
        at = label(f"at {at_s:.2f} s", "secondary")
        at.setFont(theme.number_font(at.font(), 15, bold=False))
        hl.addWidget(name)
        hl.addWidget(at)
        hl.addStretch(1)
        if rise_s is not None:
            rise = QLabel(f"rise {rise_s:.2f} s")
            rise.setFont(theme.number_font(rise.font(), 15, bold=False))
            hl.addWidget(rise)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


class RepsList(QListWidget):
    """One row per repetition; a click jumps to its seat-off. The row under the playhead is
    highlighted."""

    repActivated = Signal(int)  # the seat-off frame

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("reps")
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.seat_offs: list[int] = []
        self.itemClicked.connect(self._clicked)

    def set_reps(self, seat_offs: list[int], rise_s: list[float], fps: float) -> None:
        self.clear()
        self.seat_offs = list(seat_offs)
        for i, f in enumerate(seat_offs):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, f)
            row = _RepRow(i + 1, f / fps, rise_s[i] if i < len(rise_s) else None)
            item.setSizeHint(QSize(0, ROW_HEIGHT))
            self.addItem(item)
            self.setItemWidget(item, row)
        m = self.contentsMargins()
        self.setFixedHeight(ROW_HEIGHT * self.count() + m.top() + m.bottom() + 14)

    def set_current(self, index: int | None) -> None:
        self.blockSignals(True)
        if index is None or not 0 <= index < self.count():
            self.clearSelection()
            self.setCurrentRow(-1)
        elif self.currentRow() != index:
            self.setCurrentRow(index)
        self.blockSignals(False)

    def _clicked(self, item: QListWidgetItem) -> None:
        self.repActivated.emit(int(item.data(Qt.ItemDataRole.UserRole)))


class MeasurementRow(QWidget):
    def __init__(self, m: Metric, *, trusted: bool) -> None:
        super().__init__()
        hl = QHBoxLayout(self)
        hl.setContentsMargins(16, 10, 16, 10)
        left = QVBoxLayout()
        left.setSpacing(2)
        side = f" ({m.side})" if m.side and m.side != "bilateral" else ""
        self.name = label(m.label + side, wrap=True)
        left.addWidget(self.name)
        detail = []
        if m.error_kind == "sd" and m.n_events:
            detail.append(f"mean of {m.n_events}")
        if detail:
            left.addWidget(label(" · ".join(detail), "caption"))
        hl.addLayout(left, 1)
        right = QVBoxLayout()
        right.setSpacing(3)
        vals = QHBoxLayout()
        vals.setSpacing(6)
        self.value = QLabel(value_text(m))
        self.value.setFont(theme.number_font(self.value.font(), 16))
        self.error = label(error_text(m), "caption")
        vals.addStretch(1)
        vals.addWidget(self.value, 0, Qt.AlignmentFlag.AlignBaseline)
        vals.addWidget(self.error, 0, Qt.AlignmentFlag.AlignBaseline)
        right.addLayout(vals)
        if m.tier == 1 and not trusted:
            self.tag = tag("Check filming", "fail")
        elif m.tier == 1:
            self.tag = tag("Validated", "validated")
        else:
            self.tag = tag("Estimate", "estimate")
        right.addWidget(self.tag, 0, Qt.AlignmentFlag.AlignRight)
        hl.addLayout(right)


def section(text: str) -> QLabel:
    return label(text.upper(), "section")


class ResultColumn(QWidget):
    """The column beside the video: the cards scroll, "Open report" stays pinned below them."""

    seekRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        self.scroller = QScrollArea()
        self.scroller.setWidgetResizable(True)
        self.scroller.setFrameShape(QFrame.Shape.NoFrame)
        self.scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroller, 1)
        body = QWidget()
        self.scroller.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(12)

        self.headline = HeadlineCard()
        lay.addWidget(self.headline)
        self.skeleton_only = card()
        sl = QVBoxLayout(self.skeleton_only)
        sl.setContentsMargins(18, 16, 18, 16)
        sl.addWidget(label("Skeleton only", "headline"))
        sl.addWidget(
            label(
                "Tracking and joint angles, with no test measurements. Choose a test on the "
                "start page to measure one.",
                "secondary",
                wrap=True,
            )
        )
        lay.addWidget(self.skeleton_only)
        self.filming = FilmingCard()
        lay.addWidget(self.filming)

        self.reps_header = section("Repetitions · click to jump")
        lay.addWidget(self.reps_header)
        self.reps = RepsList()
        self.reps.repActivated.connect(self.seekRequested.emit)
        lay.addWidget(self.reps)

        self.measures_header = section("Measurements")
        lay.addWidget(self.measures_header)
        self.measures = card()
        self.measures_layout = QVBoxLayout(self.measures)
        self.measures_layout.setContentsMargins(0, 4, 0, 4)
        self.measures_layout.setSpacing(0)
        lay.addWidget(self.measures)

        self.report_button = QPushButton("Open report")
        theme.prominent(self.report_button)
        self.report_button.setToolTip("The printable report for this run, in the browser (⌘R)")
        self.report_button.setMinimumHeight(44)
        self.report_button.hide()
        lay.addStretch(1)
        outer.addWidget(self.report_button)
        self._session: TrialSession | None = None
        self.set_session(None)

    def measurement_rows(self) -> list[MeasurementRow]:
        rows = []
        for i in range(self.measures_layout.count()):
            item = self.measures_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if isinstance(w, MeasurementRow):
                rows.append(w)
        return rows

    def _clear_measures(self) -> None:
        while self.measures_layout.count():
            item = self.measures_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.setParent(None)  # off screen now, not at the next event loop
                w.deleteLater()

    def set_session(self, session: TrialSession | None) -> None:
        self._session = session
        self._clear_measures()
        has_protocol = session is not None and session.protocol is not None
        primary = session.primary_metric if session is not None else None
        trusted = timing_trusted(session.quality if session is not None else None)
        self.headline.setVisible(primary is not None)
        self.skeleton_only.setVisible(session is not None and not has_protocol)
        self.filming.setVisible(session is not None and session.quality is not None)
        if session is None:
            self.reps.set_reps([], [], 30.0)
            for w in (self.reps_header, self.reps, self.measures_header, self.measures):
                w.hide()
            return
        if primary is not None:
            subject = session.capture.subject
            ref = reference_for(
                primary, session.primary_norm, subject.age_years, subject.sex, trusted=trusted
            )
            self.headline.set_metric(primary, ref, trusted=trusted)
        if session.quality is not None:
            prov = session.provenance
            view = prov.preprocess.get("view_correction") if prov is not None else None
            self.filming.set_quality(session.quality, view if isinstance(view, dict) else None)

        seat_offs = sorted(
            m.frame for m in session.events if m.label == "seat-off" and m.rep is not None
        )
        self.reps.set_reps(seat_offs, session.rise_times(), session.fps)
        self.reps_header.setVisible(bool(seat_offs))
        self.reps.setVisible(bool(seat_offs))

        others = [m for m in session.metrics if primary is None or m.name != primary.name]
        for i, m in enumerate(others):
            if i:
                line = QFrame()
                line.setFixedHeight(1)
                line.setStyleSheet(f"background: {theme.SEPARATOR}; margin-left: 16px;")
                self.measures_layout.addWidget(line)
            self.measures_layout.addWidget(MeasurementRow(m, trusted=trusted))
        self.measures_header.setVisible(bool(others))
        self.measures.setVisible(bool(others))

    def set_frame(self, t: int) -> None:
        if not self.reps.seat_offs:
            return
        done = sum(1 for f in self.reps.seat_offs if f <= t)
        self.reps.set_current(done - 1 if done else None)
