"""[REVIEW] The pre-viewer pages (drop page + progress page), laid out like the iPhone prototype's
Start and Processing screens (ios/PTVision/Views/StartView.swift, ProcessingView.swift)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QImage, QIntValidator, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ptvision._version import __version__
from ptvision.app import theme
from ptvision.clinical.protocol import builtin_protocol_ids, load_protocol
from ptvision.trials.models import Subject

DEFAULT_PROTOCOL = "sts_5x"
POSE_ONLY = "pose"

# short names for the test picker; other protocols show their TOML name
_TEST_NAMES = {
    "sts_5x": "Five-times sit-to-stand",
    "sts_30s": "30-second chair stand",
    POSE_ONLY: "Skeleton only",
}


def card(parent: QWidget | None = None) -> QFrame:
    f = QFrame(parent)
    f.setObjectName("card")
    return f


def label(text: str = "", role: str | None = None, *, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if role:
        theme.set_role(lbl, role)
    lbl.setWordWrap(wrap)
    return lbl


class NoticeCard(QFrame):
    """An inline message on the drop page, shown in place of a modal dialog (a dialog in front of
    a class or a patient blocks the app until someone finds the OK button)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("notice")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        top = QHBoxLayout()
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.dismiss = QToolButton()
        self.dismiss.setText("Dismiss")
        self.dismiss.clicked.connect(self.clear)
        top.addWidget(self.title, 1)
        top.addWidget(self.dismiss, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(top)
        self.body = QLabel()
        self.body.setWordWrap(True)
        lay.addWidget(self.body)
        self.toggle = QToolButton()
        self.toggle.setText("Show details")
        self.toggle.setCheckable(True)
        lay.addWidget(self.toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(160)
        self.details.hide()
        lay.addWidget(self.details)
        self.toggle.toggled.connect(self._toggle_details)
        self.hide()

    def show_message(
        self, title: str, body: str, *, details: str = "", level: str = "fail"
    ) -> None:
        color = theme.RED if level == "fail" else theme.AMBER
        self.setStyleSheet(
            f"QFrame#notice {{ background: {theme.CARD}; border: 1px solid {color};"
            f" border-left: 6px solid {color}; border-radius: 14px; }}"
        )
        self.title.setText(title)
        self.title.setStyleSheet(f"font-weight: 700; color: {color};")
        self.body.setText(body)
        self.body.setVisible(bool(body))
        self.details.setPlainText(details)
        self.toggle.setChecked(False)
        self.toggle.setVisible(bool(details))
        self.show()

    def clear(self) -> None:
        self.title.clear()
        self.body.clear()
        self.details.clear()
        self.hide()

    def _toggle_details(self, on: bool) -> None:
        self.details.setVisible(on)
        self.toggle.setText("Hide details" if on else "Show details")


class DropZone(QFrame):
    """The dashed target. It lights up in the accent while a file is dragged over the window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("drop")
        self.setMinimumHeight(220)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 28, 32, 28)
        lay.setSpacing(8)
        lay.addStretch(1)
        self.title = label("Drop the video here", "headline")
        self.title.setStyleSheet("font-size: 26px;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.title)
        sub = label(
            "A side view of the whole test, from the phone or a folder from a previous run. "
            "It is processed on this laptop and never uploaded.",
            "secondary",
            wrap=True,
        )
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)
        lay.addSpacing(10)
        row = QHBoxLayout()
        row.addStretch(1)
        self.open_video = QPushButton("Choose video…")
        theme.prominent(self.open_video)
        self.open_trial = QPushButton("Open a previous trial…")
        row.addWidget(self.open_video)
        row.addWidget(self.open_trial)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)
        self.set_active(False)

    def set_active(self, on: bool) -> None:
        border = theme.ACCENT if on else theme.TEXT_3
        fill = "rgba(166, 255, 0, 0.08)" if on else theme.CARD
        self.setStyleSheet(
            f"QFrame#drop {{ background: {fill}; border: 2px dashed {border};"
            " border-radius: 22px; }"
        )
        self.title.setText("Release to process" if on else "Drop the video here")


class DropPage(QWidget):
    hide_temp_dirs = True

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.addStretch(1)
        col = QVBoxLayout()
        col.setSpacing(14)
        col.setContentsMargins(0, 28, 0, 16)
        holder = QWidget()
        holder.setLayout(col)
        holder.setMaximumWidth(920)
        holder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        outer.addWidget(holder, 6)
        outer.addStretch(1)

        col.addWidget(label("New test", "title"))
        col.addWidget(
            label(
                "Film the test on the phone, AirDrop the video to this laptop, then drop it here.",
                "secondary",
                wrap=True,
            )
        )
        self.drop_zone = DropZone()
        self.open_video = self.drop_zone.open_video
        self.open_trial = self.drop_zone.open_trial
        col.addWidget(self.drop_zone)
        self.notice = NoticeCard()
        col.addWidget(self.notice)

        # test + patient cards
        cards = QHBoxLayout()
        cards.setSpacing(14)
        test_card = card()
        tl = QVBoxLayout(test_card)
        tl.setContentsMargins(18, 14, 18, 16)
        tl.addWidget(label("TEST", "section"))
        self.protocol = QComboBox()
        for pid in self._protocol_order():
            self.protocol.addItem(self._test_name(pid), pid)
        tl.addWidget(self.protocol)
        self.protocol_summary = label("", "caption", wrap=True)
        tl.addWidget(self.protocol_summary)
        tl.addStretch(1)
        cards.addWidget(test_card, 1)

        patient = card()
        pl = QGridLayout(patient)
        pl.setContentsMargins(18, 14, 18, 16)
        pl.setHorizontalSpacing(10)
        pl.setVerticalSpacing(8)
        pl.addWidget(label("PATIENT · OPTIONAL", "section"), 0, 0, 1, 4)
        self.age_box = QLineEdit()
        self.age_box.setPlaceholderText("years")
        self.age_box.setValidator(QIntValidator(1, 120))
        self.age_box.setFixedWidth(70)
        self.age_box.setToolTip("Age in years (selects the reference band)")
        self.feet_box = QLineEdit()
        self.feet_box.setPlaceholderText("ft")
        self.feet_box.setValidator(QIntValidator(3, 7))
        self.feet_box.setFixedWidth(52)
        self.feet_box.setToolTip("Height, feet")
        self.inch_box = QLineEdit()
        self.inch_box.setPlaceholderText("in")
        self.inch_box.setValidator(QIntValidator(0, 11))
        self.inch_box.setFixedWidth(52)
        self.inch_box.setToolTip("Height, inches")
        self.sex_box = QComboBox()
        for text, value in (("not given", ""), ("female", "f"), ("male", "m")):
            self.sex_box.addItem(text, value)
        pl.addWidget(label("Age"), 1, 0)
        pl.addWidget(self.age_box, 1, 1)
        pl.addWidget(label("Height"), 2, 0)
        height = QHBoxLayout()
        height.setSpacing(6)
        height.addWidget(self.feet_box)
        height.addWidget(label("′"))
        height.addWidget(self.inch_box)
        height.addWidget(label("″"))
        height.addStretch(1)
        pl.addLayout(height, 2, 1, 1, 3)
        pl.addWidget(label("Sex"), 3, 0)
        pl.addWidget(self.sex_box, 3, 1, 1, 2)
        pl.addWidget(
            label("Age picks the reference values. No names are stored.", "caption", wrap=True),
            4,
            0,
            1,
            4,
        )
        pl.setColumnStretch(3, 1)
        cards.addWidget(patient, 1)
        col.addLayout(cards)

        # recent + sample
        head = QHBoxLayout()
        self.recent_label = label("RECENT", "section")
        head.addWidget(self.recent_label)
        head.addStretch(1)
        self.try_sample = QPushButton("Try the sample clip")
        self.try_sample.setToolTip(
            "A bundled 13-second sit-to-stand recording, filmed at an oblique angle; "
            "processes in about 15 s"
        )
        head.addWidget(self.try_sample)
        col.addLayout(head)
        self.recent = QListWidget()
        self.recent.setMaximumHeight(150)
        self.recent.setToolTip("Double-click to reopen an analysed trial")
        col.addWidget(self.recent)
        col.addStretch(1)

        foot = QHBoxLayout()
        self.note = label(f"ptvision {__version__} · measurement tool", "caption")
        foot.addWidget(self.note)
        foot.addStretch(1)
        foot.addWidget(label("Video size", "caption"))
        self.max_height_box = QComboBox()
        for text, cap in (
            ("full resolution", 0),
            ("downscale to 1080", 1080),
            ("downscale to 720", 720),
        ):
            self.max_height_box.addItem(text, cap)
        self.max_height_box.setToolTip(
            "Cap the shorter side of the frame during ingest (4K clips are slow)"
        )
        foot.addWidget(self.max_height_box)
        col.addLayout(foot)

        self._settings = QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, "ptvision", "app"
        )
        self._restore_prefs()
        self._show_summary()
        self.max_height_box.currentIndexChanged.connect(self._save_prefs)
        self.protocol.currentIndexChanged.connect(self._save_prefs)
        self.protocol.currentIndexChanged.connect(self._show_summary)
        self._load_recent()

    # ---- the test picker ---------------------------------------------------------------
    @staticmethod
    def _protocol_order() -> list[str]:
        ids = builtin_protocol_ids()
        first = [p for p in (DEFAULT_PROTOCOL,) if p in ids]
        return [*first, *(p for p in ids if p not in first), POSE_ONLY]

    @staticmethod
    def _test_name(pid: str) -> str:
        if pid in _TEST_NAMES:
            return _TEST_NAMES[pid]
        try:
            return load_protocol(pid).protocol.name
        except Exception:
            return pid

    def _show_summary(self) -> None:
        pid = self.selected_protocol()
        if pid == POSE_ONLY:
            text = "Tracks the skeleton and shows the joint angles. No test measurements."
        else:
            try:
                text = load_protocol(pid).protocol.summary
            except Exception:
                text = ""
        self.protocol_summary.setText(text)

    def selected_protocol(self) -> str:
        return str(self.protocol.currentData())

    def set_drag_active(self, on: bool) -> None:
        self.drop_zone.set_active(on)

    # ---- patient fields -------------------------------------------------------------------
    @staticmethod
    def _int(text: str) -> int | None:
        text = text.strip()
        return int(text) if text.isdigit() else None

    def height_m(self) -> float | None:
        ft = self._int(self.feet_box.text())
        if ft is None:
            return None
        inches = self._int(self.inch_box.text()) or 0
        return round((ft * 12 + inches) * 0.0254, 2)

    def subject(self) -> Subject | None:
        h = self.height_m()
        a = self._int(self.age_box.text())
        s = str(self.sex_box.currentData()) or None
        if h is None and a is None and s is None:
            return None
        return Subject(height_m=h, age_years=a, sex=s)  # type: ignore[arg-type]

    def selected_max_height(self) -> int | None:
        return int(self.max_height_box.currentData()) or None

    def set_subject(
        self, *, height_m: float | None = None, age: int | None = None, sex: str | None = None
    ) -> None:
        if height_m:
            total_in = round(height_m / 0.0254)
            self.feet_box.setText(str(total_in // 12))
            self.inch_box.setText(str(total_in % 12))
        else:
            self.feet_box.clear()
            self.inch_box.clear()
        self.age_box.setText(str(int(age)) if age else "")
        self.sex_box.setCurrentIndex(max(0, self.sex_box.findData(sex or "")))

    # ---- recent trials  --------------------------------------------------------------
    def recent_paths(self) -> list[Path]:
        raw = self._settings.value("recent/trials", [])
        items = raw if isinstance(raw, list) else [raw] if raw else []
        return [Path(str(x)) for x in items if str(x)]

    def remember(self, trial_dir: Path) -> None:
        # absolute: trials default to ./ptv_out, so a relative entry vanishes when the app is
        # launched from another folder
        trial_dir = trial_dir.resolve()
        paths = [p for p in self.recent_paths() if p != trial_dir]
        paths.insert(0, trial_dir)
        self._settings.setValue("recent/trials", [str(p) for p in paths[:8]])
        self._settings.sync()
        self._load_recent()

    def _load_recent(self) -> None:
        self.recent.clear()
        tmp = Path(tempfile.gettempdir()).resolve()
        for p in self.recent_paths():
            if not p.exists():
                continue
            if self.hide_temp_dirs and p.resolve().is_relative_to(tmp):
                continue
            item = QListWidgetItem(f"{p.name}    ·    {p.parent.name}")
            item.setToolTip(str(p.resolve()))
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.recent.addItem(item)
        has = self.recent.count() > 0
        self.recent.setVisible(has)
        self.recent_label.setVisible(has)

    def _save_prefs(self) -> None:
        self._settings.setValue("ingest/max_height", int(self.max_height_box.currentData()))
        self._settings.setValue("job/protocol", self.selected_protocol())

    def _restore_prefs(self) -> None:
        try:
            cap = int(str(self._settings.value("ingest/max_height", 0)))
        except (TypeError, ValueError):
            cap = 0
        self.max_height_box.setCurrentIndex(max(0, self.max_height_box.findData(cap)))
        # the last protocol chosen, else 5xSTS: a dropped clip should be measured, not only tracked
        pid = str(self._settings.value("job/protocol", DEFAULT_PROTOCOL))
        if self.protocol.findData(pid) < 0:
            pid = DEFAULT_PROTOCOL
        self.select_protocol(pid)
        self._settings.remove("subject")  # drop values persisted by the pre-fix build

    def select_protocol(self, pid: str) -> None:
        idx = self.protocol.findData(pid)
        if idx >= 0:
            self.protocol.setCurrentIndex(idx)


# Pipeline stage names (pipeline.py) grouped into the steps the progress page lists
_STEPS: list[tuple[str, tuple[str, ...]]] = [
    ("Preparing the video", ("ingest", "loading model")),
    ("Tracking the skeleton", ("pose", "pose (reusing cached keypoints)")),
    ("Checking the filming", ("quality checks",)),
    ("Smoothing", ("filtering",)),
    ("Finding the repetitions", ("segmenting",)),
    ("Measuring", ("metrics",)),
    ("Writing the report", ("writing artifacts", "report", "overlay")),
]
_POSE_ONLY_STEPS = 2  # pose-only jobs stop after tracking


class _StepRow(QWidget):
    def __init__(self, text: str) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        self.icon = QLabel("○")
        self.icon.setFixedWidth(22)
        self.text = QLabel(text)
        self.detail = label("", "caption")
        lay.addWidget(self.icon)
        lay.addWidget(self.text, 1)
        lay.addWidget(self.detail)
        self.set_state("pending")

    def set_state(self, state: str) -> None:
        glyph, color, weight = {
            "done": ("✓", theme.ACCENT, 400),
            "current": ("●", theme.TEXT, 700),
            "pending": ("○", theme.TEXT_3, 400),
        }[state]
        self.icon.setText(glyph)
        self.icon.setStyleSheet(f"color: {color}; font-weight: 700;")
        text_color = theme.TEXT_2 if state == "pending" else theme.TEXT
        self.text.setStyleSheet(f"color: {text_color}; font-weight: {weight};")
        if state != "current":
            self.detail.setText("")


class ProgressPage(QWidget):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(28, 28, 28, 28)
        outer.setSpacing(24)

        # live tracking preview
        self.preview_card = card()
        self.preview_card.setStyleSheet(
            f"QFrame#card {{ background: {theme.CARD}; border-radius: 22px; }}"
        )
        pv = self._preview_layout = QVBoxLayout(self.preview_card)
        pv.setContentsMargins(12, 12, 12, 12)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(1, 1)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview.hide()
        self.live_badge = QLabel(f"<span style='color:{theme.ACCENT}'>●</span>&nbsp; Live tracking")
        self.live_badge.setStyleSheet("font-weight: 600;")
        self.live_badge.hide()
        pv.addWidget(self.live_badge)
        pv.addWidget(self.preview, 1)
        self.waiting = label("The tracked skeleton appears here", "secondary")
        self.waiting.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pv.addWidget(self.waiting)
        self._preview_image: QImage | None = None
        outer.addWidget(self.preview_card, 3)

        # stage, progress and the step list
        side = QVBoxLayout()
        side.setSpacing(12)
        side.addStretch(1)
        side.addWidget(label("ANALYSING", "section"))
        self.file = label("", "secondary", wrap=True)
        side.addWidget(self.file)
        self.stage = label("Starting…", "title")
        self.stage.setWordWrap(True)
        side.addWidget(self.stage)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        side.addWidget(self.bar)
        self.status = label("", "caption")
        side.addWidget(self.status)
        steps = card()
        sl = QVBoxLayout(steps)
        sl.setContentsMargins(4, 6, 4, 6)
        sl.setSpacing(0)
        self._rows = [_StepRow(text) for text, _ in _STEPS]
        for row in self._rows:
            sl.addWidget(row)
        side.addWidget(steps)
        side.addSpacing(6)
        self.cancel = QPushButton("Cancel")
        side.addWidget(self.cancel, alignment=Qt.AlignmentFlag.AlignLeft)
        side.addStretch(1)
        holder = QWidget()
        holder.setLayout(side)
        holder.setFixedWidth(420)
        outer.addWidget(holder)

        self._current: int | None = None
        self._n_steps = len(_STEPS)
        self._pose_clock: tuple[float, int] | None = None  # (time, frames) at the first frame

    # ---- job and stages ---------------------------------------------------------------
    def set_job(self, source: Path, protocol: str | None) -> None:
        self.file.setText(source.name)
        self.file.setToolTip(str(source))
        pose_only = not protocol or protocol == POSE_ONLY
        self._n_steps = _POSE_ONLY_STEPS if pose_only else len(_STEPS)
        for i, row in enumerate(self._rows):
            row.setVisible(i < self._n_steps)
            row.set_state("pending")
        self._current = None
        self._pose_clock = None
        self.set_stage("starting")

    def set_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.bar.setRange(0, 0)
            return
        self.bar.setRange(0, total)
        self.bar.setValue(done)
        if self._current == 1:  # tracking: frames, with a time estimate
            now = time.monotonic()
            if self._pose_clock is None or done < self._pose_clock[1]:
                self._pose_clock = (now, done)
            t0, d0 = self._pose_clock
            rate = (done - d0) / (now - t0) if now > t0 and done > d0 else 0.0
            left = f" · about {round((total - done) / rate)} s left" if rate > 0 else ""
            self.status.setText(f"{done} of {total} frames{left}")
            self._rows[1].detail.setText(f"{done} / {total}")
        else:
            self.status.setText(f"{round(100 * done / total)}%")

    def set_stage(self, name: str) -> None:
        self.bar.setRange(0, 0)
        self.status.setText("")
        if name == "starting":
            self.stage.setText("Starting…")
            return
        if name.startswith("cancel"):
            self.stage.setText("Cancelling…")
            return
        step = next((i for i, (_, names) in enumerate(_STEPS) if name in names), None)
        if step is None or step >= self._n_steps:
            self.stage.setText(name[:1].upper() + name[1:])  # e.g. a one-time model download
            return
        self._current = step
        for i, row in enumerate(self._rows):
            row.set_state("done" if i < step else "current" if i == step else "pending")
        self.stage.setText(_STEPS[step][0] + "…")
        if name == "pose (reusing cached keypoints)":
            self.status.setText("using the tracking saved from the last run")

    # ---- live preview ------------------------------------------------------------------
    def set_preview(self, frame: object) -> None:
        import numpy as np

        arr = np.ascontiguousarray(frame)
        h, w = arr.shape[:2]
        img = QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_BGR888).copy()
        self._preview_image = img
        self.waiting.hide()
        self.preview.show()
        self.live_badge.show()
        self._scale_preview()

    def _scale_preview(self) -> None:
        if self._preview_image is None:
            return
        # the card's inside less the header, not the label's: the label has no size until shown
        m = self._preview_layout.contentsMargins()
        size = self.preview_card.contentsRect().size().shrunkBy(m)
        size.setHeight(size.height() - self.live_badge.sizeHint().height() - 8)
        pix = QPixmap.fromImage(self._preview_image)
        if size.width() > 1 and size.height() > 1:
            pix = pix.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.preview.setPixmap(pix)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._scale_preview()

    def clear_preview(self) -> None:
        self.preview.clear()
        self.preview.hide()
        self.live_badge.hide()
        self.waiting.show()
        self._preview_image = None
