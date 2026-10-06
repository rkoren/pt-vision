"""The two pre-viewer pages of the window: the drop page (pick a clip, protocol and subject) and
the progress page (stage, bar, live skeleton preview while the pipeline runs)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QImage, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ptvision._version import __version__
from ptvision.clinical.protocol import builtin_protocol_ids
from ptvision.trials.models import Subject


class DropPage(QWidget):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint = QLabel(
            "<h2>Drop a video here</h2>"
            "<p>a phone clip of a sit-to-stand test or walking, or a folder from a previous "
            "run. No video yet? Try the sample clip below.<br>"
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
        self.try_sample = QPushButton("Try the sample clip")  # [REVIEW]
        self.try_sample.setToolTip(
            "A bundled 13-second side-view sit-to-stand recording; processes in about 15 s"
        )
        row.addWidget(self.open_video)
        row.addWidget(self.open_trial)
        row.addWidget(self.try_sample)
        lay.addLayout(row)
        sub = QHBoxLayout()
        sub.setSpacing(6)
        sub.addStretch(1)  # stretches on both ends keep fixed-width fields clustered
        sub.addWidget(QLabel("Subject:"))

        # input fields
        self.feet_box = QLineEdit()
        self.feet_box.setPlaceholderText("ft")
        self.feet_box.setValidator(QIntValidator(3, 7))
        self.feet_box.setFixedWidth(48)
        self.feet_box.setToolTip("Height, feet")
        self.inch_box = QLineEdit()
        self.inch_box.setPlaceholderText("in")
        self.inch_box.setValidator(QIntValidator(0, 11))
        self.inch_box.setFixedWidth(48)
        self.inch_box.setToolTip("Height, inches")
        sub.addSpacing(6)
        sub.addWidget(QLabel("height"))
        sub.addWidget(self.feet_box)
        sub.addWidget(QLabel("′"))
        sub.addWidget(self.inch_box)
        sub.addWidget(QLabel("″"))
        self.age_box = QLineEdit()
        self.age_box.setPlaceholderText("age")
        self.age_box.setValidator(QIntValidator(1, 120))
        self.age_box.setFixedWidth(56)
        self.age_box.setToolTip("Age in years (selects the reference band)")
        sub.addSpacing(12)
        sub.addWidget(self.age_box)
        self.sex_box = QComboBox()
        for label, value in (("sex", ""), ("female", "f"), ("male", "m")):
            self.sex_box.addItem(label, value)
        sub.addSpacing(12)
        sub.addWidget(self.sex_box)
        self.max_height_box = QComboBox()
        for label, cap in (
            ("full resolution", 0),
            ("downscale to 1080", 1080),
            ("downscale to 720", 720),
        ):
            self.max_height_box.addItem(label, cap)
        self.max_height_box.setToolTip(
            "Cap the shorter side of the frame during ingest (4K clips are slow)"
        )
        sub.addSpacing(12)
        sub.addWidget(self.max_height_box)
        sub.addStretch(1)
        lay.addLayout(sub)
        self._settings = QSettings("ptvision", "app")
        self._restore_prefs()
        self.max_height_box.currentIndexChanged.connect(self._save_prefs)
        self.recent_label = QLabel("Recent")
        self.recent_label.setStyleSheet("color: #9aa4b2; margin-top: 12px;")
        self.recent = QListWidget()
        self.recent.setMaximumHeight(120)
        self.recent.setFixedWidth(640)
        self.recent.setToolTip("Double-click to reopen an analysed trial")
        lay.addWidget(self.recent_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(self.recent, alignment=Qt.AlignmentFlag.AlignHCenter)
        self._load_recent()
        self.note = QLabel(
            f"<span style='color:#9aa4b2'>ptvision {__version__} · measurement tool</span>"
        )
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.note)

    def selected_protocol(self) -> str:
        return str(self.protocol.currentData())

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
        paths = [p for p in self.recent_paths() if p != trial_dir]
        paths.insert(0, trial_dir)
        self._settings.setValue("recent/trials", [str(p) for p in paths[:8]])
        self._load_recent()

    def _load_recent(self) -> None:
        self.recent.clear()
        tmp = Path(tempfile.gettempdir()).resolve()
        for p in self.recent_paths():
            if p.exists() and not p.resolve().is_relative_to(tmp):  # skip test scratch dirs
                item = QListWidgetItem(f"{p.name}   ({p.parent})")
                item.setData(Qt.ItemDataRole.UserRole, str(p))
                self.recent.addItem(item)
        has = self.recent.count() > 0
        self.recent.setVisible(has)
        self.recent_label.setVisible(has)

    def _save_prefs(self) -> None:
        self._settings.setValue("ingest/max_height", int(self.max_height_box.currentData()))

    def _restore_prefs(self) -> None:
        try:
            cap = int(str(self._settings.value("ingest/max_height", 0)))
        except (TypeError, ValueError):
            cap = 0
        self.max_height_box.setCurrentIndex(max(0, self.max_height_box.findData(cap)))
        self._settings.remove("subject")  # drop values persisted by the pre-fix build

    def select_protocol(self, pid: str) -> None:
        idx = self.protocol.findData(pid)
        if idx >= 0:
            self.protocol.setCurrentIndex(idx)


class ProgressPage(QWidget):
    def __init__(self, parent=None) -> None:
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

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(1, 1)
        self.preview.hide()
        self._preview_image: QImage | None = None
        lay.addWidget(self.preview, alignment=Qt.AlignmentFlag.AlignCenter)
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

    _STAGE_LABELS = {
        "ingest": "preparing the video",
        "loading model": "loading the model",
        "pose": "tracking the skeleton",
        "pose (reusing cached keypoints)": "loading saved tracking",
        "quality checks": "checking the capture",
        "filtering": "smoothing",
        "segmenting": "finding the movements",
        "metrics": "measuring",
        "writing artifacts": "saving",
        "report": "writing the report",
        "overlay": "rendering the overlay",
    }

    def set_stage(self, name: str) -> None:
        self.stage.setText(self._STAGE_LABELS.get(name, name))
        self.bar.setRange(0, 0)

    def set_preview(self, frame: object) -> None:
        import numpy as np

        arr = np.ascontiguousarray(frame)
        h, w = arr.shape[:2]
        img = QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_BGR888).copy()
        self._preview_image = img
        avail = max(240, min(560, self.height() - 200))
        pix = QPixmap.fromImage(img)
        if pix.height() > avail:
            pix = pix.scaledToHeight(avail, Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(pix)
        self.preview.show()

    def clear_preview(self) -> None:
        self.preview.clear()
        self.preview.hide()
        self._preview_image = None
