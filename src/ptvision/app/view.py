"""Video frame + skeleton overlay in a zoomable view, with the rep, test clock and joint angles
laid over its corners (the iPhone result screen's overlays)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ptvision.app import theme
from ptvision.app.overlay import SkeletonItem
from ptvision.app.session import TrialSession
from ptvision.video import FrameSource

_NAV_KEYS = {
    Qt.Key.Key_Left,
    Qt.Key.Key_Right,
    Qt.Key.Key_Up,
    Qt.Key.Key_Down,
    Qt.Key.Key_Home,
    Qt.Key.Key_End,
    Qt.Key.Key_Space,
    Qt.Key.Key_PageUp,
    Qt.Key.Key_PageDown,
}

CHIP_NAMES = {"knee_flexion": "Knee", "hip_flexion": "Hip", "trunk_lean": "Trunk"}
_MARGIN = 14


class AngleChips(QWidget):
    """The joint angles at the current frame, one small dark chip per angle."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)
        self.values: dict[str, QLabel] = {}

    def set_names(self, names: list[str]) -> None:
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        self.values = {}
        for name in names:
            chip = QFrame()
            chip.setObjectName("chip")
            hl = QHBoxLayout(chip)
            hl.setContentsMargins(12, 4, 12, 4)
            hl.setSpacing(10)
            title = QLabel(CHIP_NAMES.get(name, name.replace("_", " ").capitalize()))
            theme.set_role(title, "secondary")
            value = QLabel("–")
            value.setFont(theme.number_font(value.font(), 22))
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setMinimumWidth(value.fontMetrics().horizontalAdvance("188°"))
            hl.addWidget(title)
            hl.addStretch(1)
            hl.addWidget(value)
            self._lay.addWidget(chip)
            self.values[name] = value
        self.adjustSize()

    def set_values(self, values: dict[str, float]) -> None:
        for name, label in self.values.items():
            v = values.get(name)
            label.setText(f"{v:.0f}°" if v is not None else "–")


class VideoView(QGraphicsView):
    frameRendered = Signal(int)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._pix_item)
        self.skeleton = SkeletonItem()
        self._scene.addItem(self.skeleton)
        # overlays: rep and phase, the test clock (top left), joint angles (top right)
        self.badge = QLabel(self.viewport())
        self.badge.setObjectName("badge")
        self.badge.hide()
        self.clock = QLabel(self.viewport())
        self.clock.setObjectName("clock")
        self.clock.setFont(theme.number_font(self.clock.font(), 34))
        self.clock.hide()
        self.chips = AngleChips(self.viewport())
        self.chips.hide()
        self._session: TrialSession | None = None
        self._source: FrameSource | None = None
        self._frame: np.ndarray | None = None  # keep the array alive while displayed
        self._user_zoom = False
        self._fit_scale = 1.0
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QColor(theme.BG))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    # ---- session ------------------------------------------------------------------------
    def set_session(self, session: TrialSession | None, source: FrameSource | None) -> None:
        self.close_source()
        self._session = session
        self._source = source
        self.skeleton.set_session(session)
        if session is not None:
            w, h = session.image_size
            self._scene.setSceneRect(QRectF(0, 0, w, h))
            self.show_frame(0)
            self.fit()
        else:
            self._pix_item.setPixmap(QPixmap())
            self.set_badge(None)
            self.set_clock(None)
            self.set_angle_names([])

    def set_badge(self, text: str | None) -> None:
        if text:
            self.badge.setText(text)
            self.badge.adjustSize()
            self.badge.show()
            self.badge.raise_()
        else:
            self.badge.hide()
        self._place_overlays()

    def set_clock(self, seconds: float | None, *, done: bool = False) -> None:
        """The stopwatch for the timed test; accent once it has stopped at the final time."""
        if seconds is None:
            self.clock.hide()
            return
        self.clock.setText(f"{seconds:.2f} s")
        if bool(self.clock.property("done")) != done:
            self.clock.setProperty("done", done)
            theme.repolish(self.clock)
        self.clock.adjustSize()
        self.clock.show()
        self.clock.raise_()
        self._place_overlays()

    def set_angle_names(self, names: list[str]) -> None:
        self.chips.set_names(names)
        self.chips.setVisible(bool(names))
        self._place_overlays()

    def set_angles(self, values: dict[str, float]) -> None:
        self.chips.set_values(values)

    def _place_overlays(self) -> None:
        y = _MARGIN
        self.badge.move(_MARGIN, y)
        if self.badge.isVisible():
            y += self.badge.height() + 8
        self.clock.move(_MARGIN, y)
        self.chips.adjustSize()
        x = self.viewport().width() - self.chips.width() - _MARGIN
        self.chips.move(max(_MARGIN, x), _MARGIN)
        self.chips.raise_()

    def close_source(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    @property
    def current_frame(self) -> np.ndarray | None:
        return self._frame

    def show_frame(self, t: int) -> None:
        if self._source is None:
            return
        try:
            frame = self._source.read(int(t))
        except IndexError:
            return
        self._frame = frame
        h, w = frame.shape[:2]
        img = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888)
        self._pix_item.setPixmap(QPixmap.fromImage(img))
        self.skeleton.set_frame(t)
        self.frameRendered.emit(int(t))

    # ---- zoom / fit ------------------------------------------------------------------
    def fit(self) -> None:
        if self._session is None:
            return
        w, h = self._session.image_size
        vw, vh = self.viewport().width(), self.viewport().height()
        if w <= 0 or h <= 0 or vw <= 0 or vh <= 0:
            return
        s = min(vw / w, vh / h)
        self._fit_scale = s
        self.resetTransform()
        self.scale(s, s)
        self.centerOn(w / 2, h / 2)
        self._user_zoom = False
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def zoom_factor(self) -> float:
        return self.transform().m11() / self._fit_scale if self._fit_scale else 1.0

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if not self._user_zoom:
            self.fit()
        self._place_overlays()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._session is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        new = self.zoom_factor() * factor
        if new < 1.0:
            self.fit()
            return
        if new > 8.0:
            return
        self.scale(factor, factor)
        self._user_zoom = True
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # navigation keys go to the window's playback actions
        if event.key() in _NAV_KEYS:
            event.ignore()
            return
        super().keyPressEvent(event)
