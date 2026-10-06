"""Video frame + skeleton overlay in a zoomable view"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QLabel

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
        # rep badge
        self.badge = QLabel(self.viewport())
        self.badge.setStyleSheet(
            "background: rgba(20, 22, 26, 170); color: white; font-size: 18px; font-weight: 600;"
            " padding: 4px 10px; border-radius: 8px;"
        )
        self.badge.move(12, 12)
        self.badge.hide()
        self._session: TrialSession | None = None
        self._source: FrameSource | None = None
        self._frame: np.ndarray | None = None  # keep the array alive while displayed
        self._user_zoom = False
        self._fit_scale = 1.0
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QColor(18, 18, 20))
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

    def set_badge(self, text: str | None) -> None:
        if text:
            self.badge.setText(text)
            self.badge.adjustSize()
            self.badge.show()
            self.badge.raise_()
        else:
            self.badge.hide()

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
