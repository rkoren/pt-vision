"""Skeleton drawn as a QGraphicsItem in image coordinates."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from ptvision.app.session import TrialSession
from ptvision.viz import status as vs
from ptvision.viz.skeleton import bone_segments


class SkeletonItem(QGraphicsItem):
    def __init__(self) -> None:
        super().__init__()
        self._session: TrialSession | None = None
        self._t = 0
        self._mode: vs.Mode = "confidence"
        self._show_others = True
        self._bone_width = 3.0
        self.setZValue(1)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    # ---- state ------------------------------------------------------------------------
    def set_session(self, session: TrialSession | None) -> None:
        self.prepareGeometryChange()
        self._session = session
        if session is not None:
            self._mode = session.default_mode()
        self.update()

    def set_frame(self, t: int) -> None:
        self._t = int(t)
        self.update()

    def set_mode(self, mode: vs.Mode) -> None:
        if self._session is not None and mode not in self._session.available_modes():
            mode = "confidence"
        self._mode = mode
        self.update()

    @property
    def mode(self) -> vs.Mode:
        return self._mode

    def set_show_others(self, show: bool) -> None:
        self._show_others = bool(show)
        self.update()

    # ---- QGraphicsItem ---------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        if self._session is None:
            return QRectF()
        w, h = self._session.image_size
        return QRectF(0, 0, w, h)

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None
    ) -> None:
        s = self._session
        if s is None or not (0 <= self._t < s.track.n_frames):
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        coords = s.track.coords[self._t]
        scores = s.track.score[self._t]
        lod = option.levelOfDetailFromTransform(painter.worldTransform()) or 1.0
        edges = s.edges

        if self._show_others:
            pen = QPen(QColor(*vs.OTHER_PERSON), 1.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for p in range(coords.shape[0]):
                if p == s.primary_slot or np.isnan(coords[p]).all():
                    continue
                for a, b in edges:
                    if (
                        scores[p, a] >= vs.CONF_HIDE
                        and scores[p, b] >= vs.CONF_HIDE
                        and not (np.isnan(coords[p, a]).any() or np.isnan(coords[p, b]).any())
                    ):
                        painter.drawLine(
                            QPointF(*coords[p, a].tolist()), QPointF(*coords[p, b].tolist())
                        )

        rgb, visible = s.bone_colors(self._t, self._mode)
        c = coords[s.primary_slot]
        for xa, ya, xb, yb, col in bone_segments(c, edges, rgb, visible):
            pen = QPen(QColor(*col), self._bone_width)
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(xa, ya), QPointF(xb, yb))

        r = 3.5 / lod
        painter.setPen(Qt.PenStyle.NoPen)
        sc = scores[s.primary_slot]
        for k in range(c.shape[0]):
            if sc[k] >= vs.CONF_HIDE and not np.isnan(c[k]).any():
                if self._mode == "rules":
                    color = QColor(*vs.JOINT_RULES_MODE)
                else:
                    color = QColor(*vs.status_rgb(vs.confidence_status(float(sc[k]))).tolist())
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(float(c[k, 0]), float(c[k, 1])), r, r)
