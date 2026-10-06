"""Joint-angle timeline synced to the player"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal

from ptvision.app.session import TrialSession

PHASE_COLORS = {
    "rise": (200, 230, 201, 110),
    "stand": (227, 242, 253, 110),
    "descent": (255, 224, 178, 110),
    "stance": (200, 230, 201, 90),
    "swing": (227, 242, 253, 90),
}


class AngleTimeline(pg.GraphicsLayoutWidget):
    seekRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setBackground("#1a1b1e")
        self._session: TrialSession | None = None
        self._names: list[str] = []
        self._plots: list[pg.PlotItem] = []
        self._cursors: list[pg.InfiniteLine] = []
        self._frame = 0
        self._block = False
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_session(self, session: TrialSession | None) -> None:
        self._session = session
        self._names = session.default_angle_names() if session else []
        self._rebuild()

    def set_angles(self, names: list[str]) -> None:
        if self._session is None:
            return
        self._names = [n for n in names if n in self._session.angles.names]
        self._rebuild()

    @property
    def angle_names(self) -> list[str]:
        return list(self._names)

    def _rebuild(self) -> None:
        self.clear()
        self._plots = []
        self._cursors = []
        s = self._session
        if s is None or not self._names:
            return
        t = np.arange(s.angles.frame.shape[0]) / s.fps
        first: pg.PlotItem | None = None
        for i, name in enumerate(self._names):
            plot = self.addPlot(row=i, col=0)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=True, y=False)
            plot.hideButtons()
            plot.setLabel("left", name.replace("_", " "), units="°")
            plot.showGrid(x=False, y=True, alpha=0.15)
            if first is None:
                first = plot
            else:
                plot.setXLink(first)
            for ph in s.phases:
                region = pg.LinearRegionItem(
                    values=(ph.start / s.fps, ph.end / s.fps),
                    movable=False,
                    brush=pg.mkBrush(*PHASE_COLORS[ph.kind]),
                    pen=pg.mkPen(None),
                )
                region.setZValue(-20)
                plot.addItem(region)
            for rule in s.rules_for(name):
                lo, hi = rule.band.ok
                band = pg.LinearRegionItem(
                    values=(lo, hi),
                    orientation="horizontal",
                    movable=False,
                    brush=pg.mkBrush(43, 168, 74, 40),
                    pen=pg.mkPen(43, 168, 74, 120, style=Qt.PenStyle.DashLine),
                )
                band.setZValue(-10)
                plot.addItem(band)
            if s.test_window is not None:
                for f in s.test_window:
                    plot.addItem(
                        pg.InfiniteLine(
                            pos=f / s.fps,
                            angle=90,
                            pen=pg.mkPen(220, 220, 220, 160, style=Qt.PenStyle.DashLine),
                        )
                    )
            values = s.angles[name]
            plot.plot(t, values, pen=pg.mkPen("#7fb3ff", width=1.6))
            cursor = pg.InfiniteLine(
                pos=self._frame / s.fps,
                angle=90,
                movable=True,
                pen=pg.mkPen("#ffd166", width=2),
                hoverPen=pg.mkPen("#ffe08a", width=3),
            )
            cursor.setZValue(50)
            cursor.sigDragged.connect(self._on_cursor_dragged)
            plot.addItem(cursor)
            self._plots.append(plot)
            self._cursors.append(cursor)
            if i < len(self._names) - 1:
                plot.hideAxis("bottom")
            else:
                plot.setLabel("bottom", "time", units="s")

    def set_frame(self, t: int) -> None:
        self._frame = int(t)
        if self._session is None:
            return
        x = self._frame / self._session.fps
        self._block = True
        try:
            for c in self._cursors:
                c.setPos(x)
        finally:
            self._block = False

    def _on_cursor_dragged(self, line: pg.InfiniteLine) -> None:
        if self._block or self._session is None:
            return
        self.seekRequested.emit(round(float(line.value()) * self._session.fps))

    def _on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._session is None or not self._plots:
            return
        pos = event.scenePos()
        for plot in self._plots:
            vb = plot.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                x = float(vb.mapSceneToView(pos).x())
                self.seekRequested.emit(round(x * self._session.fps))
                return

    def seek_to_x(self, seconds: float) -> None:
        """Programmatic equivalent of a click, used by tests."""
        if self._session is not None:
            self.seekRequested.emit(round(seconds * self._session.fps))
