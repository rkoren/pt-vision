"""[REVIEW] Playback controls under the video, after the iPhone result screen.

A scrubber with the movement phases drawn on its track and a numbered chip at each seat-off, then
step back, play/pause, step forward, the time, and a 1× / ½× / ¼× speed button. Every control
is NoFocus: the window's key handling keeps Space and the arrows, and no focus ring shows on a
projector."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ptvision.app import theme
from ptvision.app.session import Phase, TrialSession

# iOS riseFill / systemGray5 / sitFill, brighter than the phone's 28% so they read on a dark track
PHASE_FILL = {
    "rise": QColor(64, 140, 255, 150),
    "stand": QColor(99, 99, 102, 150),
    "descent": QColor(255, 153, 51, 150),
}
RATE_LABELS = {1.0: "1×", 0.5: "½×", 0.25: "¼×"}


def _icon(kind: str, color: str, px: int = 18) -> QIcon:
    """Play, pause and frame-step glyphs, drawn so they look the same on any font."""
    scale = 3  # sharp on a Retina screen and a projector
    pm = QPixmap(px * scale, px * scale)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(scale, scale)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    s = float(px)

    def triangle(x0: float, x1: float) -> None:  # pointing from x0 to x1
        path = QPainterPath(QPointF(x0, s * 0.18))
        path.lineTo(QPointF(x1, s * 0.5))
        path.lineTo(QPointF(x0, s * 0.82))
        path.closeSubpath()
        p.drawPath(path)

    if kind == "play":
        triangle(s * 0.26, s * 0.84)
    elif kind == "pause":
        p.drawRoundedRect(QRectF(s * 0.22, s * 0.18, s * 0.2, s * 0.64), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.58, s * 0.18, s * 0.2, s * 0.64), 1.5, 1.5)
    elif kind == "back":
        p.drawRoundedRect(QRectF(s * 0.16, s * 0.2, s * 0.13, s * 0.6), 1, 1)
        triangle(s * 0.82, s * 0.34)
    elif kind == "forward":
        triangle(s * 0.18, s * 0.66)
        p.drawRoundedRect(QRectF(s * 0.71, s * 0.2, s * 0.13, s * 0.6), 1, 1)
    p.end()
    return QIcon(pm)


class Scrubber(QWidget):
    """The clip as a track: phases coloured in, a chip per repetition, a draggable playhead."""

    scrubbed = Signal(int)  # a frame the user picked (pause, then seek)

    CHIP = 24
    TRACK = 8
    HANDLE = 18

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(self.CHIP + 8 + self.HANDLE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.n_frames = 1
        self.frame = 0
        self.seat_offs: list[int] = []
        self.phases: list[Phase] = []
        self._dragging = False

    def set_session(self, session: TrialSession | None) -> None:
        self.n_frames = max(1, session.n_frames) if session else 1
        self.seat_offs = (
            sorted(m.frame for m in session.events if m.label == "seat-off" and m.rep is not None)
            if session
            else []
        )
        self.phases = list(session.phases) if session else []
        self.frame = 0
        self.update()

    def set_frame(self, t: int) -> None:
        self.frame = int(t)
        self.update()

    @property
    def current_rep(self) -> int | None:
        """0-based index of the repetition under the playhead, or None before the first."""
        done = sum(1 for f in self.seat_offs if f <= self.frame)
        return done - 1 if done else None

    # ---- geometry ------------------------------------------------------------------------
    def _x(self, frame: float) -> float:
        pad = self.CHIP / 2
        span = max(1, self.n_frames - 1)
        return pad + (self.width() - 2 * pad) * min(max(frame, 0), span) / span

    def frame_at(self, x: float) -> int:
        pad = self.CHIP / 2
        w = max(1.0, self.width() - 2 * pad)
        return round(min(max((x - pad) / w, 0.0), 1.0) * max(0, self.n_frames - 1))

    def _track_y(self) -> float:
        return self.CHIP + 8 + self.HANDLE / 2

    def chip_rect(self, i: int) -> QRectF:
        x = self._x(self.seat_offs[i])
        return QRectF(x - self.CHIP / 2, 0, self.CHIP, self.CHIP)

    # ---- painting ------------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self._track_y()
        left, right = self._x(0), self._x(self.n_frames - 1)
        track = QRectF(left, y - self.TRACK / 2, right - left, self.TRACK)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(118, 118, 128, 61))
        p.drawRoundedRect(track, self.TRACK / 2, self.TRACK / 2)
        for ph in self.phases:
            x0, x1 = self._x(ph.start), self._x(ph.end)
            p.setBrush(PHASE_FILL.get(ph.kind, PHASE_FILL["stand"]))
            p.drawRect(QRectF(x0, track.top(), max(1.0, x1 - x0), self.TRACK))

        current = self.current_rep
        font = theme.number_font(self.font(), 12)
        p.setFont(font)
        for i in range(len(self.seat_offs)):
            r = self.chip_rect(i)
            on = i == current
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.ACCENT if on else theme.RAISED))
            p.drawEllipse(r)
            p.setPen(QColor("#000000" if on else theme.TEXT))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(i + 1))
            # a hairline from the chip down to its seat-off on the track
            p.setPen(QColor(theme.TEXT_3))
            p.drawLine(QPointF(r.center().x(), r.bottom() + 1), QPointF(r.center().x(), y - 6))

        hx = self._x(self.frame)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.ACCENT))
        p.drawEllipse(QPointF(hx, y), self.HANDLE / 2, self.HANDLE / 2)
        p.setBrush(QColor(theme.TEXT))
        p.drawEllipse(QPointF(hx, y), self.HANDLE / 2 - 3, self.HANDLE / 2 - 3)
        p.end()

    # ---- mouse ----------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        for i in range(len(self.seat_offs)):
            if self.chip_rect(i).adjusted(-3, -3, 3, 3).contains(pos):
                self.scrubbed.emit(self.seat_offs[i])
                return
        self._dragging = True
        self.scrubbed.emit(self.frame_at(pos.x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self.scrubbed.emit(self.frame_at(event.position().x()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False


class TransportBar(QFrame):
    playToggled = Signal()
    stepRequested = Signal(int)
    scrubbed = Signal(int)
    speedRequested = Signal()  # cycle 1× → ½× → ¼×

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("transport")
        self._fps = 30.0
        self._n_frames = 1
        self._playing = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 12)
        lay.setSpacing(6)
        self.scrubber = Scrubber()
        self.scrubber.scrubbed.connect(self.scrubbed.emit)
        lay.addWidget(self.scrubber)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.back = self._button("round", "back", "Back one frame (←)")
        self.play = self._button("play", "play", "Play / pause (Space)")
        theme.prominent(self.play)
        self.forward = self._button("round", "forward", "Forward one frame (→)")
        self.back.clicked.connect(lambda: self.stepRequested.emit(-1))
        self.forward.clicked.connect(lambda: self.stepRequested.emit(1))
        self.play.clicked.connect(self.playToggled.emit)
        self.time = QLabel()
        self.time.setFont(theme.number_font(self.time.font(), 17, bold=False))
        self.speed = QPushButton("1×")
        self.speed.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.speed.setFont(theme.number_font(self.speed.font(), 16))
        self.speed.setMinimumWidth(64)
        self.speed.setToolTip("Playback speed: 1×, ½× or ¼× (slow motion)")
        self.speed.clicked.connect(self.speedRequested.emit)
        for w in (self.back, self.play, self.forward):
            row.addWidget(w)
        row.addSpacing(8)
        row.addWidget(self.time)
        row.addStretch(1)
        row.addWidget(self.speed)
        lay.addLayout(row)
        self._show_time(0)

    def _button(self, name: str, icon: str, tip: str) -> QPushButton:
        b = QPushButton()
        b.setObjectName(name)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setToolTip(tip)
        color = "#000000" if name == "play" else theme.TEXT
        b.setIcon(_icon(icon, color))
        b.setIconSize(QSize(20, 20) if name == "play" else QSize(16, 16))
        return b

    def set_session(self, session: TrialSession | None) -> None:
        self._fps = session.fps if session else 30.0
        self._n_frames = session.n_frames if session else 1
        self.scrubber.set_session(session)
        self.set_playing(False)
        self.set_rate(1.0)
        self._show_time(0)

    def set_frame(self, t: int) -> None:
        self.scrubber.set_frame(t)
        self._show_time(t)

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self.play.setIcon(_icon("pause" if playing else "play", "#000000"))
        self.play.setToolTip(("Pause" if playing else "Play") + " (Space)")

    def set_rate(self, rate: float) -> None:
        self.speed.setText(RATE_LABELS.get(rate, f"{rate:g}×"))

    def _show_time(self, t: int) -> None:
        total = max(0, self._n_frames - 1) / self._fps
        self.time.setText(
            f"<b>{t / self._fps:.2f}</b> <span style='color:{theme.TEXT_2}'>/ {total:.2f} s</span>"
        )
