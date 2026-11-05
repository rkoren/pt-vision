"""Frame clock for playback"""

from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal

RATES = (1.0, 0.5, 0.25)  # normal, half and quarter speed (slow motion for a class)


class Player(QObject):
    frameChanged = Signal(int)
    playingChanged = Signal(bool)
    rateChanged = Signal(float)

    def __init__(self, n_frames: int, fps: float, parent: QObject | None = None):
        super().__init__(parent)
        self.n_frames = max(1, int(n_frames))
        self.fps = float(fps) if fps > 0 else 30.0
        self._frame = 0
        self._playing = False
        self._base_frame = 0
        self._rate = 1.0
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(max(1, int(500 / self.fps)))
        self._timer.timeout.connect(self._tick)

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def time_s(self) -> float:
        return self._frame / self.fps

    @property
    def rate(self) -> float:
        return self._rate

    def set_rate(self, rate: float) -> None:
        """Playback speed as a fraction of real time. Changing it mid-play continues from the
        current frame instead of jumping."""
        rate = float(rate)
        if rate <= 0 or rate == self._rate:
            return
        self._rate = rate
        if self._playing:
            self._base_frame = self._frame
            self._clock.restart()
        self.rateChanged.emit(rate)

    def cycle_rate(self) -> float:
        """1× → ½× → ¼× → 1×."""
        i = RATES.index(self._rate) if self._rate in RATES else -1
        self.set_rate(RATES[(i + 1) % len(RATES)])
        return self._rate

    def _set_frame(self, i: int) -> None:
        i = max(0, min(self.n_frames - 1, int(i)))
        if i != self._frame:
            self._frame = i
            self.frameChanged.emit(i)

    def seek(self, i: int) -> None:
        self._set_frame(i)
        if self._playing:
            self._base_frame = self._frame
            self._clock.restart()

    def step(self, delta: int) -> None:
        self.pause()
        self._set_frame(self._frame + delta)

    def play(self) -> None:
        if self._playing:
            return
        if self._frame >= self.n_frames - 1:
            self._set_frame(0)
        self._playing = True
        self._base_frame = self._frame
        self._clock.restart()
        self._timer.start()
        self.playingChanged.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self.playingChanged.emit(False)

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def _tick(self) -> None:
        target = self._base_frame + int(self._clock.elapsed() * self.fps * self._rate / 1000.0)
        if target >= self.n_frames - 1:
            self._set_frame(self.n_frames - 1)
            self.pause()
            return
        if target != self._frame:
            self._set_frame(target)
