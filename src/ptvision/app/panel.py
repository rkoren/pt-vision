"""Side panel: quality banner, metrics, events, display options, legend."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ptvision.app.session import TrialSession
from ptvision.viz import status as vs

_STATUS_COLORS = {"pass": "#2BA84A", "warn": "#F2A900", "fail": "#D62828"}


class LegendWidget(QWidget):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._mode: vs.Mode = "confidence"
        self.setMinimumHeight(20 * 5 + 8)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_mode(self, mode: vs.Mode) -> None:
        self._mode = mode
        self.setMinimumHeight(20 * len(vs.legend(mode)) + 8)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = 4
        for text, hx in vs.legend(self._mode):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(hx))
            p.drawRoundedRect(4, y + 3, 26, 12, 3, 3)
            p.setPen(self.palette().text().color())
            p.drawText(38, y + 14, text)
            y += 20
        p.end()


class SidePanel(QWidget):
    modeChanged = Signal(str)
    angleToggled = Signal(str, bool)
    eventActivated = Signal(int)
    showOthersToggled = Signal(bool)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._session: TrialSession | None = None
        self.setMinimumWidth(380)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setSpacing(8)

        self.title = QLabel("")
        self.title.setStyleSheet("font-weight: 600; font-size: 14px;")
        self.title.setWordWrap(True)
        lay.addWidget(self.title)

        self.quality = QLabel("")
        self.quality.setWordWrap(True)
        self.quality.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.quality)

        # display options
        opts = QGroupBox("Colour bones by")
        ol = QVBoxLayout(opts)
        self.radio_rules = QRadioButton("protocol angle rules")
        self.radio_conf = QRadioButton("tracking confidence")
        ol.addWidget(self.radio_rules)
        ol.addWidget(self.radio_conf)
        self.radio_rules.toggled.connect(self._on_rules_toggled)
        self.radio_conf.toggled.connect(self._on_conf_toggled)
        self.legend = LegendWidget()
        ol.addWidget(self.legend)
        self.show_others = QCheckBox("show other people")
        self.show_others.setChecked(True)
        self.show_others.toggled.connect(self.showOthersToggled.emit)
        ol.addWidget(self.show_others)
        lay.addWidget(opts)

        self.angles_box = QGroupBox("Timeline angles")
        self.angles_layout = QVBoxLayout(self.angles_box)
        self._angle_checks: dict[str, QCheckBox] = {}
        lay.addWidget(self.angles_box)

        mb = QGroupBox("Measurements")
        ml = QVBoxLayout(mb)
        self.metrics = QTableWidget(0, 3)
        self.metrics.setHorizontalHeaderLabels(["measure", "value", "tier"])
        self.metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.metrics.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.metrics.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.metrics.verticalHeader().setVisible(False)
        self.metrics.setWordWrap(True)
        self.metrics.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.metrics.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        ml.addWidget(self.metrics)
        self.norm_label = QLabel("")
        self.norm_label.setWordWrap(True)
        self.norm_label.setStyleSheet("color: #9aa4b2; font-size: 12px;")
        ml.addWidget(self.norm_label)
        lay.addWidget(mb)

        eb = QGroupBox("Events (click to jump)")
        el = QVBoxLayout(eb)
        self.events = QListWidget()
        self.events.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.events.itemActivated.connect(self._on_event)
        self.events.itemClicked.connect(self._on_event)
        el.addWidget(self.events)
        lay.addWidget(eb)

        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet("color: #F2A900; font-size: 12px;")
        lay.addWidget(self.warnings)
        lay.addStretch(1)

    # ---- population -------------------------------------------------------------------
    def set_session(self, session: TrialSession | None) -> None:
        self._session = session
        for w in list(self._angle_checks.values()):
            self.angles_layout.removeWidget(w)
            w.deleteLater()
        self._angle_checks.clear()
        self.metrics.setRowCount(0)
        self.events.clear()
        if session is None:
            self.title.setText("")
            self.quality.setText("")
            self.norm_label.setText("")
            self.warnings.setText("")
            return
        self.title.setText(session.title)

        if session.quality is not None:
            q = session.quality
            color = _STATUS_COLORS[q.status]
            bad = [c for c in q.checks if c.status != "pass"]
            lines = [f"<b style='color:{color}'>capture quality: {q.status.upper()}</b>"]
            lines += [f"• <b>{c.name}</b> ({c.status}): {c.message}" for c in bad]
            self.quality.setText("<br>".join(lines))
        else:
            self.quality.setText("<span style='color:#9aa4b2'>pose only, no protocol checks</span>")

        modes = session.available_modes()
        self.radio_rules.setEnabled("rules" in modes)
        default = session.default_mode()
        self.radio_rules.blockSignals(True)
        self.radio_conf.blockSignals(True)
        (self.radio_rules if default == "rules" else self.radio_conf).setChecked(True)
        self.radio_rules.blockSignals(False)
        self.radio_conf.blockSignals(False)
        self.legend.set_mode(default)

        defaults = set(session.default_angle_names())
        for name in session.angles.names:
            cb = QCheckBox(name.replace("_", " "))
            cb.setChecked(name in defaults)
            cb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            cb.toggled.connect(lambda on, n=name: self.angleToggled.emit(n, on))
            self.angles_layout.addWidget(cb)
            self._angle_checks[name] = cb

        self.metrics.setRowCount(len(session.metrics))
        for i, m in enumerate(session.metrics):
            side = f" ({m.side})" if m.side else ""
            self.metrics.setItem(i, 0, QTableWidgetItem(m.label + side))
            v = QTableWidgetItem(m.formatted())
            v.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.metrics.setItem(i, 1, v)
            self.metrics.setItem(i, 2, QTableWidgetItem(str(m.tier)))
        self.metrics.resizeRowsToContents()
        self.metrics.setMinimumHeight(min(320, 28 * (len(session.metrics) + 1) + 4))
        applicable = [n for n in session.norms if n.applicable]
        self.norm_label.setText(applicable[0].statement if applicable else "")

        for ev in session.events:
            item = QListWidgetItem(f"{ev.text}   f {ev.frame}   {ev.frame / session.fps:.2f} s")
            item.setData(Qt.ItemDataRole.UserRole, ev.frame)
            self.events.addItem(item)
        self.warnings.setText("<br>".join(session.warnings))

    def selected_angles(self) -> list[str]:
        return [n for n, cb in self._angle_checks.items() if cb.isChecked()]

    def set_mode(self, mode: vs.Mode) -> None:
        self.legend.set_mode(mode)
        target = self.radio_rules if mode == "rules" else self.radio_conf
        if not target.isChecked():
            target.setChecked(True)

    def _on_rules_toggled(self, on: bool) -> None:
        if on:
            self.modeChanged.emit("rules")

    def _on_conf_toggled(self, on: bool) -> None:
        if on:
            self.modeChanged.emit("confidence")

    def _on_event(self, item: QListWidgetItem) -> None:
        self.eventActivated.emit(int(item.data(Qt.ItemDataRole.UserRole)))
