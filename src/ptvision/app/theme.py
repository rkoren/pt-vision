"""[REVIEW] The desktop app's look, matched to the iPhone prototype (`ios/PTVision`).

Dark grouped surfaces, a lime accent that always carries black text, and the status colors the
skeleton already uses (viz/status.py). It stays dark whatever the macOS appearance setting, so the
screen looks the same on any laptop and projector."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QWidget

from ptvision.viz import status as vs

# iOS tokens
ACCENT = "#A6FF00"  # fill under black text, or text on dark; never text on white
ACCENT_INK = "#3D6B00"  # the accent for text and lines on a white page (the report)
GREEN = vs.hex_of(vs.STOPS[2][1])  # #2BA84A pass / in range
AMBER = vs.hex_of(vs.STOPS[1][1])  # #F2A900 warn / near limit
RED = vs.hex_of(vs.STOPS[0][1])  # #D62828 fail / outside range
NEUTRAL = vs.hex_of(vs.NEUTRAL)  # #8C9BAB not assessed
NOTE_INK = "#FFCC73"
NOTE_FILL = "rgba(255, 158, 26, 0.14)"

# iOS dark grouped surfaces
BG = "#000000"
CARD = "#1C1C1E"
RAISED = "#2C2C2E"  # controls on a card
HOVER = "#3A3A3C"
PRESSED = "#48484A"
SEPARATOR = "#38383A"
TEXT = "#FFFFFF"
TEXT_2 = "#98989F"  # secondary label (rgba(235,235,245,0.6) over black)
TEXT_3 = "#5A5A5F"  # tertiary / disabled
FILL = "rgba(118, 118, 128, 0.24)"

STATUS = {"pass": GREEN, "warn": AMBER, "fail": RED}

BASE_POINT_SIZE = 14  # one step up from the macOS default, for a projector

QSS = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; }}
QFrame#card {{ background: {CARD}; border-radius: 20px; }}
QLabel {{ background: transparent; }}
QLabel#badge {{
    background: rgba(28, 28, 30, 200); color: {TEXT}; font-weight: 600;
    padding: 5px 12px; border-radius: 13px;
}}
QLabel[role="title"] {{ font-size: 30px; font-weight: 700; }}
QLabel[role="headline"] {{ font-size: 20px; font-weight: 700; }}
QLabel[role="section"] {{ color: {TEXT_2}; font-size: 12px; font-weight: 600; }}
QLabel[role="secondary"] {{ color: {TEXT_2}; }}
QLabel[role="caption"] {{ color: {TEXT_2}; font-size: 12px; }}
QPushButton {{
    background: {RAISED}; color: {TEXT}; border: none; border-radius: 17px;
    padding: 8px 18px; font-weight: 600;
}}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {PRESSED}; }}
QPushButton:disabled {{ color: {TEXT_3}; }}
QPushButton[prominent="true"] {{ background: {ACCENT}; color: #000000; }}
QPushButton[prominent="true"]:hover {{ background: #B8FF33; }}
QPushButton[prominent="true"]:pressed {{ background: #8FDB00; }}
QToolButton {{ background: transparent; color: {ACCENT}; border: none; font-weight: 600; }}
QComboBox, QLineEdit {{
    background: {RAISED}; border: none; border-radius: 8px; padding: 6px 10px;
    selection-background-color: {ACCENT}; selection-color: #000000;
}}
QComboBox QAbstractItemView {{
    background: {RAISED}; border: none; selection-background-color: {HOVER};
    selection-color: {TEXT};
}}
QPlainTextEdit {{ background: {RAISED}; border: none; border-radius: 8px; padding: 6px; }}
QProgressBar {{
    background: {FILL}; border: none; border-radius: 4px; max-height: 8px; min-height: 8px;
    color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QListWidget {{ background: {CARD}; border: none; border-radius: 14px; padding: 6px; }}
QListWidget::item {{ padding: 8px 10px; border-radius: 8px; }}
QListWidget::item:hover {{ background: {RAISED}; }}
QListWidget::item:selected {{ background: rgba(166, 255, 0, 0.14); color: {TEXT}; }}
QTableWidget {{
    background: {CARD}; border: none; border-radius: 12px; gridline-color: {SEPARATOR};
}}
QHeaderView::section {{ background: {RAISED}; color: {TEXT_2}; border: none; padding: 4px; }}
QGroupBox {{
    background: {CARD}; border: none; border-radius: 14px; margin-top: 22px; padding: 10px;
    font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 4px; color: {TEXT_2}; }}
QCheckBox, QRadioButton {{ spacing: 8px; }}
QSplitter::handle {{ background: {BG}; }}
QStatusBar {{ background: {BG}; color: {TEXT_2}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {SEPARATOR}; padding: 4px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {HOVER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""


def palette() -> QPalette:
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: BG,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: CARD,
        QPalette.ColorRole.AlternateBase: RAISED,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: RAISED,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.BrightText: ACCENT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: "#000000",
        QPalette.ColorRole.ToolTipBase: RAISED,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.PlaceholderText: TEXT_2,
        QPalette.ColorRole.Link: ACCENT,
        QPalette.ColorRole.Mid: SEPARATOR,
    }
    for role, hx in roles.items():
        pal.setColor(role, QColor(hx))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(TEXT_3))
    return pal


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")  # the native macOS style ignores most of the palette
    app.setPalette(palette())
    font = QFont(app.font())
    font.setPointSize(BASE_POINT_SIZE)
    app.setFont(font)
    app.setStyleSheet(QSS)


def set_role(widget: QWidget, role: str) -> None:
    """Tag a label for the stylesheet: title, headline, section, secondary or caption."""
    widget.setProperty("role", role)


def prominent(button: QWidget) -> None:
    """The accent button with black text, for the one main action on a screen."""
    button.setProperty("prominent", True)
