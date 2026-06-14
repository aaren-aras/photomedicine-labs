# theme.py
# ========
# Qt stylesheet for the dark scientific theme.
# Apply once with: app.setStyleSheet(STYLESHEET)

from constants import (
    BG_DARK, BG_PANEL, BG_CARD,
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED,
    TEXT_PRIMARY, TEXT_DIM, BORDER
)

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'Courier New', monospace;
    font-size: 12px;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px;
    font-weight: bold;
    color: {TEXT_DIM};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    color: {ACCENT_CYAN};
    font-size: 11px;
}}
QPushButton {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 14px;
    font-family: 'Courier New', monospace;
}}
QPushButton:hover {{
    background-color: {BORDER};
    border-color: {ACCENT_CYAN};
}}
QPushButton#primary {{
    background-color: {ACCENT_CYAN};
    color: {BG_DARK};
    font-weight: bold;
    border: none;
}}
QPushButton#primary:hover {{
    background-color: #5de8ef;
}}
QPushButton#danger {{
    background-color: transparent;
    color: {ACCENT_RED};
    border: 1px solid {ACCENT_RED};
}}
QPushButton#danger:hover {{
    background-color: rgba(248,81,73,0.15);
}}
QDoubleSpinBox, QSpinBox, QComboBox {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: {ACCENT_CYAN};
}}
QLabel#header {{
    font-size: 15px;
    font-weight: bold;
    color: {ACCENT_CYAN};
    letter-spacing: 2px;
}}
QLabel#voltage {{
    font-size: 13px;
    font-weight: bold;
    color: {ACCENT_GREEN};
    font-family: 'Courier New', monospace;
}}
QLabel#dim {{
    color: {TEXT_DIM};
    font-size: 11px;
}}
QSlider::groove:horizontal {{
    background: {BG_CARD};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT_CYAN};
    width: 12px; height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
QStatusBar {{
    background-color: {BG_PANEL};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background-color: {BG_PANEL};
}}
QTabBar::tab {{
    background-color: {BG_CARD};
    color: {TEXT_DIM};
    padding: 6px 16px;
    border: 1px solid {BORDER};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background-color: {BG_PANEL};
    color: {ACCENT_CYAN};
    border-bottom: 2px solid {ACCENT_CYAN};
}}
QFrame#divider {{
    background-color: {BORDER};
    max-height: 1px;
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {BG_CARD};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_CYAN};
    border-color: {ACCENT_CYAN};
}}
"""
