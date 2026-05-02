"""
OCT ROI Selector GUI — Galvo Mirror Waveform Generator
For Andrei's OCT scanning system (OCTA-500 dataset)

Assumptions:
  - Full scan: X ∈ [0, 2V], Y ∈ [0, 2V]  (galvo mirror full range)
  - OCTA-500 image resolution: 500×500 px (standard)
  - X axis = fast axis → sawtooth waveform
  - Y axis = slow axis → stair-step waveform
  - ROI selection maps pixel coords → voltage proportionally

Usage:
  python oct_roi_selector.py
  OR with a pre-segmented image:
  python oct_roi_selector.py --image path/to/segmented.png
"""

import sys
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import argparse
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QSlider, QFileDialog, QGroupBox,
    QGridLayout, QSpinBox, QDoubleSpinBox, QComboBox, QSplitter,
    QFrame, QTabWidget, QCheckBox, QMessageBox, QSizePolicy,
    QStatusBar, QToolBar, QAction, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QRect, QPoint, QSize, pyqtSignal, QThread, QTimer
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush, QFont,
    QCursor, QIcon, QPalette, QLinearGradient
)

try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

ROOT_DIR  = Path(__file__).resolve().parent.parent


# ─── Constants ────────────────────────────────────────────────────────────────

OCTA_PIXEL_SIZE = 500          # OCTA-500 image is 500×500 px
GALVO_V_MAX     = 2.0          # Full sweep = 2V
GALVO_V_MIN     = 0.0          # Start = 0V

# Dark scientific theme palette
BG_DARK      = "#0d1117"
BG_PANEL     = "#161b22"
BG_CARD      = "#1c2128"
ACCENT_CYAN  = "#39d0d8"
ACCENT_GREEN = "#3fb950"
ACCENT_AMBER = "#e3b341"
ACCENT_RED   = "#f85149"
TEXT_PRIMARY = "#e6edf3"
TEXT_DIM     = "#7d8590"
BORDER       = "#30363d"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def px_to_volts(px, img_size, v_min=GALVO_V_MIN, v_max=GALVO_V_MAX):
    """Convert pixel coordinate to galvo voltage (proportional)."""
    return v_min + (px / img_size) * (v_max - v_min)


def volts_to_px(v, img_size, v_min=GALVO_V_MIN, v_max=GALVO_V_MAX):
    """Convert galvo voltage to pixel coordinate."""
    return int((v - v_min) / (v_max - v_min) * img_size)


def generate_sawtooth(v_start, v_end, n_lines, samples_per_line):
    """
    Generate X (fast axis) sawtooth waveform.
    Each line: ramp from v_start → v_end, then fly-back.
    Returns flat array of length n_lines * samples_per_line.
    """
    line = np.linspace(v_start, v_end, samples_per_line)
    return np.tile(line, n_lines)


def generate_stair_step(v_start, v_end, n_lines, samples_per_line):
    """
    Generate Y (slow axis) stair-step waveform.
    Each step is held for samples_per_line samples.
    Returns flat array of length n_lines * samples_per_line.
    """
    steps = np.linspace(v_start, v_end, n_lines)
    return np.repeat(steps, samples_per_line)


# ─── ROI Canvas Widget ────────────────────────────────────────────────────────

class ROICanvas(QLabel):
    """
    Interactive image canvas supporting drag-to-draw rectangular ROIs.
    Emits roi_changed(x0, y0, x1, y1) in image-pixel coordinates.
    """
    roi_changed = pyqtSignal(int, int, int, int)   # x0, y0, x1, y1 in img px
    roi_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._pixmap_orig = None   # original loaded image pixmap
        self._display_px  = None   # scaled pixmap for display

        # ROI state (in display coords)
        self._drawing   = False
        self._roi_start = None
        self._roi_end   = None
        self._roi_rect  = None     # finalised QRect in display coords

        # AI / vessel segmentation overlay pixmap
        self._seg_overlay = None
        self._show_seg    = True

        # Multiple ROIs support
        self._roi_list = []        # list of QRect in display coords

        self.setStyleSheet(f"""
            QLabel {{
                background-color: {BG_DARK};
                border: 2px solid {BORDER};
                border-radius: 6px;
            }}
        """)

    # ── Image loading ─────────────────────────────────────────────────────────

    def load_image(self, path):
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self._pixmap_orig = pm
        self._roi_list.clear()
        self._roi_rect = None
        self._update_display()
        return True

    def load_ndarray(self, arr):
        """Accept a numpy HxW or HxWx3 uint8 array."""
        if arr.dtype != np.uint8:
            arr = (arr / arr.max() * 255).astype(np.uint8)
        if arr.ndim == 2:
            h, w = arr.shape
            qi = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, c = arr.shape
            qi = QImage(arr.data, w, h, w * c, QImage.Format_RGB888)
        self._pixmap_orig = QPixmap.fromImage(qi)
        self._roi_list.clear()
        self._roi_rect = None
        self._update_display()

    def set_segmentation_overlay(self, arr_rgba):
        """arr_rgba: HxWx4 uint8 RGBA overlay."""
        h, w, _ = arr_rgba.shape
        qi = QImage(arr_rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
        self._seg_overlay = QPixmap.fromImage(qi)
        self._update_display()

    def toggle_segmentation(self, show):
        self._show_seg = show
        self._update_display()

    # ── Layout / scaling ──────────────────────────────────────────────────────

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)

    def _update_display(self):
        if self._pixmap_orig is None:
            self._draw_placeholder()
            return

        # Scale to fit while keeping aspect ratio
        scaled = self._pixmap_orig.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Composite segmentation overlay if present
        if self._seg_overlay and self._show_seg:
            seg_scaled = self._seg_overlay.scaled(
                scaled.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            painter = QPainter(scaled)
            painter.setOpacity(0.45)
            painter.drawPixmap(0, 0, seg_scaled)
            painter.end()

        # Draw all ROI rectangles
        painter = QPainter(scaled)
        for i, rect in enumerate(self._roi_list):
            dr = self._img_rect_to_display(rect, scaled.size())
            self._draw_roi_rect(painter, dr, i, active=False)

        # Draw current in-progress rect
        if self._roi_start and self._roi_end:
            r = QRect(self._roi_start, self._roi_end).normalized()
            # map to scaled image space
            dr = self._display_to_scaled(r, scaled.size())
            self._draw_roi_rect(painter, dr, len(self._roi_list), active=True)
        elif self._roi_rect:
            dr = self._img_rect_to_display(self._roi_rect, scaled.size())
            self._draw_roi_rect(painter, dr, 0, active=True)

        painter.end()

        self._display_px = scaled
        self.setPixmap(scaled)

    def _draw_roi_rect(self, painter, rect, index, active):
        color = QColor(ACCENT_CYAN) if active else QColor(ACCENT_AMBER)
        pen = QPen(color, 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
        painter.drawRect(rect)

        # Corner handles
        handle_size = 6
        for pt in [rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()]:
            painter.fillRect(
                QRect(pt.x() - handle_size // 2, pt.y() - handle_size // 2,
                      handle_size, handle_size),
                color
            )

        # Label
        painter.setPen(QPen(color))
        painter.setFont(QFont("Courier New", 8, QFont.Bold))
        painter.drawText(rect.topLeft() + QPoint(4, -4), f"ROI {index + 1}")

    def _draw_placeholder(self):
        pm = QPixmap(self.size())
        pm.fill(QColor(BG_DARK))
        painter = QPainter(pm)
        painter.setPen(QPen(QColor(TEXT_DIM)))
        painter.setFont(QFont("Courier New", 12))
        painter.drawText(pm.rect(), Qt.AlignCenter,
                         "Load an OCT image\nor click 'Open Image'")
        painter.end()
        self.setPixmap(pm)

    # ── Coord mapping ─────────────────────────────────────────────────────────

    def _offset(self, display_size):
        """Top-left offset of the scaled pixmap within this label."""
        lw, lh = self.width(), self.height()
        pw, ph = display_size.width(), display_size.height()
        return QPoint((lw - pw) // 2, (lh - ph) // 2)

    def _label_to_img(self, pt, display_size):
        """Map label pixel → image pixel (0..OCTA_PIXEL_SIZE)."""
        off = self._offset(display_size)
        rel = pt - off
        sx = self._pixmap_orig.width()  / display_size.width()
        sy = self._pixmap_orig.height() / display_size.height()
        return QPoint(int(rel.x() * sx), int(rel.y() * sy))

    def _img_rect_to_display(self, img_rect, display_size):
        """Map image QRect → display QRect."""
        off = self._offset(display_size)
        sx = display_size.width()  / self._pixmap_orig.width()
        sy = display_size.height() / self._pixmap_orig.height()
        return QRect(
            off.x() + int(img_rect.x() * sx),
            off.y() + int(img_rect.y() * sy),
            int(img_rect.width() * sx),
            int(img_rect.height() * sy)
        )

    def _display_to_scaled(self, rect, display_size):
        """Map label-space QRect → scaled-pixmap-space QRect."""
        off = self._offset(display_size)
        return QRect(rect.x() - off.x(), rect.y() - off.y(),
                     rect.width(), rect.height())

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap_orig:
            self._drawing   = True
            self._roi_start = event.pos()
            self._roi_end   = event.pos()
            self._roi_rect  = None

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._roi_end = event.pos()
            self._update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            self._roi_end = event.pos()

            if self._pixmap_orig and self._display_px:
                ds = self._display_px.size()
                p0 = self._label_to_img(self._roi_start, ds)
                p1 = self._label_to_img(self._roi_end,   ds)

                # Clamp to image bounds
                iw = self._pixmap_orig.width()
                ih = self._pixmap_orig.height()
                p0 = QPoint(max(0, min(iw, p0.x())), max(0, min(ih, p0.y())))
                p1 = QPoint(max(0, min(iw, p1.x())), max(0, min(ih, p1.y())))

                self._roi_rect = QRect(p0, p1).normalized()
                if self._roi_rect.width() > 5 and self._roi_rect.height() > 5:
                    self.roi_changed.emit(
                        self._roi_rect.x(), self._roi_rect.y(),
                        self._roi_rect.right(), self._roi_rect.bottom()
                    )

            self._roi_start = None
            self._roi_end   = None
            self._update_display()

    def add_roi_to_list(self):
        if self._roi_rect:
            self._roi_list.append(self._roi_rect)
            self._roi_rect = None
            self._update_display()

    def clear_rois(self):
        self._roi_list.clear()
        self._roi_rect = None
        self._roi_start = None
        self._roi_end   = None
        self._update_display()
        self.roi_cleared.emit()

    def get_image_size(self):
        if self._pixmap_orig:
            return self._pixmap_orig.width(), self._pixmap_orig.height()
        return OCTA_PIXEL_SIZE, OCTA_PIXEL_SIZE


# ─── Waveform Preview (Matplotlib) ────────────────────────────────────────────

class WaveformCanvas(FigureCanvas if HAS_MPL else QWidget):
    def __init__(self, parent=None):
        if not HAS_MPL:
            super().__init__(parent)
            return
        self.fig = Figure(figsize=(5, 3), facecolor=BG_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_axes()

    def _setup_axes(self):
        self.fig.clear()
        self.ax_x = self.fig.add_subplot(211)
        self.ax_y = self.fig.add_subplot(212)
        for ax in (self.ax_x, self.ax_y):
            ax.set_facecolor(BG_PANEL)
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
        self.ax_x.set_title("X — Fast Axis (Sawtooth)", color=ACCENT_CYAN, fontsize=8)
        self.ax_y.set_title("Y — Slow Axis (Stair-Step)", color=ACCENT_AMBER, fontsize=8)
        self.ax_x.set_ylabel("Voltage (V)", color=TEXT_DIM, fontsize=7)
        self.ax_y.set_ylabel("Voltage (V)", color=TEXT_DIM, fontsize=7)
        self.ax_y.set_xlabel("Samples", color=TEXT_DIM, fontsize=7)
        self.fig.tight_layout(pad=1.5)

    def update_waveforms(self, x_start, x_end, y_start, y_end,
                         n_lines=50, samples_per_line=500):
        if not HAS_MPL:
            return
        self._setup_axes()
        x_wave = generate_sawtooth(x_start, x_end, n_lines, samples_per_line)
        y_wave = generate_stair_step(y_start, y_end, n_lines, samples_per_line)
        t = np.arange(len(x_wave))
        self.ax_x.plot(t, x_wave, color=ACCENT_CYAN, lw=0.8)
        self.ax_y.plot(t, y_wave, color=ACCENT_AMBER, lw=0.8)
        # Mark ROI voltage extents
        for ax, v0, v1, col in [
            (self.ax_x, x_start, x_end, ACCENT_CYAN),
            (self.ax_y, y_start, y_end, ACCENT_AMBER),
        ]:
            ax.axhline(v0, color=ACCENT_GREEN, lw=0.7, ls="--", label=f"start {v0:.3f}V")
            ax.axhline(v1, color=ACCENT_RED,   lw=0.7, ls="--", label=f"end {v1:.3f}V")
            ax.legend(fontsize=6, facecolor=BG_PANEL, edgecolor=BORDER,
                      labelcolor=TEXT_PRIMARY)
        self.fig.tight_layout(pad=1.5)
        self.draw()


# ─── Main Window ──────────────────────────────────────────────────────────────

class OCTROISelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCT ROI Selector — Galvo Waveform Generator")
        self.setMinimumSize(1200, 750)
        self._apply_theme()
        self._build_ui()
        self._current_roi = None   # (x0_px, y0_px, x1_px, y1_px) image coords

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet(f"""
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
                width: 12px;
                height: 12px;
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
        """)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Left panel: controls
        left = self._build_left_panel()
        root.addWidget(left, stretch=0)

        # Centre: image canvas
        centre = self._build_centre_panel()
        root.addWidget(centre, stretch=3)

        # Right: waveform / export
        right = self._build_right_panel()
        root.addWidget(right, stretch=2)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load an OCT image to begin")

    # ─── Left panel ───────────────────────────────────────────────────────────

    def _build_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(230)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Header
        hdr = QLabel("OCT ROI\nSELECTOR")
        hdr.setObjectName("header")
        hdr.setAlignment(Qt.AlignCenter)
        layout.addWidget(hdr)

        sub = QLabel("Galvo Waveform Generator\nfor OCTA-500 Dataset")
        sub.setObjectName("dim")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        div = QFrame(); div.setObjectName("divider")
        layout.addWidget(div)

        # ── Image load ────────────────────────────────────────────────────────
        img_grp = QGroupBox("Image")
        ig = QVBoxLayout(img_grp)

        self.btn_open = QPushButton("📂  Open Image")
        self.btn_open.clicked.connect(self._open_image)
        ig.addWidget(self.btn_open)

        self.btn_demo = QPushButton("🔬  Load Demo (synthetic)")
        self.btn_demo.clicked.connect(self._load_demo)
        ig.addWidget(self.btn_demo)

        self.chk_seg = QCheckBox("Show Segmentation Overlay")
        self.chk_seg.setChecked(True)
        self.chk_seg.toggled.connect(self._toggle_seg)
        ig.addWidget(self.chk_seg)

        layout.addWidget(img_grp)

        # ── Full scan parameters ──────────────────────────────────────────────
        scan_grp = QGroupBox("Full Scan Parameters")
        sg = QGridLayout(scan_grp)

        sg.addWidget(QLabel("V max (X):"), 0, 0)
        self.spin_vmax_x = QDoubleSpinBox()
        self.spin_vmax_x.setRange(0.1, 10.0); self.spin_vmax_x.setValue(2.0)
        self.spin_vmax_x.setSuffix(" V"); self.spin_vmax_x.setSingleStep(0.1)
        sg.addWidget(self.spin_vmax_x, 0, 1)

        sg.addWidget(QLabel("V max (Y):"), 1, 0)
        self.spin_vmax_y = QDoubleSpinBox()
        self.spin_vmax_y.setRange(0.1, 10.0); self.spin_vmax_y.setValue(2.0)
        self.spin_vmax_y.setSuffix(" V"); self.spin_vmax_y.setSingleStep(0.1)
        sg.addWidget(self.spin_vmax_y, 1, 1)

        sg.addWidget(QLabel("Lines (Y):"), 2, 0)
        self.spin_nlines = QSpinBox()
        self.spin_nlines.setRange(10, 2000); self.spin_nlines.setValue(500)
        sg.addWidget(self.spin_nlines, 2, 1)

        sg.addWidget(QLabel("Samples/line:"), 3, 0)
        self.spin_spl = QSpinBox()
        self.spin_spl.setRange(10, 5000); self.spin_spl.setValue(500)
        sg.addWidget(self.spin_spl, 3, 1)

        layout.addWidget(scan_grp)

        # ── ROI controls ──────────────────────────────────────────────────────
        roi_grp = QGroupBox("ROI Controls")
        rg = QVBoxLayout(roi_grp)

        self.btn_add_roi = QPushButton("➕  Add ROI to List")
        self.btn_add_roi.clicked.connect(self._add_roi)
        rg.addWidget(self.btn_add_roi)

        self.btn_clear = QPushButton("🗑  Clear All ROIs")
        self.btn_clear.setObjectName("danger")
        self.btn_clear.clicked.connect(self._clear_rois)
        rg.addWidget(self.btn_clear)

        layout.addWidget(roi_grp)

        # ── Segmentation ──────────────────────────────────────────────────────
        seg_grp = QGroupBox("Vessel Segmentation")
        segg = QVBoxLayout(seg_grp)

        self.btn_run_seg = QPushButton("🫀  Run Segmentation")
        self.btn_run_seg.setObjectName("primary")
        self.btn_run_seg.clicked.connect(self._run_segmentation)
        segg.addWidget(self.btn_run_seg)

        seg_note = QLabel("Runs your existing\nsegmentation pipeline\nand overlays result.")
        seg_note.setObjectName("dim")
        segg.addWidget(seg_note)

        layout.addWidget(seg_grp)

        layout.addStretch()

        # Version tag
        ver = QLabel("v1.0  |  OCTA-500\n0–2 V galvo range")
        ver.setObjectName("dim")
        ver.setAlignment(Qt.AlignCenter)
        layout.addWidget(ver)

        return panel

    # ─── Centre panel ─────────────────────────────────────────────────────────

    def _build_centre_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        lbl = QLabel("OCT Image — Drag to Select ROI")
        lbl.setObjectName("dim")
        top_row.addWidget(lbl)
        top_row.addStretch()
        self.lbl_img_info = QLabel("No image loaded")
        self.lbl_img_info.setObjectName("dim")
        top_row.addWidget(self.lbl_img_info)
        layout.addLayout(top_row)

        self.canvas = ROICanvas()
        self.canvas.roi_changed.connect(self._on_roi_changed)
        self.canvas.roi_cleared.connect(self._on_roi_cleared)
        layout.addWidget(self.canvas)

        # ROI voltage readout bar
        roi_bar = QGroupBox("Current ROI — Galvo Voltages")
        rb = QGridLayout(roi_bar)
        rb.setHorizontalSpacing(20)

        def voltage_label():
            l = QLabel("—")
            l.setObjectName("voltage")
            return l

        self.lbl_x0 = voltage_label(); self.lbl_x1 = voltage_label()
        self.lbl_y0 = voltage_label(); self.lbl_y1 = voltage_label()
        self.lbl_px_x0 = QLabel(""); self.lbl_px_x0.setObjectName("dim")
        self.lbl_px_x1 = QLabel(""); self.lbl_px_x1.setObjectName("dim")
        self.lbl_px_y0 = QLabel(""); self.lbl_px_y0.setObjectName("dim")
        self.lbl_px_y1 = QLabel(""); self.lbl_px_y1.setObjectName("dim")

        rb.addWidget(QLabel("X start (fast):"), 0, 0)
        rb.addWidget(self.lbl_x0,               0, 1)
        rb.addWidget(self.lbl_px_x0,            0, 2)
        rb.addWidget(QLabel("X end:"),           0, 3)
        rb.addWidget(self.lbl_x1,               0, 4)
        rb.addWidget(self.lbl_px_x1,            0, 5)

        rb.addWidget(QLabel("Y start (slow):"), 1, 0)
        rb.addWidget(self.lbl_y0,               1, 1)
        rb.addWidget(self.lbl_px_y0,            1, 2)
        rb.addWidget(QLabel("Y end:"),           1, 3)
        rb.addWidget(self.lbl_y1,               1, 4)
        rb.addWidget(self.lbl_px_y1,            1, 5)

        layout.addWidget(roi_bar)
        return panel

    # ─── Right panel ──────────────────────────────────────────────────────────

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        tabs = QTabWidget()

        # Tab 1: Waveforms
        wave_tab = QWidget()
        wl = QVBoxLayout(wave_tab)

        if HAS_MPL:
            self.waveform_canvas = WaveformCanvas()
            wl.addWidget(self.waveform_canvas)
        else:
            wl.addWidget(QLabel("Install matplotlib for waveform preview"))

        self.btn_update_wave = QPushButton("⟳  Preview Waveforms")
        self.btn_update_wave.setObjectName("primary")
        self.btn_update_wave.clicked.connect(self._update_waveforms)
        wl.addWidget(self.btn_update_wave)

        tabs.addTab(wave_tab, "Waveforms")

        # Tab 2: Export
        export_tab = QWidget()
        el = QVBoxLayout(export_tab)

        el.addWidget(QLabel("Export ROI Parameters:"))

        self.btn_export_txt = QPushButton("📋  Copy to Clipboard")
        self.btn_export_txt.clicked.connect(self._export_clipboard)
        el.addWidget(self.btn_export_txt)

        self.btn_export_npy = QPushButton("💾  Save Waveforms (.npy)")
        self.btn_export_npy.clicked.connect(self._export_npy)
        el.addWidget(self.btn_export_npy)

        self.btn_export_csv = QPushButton("📊  Save Waveforms (.csv)")
        self.btn_export_csv.clicked.connect(self._export_csv)
        el.addWidget(self.btn_export_csv)

        el.addSpacing(10)
        el.addWidget(QLabel("Parameter Summary:"))
        self.lbl_summary = QLabel("No ROI selected yet.")
        self.lbl_summary.setObjectName("dim")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        el.addWidget(self.lbl_summary)

        el.addStretch()
        tabs.addTab(export_tab, "Export")

        # Tab 3: Help
        help_tab = QWidget()
        hl = QVBoxLayout(help_tab)
        help_text = QLabel(
            "<b>How to use:</b><br><br>"
            "1. Load an OCT image (or use Demo)<br>"
            "2. Click + drag on the image to draw an ROI rectangle<br>"
            "3. Read off X/Y galvo voltages in the readout bar<br>"
            "4. Click 'Preview Waveforms' to visualise sawtooth + stair-step<br>"
            "5. Export waveforms as .npy or .csv for your DAQ<br><br>"
            "<b>Coordinate convention:</b><br>"
            "X = fast axis → sawtooth (full sweep each line)<br>"
            "Y = slow axis → stair-step (one step per line)<br><br>"
            "<b>Voltage mapping:</b><br>"
            "Image pixel 0 → 0 V<br>"
            "Image pixel W → V_max (default 2 V)<br>"
            "Proportional scaling for ROI sub-region<br><br>"
            "<b>Segmentation:</b><br>"
            "Hook your segmentation function into<br>"
            "<i>_run_segmentation()</i> below.<br>"
            "The overlay is displayed semi-transparently."
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("dim")
        hl.addWidget(help_text)
        hl.addStretch()
        tabs.addTab(help_tab, "Help")

        layout.addWidget(tabs)
        return panel

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open OCT Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
        )
        if path:
            if self.canvas.load_image(path):
                w, h = self.canvas.get_image_size()
                self.lbl_img_info.setText(f"{os.path.basename(path)}  {w}×{h}px")
                self.status.showMessage(f"Loaded: {path}")
            else:
                QMessageBox.warning(self, "Error", "Could not load image.")

    def _load_demo(self):
        """Generate a synthetic OCTA-like image for demo purposes."""
        sz = OCTA_PIXEL_SIZE
        img = np.zeros((sz, sz), dtype=np.uint8)

        rng = np.random.default_rng(42)
        # Background speckle
        img += rng.integers(0, 40, (sz, sz), dtype=np.uint8)

        # Simulate vessel network: random branching lines
        from itertools import product as itp
        for _ in range(80):
            x0, y0 = rng.integers(0, sz, 2)
            length  = rng.integers(30, 150)
            angle   = rng.uniform(0, 2 * np.pi)
            thickness = rng.integers(1, 4)
            for step in range(length):
                xi = int(x0 + step * np.cos(angle)) % sz
                yi = int(y0 + step * np.sin(angle)) % sz
                for di in range(-thickness, thickness + 1):
                    for dj in range(-thickness, thickness + 1):
                        img[
                            np.clip(yi + di, 0, sz - 1),
                            np.clip(xi + dj, 0, sz - 1)
                        ] = rng.integers(180, 255)

        self.canvas.load_ndarray(img)

        # Auto-generate a simple segmentation overlay (thresholded)
        mask = (img > 150).astype(np.uint8)
        overlay = np.zeros((sz, sz, 4), dtype=np.uint8)
        overlay[mask == 1, 0] = 57   # R
        overlay[mask == 1, 1] = 208  # G
        overlay[mask == 1, 2] = 216  # B  (ACCENT_CYAN ≈ #39d0d8)
        overlay[mask == 1, 3] = 180  # A
        self.canvas.set_segmentation_overlay(overlay)

        self.lbl_img_info.setText(f"Demo (synthetic OCTA)  {sz}×{sz}px")
        self.status.showMessage("Demo image loaded — drag to select ROI")

    def _toggle_seg(self, checked):
        self.canvas.toggle_segmentation(checked)

    def _on_roi_changed(self, x0, y0, x1, y1):
        self._current_roi = (x0, y0, x1, y1)
        iw, ih = self.canvas.get_image_size()
        vx_max = self.spin_vmax_x.value()
        vy_max = self.spin_vmax_y.value()

        vx0 = px_to_volts(x0, iw, 0, vx_max)
        vx1 = px_to_volts(x1, iw, 0, vx_max)
        vy0 = px_to_volts(y0, ih, 0, vy_max)
        vy1 = px_to_volts(y1, ih, 0, vy_max)

        self.lbl_x0.setText(f"{vx0:.4f} V")
        self.lbl_x1.setText(f"{vx1:.4f} V")
        self.lbl_y0.setText(f"{vy0:.4f} V")
        self.lbl_y1.setText(f"{vy1:.4f} V")

        self.lbl_px_x0.setText(f"(px {x0})")
        self.lbl_px_x1.setText(f"(px {x1})")
        self.lbl_px_y0.setText(f"(px {y0})")
        self.lbl_px_y1.setText(f"(px {y1})")

        summary = (
            f"ROI pixels:  x=[{x0}, {x1}]  y=[{y0}, {y1}]\n"
            f"ROI size:    {x1-x0} × {y1-y0} px\n\n"
            f"X (fast):  {vx0:.4f} V → {vx1:.4f} V\n"
            f"Y (slow):  {vy0:.4f} V → {vy1:.4f} V\n\n"
            f"Sawtooth ramps from {vx0:.4f}V to {vx1:.4f}V\n"
            f"Stair-step from    {vy0:.4f}V to {vy1:.4f}V"
        )
        self.lbl_summary.setText(summary)
        self.status.showMessage(
            f"ROI selected  X: {vx0:.3f}→{vx1:.3f} V   Y: {vy0:.3f}→{vy1:.3f} V"
        )
        # Auto-update waveform preview
        self._update_waveforms()

    def _on_roi_cleared(self):
        for lbl in (self.lbl_x0, self.lbl_x1, self.lbl_y0, self.lbl_y1):
            lbl.setText("—")
        for lbl in (self.lbl_px_x0, self.lbl_px_x1, self.lbl_px_y0, self.lbl_px_y1):
            lbl.setText("")
        self.lbl_summary.setText("No ROI selected yet.")
        self._current_roi = None
        self.status.showMessage("ROI cleared")

    def _add_roi(self):
        self.canvas.add_roi_to_list()
        self.status.showMessage("ROI added to list")

    def _clear_rois(self):
        self.canvas.clear_rois()

    def _update_waveforms(self):
        if not HAS_MPL or self._current_roi is None:
            return
        x0, y0, x1, y1 = self._current_roi
        iw, ih = self.canvas.get_image_size()
        vx_max = self.spin_vmax_x.value()
        vy_max = self.spin_vmax_y.value()
        vx0 = px_to_volts(x0, iw, 0, vx_max)
        vx1 = px_to_volts(x1, iw, 0, vx_max)
        vy0 = px_to_volts(y0, ih, 0, vy_max)
        vy1 = px_to_volts(y1, ih, 0, vy_max)
        self.waveform_canvas.update_waveforms(
            vx0, vx1, vy0, vy1,
            n_lines=self.spin_nlines.value(),
            samples_per_line=self.spin_spl.value()
        )

    def _run_segmentation(self):
        import tensorflow as tf

        if self.canvas._pixmap_orig is None:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return

        # ── Load model ─────────────────────────────────────────────
        seg_path =  ROOT_DIR / 'models' / 'segmentation_best.keras'
        if not seg_path.exists():
            QMessageBox.warning(self, "Model Missing",
                                "segmentation_best.keras not found.")
            return

        try:
            seg_model = tf.keras.models.load_model(
                str(seg_path),
                compile=False
            )
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))
            return

        # ── Get image from canvas ─────────────────────────────────
        qimg = self.canvas._pixmap_orig.toImage()
        w = qimg.width()
        h = qimg.height()

        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())

        arr = np.array(ptr).reshape(h, w, 4)  # RGBA
        img = arr[:, :, :3]  # drop alpha

        # convert to grayscale if needed
        if img.ndim == 3:
            img = np.mean(img, axis=-1)

        # normalize
        img = img.astype(np.float32) / 255.0

        # resize to model input (assuming 224x224)
        import cv2
        input_size = 224
        img_resized = cv2.resize(img, (input_size, input_size))

        # ── Predict ───────────────────────────────────────────────
        pred = seg_model.predict(
            img_resized[np.newaxis, :, :, np.newaxis],
            verbose=0
        )[0, :, :, 0]

        # ── Post-process ──────────────────────────────────────────
        pred_mask = (pred > 0.5).astype(np.uint8)

        # resize back to original image size
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # ── Create overlay ────────────────────────────────────────
        rgba_array = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_array[pred_mask == 1] = [57, 208, 216, 180]  # cyan

        # ── Display ───────────────────────────────────────────────
        self.canvas.set_segmentation_overlay(rgba_array)

        self.status.showMessage("Segmentation complete")

    # ── Export ────────────────────────────────────────────────────────────────

    def _get_waveform_arrays(self):
        if self._current_roi is None:
            return None, None
        x0, y0, x1, y1 = self._current_roi
        iw, ih = self.canvas.get_image_size()
        vx0 = px_to_volts(x0, iw, 0, self.spin_vmax_x.value())
        vx1 = px_to_volts(x1, iw, 0, self.spin_vmax_x.value())
        vy0 = px_to_volts(y0, ih, 0, self.spin_vmax_y.value())
        vy1 = px_to_volts(y1, ih, 0, self.spin_vmax_y.value())
        n  = self.spin_nlines.value()
        sp = self.spin_spl.value()
        return (
            generate_sawtooth(vx0, vx1, n, sp),
            generate_stair_step(vy0, vy1, n, sp)
        )

    def _export_clipboard(self):
        QApplication.clipboard().setText(self.lbl_summary.text())
        self.status.showMessage("Copied to clipboard")

    def _export_npy(self):
        x_wave, y_wave = self._get_waveform_arrays()
        if x_wave is None:
            QMessageBox.warning(self, "No ROI", "Select an ROI first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Waveforms", "oct_roi_waveforms.npy", "NumPy (*.npy)"
        )
        if path:
            np.save(path, {"x_sawtooth": x_wave, "y_stairstep": y_wave})
            self.status.showMessage(f"Saved: {path}")

    def _export_csv(self):
        x_wave, y_wave = self._get_waveform_arrays()
        if x_wave is None:
            QMessageBox.warning(self, "No ROI", "Select an ROI first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Waveforms CSV", "oct_roi_waveforms.csv", "CSV (*.csv)"
        )
        if path:
            data = np.column_stack([x_wave, y_wave])
            np.savetxt(path, data, delimiter=",",
                       header="x_sawtooth_V,y_stairstep_V", comments="")
            self.status.showMessage(f"Saved: {path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OCT ROI Selector — Galvo Waveform Generator")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to OCT image to load on startup")
    args = parser.parse_args()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("OCT ROI Selector")

    window = OCTROISelector()
    window.show()

    if args.image and os.path.isfile(args.image):
        window.canvas.load_image(args.image)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()