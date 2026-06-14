# main_window.py
# ==============
# OCTROISelector: main application window.
# Pure UI wiring — all logic lives in the imported modules:
#   waveforms.py     → voltage math
#   segmentation.py  → model inference
#   canvas.py    → image interaction
#   waveform_canvas.py → waveform plots

import os
import sys
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QGroupBox, QGridLayout,
    QSpinBox, QDoubleSpinBox, QTabWidget, QCheckBox,
    QMessageBox, QSizePolicy, QStatusBar, QFrame, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from constants import OCTA_PIXEL_SIZE, GALVO_V_MAX, SEG_MODEL_PATH
from roi_canvas import ROICanvas
from waveform_canvas import WaveformCanvas
from waveforms import roi_to_voltages, generate_sawtooth, generate_stair_step
from segmentation import SegmentationModel
from theme import STYLESHEET


# ─── Background worker for segmentation (keeps UI responsive) ────────────────

class SegWorker(QThread):
    """Run segmentation inference on a background thread."""
    done    = pyqtSignal(object, object)   # mask, overlay
    error   = pyqtSignal(str)

    def __init__(self, model: SegmentationModel, img: np.ndarray):
        super().__init__()
        self._model = model
        self._img   = img

    def run(self):
        try:
            mask, overlay = self._model.predict(self._img)
            self.done.emit(mask, overlay)
        except Exception as e:
            self.error.emit(str(e))


# ─── Main Window ──────────────────────────────────────────────────────────────

class OCTROISelector(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCTA ROI Selector->Galvo Waveform Generator")
        self.setMinimumSize(1200, 760)
        self.setStyleSheet(STYLESHEET)

        self._current_roi  = None    # (x0, y0, x1, y1) image px
        self._seg_model    = SegmentationModel()
        self._seg_worker   = None    # background thread handle

        self._build_ui()

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load an OCT image to begin")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        centre = self._build_centre()   # ← build centre FIRST (creates self.canvas)
        left   = self._build_left()     # ← left references self.canvas, so must be after
        right  = self._build_right()

        root.addWidget(left,   stretch=0)   # still added to layout in correct visual order
        root.addWidget(centre, stretch=3)
        root.addWidget(right,  stretch=2)

    # ─── Left panel ───────────────────────────────────────────────────────────

    def _build_left(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(235)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        hdr = QLabel("OCTA ROI\nSELECTOR")
        hdr.setObjectName("header")
        hdr.setAlignment(Qt.AlignCenter)
        lay.addWidget(hdr)

        sub = QLabel("Galvo Waveform Generator\nOCTA-500  · assume 0–2 V range")
        sub.setObjectName("dim")
        sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(sub)

        div = QFrame(); div.setObjectName("divider")
        lay.addWidget(div)

        # Image group
        ig = QGroupBox("Image")
        il = QVBoxLayout(ig)
        self.btn_open = QPushButton("📂  Open Image")
        self.btn_open.clicked.connect(self._open_image)
        il.addWidget(self.btn_open)

        self.btn_demo = QPushButton("🔬  Load Demo (synthetic)")
        self.btn_demo.clicked.connect(self._load_demo)
        il.addWidget(self.btn_demo)

        self.chk_seg = QCheckBox("Show Segmentation Overlay")
        self.chk_seg.setChecked(True)
        self.chk_seg.toggled.connect(lambda v: self.canvas.toggle_segmentation(v))
        il.addWidget(self.chk_seg)
        lay.addWidget(ig)

        # Scan parameters group
        sg = QGroupBox("Full Scan Parameters")
        sl = QGridLayout(sg)

        sl.addWidget(QLabel("V max  (X):"), 0, 0)
        self.spin_vmax_x = QDoubleSpinBox()
        self.spin_vmax_x.setRange(0.1, 10.0)
        self.spin_vmax_x.setValue(GALVO_V_MAX)
        self.spin_vmax_x.setSuffix(" V")
        self.spin_vmax_x.setSingleStep(0.1)
        sl.addWidget(self.spin_vmax_x, 0, 1)

        sl.addWidget(QLabel("V max  (Y):"), 1, 0)
        self.spin_vmax_y = QDoubleSpinBox()
        self.spin_vmax_y.setRange(0.1, 10.0)
        self.spin_vmax_y.setValue(GALVO_V_MAX)
        self.spin_vmax_y.setSuffix(" V")
        self.spin_vmax_y.setSingleStep(0.1)
        sl.addWidget(self.spin_vmax_y, 1, 1)

        sl.addWidget(QLabel("Lines (Y):"), 2, 0)
        self.spin_nlines = QSpinBox()
        self.spin_nlines.setRange(10, 2000)
        self.spin_nlines.setValue(500)
        sl.addWidget(self.spin_nlines, 2, 1)

        sl.addWidget(QLabel("Samples/line:"), 3, 0)
        self.spin_spl = QSpinBox()
        self.spin_spl.setRange(10, 5000)
        self.spin_spl.setValue(500)
        sl.addWidget(self.spin_spl, 3, 1)
        lay.addWidget(sg)

        # ROI controls
        rg = QGroupBox("ROI Controls")
        rl = QVBoxLayout(rg)
        self.btn_add_roi = QPushButton("➕  Add ROI to List")
        self.btn_add_roi.clicked.connect(lambda: self.canvas.add_roi_to_list())
        rl.addWidget(self.btn_add_roi)

        self.btn_clear = QPushButton("🗑  Clear All ROIs")
        self.btn_clear.setObjectName("danger")
        self.btn_clear.clicked.connect(self.canvas.clear_rois)
        rl.addWidget(self.btn_clear)
        lay.addWidget(rg)

        # Segmentation
        seg_g = QGroupBox("Vessel Segmentation")
        seg_l = QVBoxLayout(seg_g)
        self.btn_run_seg = QPushButton("🫀  Run Segmentation")
        self.btn_run_seg.setObjectName("primary")
        self.btn_run_seg.clicked.connect(self._run_segmentation)
        seg_l.addWidget(self.btn_run_seg)

        model_status = "✓ Model found" if SEG_MODEL_PATH.exists() else "⚠ Model not found"
        self.lbl_model_status = QLabel(model_status)
        self.lbl_model_status.setObjectName("dim")
        seg_l.addWidget(self.lbl_model_status)
        lay.addWidget(seg_g)

        lay.addStretch()
        return panel

    # ─── Centre panel ─────────────────────────────────────────────────────────

    def _build_centre(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Top info bar
        top = QHBoxLayout()
        top.addWidget(QLabel("OCT Image  —  Drag to Select ROI"))
        top.addStretch()
        self.lbl_img_info = QLabel("No image loaded")
        self.lbl_img_info.setObjectName("dim")
        top.addWidget(self.lbl_img_info)
        lay.addLayout(top)

        # Canvas
        self.canvas = ROICanvas()
        self.canvas.roi_changed.connect(self._on_roi_changed)
        self.canvas.roi_cleared.connect(self._on_roi_cleared)
        lay.addWidget(self.canvas)

        # Voltage readout bar
        vg = QGroupBox("Current ROI  —  Galvo Voltages")
        vl = QGridLayout(vg)
        vl.setHorizontalSpacing(18)

        def vlabel():
            l = QLabel("—"); l.setObjectName("voltage"); return l

        def dlabel():
            l = QLabel(""); l.setObjectName("dim"); return l

        self.lbl_x0 = vlabel(); self.lbl_x1 = vlabel()
        self.lbl_y0 = vlabel(); self.lbl_y1 = vlabel()
        self.lbl_px_x0 = dlabel(); self.lbl_px_x1 = dlabel()
        self.lbl_px_y0 = dlabel(); self.lbl_px_y1 = dlabel()

        vl.addWidget(QLabel("X start (fast):"), 0, 0)
        vl.addWidget(self.lbl_x0,               0, 1)
        vl.addWidget(self.lbl_px_x0,            0, 2)
        vl.addWidget(QLabel("X end:"),           0, 3)
        vl.addWidget(self.lbl_x1,               0, 4)
        vl.addWidget(self.lbl_px_x1,            0, 5)

        vl.addWidget(QLabel("Y start (slow):"), 1, 0)
        vl.addWidget(self.lbl_y0,               1, 1)
        vl.addWidget(self.lbl_px_y0,            1, 2)
        vl.addWidget(QLabel("Y end:"),           1, 3)
        vl.addWidget(self.lbl_y1,               1, 4)
        vl.addWidget(self.lbl_px_y1,            1, 5)

        lay.addWidget(vg)
        return panel

    # ─── Right panel ──────────────────────────────────────────────────────────

    def _build_right(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()

        # Waveform tab
        wt = QWidget()
        wl = QVBoxLayout(wt)
        self.waveform_canvas = WaveformCanvas()
        wl.addWidget(self.waveform_canvas)
        self.btn_wave = QPushButton("⟳  Preview Waveforms")
        self.btn_wave.setObjectName("primary")
        self.btn_wave.clicked.connect(self._update_waveforms)
        wl.addWidget(self.btn_wave)
        tabs.addTab(wt, "Waveforms")

        # Export tab
        et = QWidget()
        el = QVBoxLayout(et)
        el.addWidget(QLabel("Export ROI Parameters:"))

        self.btn_copy = QPushButton("📋  Copy to Clipboard")
        self.btn_copy.clicked.connect(self._export_clipboard)
        el.addWidget(self.btn_copy)

        self.btn_npy = QPushButton("💾  Save Waveforms (.npy)")
        self.btn_npy.clicked.connect(self._export_npy)
        el.addWidget(self.btn_npy)

        self.btn_csv = QPushButton("📊  Save Waveforms (.csv)")
        self.btn_csv.clicked.connect(self._export_csv)
        el.addWidget(self.btn_csv)

        el.addSpacing(10)
        el.addWidget(QLabel("Parameter Summary:"))
        self.lbl_summary = QLabel("No ROI selected yet.")
        self.lbl_summary.setObjectName("dim")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        el.addWidget(self.lbl_summary)
        el.addStretch()
        tabs.addTab(et, "Export")

        # Help tab
        ht = QWidget()
        hl = QVBoxLayout(ht)
        help_lbl = QLabel(
            "<b>How to use:</b><br><br>"
            "1. Load an OCTA image (PNG/BMP/TIFF) or use Demo<br>"
            "2. Click + drag to draw an ROI rectangle<br>"
            "3. Read off X/Y galvo voltages in the readout bar<br>"
            "4. Click 'Preview Waveforms' to see sawtooth + stair-step<br>"
            "5. Export waveforms as .npy or .csv for your data acquisition <br><br>"
            "<b>Coordinate convention:</b><br>"
            "X  =  fast axis  →  sawtooth (sweeps each line)<br>"
            "Y  =  slow axis  →  stair-step (one step per line)<br><br>"
            "<b>Voltage mapping:</b><br>"
            "pixel 0  →  0 V<br>"
            "pixel W  →  V_max  (default 2 V)<br>"
            "ROI sub-region scales proportionally<br><br>"
            "<b>Segmentation:</b><br>"
            "Loads <i>segmentation_best.keras</i> directly — no Docker needed.<br>"
            "CPU inference ~1s per image."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setObjectName("dim")
        hl.addWidget(help_lbl)
        hl.addStretch()
        tabs.addTab(ht, "Help")

        lay.addWidget(tabs)
        return panel

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open OCT Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
        )
        if not path:
            return
        if self.canvas.load_image(path):
            w, h = self.canvas.get_image_size()
            self.lbl_img_info.setText(f"{os.path.basename(path)}  {w}×{h}px")
            self.status.showMessage(f"Loaded: {path}")
        else:
            QMessageBox.warning(self, "Error", "Could not load image.")

    def _load_demo(self):
        sz = OCTA_PIXEL_SIZE
        rng = np.random.default_rng(42)
        img = rng.integers(0, 40, (sz, sz), dtype=np.uint8)

        # Synthetic vessel network
        for _ in range(80):
            x0, y0   = rng.integers(0, sz, 2)
            length   = rng.integers(30, 150)
            angle    = rng.uniform(0, 2 * np.pi)
            thick    = rng.integers(1, 4)
            for s in range(length):
                xi = int(x0 + s * np.cos(angle)) % sz
                yi = int(y0 + s * np.sin(angle)) % sz
                for di in range(-thick, thick + 1):
                    for dj in range(-thick, thick + 1):
                        img[
                            np.clip(yi + di, 0, sz - 1),
                            np.clip(xi + dj, 0, sz - 1)
                        ] = rng.integers(180, 255)

        self.canvas.load_ndarray(img)

        # Simple threshold overlay for demo
        mask = (img > 150).astype(np.uint8)
        overlay = np.zeros((sz, sz, 4), dtype=np.uint8)
        overlay[mask == 1] = [57, 208, 216, 150]
        self.canvas.set_segmentation_overlay(overlay)

        self.lbl_img_info.setText(f"Demo (synthetic OCTA)  {sz}×{sz}px")
        self.status.showMessage("Demo loaded — drag to select ROI")

    def _on_roi_changed(self, x0: int, y0: int, x1: int, y1: int):
        self._current_roi = (x0, y0, x1, y1)
        iw, ih = self.canvas.get_image_size()
        v = roi_to_voltages(x0, y0, x1, y1, iw, ih,
                            self.spin_vmax_x.value(),
                            self.spin_vmax_y.value())

        self.lbl_x0.setText(f"{v['vx0']:.4f} V")
        self.lbl_x1.setText(f"{v['vx1']:.4f} V")
        self.lbl_y0.setText(f"{v['vy0']:.4f} V")
        self.lbl_y1.setText(f"{v['vy1']:.4f} V")
        self.lbl_px_x0.setText(f"(px {x0})")
        self.lbl_px_x1.setText(f"(px {x1})")
        self.lbl_px_y0.setText(f"(px {y0})")
        self.lbl_px_y1.setText(f"(px {y1})")

        self.lbl_summary.setText(
            f"ROI pixels:  x=[{x0}, {x1}]  y=[{y0}, {y1}]\n"
            f"ROI size:    {x1-x0} × {y1-y0} px\n\n"
            f"X (fast):  {v['vx0']:.4f} V  →  {v['vx1']:.4f} V\n"
            f"Y (slow):  {v['vy0']:.4f} V  →  {v['vy1']:.4f} V\n\n"
            f"Sawtooth:   {v['vx0']:.4f} V  →  {v['vx1']:.4f} V\n"
            f"Stair-step: {v['vy0']:.4f} V  →  {v['vy1']:.4f} V"
        )
        self.status.showMessage(
            f"ROI  X: {v['vx0']:.3f}→{v['vx1']:.3f} V   "
            f"Y: {v['vy0']:.3f}→{v['vy1']:.3f} V"
        )
        self._update_waveforms()

    def _on_roi_cleared(self):
        for l in (self.lbl_x0, self.lbl_x1, self.lbl_y0, self.lbl_y1):
            l.setText("—")
        for l in (self.lbl_px_x0, self.lbl_px_x1,
                  self.lbl_px_y0, self.lbl_px_y1):
            l.setText("")
        self.lbl_summary.setText("No ROI selected yet.")
        self._current_roi = None
        self.status.showMessage("ROI cleared")

    def _update_waveforms(self):
        if self._current_roi is None:
            return
        x0, y0, x1, y1 = self._current_roi
        iw, ih = self.canvas.get_image_size()
        v = roi_to_voltages(x0, y0, x1, y1, iw, ih,
                            self.spin_vmax_x.value(),
                            self.spin_vmax_y.value())
        self.waveform_canvas.update_waveforms(
            v['vx0'], v['vx1'], v['vy0'], v['vy1'],
            n_lines=self.spin_nlines.value(),
            samples_per_line=self.spin_spl.value(),
        )

    def _run_segmentation(self):
        """
        Run vessel segmentation on the currently loaded image.
        Uses SegmentationModel which loads segmentation_best.keras directly —
        no Docker, no training environment needed. CPU inference only.
        """
        if self.canvas._pixmap_orig is None:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return

        if not self._seg_model.is_available():
            QMessageBox.warning(
                self, "Model Not Found",
                f"Could not find model at:\n{SEG_MODEL_PATH}\n\n"
                "Make sure segmentation_best.keras is in the models/ directory."
            )
            return

        # Extract numpy array from canvas pixmap
        qimg = self.canvas._pixmap_orig.toImage().convertToFormat(
            __import__('PyQt5.QtGui', fromlist=['QImage']).QImage.Format_RGBA8888
        )
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w * 4)
        img_arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()

        # Disable button, show progress
        self.btn_run_seg.setEnabled(False)
        self.btn_run_seg.setText("⏳  Running...")
        self.status.showMessage("Running segmentation…")

        # Run on background thread so UI stays responsive
        self._seg_worker = SegWorker(self._seg_model, img_arr)
        self._seg_worker.done.connect(self._on_seg_done)
        self._seg_worker.error.connect(self._on_seg_error)
        self._seg_worker.start()

    def _on_seg_done(self, mask: np.ndarray, overlay: np.ndarray):
        self.canvas.set_segmentation_overlay(overlay)
        vessel_pct = mask.mean() * 100
        self.btn_run_seg.setEnabled(True)
        self.btn_run_seg.setText("🫀  Run Segmentation")
        self.status.showMessage(
            f"Segmentation complete  —  {vessel_pct:.1f}% vessel pixels detected"
        )

    def _on_seg_error(self, msg: str):
        self.btn_run_seg.setEnabled(True)
        self.btn_run_seg.setText("🫀  Run Segmentation")
        QMessageBox.critical(self, "Segmentation Error", msg)
        self.status.showMessage("Segmentation failed")

    # ── Export ────────────────────────────────────────────────────────────────

    def _get_waveforms(self):
        if self._current_roi is None:
            return None, None
        x0, y0, x1, y1 = self._current_roi
        iw, ih = self.canvas.get_image_size()
        v  = roi_to_voltages(x0, y0, x1, y1, iw, ih,
                             self.spin_vmax_x.value(),
                             self.spin_vmax_y.value())
        n  = self.spin_nlines.value()
        sp = self.spin_spl.value()
        return (
            generate_sawtooth(v['vx0'], v['vx1'], n, sp),
            generate_stair_step(v['vy0'], v['vy1'], n, sp),
        )

    def _export_clipboard(self):
        QApplication.clipboard().setText(self.lbl_summary.text())
        self.status.showMessage("Copied to clipboard ✓")

    def _export_npy(self):
        xw, yw = self._get_waveforms()
        if xw is None:
            QMessageBox.warning(self, "No ROI", "Select an ROI first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Waveforms", "oct_roi_waveforms.npy", "NumPy (*.npy)"
        )
        if path:
            np.save(path, {"x_sawtooth": xw, "y_stairstep": yw})
            self.status.showMessage(f"Saved: {path}")

    def _export_csv(self):
        xw, yw = self._get_waveforms()
        if xw is None:
            QMessageBox.warning(self, "No ROI", "Select an ROI first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Waveforms CSV", "oct_roi_waveforms.csv", "CSV (*.csv)"
        )
        if path:
            np.savetxt(path, np.column_stack([xw, yw]),
                       delimiter=",",
                       header="x_sawtooth_V,y_stairstep_V",
                       comments="")
            self.status.showMessage(f"Saved: {path}")
