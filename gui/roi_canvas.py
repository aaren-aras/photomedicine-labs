# roi_canvas.py
# =============
# ROICanvas: interactive Qt widget for drag-to-select rectangular ROIs
# on an OCTA image, with segmentation overlay compositing.
#
# Emits:
#   roi_changed(x0, y0, x1, y1) — image pixel coordinates, on mouse release
#   roi_cleared()               — when all ROIs are cleared

import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush,
    QFont, QCursor
)
from constants import (
    BG_DARK, BORDER, ACCENT_CYAN, ACCENT_AMBER, TEXT_DIM,
    OCTA_PIXEL_SIZE
)


class ROICanvas(QLabel):
    """
    Interactive image canvas — drag to draw rectangular ROIs.
    Handles:
        - Image loading from file path or numpy array
        - Segmentation overlay compositing (semi-transparent)
        - ROI drawing with corner handles and voltage labels
        - Multiple ROI accumulation
        - Coordinate mapping: label space ↔ image pixel space
    """

    roi_changed = pyqtSignal(int, int, int, int)   # x0, y0, x1, y1 (img px)
    roi_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._pixmap_orig  = None   # original image pixmap (full res)
        self._display_px   = None   # current scaled pixmap
        self._seg_overlay  = None   # segmentation overlay pixmap
        self._show_seg     = True

        # ROI drawing state (label coords)
        self._drawing    = False
        self._roi_start  = None
        self._roi_end    = None
        self._roi_rect   = None    # finalised QRect in IMAGE coords

        # Accumulated ROI list (image coords)
        self._roi_list   = []

        self.setStyleSheet(f"""
            QLabel {{
                background-color: {BG_DARK};
                border: 2px solid {BORDER};
                border-radius: 6px;
            }}
        """)

    # ── Image loading ─────────────────────────────────────────────────────────

    def load_image(self, path: str) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self._pixmap_orig = pm
        self._roi_list.clear()
        self._roi_rect = None
        self._update_display()
        return True

    def load_ndarray(self, arr: np.ndarray):
        """Accept HxW or HxWx3 uint8 numpy array."""
        if arr.dtype != np.uint8:
            arr = (arr / arr.max() * 255).astype(np.uint8)
        if arr.ndim == 2:
            h, w = arr.shape
            qi = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, _ = arr.shape
            arr = np.ascontiguousarray(arr)
            qi = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap_orig = QPixmap.fromImage(qi)
        self._roi_list.clear()
        self._roi_rect = None
        self._update_display()

    def set_segmentation_overlay(self, arr_rgba: np.ndarray):
        """
        Set the vessel segmentation overlay.
        arr_rgba: HxWx4 uint8 RGBA — vessel pixels have alpha > 0.
        """
        arr_rgba = np.ascontiguousarray(arr_rgba)
        h, w, _ = arr_rgba.shape
        qi = QImage(arr_rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
        self._seg_overlay = QPixmap.fromImage(qi)
        self._update_display()

    def toggle_segmentation(self, show: bool):
        self._show_seg = show
        self._update_display()

    def get_image_size(self) -> tuple[int, int]:
        if self._pixmap_orig:
            return self._pixmap_orig.width(), self._pixmap_orig.height()
        return OCTA_PIXEL_SIZE, OCTA_PIXEL_SIZE

    # ── Display ───────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)

    def _update_display(self):
        if self._pixmap_orig is None:
            self._draw_placeholder()
            return

        scaled = self._pixmap_orig.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Composite segmentation overlay
        if self._seg_overlay and self._show_seg:
            seg_s = self._seg_overlay.scaled(
                scaled.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            p = QPainter(scaled)
            p.setOpacity(0.55)
            p.drawPixmap(0, 0, seg_s)
            p.end()

        # Draw ROI rectangles
        p = QPainter(scaled)
        for i, rect in enumerate(self._roi_list):
            dr = self._img_to_scaled(rect, scaled.size())
            self._draw_roi(p, dr, i, active=False)

        # Draw active (in-progress or finalised) ROI
        if self._drawing and self._roi_start and self._roi_end:
            r = QRect(self._roi_start, self._roi_end).normalized()
            dr = self._label_to_scaled(r, scaled.size())
            self._draw_roi(p, dr, len(self._roi_list), active=True)
        elif self._roi_rect:
            dr = self._img_to_scaled(self._roi_rect, scaled.size())
            self._draw_roi(p, dr, len(self._roi_list), active=True)
        p.end()

        self._display_px = scaled
        self.setPixmap(scaled)

    def _draw_roi(self, painter: QPainter, rect: QRect,
                  index: int, active: bool):
        color = QColor(ACCENT_CYAN) if active else QColor(ACCENT_AMBER)
        painter.setPen(QPen(color, 2, Qt.SolidLine))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 35)))
        painter.drawRect(rect)

        # Corner handles
        hs = 6
        for pt in [rect.topLeft(), rect.topRight(),
                   rect.bottomLeft(), rect.bottomRight()]:
            painter.fillRect(
                QRect(pt.x() - hs // 2, pt.y() - hs // 2, hs, hs), color
            )

        painter.setPen(QPen(color))
        painter.setFont(QFont("Courier New", 8, QFont.Bold))
        painter.drawText(rect.topLeft() + QPoint(4, -4), f"ROI {index + 1}")

    def _draw_placeholder(self):
        pm = QPixmap(self.size())
        pm.fill(QColor(BG_DARK))
        p = QPainter(pm)
        p.setPen(QPen(QColor(TEXT_DIM)))
        p.setFont(QFont("Courier New", 12))
        p.drawText(pm.rect(), Qt.AlignCenter,
                   "Load an OCT image\nor click 'Open Image'")
        p.end()
        self.setPixmap(pm)

    # ── Coordinate mapping ────────────────────────────────────────────────────

    def _scaled_offset(self, scaled_size) -> QPoint:
        """Top-left offset of scaled pixmap within the label (letterbox)."""
        return QPoint(
            (self.width()  - scaled_size.width())  // 2,
            (self.height() - scaled_size.height()) // 2,
        )

    def _label_to_img(self, pt: QPoint, scaled_size) -> QPoint:
        """Label pixel → image pixel."""
        off = self._scaled_offset(scaled_size)
        rel = pt - off
        sx = self._pixmap_orig.width()  / scaled_size.width()
        sy = self._pixmap_orig.height() / scaled_size.height()
        return QPoint(int(rel.x() * sx), int(rel.y() * sy))

    def _img_to_scaled(self, img_rect: QRect, scaled_size) -> QRect:
        """Image QRect → scaled-pixmap-space QRect (for drawing)."""
        off = self._scaled_offset(scaled_size)
        sx = scaled_size.width()  / self._pixmap_orig.width()
        sy = scaled_size.height() / self._pixmap_orig.height()
        return QRect(
            off.x() + int(img_rect.x() * sx),
            off.y() + int(img_rect.y() * sy),
            int(img_rect.width()  * sx),
            int(img_rect.height() * sy),
        )

    def _label_to_scaled(self, rect: QRect, scaled_size) -> QRect:
        """Label-space QRect → scaled-pixmap-space QRect (subtract offset)."""
        off = self._scaled_offset(scaled_size)
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
        if event.button() != Qt.LeftButton or not self._drawing:
            return
        self._drawing = False
        self._roi_end = event.pos()

        if self._pixmap_orig and self._display_px:
            ds = self._display_px.size()
            iw = self._pixmap_orig.width()
            ih = self._pixmap_orig.height()

            p0 = self._label_to_img(self._roi_start, ds)
            p1 = self._label_to_img(self._roi_end,   ds)

            # Clamp to image bounds
            def clamp(pt):
                return QPoint(
                    max(0, min(iw, pt.x())),
                    max(0, min(ih, pt.y()))
                )
            p0, p1 = clamp(p0), clamp(p1)
            self._roi_rect = QRect(p0, p1).normalized()

            if self._roi_rect.width() > 5 and self._roi_rect.height() > 5:
                self.roi_changed.emit(
                    self._roi_rect.x(), self._roi_rect.y(),
                    self._roi_rect.right(), self._roi_rect.bottom()
                )

        self._roi_start = None
        self._roi_end   = None
        self._update_display()

    # ── Public controls ───────────────────────────────────────────────────────

    def add_roi_to_list(self):
        """Commit current active ROI to the accumulated list."""
        if self._roi_rect:
            self._roi_list.append(self._roi_rect)
            self._roi_rect = None
            self._update_display()

    def clear_rois(self):
        self._roi_list.clear()
        self._roi_rect  = None
        self._roi_start = None
        self._roi_end   = None
        self._update_display()
        self.roi_cleared.emit()
