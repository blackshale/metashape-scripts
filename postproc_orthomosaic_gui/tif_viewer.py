# file: tif_viewer_qt_fixed.py
import sys, os, traceback
import numpy as np
import tifffile

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog,
    QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox, QScrollArea
)

LOG_FILE = "tif_viewer_error.log"

def log_exc():
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== ERROR ===\n")
        traceback.print_exc(file=f)

def percentile_to_uint8(arr, p_low=1, p_high=99):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (3,4) and arr.shape[-1] not in (3,4):
        arr = np.moveaxis(arr, 0, -1)
    arrf = arr.astype(np.float64, copy=False)
    low = np.nanpercentile(arrf, p_low)
    high = np.nanpercentile(arrf, p_high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(arrf)); high = float(np.nanmax(arrf))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            high = low + 1.0
    arrf = (np.clip(arrf, low, high) - low) / (high - low)
    arr8 = (arrf * 255.0 + 0.5).astype(np.uint8)
    if arr8.ndim == 2:
        arr8 = np.stack([arr8, arr8, arr8], axis=-1)
    elif arr8.ndim == 3:
        if arr8.shape[-1] == 1:
            arr8 = np.repeat(arr8, 3, axis=-1)
        elif arr8.shape[-1] > 4:
            arr8 = arr8[..., :3]
    return arr8

def pick_reduced_level(series, target_long_edge=3000):
    try:
        levels = series.levels
        if not levels:
            return 0
        sizes = []
        for i, lvl in enumerate(levels):
            h, w = lvl.shape[:2]
            sizes.append((i, max(h, w)))
        under = [i for i, L in sizes if L <= target_long_edge]
        return (max(under, key=lambda i: sizes[i][1]) if under
                else min(range(len(levels)), key=lambda i: sizes[i][1]))
    except Exception:
        return 0

def read_tiff_preview(path, target_long_edge=3000):
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        lvl_idx = pick_reduced_level(series, target_long_edge)
        try:
            arr = series.asarray(level=lvl_idx, maxworkers=1)
        except TypeError:
            arr = (series.asarray(maxworkers=1) if lvl_idx == 0
                   else series.levels[lvl_idx].asarray(maxworkers=1))
    arr8 = percentile_to_uint8(arr)
    h, w = arr8.shape[:2]
    long_edge = max(h, w)
    if long_edge > target_long_edge:
        scale = target_long_edge / long_edge
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        try:
            import cv2
            arr8 = cv2.resize(arr8, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            step = max(1, int(1/scale))
            arr8 = arr8[::step, ::step].copy()
    return arr8

def np_to_qimage(arr8):
    h, w, c = arr8.shape
    if c == 3:
        qimg = QImage(arr8.data, w, h, 3*w, QImage.Format.Format_RGB888)
        return qimg.copy()  # ensure data ownership
    elif c == 4:
        qimg = QImage(arr8.data, w, h, 4*w, QImage.Format.Format_RGBA8888)
        return qimg.copy()
    else:
        raise ValueError("Expected 3 or 4 channels")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF Viewer (Qt)")
        self.resize(1200, 800)

        # Top buttons
        btn_open = QPushButton("Open")
        btn_redraw = QPushButton("Redraw")
        btn_exit = QPushButton("Exit")
        btn_open.clicked.connect(self.on_open)
        btn_redraw.clicked.connect(self.on_redraw)
        btn_exit.clicked.connect(self.close)

        top = QHBoxLayout()
        top.addWidget(btn_open); top.addWidget(btn_redraw); top.addWidget(btn_exit); top.addStretch(1)

        # Image area (scrollable)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setBackgroundRole(QPalette.Base)  # <-- fixed: valid ColorRole
        self.image_label.setAutoFillBackground(True)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.image_label)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(top); lay.addWidget(self.scroll, stretch=1)
        self.setCentralWidget(central)

        self._pixmap = None

    def on_open(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open TIFF Image", "", "TIFF Images (*.tif *.tiff);;All Files (*)"
            )
            if not path:
                return
            arr8 = read_tiff_preview(path, target_long_edge=3000)
            qimg = np_to_qimage(arr8)     # owns its buffer
            self._pixmap = QPixmap.fromImage(qimg)
            self._set_scaled_pixmap()
            self.setWindowTitle(f"TIFF Viewer (Qt) — {os.path.basename(path)}")
        except Exception:
            log_exc()
            QMessageBox.critical(self, "Error",
                                 f"Failed to open image.\nSee log: {os.path.abspath(LOG_FILE)}")

    def on_redraw(self):
        if self._pixmap is None:
            QMessageBox.information(self, "Redraw", "No image loaded. Click Open first.")
            return
        try:
            self._set_scaled_pixmap()
        except Exception:
            log_exc()
            QMessageBox.critical(self, "Error", "Redraw failed.")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._pixmap is not None:
            self._set_scaled_pixmap()

    def _set_scaled_pixmap(self):
        avail = self.scroll.viewport().size()
        scaled = self._pixmap.scaled(
            QSize(avail.width(), avail.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

def main():
    try:
        app = QApplication(sys.argv)
        w = MainWindow()
        w.show()
        sys.exit(app.exec())
    except Exception:
        log_exc()
        print("Fatal error. See log:", os.path.abspath(LOG_FILE))

if __name__ == "__main__":
    main()
