# file: tif_viewer_qt_crop.py
import sys, os, traceback
import numpy as np
import tifffile

from PySide6.QtCore import Qt, QSize, QPointF, QRectF, Signal
from PySide6.QtGui import QImage, QPixmap, QPalette, QPainter, QPen, QBrush, QAction, QCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog,
    QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox, QScrollArea
)
from PySide6.QtGui import QPolygonF

# Optional but recommended for compressed TIFFs (LZW/Deflate/JPEG)
try:
    import imagecodecs  # noqa: F401
except Exception:
    imagecodecs = None

# For building the mask efficiently on full-res via pillow
from PIL import Image as PILImage, ImageDraw

# For shapefile output (ESRI Shapefile)
import shapefile  # pyshp

LOG_FILE = "tif_viewer_error.log"
NODATA_VALUE = -32767  # as requested

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
        return qimg.copy()  # take ownership
    elif c == 4:
        qimg = QImage(arr8.data, w, h, 4*w, QImage.Format.Format_RGBA8888)
        return qimg.copy()
    else:
        raise ValueError("Expected 3 or 4 channels")

class ImageLabel(QLabel):
    """
    QLabel that draws a polygon overlay and translates mouse clicks
    into image-space coordinates (of the PREVIEW image).
    Emits polygonFinished(list_of_xy) when the user completes the polygon.
    """
    SNAP_PIX = 12  # snap radius in widget pixels
    polygonFinished = Signal(list)  # list of (x, y) tuples in PREVIEW image coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundRole(QPalette.Base)
        self.setAutoFillBackground(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # receive Enter/Esc

        self._scaled_pixmap = None
        self._base_pixmap = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)  # top-left of the drawn pixmap
        self.selecting = False
        self.points_img = []  # list[QPointF] in PREVIEW image coords

    def set_pixmaps(self, base_pixmap: QPixmap, scaled_pixmap: QPixmap, scale: float, offset_xy):
        self._base_pixmap = base_pixmap
        self._scaled_pixmap = scaled_pixmap
        self._scale = scale
        self._offset = QPointF(*offset_xy)

    def start_selection(self):
        self.points_img = []
        self.selecting = True
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setFocus()
        self.update()

    def cancel_selection(self):
        self.selecting = False
        self.points_img = []
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.clearFocus()
        self.update()

    def finish_selection(self, snap_close=False):
        """Finish polygon; optionally snap the last point to the first."""
        if snap_close and len(self.points_img) >= 2:
            first = self.points_img[0]
            self.points_img[-1] = QPointF(first.x(), first.y())

        self.selecting = False
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.clearFocus()
        self.update()

        pts = [(p.x(), p.y()) for p in self.points_img]
        self.polygonFinished.emit(pts)  # <-- emit instead of calling parent
        return pts

    # --- coord helpers ---
    def img_to_widget(self, p_img: QPointF) -> QPointF:
        xw = self._offset.x() + p_img.x() * self._scale
        yw = self._offset.y() + p_img.y() * self._scale
        return QPointF(xw, yw)

    def widget_to_img(self, p_w: QPointF) -> QPointF:
        xi = (p_w.x() - self._offset.x()) / self._scale
        yi = (p_w.y() - self._offset.y()) / self._scale
        return QPointF(xi, yi)

    def _near_first_vertex(self, p_w: QPointF) -> bool:
        if not self.points_img:
            return False
        first_w = self.img_to_widget(self.points_img[0])
        dx = p_w.x() - first_w.x()
        dy = p_w.y() - first_w.y()
        return (dx*dx + dy*dy) ** 0.5 <= self.SNAP_PIX

    # --- events ---
    def mousePressEvent(self, e):
        if not self.selecting or self._scaled_pixmap is None:
            return super().mousePressEvent(e)

        if e.button() == Qt.MouseButton.LeftButton:
            pos = e.position() if hasattr(e, "position") else e.posF()
            xw, yw = pos.x(), pos.y()

            # inside drawn pixmap?
            x0, y0 = self._offset.x(), self._offset.y()
            sw = self._scaled_pixmap.width()
            sh = self._scaled_pixmap.height()
            if not (x0 <= xw <= x0 + sw and y0 <= yw <= y0 + sh):
                return

            p_w = QPointF(xw, yw)

            # Snap-close if near first and we already have a polygon
            if len(self.points_img) >= 3 and self._near_first_vertex(p_w):
                self.points_img.append(QPointF(self.points_img[0].x(), self.points_img[0].y()))
                self.finish_selection(snap_close=True)
                return

            # Otherwise append a new vertex (in image coords)
            p_img = self.widget_to_img(p_w)
            self.points_img.append(p_img)
            self.update()

        elif e.button() == Qt.MouseButton.RightButton:
            # Right-click finishes; snap if near first
            pos = e.position() if hasattr(e, "position") else e.posF()
            p_w = QPointF(pos.x(), pos.y())
            snap = len(self.points_img) >= 3 and self._near_first_vertex(p_w)
            self.finish_selection(snap_close=snap)

    def keyPressEvent(self, e):
        if self.selecting and e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if len(self.points_img) >= 3:
                self.finish_selection(snap_close=False)
                return
        elif self.selecting and e.key() == Qt.Key.Key_Escape:
            self.cancel_selection()
            return
        super().keyPressEvent(e)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._scaled_pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self.points_img:
            pts_widget = [self.img_to_widget(p) for p in self.points_img]

            painter.setPen(QPen(Qt.GlobalColor.cyan, 2))
            for i in range(1, len(pts_widget)):
                painter.drawLine(pts_widget[i-1], pts_widget[i])

            painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
            for pt in pts_widget:
                painter.drawEllipse(QRectF(pt.x()-3, pt.y()-3, 6, 6))

            if len(pts_widget) >= 3:
                from PySide6.QtGui import QPolygonF
                painter.setPen(QPen(Qt.GlobalColor.cyan, 1))
                painter.setBrush(QBrush(Qt.GlobalColor.cyan, Qt.BrushStyle.Dense4Pattern))
                painter.drawPolygon(QPolygonF(pts_widget))  # correct usage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF Viewer (Qt) + Crop")
        self.resize(1200, 800)

        # Top buttons
        btn_open = QPushButton("Open")
        btn_redraw = QPushButton("Redraw")
        btn_crop = QPushButton("Crop")
        btn_exit = QPushButton("Exit")
        btn_open.clicked.connect(self.on_open)
        btn_redraw.clicked.connect(self.on_redraw)
        btn_crop.clicked.connect(self.on_crop)
        btn_exit.clicked.connect(self.close)

        top = QHBoxLayout()
        top.addWidget(btn_open); top.addWidget(btn_redraw); top.addWidget(btn_crop); top.addWidget(btn_exit); top.addStretch(1)

        # Image area (scrollable)
        self.image_label = ImageLabel(self)  # custom overlay label
        self.image_label.polygonFinished.connect(self.on_polygon_finished)  # <-- required

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.image_label)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(top); lay.addWidget(self.scroll, stretch=1)
        self.setCentralWidget(central)

        self._pixmap = None           # original pixmap (from preview arr8)
        self._scaled = None           # scaled pixmap shown
        self._scale = 1.0
        self._offset_xy = (0.0, 0.0)
        self.preview_size = None      # (w, h) of preview image
        self.current_path = None      # full-res file path
        self.full_shape = None        # (H, W, C) or (H, W)
        self.full_dtype = None        # dtype of full-res
        self.series_index = 0         # assume first series

    # ---------- UI plumbing ----------
    def _set_scaled_pixmap(self):
        if self._pixmap is None:
            self.image_label.setPixmap(QPixmap())  # clear
            return
        avail = self.scroll.viewport().size()
        scaled = self._pixmap.scaled(
            QSize(avail.width(), avail.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._scaled = scaled
        self.image_label.setPixmap(scaled)

        # compute scale and offset for mapping clicks
        base_w, base_h = self._pixmap.width(), self._pixmap.height()
        sc = scaled.width() / base_w if base_w else 1.0
        x0 = (self.image_label.width() - scaled.width()) / 2
        y0 = (self.image_label.height() - scaled.height()) / 2
        self._scale = sc
        self._offset_xy = (x0, y0)
        self.image_label.set_pixmaps(self._pixmap, self._scaled, self._scale, self._offset_xy)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._pixmap is not None:
            self._set_scaled_pixmap()

    # ---------- Buttons ----------
    def on_open(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open TIFF Image", "", "TIFF Images (*.tif *.tiff);;All Files (*)"
            )
            if not path:
                return
            arr8 = read_tiff_preview(path, target_long_edge=3000)
            qimg = np_to_qimage(arr8)
            self._pixmap = QPixmap.fromImage(qimg)
            self.preview_size = (arr8.shape[1], arr8.shape[0])  # (w,h)

            # also store full-res shape & dtype for later masking
            with tifffile.TiffFile(path) as tif:
                ser = tif.series[self.series_index]
                shp = ser.shape  # could be (H,W), (H,W,C), (C,H,W)
                # normalize to (H,W, C?)
                if len(shp) == 2:
                    H, W = shp
                    C = 1
                elif len(shp) == 3:
                    # guess channel-last vs channel-first
                    if shp[-1] in (1, 3, 4):
                        H, W, C = shp
                    else:
                        C, H, W = shp
                else:
                    raise RuntimeError(f"Unsupported TIFF shape: {shp}")
                self.full_shape = (H, W, C)
                self.full_dtype = ser.dtype

            self.current_path = path
            self._set_scaled_pixmap()
            self.setWindowTitle(f"TIFF Viewer (Qt) + Crop — {os.path.basename(path)}")
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

    def on_crop(self):
        if self._pixmap is None or self.current_path is None:
            QMessageBox.information(self, "Crop", "Open a TIFF first.")
            return
        self.image_label.start_selection()
        self.image_label.setFocus()
        QMessageBox.information(
            self, "Polygon selection",
            "Left-click to add vertices.\nRight-click (or Enter) to finish.\nEsc to cancel.\n"
            "Tip: click near the first vertex to snap-close."
        )

    def on_polygon_finished(self, pts_preview):
        # pts_preview: list[(x,y)] in PREVIEW pixel coords
        if len(pts_preview) < 3:
            QMessageBox.warning(self, "Crop", "Need at least 3 points.")
            self.image_label.cancel_selection()
            return

        # map preview->full
        (prev_w, prev_h) = self.preview_size
        (full_h, full_w, _c) = self.full_shape
        sx = full_w / float(prev_w)
        sy = full_h / float(prev_h)
        poly_full = [(x * sx, y * sy) for (x, y) in pts_preview]

        # default filenames
        base, _ = os.path.splitext(self.current_path)
        out_tif, _ = QFileDialog.getSaveFileName(
            self, "Save clipped TIFF as", f"{base}_clipped.tif",
            "TIFF Images (*.tif *.tiff)"
        )
        if not out_tif:
            self.image_label.cancel_selection()
            return

        out_shp, _ = QFileDialog.getSaveFileName(
            self, "Save polygon shapefile as", f"{base}_poly.shp",
            "ESRI Shapefile (*.shp)"
        )
        if not out_shp:
            self.image_label.cancel_selection()
            return

        try:
            self._apply_polygon_and_save(poly_full, out_tif, out_shp)
            QMessageBox.information(self, "Done", f"Saved:\n{out_tif}\n{out_shp}\n(.dbf/.shx written alongside)")
        except Exception as e:
            log_exc()
            QMessageBox.critical(self, "Error", f"Cropping failed:\n{e}")
        finally:
            self.image_label.cancel_selection()

    # ---------- Core masking & saving ----------
    def _apply_polygon_and_save(self, poly_full_xy, out_tif, out_shp):
        """
        poly_full_xy: list of (x,y) in FULL image pixel coordinates.
        Writes TIFF where pixels OUTSIDE polygon are set to NODATA_VALUE (-32767),
        and writes the polygon to a .shp (pixel coordinates).
        """
        if self.current_path is None or self.full_shape is None:
            raise RuntimeError("No image loaded.")

        H, W, C = self.full_shape

        # Full-res mask via Pillow (fast, low-RAM)
        from PIL import Image as PILImage, ImageDraw
        mask_img = PILImage.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask_img, 'L')

        # ensure closed ring for fill
        ring = [(float(x), float(y)) for (x, y) in poly_full_xy]
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        draw.polygon(ring, outline=1, fill=1)
        mask = np.array(mask_img, dtype=np.uint8).astype(bool)  # True inside

        # Read full image (single-threaded)
        with tifffile.TiffFile(self.current_path) as tif:
            ser = tif.series[0]
            data = ser.asarray(maxworkers=1)

        # Normalize to (H, W, C)
        if data.ndim == 2:
            data = data[:, :, None]
        elif data.ndim == 3 and data.shape[0] in (1, 3, 4) and data.shape[-1] not in (1, 3, 4):
            data = np.moveaxis(data, 0, -1)

        # Convert to int16 so we can store -32767
        data_i16 = data.astype(np.int16, copy=False)

        # Apply mask (outside -> NODATA)
        m = mask[:, :, None]
        data_i16[~m] = NODATA_VALUE

        # Choose compression safely
        use_compression = (imagecodecs is not None)
        comp = 'deflate' if use_compression else None  # None = no compression

        # Add GDAL_NODATA tag so GIS reads NoData properly.
        nodata_str = str(NODATA_VALUE)
        extratags = [
            # (tag, type, count, value, write_once)
            (42113, 's', len(nodata_str) + 1, nodata_str, False),  # GDAL_NODATA
        ]

        # BigTIFF if > ~2GB
        big = (H * W * data_i16.shape[-1] * 2) > 2_000_000_000

        # Write TIFF
        tifffile.imwrite(
            out_tif,
            data_i16 if C > 1 else data_i16[:, :, 0],
            bigtiff=big,
            compression=comp,
            dtype=np.int16,
            photometric='minisblack' if C == 1 else 'rgb',
            extratags=extratags,
        )

        # --- Write ESRI Shapefile (pixel coords) ---
        import shapefile
        shp_base, ext = os.path.splitext(out_shp)
        if ext.lower() != '.shp':
            # user passed no extension; create .shp/.shx/.dbf with this base
            shp_base = out_shp

        w = shapefile.Writer(shp_base, shapeType=shapefile.POLYGON)
        w.field('id', 'N')
        # shapefile expects a list of parts; each part is a closed ring
        w.poly([ring])
        w.record(1)
        w.close()

        # Optional: write a minimal .prj (none/unknown). Skip for now.

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
