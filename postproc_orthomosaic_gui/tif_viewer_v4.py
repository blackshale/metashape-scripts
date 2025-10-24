# file: tif_viewer_v4.py
# Requires: PySide6, numpy, tifffile, pillow, pyshp
# Optional (recommended): imagecodecs, rasterio, opencv-python
import sys, os, traceback
import numpy as np
import tifffile

from PySide6.QtCore import Qt, QSize, QPointF, QRectF, Signal
from PySide6.QtGui import QImage, QPixmap, QPalette, QPainter, QPen, QBrush, QCursor, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog,
    QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox, QScrollArea, QSlider
)

# Optional but recommended for compressed TIFFs (LZW/Deflate/JPEG)
try:
    import imagecodecs  # noqa: F401
except Exception:
    imagecodecs = None

# Optional, best-quality CRS/transform if available
try:
    import rasterio
    from rasterio.transform import Affine
    HAVE_RASTERIO = True
except Exception:
    HAVE_RASTERIO = False

# For masks & polygon rasterization
from PIL import Image as PILImage, ImageDraw

# ESRI Shapefile
import shapefile  # pyshp

LOG_FILE = "tif_viewer_error.log"
NODATA_VALUE = -32767  # requested

def log_exc():
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== ERROR ===\n")
        traceback.print_exc(file=f)

# ------------------------ Preview / NoData helpers ------------------------

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

def _read_gdal_nodata_from_page(page):
    """Read GDAL_NODATA (42113) if present. Returns float or [float,...] or None."""
    tag = page.tags.get(42113) or page.tags.get('GDAL_NODATA')
    if tag is None:
        return None
    val = tag.value
    if isinstance(val, (bytes, bytearray)):
        val = val.decode('utf-8', errors='ignore')
    s = str(val).strip()
    parts = s.split()
    try:
        nums = [float(p) for p in parts]
    except Exception:
        try:
            nums = [float(s)]
        except Exception:
            return None
    return nums if len(nums) > 1 else nums[0]

def read_tiff_preview_raw(path, target_long_edge=3000):
    """
    Return downsized RAW array (H,W,C) + valid_mask (H,W) + min/max across valid.
    Does NOT scale to 8-bit. Honors GDAL_NODATA (transparent later).
    """
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        page0 = tif.pages[0]
        nodata_tag = _read_gdal_nodata_from_page(page0)

        lvl_idx = pick_reduced_level(series, target_long_edge)
        try:
            arr = series.asarray(level=lvl_idx, maxworkers=1)
        except TypeError:
            arr = (series.asarray(maxworkers=1) if lvl_idx == 0
                   else series.levels[lvl_idx].asarray(maxworkers=1))

    # Normalize to (H,W,C)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3 and arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
        arr = np.moveaxis(arr, 0, -1)

    H, W, C = arr.shape

    # NoData mask (True = valid)
    valid = np.ones((H, W), dtype=bool)
    if nodata_tag is not None:
        if isinstance(nodata_tag, (list, tuple)) and C >= 1:
            invalid = np.ones((H, W), dtype=bool)
            for b in range(min(C, len(nodata_tag))):
                nd = nodata_tag[b]
                if np.isnan(nd):
                    invalid &= np.isnan(arr[..., b])
                else:
                    invalid &= (arr[..., b] == nd)
            valid = ~invalid
        else:
            nd = float(nodata_tag)
            if np.isnan(nd):
                valid = ~np.isnan(arr).all(axis=-1)
            else:
                equal_nd = np.all(arr == nd, axis=-1) if C > 1 else (arr[..., 0] == nd)
                valid = ~equal_nd

    # Min/Max on valid only
    if valid.any():
        vmin = float(np.nanmin(arr[valid]))
        vmax = float(np.nanmax(arr[valid]))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    # If very large (no overview), shrink here
    long_edge = max(H, W)
    if long_edge > target_long_edge:
        scale = target_long_edge / float(long_edge)
        new_w = max(1, int(W*scale))
        new_h = max(1, int(H*scale))
        try:
            import cv2
            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            valid = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        except Exception:
            step = max(1, int(1/scale))
            arr = arr[::step, ::step, :].copy()
            valid = valid[::step, ::step].copy()
    return arr, valid, vmin, vmax

def _render_window_to_rgba(arr_raw, valid_mask, vmin, vmax, slider_frac):
    """
    Convert raw (H,W,C) to 8-bit RGBA using window:
      low=vmin, high=vmin + frac*(vmax-vmin). NoData -> alpha=0.
    """
    H, W, C = arr_raw.shape
    low = vmin
    high = vmin + float(slider_frac) * (vmax - vmin)
    if high <= low:
        high = low + 1e-6

    arr = arr_raw.astype(np.float64, copy=False)
    scaled = (np.clip(arr, low, high) - low) / (high - low)
    scaled8 = (scaled * 255.0 + 0.5).astype(np.uint8)

    if C == 1:
        rgb8 = np.repeat(scaled8, 3, axis=-1)
    else:
        rgb8 = scaled8[..., :3]

    alpha = (valid_mask.astype(np.uint8) * 255)[..., None]
    rgba8 = np.concatenate([rgb8, alpha], axis=-1)
    return rgba8

def np_to_qimage(arr8):
    h, w, c = arr8.shape
    if c == 3:
        qimg = QImage(arr8.data, w, h, 3*w, QImage.Format.Format_RGB888)
        return qimg.copy()
    elif c == 4:
        qimg = QImage(arr8.data, w, h, 4*w, QImage.Format.Format_RGBA8888)
        return qimg.copy()
    else:
        raise ValueError("Expected 3 or 4 channels")

# ------------------------ GeoTIFF tag snapshot & CRS helpers ------------------------

def _snapshot_geo_tags(src_page):
    """Copy GeoTIFF/GDAL/Resolution tag VALUES while file is OPEN (avoids warnings)."""
    tg = src_page.tags
    def get_copy(tag_id):
        t = tg.get(tag_id)
        if t is None:
            return None
        v = t.value
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        if isinstance(v, str):
            return str(v)
        try:
            arr = np.array(v)
            return arr.copy()
        except Exception:
            return v
    return {
        33550: get_copy(33550),  # ModelPixelScale
        33922: get_copy(33922),  # ModelTiepoint
        34264: get_copy(34264),  # ModelTransformation
        34735: get_copy(34735),  # GeoKeyDirectory
        34736: get_copy(34736),  # GeoDoubleParams
        34737: get_copy(34737),  # GeoAsciiParams
        42112: get_copy(42112),  # GDAL_METADATA
        282:   get_copy(282),    # XResolution
        283:   get_copy(283),    # YResolution
        296:   get_copy(296),    # ResolutionUnit
    }

def _build_extratags_from_snapshot(snap_dict, nodata_value):
    """Build tifffile.imwrite extratags from snapshot (no file I/O now)."""
    extratags = []
    def add(tag, tifftype, value):
        if value is None:
            return
        if tifftype == 's':
            if isinstance(value, (bytes, bytearray)):
                value = value.decode('utf-8', errors='ignore')
            s = str(value)
            extratags.append((tag, 's', len(s)+1, s, False))
        elif tifftype in ('H','I','d'):
            dt = {'H': np.uint16, 'I': np.uint32, 'd': np.float64}[tifftype]
            arr = np.array(value, dtype=dt).ravel()
            extratags.append((tag, tifftype, int(arr.size), arr, False))
        elif tifftype == 'r':
            extratags.append((tag, 'r', 1, (int(value[0]), int(value[1])), False))
    # GeoTIFF & GDAL
    add(33550,'d',snap_dict.get(33550))
    add(33922,'d',snap_dict.get(33922))
    add(34264,'d',snap_dict.get(34264))
    add(34735,'H',snap_dict.get(34735))
    add(34736,'d',snap_dict.get(34736))
    add(34737,'s',snap_dict.get(34737))
    add(42112,'s',snap_dict.get(42112))
    # Resolution
    xr = snap_dict.get(282); yr = snap_dict.get(283); ru = snap_dict.get(296)
    if isinstance(xr,(tuple,list)) and len(xr)==2: add(282,'r',xr)
    if isinstance(yr,(tuple,list)) and len(yr)==2: add(283,'r',yr)
    if ru is not None: add(296,'H',ru)
    # GDAL_NODATA for the clip
    nds = str(nodata_value)
    extratags.append((42113,'s',len(nds)+1, nds, False))
    return extratags

def _get_affine_and_wkt_from_raster(path):
    """Prefer rasterio for CRS/transform."""
    if not HAVE_RASTERIO:
        return None, None
    try:
        with rasterio.open(path) as src:
            return src.transform, (src.crs.to_wkt() if src.crs else None)
    except Exception:
        return None, None

def _get_affine_from_tifftags(src_page):
    """Fallback affine from GeoTIFF tags."""
    if HAVE_RASTERIO:
        Aff = Affine
    else:
        # stub-like tuple fallback
        Aff = lambda a,b,c,d,e,f: (a,b,c,d,e,f)

    tr = src_page.tags.get(34264)
    if tr is not None:
        vals = np.array(tr.value, dtype=float).ravel()
        if vals.size == 16:
            m = vals.reshape(4,4)
            a,b,c = m[0,0], m[0,1], m[0,3]
            d,e,f = m[1,0], m[1,1], m[1,3]
            return Aff(a,b,c,d,e,f)

    scale_tag = src_page.tags.get(33550)
    tie_tag   = src_page.tags.get(33922)
    if scale_tag is None or tie_tag is None:
        return None
    sx, sy = float(scale_tag.value[0]), float(scale_tag.value[1])
    tp = tie_tag.value
    i,j,_k, X,Y,_Z = map(float, tp[:6])
    a,b,c = sx, 0.0, X - i*sx
    d,e,f = 0.0, -sy, Y - j*(-sy)  # Y + j*sy
    return Aff(a,b,c,d,e,f)

def _pixel_to_map_xy(affine_like, x_col, y_row):
    """Apply affine (col,row)->(X,Y)."""
    if HAVE_RASTERIO and isinstance(affine_like, Affine):
        X, Y = affine_like * (x_col, y_row)
        return float(X), float(Y)
    else:
        a,b,c,d,e,f = affine_like
        X = a*x_col + b*y_row + c
        Y = d*x_col + e*y_row + f
        return float(X), float(Y)

def _extract_wkt_from_tifftags(src_page):
    """Best-effort WKT finding."""
    t = src_page.tags.get(34737)
    if t and t.value:
        val = t.value
        if isinstance(val,(bytes,bytearray)): val = val.decode('utf-8','ignore')
        s = str(val).strip()
        if any(k in s for k in ("GEOGCS","PROJCS","PROJCRS","GEOGCRS")):
            return s
    t = src_page.tags.get(42112)
    if t and t.value:
        val = t.value
        if isinstance(val,(bytes,bytearray)): val = val.decode('utf-8','ignore')
        s = str(val)
        start, end = s.find("<SRS>"), s.find("</SRS>")
        if 0 <= start < end:
            w = s[start+5:end].strip()
            if w: return w
    return None

def _write_prj(shp_base, wkt):
    if not wkt:
        return
    with open(shp_base + ".prj", "w", encoding="utf-8") as f:
        f.write(wkt)

# ------------------------ Interactive image label (polygon) ------------------------

class ImageLabel(QLabel):
    """QLabel overlay for polygon selection in PREVIEW image coords."""
    SNAP_PIX = 12
    polygonFinished = Signal(list)  # list[(x,y)] in preview pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundRole(QPalette.Base)
        self.setAutoFillBackground(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._scaled_pixmap = None
        self._base_pixmap = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self.selecting = False
        self.points_img = []

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
        if snap_close and len(self.points_img) >= 2:
            first = self.points_img[0]
            self.points_img[-1] = QPointF(first.x(), first.y())
        self.selecting = False
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.clearFocus()
        self.update()
        self.polygonFinished.emit([(p.x(), p.y()) for p in self.points_img])

    # --- coord helpers ---
    def img_to_widget(self, p_img: QPointF) -> QPointF:
        return QPointF(self._offset.x() + p_img.x()*self._scale,
                       self._offset.y() + p_img.y()*self._scale)

    def widget_to_img(self, p_w: QPointF) -> QPointF:
        return QPointF((p_w.x()-self._offset.x())/self._scale,
                       (p_w.y()-self._offset.y())/self._scale)

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
            x0, y0 = self._offset.x(), self._offset.y()
            sw, sh = self._scaled_pixmap.width(), self._scaled_pixmap.height()
            if not (x0 <= xw <= x0+sw and y0 <= yw <= y0+sh):
                return
            p_w = QPointF(xw, yw)
            if len(self.points_img) >= 3 and self._near_first_vertex(p_w):
                self.points_img.append(QPointF(self.points_img[0].x(), self.points_img[0].y()))
                self.finish_selection(snap_close=True)
                return
            self.points_img.append(self.widget_to_img(p_w))
            self.update()
        elif e.button() == Qt.MouseButton.RightButton:
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
        if self.points_img:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pts_widget = [self.img_to_widget(p) for p in self.points_img]
            painter.setPen(QPen(Qt.GlobalColor.cyan, 2))
            for i in range(1, len(pts_widget)):
                painter.drawLine(pts_widget[i-1], pts_widget[i])
            painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
            for pt in pts_widget:
                painter.drawEllipse(QRectF(pt.x()-3, pt.y()-3, 6, 6))
            if len(pts_widget) >= 3:
                painter.setPen(QPen(Qt.GlobalColor.cyan, 1))
                painter.setBrush(QBrush(Qt.GlobalColor.cyan, Qt.BrushStyle.Dense4Pattern))
                painter.drawPolygon(QPolygonF(pts_widget))

# ------------------------ Main Window ------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF Viewer (Qt) + Crop + Slider")
        self.resize(1280, 860)

        # Buttons
        btn_open = QPushButton("Open")
        btn_redraw = QPushButton("Redraw")
        btn_crop = QPushButton("Crop")
        btn_exit = QPushButton("Exit")
        btn_open.clicked.connect(self.on_open)
        btn_redraw.clicked.connect(self.on_redraw)
        btn_crop.clicked.connect(self.on_crop)
        btn_exit.clicked.connect(self.close)

        row_buttons = QHBoxLayout()
        row_buttons.addWidget(btn_open)
        row_buttons.addWidget(btn_redraw)
        row_buttons.addWidget(btn_crop)
        row_buttons.addWidget(btn_exit)
        row_buttons.addStretch(1)

        # Slider row (min .. max)
        self.lbl_min = QLabel("min")
        self.lbl_min.setMinimumWidth(140)
        self.lbl_min.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)   # 0..100%
        self.slider.setValue(1000)      # default: full range
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.setToolTip("Adjust upper display bound between data min and max")

        self.lbl_max = QLabel("max")
        self.lbl_max.setMinimumWidth(140)
        self.lbl_max.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row_slider = QHBoxLayout()
        row_slider.addWidget(self.lbl_min)
        row_slider.addWidget(self.slider, stretch=1)
        row_slider.addWidget(self.lbl_max)

        # Image area
        self.image_label = ImageLabel(self)
        self.image_label.polygonFinished.connect(self.on_polygon_finished)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.image_label)

        # Central layout
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(row_buttons)
        lay.addLayout(row_slider)
        lay.addWidget(self.scroll, stretch=1)
        self.setCentralWidget(central)

        # State
        self._pixmap = None
        self._scaled = None
        self._scale = 1.0
        self._offset_xy = (0.0, 0.0)
        self.preview_size = None
        self.current_path = None
        self.full_shape = None
        self.full_dtype = None
        self.series_index = 0

        # Preview raw + mask + range
        self.preview_raw = None
        self.preview_valid = None
        self.preview_min = None
        self.preview_max = None

    # ---------- UI plumbing ----------
    def _set_scaled_pixmap(self):
        if self._pixmap is None:
            self.image_label.setPixmap(QPixmap())
            return
        avail = self.scroll.viewport().size()
        scaled = self._pixmap.scaled(
            QSize(avail.width(), avail.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._scaled = scaled
        self.image_label.setPixmap(scaled)
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

            # Load RAW preview + mask + min/max
            arr_raw, valid, vmin, vmax = read_tiff_preview_raw(path, target_long_edge=3000)
            self.preview_raw = arr_raw
            self.preview_valid = valid
            self.preview_min = vmin
            self.preview_max = vmax
            self.lbl_min.setText(f"min: {vmin:.6g}")
            self.lbl_max.setText(f"max: {vmax:.6g}")

            frac = self.slider.value() / self.slider.maximum()
            rgba8 = _render_window_to_rgba(arr_raw, valid, vmin, vmax, frac)
            qimg = np_to_qimage(rgba8)
            self._pixmap = QPixmap.fromImage(qimg)
            self.preview_size = (rgba8.shape[1], rgba8.shape[0])  # (w,h)

            # store full-res shape & dtype
            with tifffile.TiffFile(path) as tif:
                ser = tif.series[self.series_index]
                shp = ser.shape
                if len(shp) == 2:
                    H, W = shp; C = 1
                elif len(shp) == 3:
                    if shp[-1] in (1,3,4):
                        H, W, C = shp
                    else:
                        C, H, W = shp
                else:
                    raise RuntimeError(f"Unsupported TIFF shape: {shp}")
                self.full_shape = (H, W, C)
                self.full_dtype = ser.dtype

            self.current_path = path
            self._set_scaled_pixmap()
            self.setWindowTitle(f"TIFF Viewer (Qt) + Crop + Slider — {os.path.basename(path)}")
        except Exception:
            log_exc()
            QMessageBox.critical(self, "Error",
                                 f"Failed to open image.\nSee log: {os.path.abspath(LOG_FILE)}")

    def on_redraw(self):
        if self._pixmap is None:
            QMessageBox.information(self, "Redraw", "No image loaded. Click Open first.")
            return
        try:
            self._on_slider_changed(self.slider.value())
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

    # ---------- Slider callback ----------
    def _on_slider_changed(self, value):
        if self.preview_raw is None or self.preview_min is None:
            return
        frac = value / self.slider.maximum()
        rgba8 = _render_window_to_rgba(self.preview_raw, self.preview_valid,
                                       self.preview_min, self.preview_max, frac)
        qimg = np_to_qimage(rgba8)
        self._pixmap = QPixmap.fromImage(qimg)
        self.preview_size = (rgba8.shape[1], rgba8.shape[0])
        self._set_scaled_pixmap()

    # ---------- Polygon finished ----------
    def on_polygon_finished(self, pts_preview):
        if len(pts_preview) < 3:
            QMessageBox.warning(self, "Crop", "Need at least 3 points.")
            self.image_label.cancel_selection()
            return

        (prev_w, prev_h) = self.preview_size
        (full_h, full_w, _c) = self.full_shape
        sx = full_w / float(prev_w)
        sy = full_h / float(prev_h)
        poly_full = [(x*sx, y*sy) for (x, y) in pts_preview]

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
        if self.current_path is None or self.full_shape is None:
            raise RuntimeError("No image loaded.")
        H, W, C = self.full_shape

        # Build full-res mask from polygon
        ring = [(float(x), float(y)) for (x, y) in poly_full_xy]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        mask_img = PILImage.new('L', (W, H), 0)
        ImageDraw.Draw(mask_img, 'L').polygon(ring, outline=1, fill=1)
        mask = np.array(mask_img, dtype=np.uint8).astype(bool)

        # Read full image + SNAPSHOT tags while open
        with tifffile.TiffFile(self.current_path) as tif:
            ser = tif.series[0]
            src_page = tif.pages[0]
            data = ser.asarray(maxworkers=1)
            geo_snapshot = _snapshot_geo_tags(src_page)

        # Normalize to (H,W,C)
        if data.ndim == 2:
            data = data[:, :, None]
        elif data.ndim == 3 and data.shape[0] in (1,3,4) and data.shape[-1] not in (1,3,4):
            data = np.moveaxis(data, 0, -1)

        # Apply NODATA outside polygon
        data_i16 = data.astype(np.int16, copy=False)
        m = mask[:, :, None]
        data_i16[~m] = NODATA_VALUE

        extratags = _build_extratags_from_snapshot(geo_snapshot, NODATA_VALUE)
        comp = 'deflate' if (imagecodecs is not None) else None
        big = (H * W * data_i16.shape[-1] * 2) > 2_000_000_000
        photometric = 'minisblack' if data_i16.shape[-1] == 1 else 'rgb'

        tifffile.imwrite(
            out_tif,
            data_i16 if data_i16.shape[-1] > 1 else data_i16[:, :, 0],
            bigtiff=big,
            compression=comp,
            dtype=np.int16,
            photometric=photometric,
            extratags=extratags,
        )

        # Shapefile in MAP COORDINATES (+ .prj)
        # Affine & WKT from rasterio or GeoTIFF tags
        aff, wkt = _get_affine_and_wkt_from_raster(self.current_path)
        if aff is None:
            with tifffile.TiffFile(self.current_path) as tif:
                src_page = tif.pages[0]
                aff = _get_affine_from_tifftags(src_page)
                if wkt is None:
                    wkt = _extract_wkt_from_tifftags(src_page)

        if aff is not None:
            ring_map = [_pixel_to_map_xy(aff, x, y) for (x, y) in ring]
        else:
            ring_map = ring  # fallback: pixel coords

        shp_base, ext = os.path.splitext(out_shp)
        if ext.lower() != '.shp':
            shp_base = out_shp
        w = shapefile.Writer(shp_base, shapeType=shapefile.POLYGON)
        w.field('id', 'N')
        w.poly([ring_map])
        w.record(1)
        w.close()
        _write_prj(shp_base, wkt)

# ------------------------ Entrypoint ------------------------

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
