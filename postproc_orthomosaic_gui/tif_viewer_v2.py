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

# Optional, but best-quality CRS/transform if available
try:
    import rasterio
    from rasterio.transform import Affine
    HAVE_RASTERIO = True
except Exception:
    HAVE_RASTERIO = False
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

def percentile_to_uint8(arr, p_low=1, p_high=99, valid_mask=None):
    """
    Scale numeric array to 8-bit for display using percentiles.
    If valid_mask is provided (H,W), only valid pixels are used for statistics,
    and invalid pixels are left as-is (we'll make them transparent later).
    Returns uint8 array in channel-last shape with 3 or 4 channels.
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    # Move channel-first -> channel-last
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    # Build a mask of valid pixels (H,W)
    if arr.ndim == 2:
        H, W = arr.shape
        C = 1
        arr_ = arr[..., None]
    elif arr.ndim == 3:
        H, W, C = arr.shape
        arr_ = arr
    else:
        raise ValueError(f"Unsupported shape: {arr.shape}")

    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=bool)

    # Gather values for stats from valid pixels only
    vals = arr_[valid_mask]  # shape (N*C,)
    if vals.size == 0:
        # fall back to entire image
        vals = arr_.ravel()

    vals = vals.astype(np.float64, copy=False)
    low = np.nanpercentile(vals, p_low)
    high = np.nanpercentile(vals, p_high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(vals))
        high = float(np.nanmax(vals))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            high = low + 1.0

    # Scale whole image (we'll mask invalids later)
    arrf = arr_.astype(np.float64, copy=False)
    arrf = (np.clip(arrf, low, high) - low) / (high - low)
    arr8 = (arrf * 255.0 + 0.5).astype(np.uint8)

    # Ensure 3 channels
    if C == 1:
        arr8 = np.repeat(arr8, 3, axis=-1)
        C = 3
    elif C > 4:
        arr8 = arr8[..., :3]
        C = 3

    return arr8  # RGB (no alpha yet)


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
    """
    Try to read GDAL_NODATA (tag 42113) from a TiffPage.
    Returns: None if not present, or a list[float] (per-band) or single float.
    """
    tag = page.tags.get(42113) or page.tags.get('GDAL_NODATA')
    if tag is None:
        return None
    val = tag.value
    if isinstance(val, (bytes, bytearray)):
        val = val.decode('utf-8', errors='ignore')
    s = str(val).strip()
    # GDAL stores a single string; sometimes per-band values are separated by spaces.
    parts = s.split()
    try:
        nums = [float(p) for p in parts]
    except Exception:
        # could be something like "nan"
        try:
            nums = [float(s)]
        except Exception:
            return None
    if len(nums) == 1:
        return nums[0]
    return nums  # per-band

def read_tiff_preview(path, target_long_edge=3000):
    """
    Build a display-ready preview as uint8 RGBA, honoring GDAL_NODATA (tag 42113):
    - NoData pixels are transparent (alpha=0).
    - Percentile stretch ignores NoData values.
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

    # Normalize shape to (H,W,C)
    if arr.ndim == 2:
        H, W = arr.shape
        C = 1
        arr = arr[:, :, None]
    elif arr.ndim == 3:
        if arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
            arr = np.moveaxis(arr, 0, -1)  # (C,H,W) -> (H,W,C)
        H, W, C = arr.shape
    else:
        raise ValueError(f"Unsupported TIFF array shape: {arr.shape}")

    # Build NoData mask (True=valid, False=NoData)
    valid_mask = np.ones((H, W), dtype=bool)
    if nodata_tag is not None:
        if isinstance(nodata_tag, (list, tuple)) and C >= 1:
            # per-band: mark pixel invalid if *all* bands equal their nodata
            invalid = np.ones((H, W), dtype=bool)
            for b in range(min(C, len(nodata_tag))):
                nd = nodata_tag[b]
                if np.isnan(nd):
                    invalid &= np.isnan(arr[..., b])
                else:
                    invalid &= (arr[..., b] == nd)
            valid_mask = ~invalid
        else:
            nd = float(nodata_tag)
            if np.isnan(nd):
                valid_mask = ~np.isnan(arr).all(axis=-1)
            else:
                # if multi-band, consider NoData when all bands equal nd (common RGB convention)
                equal_nd = np.all(arr == nd, axis=-1) if C > 1 else (arr[..., 0] == nd)
                valid_mask = ~equal_nd

    # Downsample to target size if still too large (fast path)
    long_edge = max(H, W)
    if long_edge > target_long_edge:
        scale = target_long_edge / float(long_edge)
        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))
        try:
            import cv2
            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            valid_mask = cv2.resize(valid_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            H, W = new_h, new_w
        except Exception:
            step = max(1, int(1/scale))
            arr = arr[::step, ::step, ...].copy()
            valid_mask = valid_mask[::step, ::step].copy()
            H, W = arr.shape[:2]

    # Make an 8-bit RGB preview ignoring NoData in the stats
    rgb8 = percentile_to_uint8(arr, p_low=1, p_high=99, valid_mask=valid_mask)

    # Compose RGBA with alpha=0 for NoData
    alpha = (valid_mask.astype(np.uint8) * 255)[..., None]
    rgba8 = np.concatenate([rgb8, alpha], axis=-1)  # (H,W,4)
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
        poly_full_xy in FULL image pixel coords.
        - Writes clipped GeoTIFF with preserved geotags & GDAL_NODATA.
        - Writes ESRI Shapefile in MAP COORDINATES matching the GeoTIFF, plus .prj.
        """
        if self.current_path is None or self.full_shape is None:
            raise RuntimeError("No image loaded.")

        H, W, C = self.full_shape

        # --- Build mask (unchanged) ---
        from PIL import Image as PILImage, ImageDraw
        mask_img = PILImage.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask_img, 'L')
        ring_px = [(float(x), float(y)) for (x, y) in poly_full_xy]
        if ring_px[0] != ring_px[-1]:
            ring_px.append(ring_px[0])
        draw.polygon(ring_px, outline=1, fill=1)
        mask = np.array(mask_img, dtype=np.uint8).astype(bool)

        # --- Read raster + src page for tags ---
        with tifffile.TiffFile(self.current_path) as tif:
            ser = tif.series[0]
            src_page = tif.pages[0]
            data = ser.asarray(maxworkers=1)

        # normalize to (H,W,C)
        if data.ndim == 2:
            data = data[:, :, None]
        elif data.ndim == 3 and data.shape[0] in (1, 3, 4) and data.shape[-1] not in (1, 3, 4):
            data = np.moveaxis(data, 0, -1)

        # int16 to hold -32767
        data_i16 = data.astype(np.int16, copy=False)
        m = mask[:, :, None]
        data_i16[~m] = NODATA_VALUE

        # preserve geotags
        extratags = _collect_geo_extratags(src_page, NODATA_VALUE)
        use_compression = (imagecodecs is not None)
        comp = 'deflate' if use_compression else None
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

        # --- Build shapefile in MAP coordinates ---
        # 1) Get affine + WKT
        aff, wkt = _get_affine_and_wkt_from_raster(self.current_path)
        if aff is None:
            # fallback from tags
            aff = _get_affine_from_tifftags(src_page)
            if aff is None:
                # Can't georeference shapefile, write pixel coords as last resort
                aff = None
            if wkt is None:
                wkt = _extract_wkt_from_tifftags(src_page)

        # 2) Convert ring from pixel (col=x,row=y) to map (X,Y)
        if aff is not None:
            ring_map = [_pixel_to_map_xy(aff, x, y) for (x, y) in ring_px]
        else:
            ring_map = ring_px  # fallback: pixel coords (no .prj)

        # 3) Write shapefile
        import shapefile
        shp_base, ext = os.path.splitext(out_shp)
        if ext.lower() != '.shp':
            shp_base = out_shp

        w = shapefile.Writer(shp_base, shapeType=shapefile.POLYGON)
        w.field('id', 'N')
        w.poly([ring_map])
        w.record(1)
        w.close()

        # 4) Write .prj if we have WKT (so GIS aligns it to the raster)
        _write_prj(shp_base, wkt)


def _collect_geo_extratags(src_page, nodata_value):
    """
    Read GeoTIFF and GDAL tags from the source first page and return a tifffile
    'extratags' list suitable for tifffile.imwrite.

    We preserve:
      - 33550 ModelPixelScaleTag        (DOUBLE[3])
      - 33922 ModelTiepointTag          (DOUBLE[6*n])
      - 34264 ModelTransformationTag    (DOUBLE[16])
      - 34735 GeoKeyDirectoryTag        (SHORT[])
      - 34736 GeoDoubleParamsTag        (DOUBLE[])
      - 34737 GeoAsciiParamsTag         (ASCII)
      - 42112 GDAL_METADATA             (ASCII)
      - 42113 GDAL_NODATA               (ASCII)  <-- we will overwrite to match clip
      - 282/283 X/YResolution           (RATIONAL)
      - 296   ResolutionUnit            (SHORT)
    """
    tags = src_page.tags
    extratags = []

    def add_tag(tag_id, tifftype, value):
        # Compute count and coerce to correct container for tifffile
        if tifftype == 's':  # ASCII
            if isinstance(value, (bytes, bytearray)):
                value = value.decode('utf-8', errors='ignore')
            val = str(value)
            cnt = len(val) + 1  # include terminating NUL
        elif tifftype in ('H', 'I', 'd'):  # SHORT, LONG, DOUBLE arrays
            arr = np.array(value, dtype={'H': np.uint16, 'I': np.uint32, 'd': np.float64}[tifftype]).ravel()
            val = arr
            cnt = int(arr.size)
        elif tifftype == 'r':  # RATIONAL (num, den)
            # tifffile accepts a tuple (num, den)
            val = value
            cnt = 1
        else:
            return  # unsupported here
        extratags.append((tag_id, tifftype, cnt, val, False))

    def get_value(tag_key):
        t = tags.get(tag_key)
        return None if t is None else t.value

    # GeoTIFF core
    v = get_value(33550);  v is not None and add_tag(33550, 'd', v)     # ModelPixelScaleTag
    v = get_value(33922);  v is not None and add_tag(33922, 'd', v)     # ModelTiepointTag
    v = get_value(34264);  v is not None and add_tag(34264, 'd', v)     # ModelTransformationTag
    v = get_value(34735);  v is not None and add_tag(34735, 'H', v)     # GeoKeyDirectoryTag
    v = get_value(34736);  v is not None and add_tag(34736, 'd', v)     # GeoDoubleParamsTag
    v = get_value(34737);  v is not None and add_tag(34737, 's', v)     # GeoAsciiParamsTag

    # GDAL metadata (keep as-is)
    v = get_value(42112);  v is not None and add_tag(42112, 's', v)     # GDAL_METADATA

    # We will explicitly set GDAL_NODATA (42113) to nodata_value below.

    # Resolution tags if present
    v = get_value(282)   # XResolution (RATIONAL)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        add_tag(282, 'r', (int(v[0]), int(v[1])))
    v = get_value(283)   # YResolution (RATIONAL)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        add_tag(283, 'r', (int(v[0]), int(v[1])))
    v = get_value(296)   # ResolutionUnit (SHORT)
    if v is not None:
        add_tag(296, 'H', v)

    # Always set/update GDAL_NODATA to the clip's nodata
    nds = str(nodata_value)
    extratags.append((42113, 's', len(nds) + 1, nds, False))

    return extratags

def _get_affine_and_wkt_from_raster(path):
    """
    Prefer rasterio for robust CRS + transform.
    Returns (affine, wkt_or_None) where affine maps pixel (col=x, row=y) to (X,Y):
      X = a*col + b*row + c,  Y = d*col + e*row + f
    """
    if not HAVE_RASTERIO:
        return None, None
    try:
        with rasterio.open(path) as src:
            aff = src.transform
            wkt = src.crs.to_wkt() if src.crs else None
            return aff, wkt
    except Exception:
        return None, None

def _get_affine_from_tifftags(src_page):
    """
    Fallback: derive affine from GeoTIFF tags.
    Priority:
      1) ModelTransformationTag (34264) 4x4
      2) ModelPixelScale (33550) + ModelTiepoint (33922), north-up assumption.
    Returns Affine(a, b, c, d, e, f) or None if not possible.
    """
    # 34264 ModelTransformationTag
    tr = src_page.tags.get(34264)
    if tr is not None:
        vals = np.array(tr.value, dtype=float).ravel()
        if vals.size == 16:
            m = vals.reshape(4, 4)
            # GeoTIFF 4x4 maps homogeneous pixel [col,row,0,1] to map
            a, b, c = m[0, 0], m[0, 1], m[0, 3]
            d, e, f = m[1, 0], m[1, 1], m[1, 3]
            return Affine(a, b, c, d, e, f) if HAVE_RASTERIO else (a, b, c, d, e, f)

    # 33550 ModelPixelScale & 33922 ModelTiepoint
    scale_tag = src_page.tags.get(33550)
    tie_tag   = src_page.tags.get(33922)
    if scale_tag is None or tie_tag is None:
        return None

    sx, sy = float(scale_tag.value[0]), float(scale_tag.value[1])
    # Common convention: tiepoint maps pixel (i,j) -> (X,Y) in map coords.
    tp = tie_tag.value
    # Use first tiepoint (most rasters have only one at upper-left)
    i, j, _k, X, Y, _Z = map(float, tp[:6])

    # For north-up rasters, Y decreases as row increases → negative sy in affine
    a, b, c = sx, 0.0, X - i * sx
    d, e, f = 0.0, -sy, Y - j * (-sy)  # i.e., f = Y + j*sy
    return Affine(a, b, c, d, e, f) if HAVE_RASTERIO else (a, b, c, d, e, f)

def _pixel_to_map_xy(affine_like, x_col, y_row):
    """
    Apply affine to (col,row) -> (X,Y).
    Accepts rasterio Affine or a 6-tuple (a,b,c,d,e,f).
    """
    if HAVE_RASTERIO and isinstance(affine_like, Affine):
        X, Y = affine_like * (x_col, y_row)
        return float(X), float(Y)
    else:
        a, b, c, d, e, f = affine_like
        X = a * x_col + b * y_row + c
        Y = d * x_col + e * y_row + f
        return float(X), float(Y)

def _extract_wkt_from_tifftags(src_page):
    """
    Best-effort WKT from tags if rasterio is unavailable.
    We try:
      - GeoAsciiParamsTag (34737): sometimes contains WKT or parts
      - GDAL_METADATA (42112): may contain SRS WKT/XML
    If nothing usable, return None (shapefile will have no .prj).
    """
    # 34737 GeoAsciiParamsTag
    t = src_page.tags.get(34737)
    if t and t.value:
        val = t.value
        if isinstance(val, (bytes, bytearray)):
            val = val.decode('utf-8', errors='ignore')
        s = str(val).strip()
        if "GEOGCS" in s or "PROJCS" in s or "PROJCRS" in s or "GEOGCRS" in s:
            return s

    # 42112 GDAL_METADATA (XML), try to find SRS WKT inside
    t = src_page.tags.get(42112)
    if t and t.value:
        val = t.value
        if isinstance(val, (bytes, bytearray)):
            val = val.decode('utf-8', errors='ignore')
        s = str(val)
        # crude extraction
        start = s.find("<SRS>")
        end   = s.find("</SRS>")
        if 0 <= start < end:
            wkt = s[start+5:end].strip()
            if wkt:
                return wkt
    return None

def _write_prj(shp_base, wkt):
    """Write a .prj file next to shapefile base if WKT is available."""
    if not wkt:
        return
    prj_path = shp_base + ".prj"
    with open(prj_path, "w", encoding="utf-8") as f:
        f.write(wkt)



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
