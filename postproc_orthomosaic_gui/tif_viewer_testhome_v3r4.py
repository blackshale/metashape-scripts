# file: tif_viewer_testhome_v3r4.py
# GeoTIFF viewer with:
# - Open / Redraw / Crop (polygon to nodata) / Background reprojection
# - Colormap & range slider with value bubbles
# - Legend + scale bar
# - Align: pick pt1/pt2 with markers; show map coords to 7 decimals
# - Translate prompt (pt1 -> pt2) via translate_image.py
# - After opening the translated image, fully clears lower-right coord box and reapplies previous background
#
# Requires: PySide6, numpy, tifffile, pillow, pyshp
# Optional (recommended): imagecodecs, rasterio, matplotlib, opencv-python

import sys, os, traceback
import numpy as np
import tifffile

from PySide6.QtCore import Qt, QSize, QPointF, QRectF, Signal, QRect, QEvent
from PySide6.QtGui import (
    QImage, QPixmap, QPalette, QPainter, QPen, QBrush,
    QCursor, QColor, QFont, QFontMetrics
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog,
    QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox, QScrollArea,
    QComboBox
)

# Optional (TIFF compression support)
try:
    import imagecodecs  # noqa: F401
except Exception:
    imagecodecs = None

# Optional colormaps via Matplotlib
try:
    from matplotlib import cm as mpl_cm
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Optional: OpenCV for colormap fallback & resizing
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

# Optional (best CRS/transform + reprojection for background)
try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import reproject, Resampling
    HAVE_RASTERIO = True
except Exception:
    HAVE_RASTERIO = False

from PIL import Image as PILImage, ImageDraw
import shapefile  # pyshp

LOG_FILE = "tif_viewer_error.log"
NODATA_VALUE = -32767

COLORMAPS = [
    "jet", "viridis", "plasma", "inferno", "magma",
    "cividis", "turbo", "gray", "hot", "terrain",
]

# Map our names to OpenCV colormap enums (if OpenCV is available)
CV2_CMAPS = {
    "jet": getattr(cv2, "COLORMAP_JET", None) if 'cv2' in globals() else None,
    "viridis": getattr(cv2, "COLORMAP_VIRIDIS", None) if 'cv2' in globals() else None,
    "plasma": getattr(cv2, "COLORMAP_PLASMA", None) if 'cv2' in globals() else None,
    "inferno": getattr(cv2, "COLORMAP_INFERNO", None) if 'cv2' in globals() else None,
    "magma": getattr(cv2, "COLORMAP_MAGMA", None) if 'cv2' in globals() else None,
    "cividis": getattr(cv2, "COLORMAP_CIVIDIS", None) if 'cv2' in globals() else None,
    "turbo": getattr(cv2, "COLORMAP_TURBO", None) if 'cv2' in globals() else None,
    "gray": None,
    "hot": getattr(cv2, "COLORMAP_HOT", None) if 'cv2' in globals() else None,
    "terrain": getattr(cv2, "COLORMAP_TERRAIN", None) if 'cv2' in globals() else None,
}

def log_exc():
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== ERROR ===\n")
        traceback.print_exc(file=f)

# ------------------------ Range slider (two handles + editable bubbles) ------------------------

class RangeSlider(QWidget):
    valueChanged = Signal(int, int)

    def __init__(self, parent=None, minimum=0, maximum=1000, low=0, high=1000):
        super().__init__(parent)
        self._min = int(minimum)
        self._max = int(maximum)
        self._low = int(low)
        self._high = int(high)
        self._pressed = None
        self._vmin = 0.0
        self._vmax = 1.0
        self._fmt = "{:.6g}"
        self.setMinimumHeight(56)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        from PySide6.QtWidgets import QLineEdit
        from PySide6.QtGui import QDoubleValidator
        self._edit = QLineEdit(self); self._edit.hide()
        self._edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._edit.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._edit.setValidator(QDoubleValidator(-1e30, 1e30, 10, self))
        self._edit.editingFinished.connect(self._commit_editor)
        self._edit.installEventFilter(self)
        self._editing = None

        self._rect_low_bubble = QRect()
        self._rect_high_bubble = QRect()

    def minimum(self): return self._min
    def maximum(self): return self._max
    def lowValue(self): return self._low
    def highValue(self): return self._high

    def setRange(self, minimum, maximum):
        self._min, self._max = int(minimum), int(maximum)
        self._low = max(self._min, min(self._low, self._max))
        self._high = max(self._min, min(self._high, self._max))
        if self._low > self._high: self._low = self._high
        self.update()

    def setValues(self, low, high, emit=True):
        low = max(self._min, min(int(low), self._max))
        high = max(self._min, min(int(high), self._max))
        if low > high: low, high = high, low
        changed = (low != self._low) or (high != self._high)
        self._low, self._high = low, high
        if changed and emit:
            self.valueChanged.emit(self._low, self._high)
        self.update()

    def setDataRange(self, vmin, vmax, fmt="{:.6g}"):
        self._vmin = float(vmin); self._vmax = float(vmax) if float(vmax) != float(vmin) else float(vmin) + 1.0
        self._fmt = fmt; self.update()

    def _groove_rect(self):
        margin_side = 10; groove_h = 8; bottom_pad = 12
        top = self.height() - (groove_h + bottom_pad)
        width = self.width() - 2 * margin_side
        return QRect(margin_side, max(0, top), width, groove_h)

    def _pos_to_value(self, x):
        gr = self._groove_rect()
        if gr.width() <= 0: return self._min
        t = (x - gr.left()) / gr.width()
        return int(round(self._min + t * (self._max - self._min)))

    def _value_to_pos(self, v):
        gr = self._groove_rect()
        if self._max == self._min: return gr.left()
        t = (v - self._min) / (self._max - self._min)
        return int(gr.left() + t * gr.width())

    def _map_internal_to_data(self, val):
        frac = (val - self._min) / (self._max - self._min) if self._max != self._min else 0.0
        return self._vmin + frac * (self._vmax - self._vmin)

    def _map_data_to_internal(self, data_val: float) -> int:
        if self._vmax == self._vmin: return self._min
        frac = (float(data_val) - self._vmin) / (self._vmax - self._vmin)
        v = self._min + frac * (self._max - self._min)
        return int(round(max(self._min, min(self._max, v))))

    def _draw_value_bubble(self, p: QPainter, x_center: int, text: str, which: str):
        fm = QFontMetrics(p.font()); pad_x, pad_y = 6, 3
        tw = fm.horizontalAdvance(text); th = fm.height()
        w = tw + 2*pad_x; h = th + 2*pad_y
        gr = self._groove_rect()
        rect = QRect(x_center - w//2, gr.top() - h - 10, w, h)
        if rect.left() < 2: rect.moveLeft(2)
        if rect.right() > self.width()-2: rect.moveRight(self.width()-2)
        if which == 'low': self._rect_low_bubble = QRect(rect)
        else: self._rect_high_bubble = QRect(rect)
        p.setPen(QPen(QColor(70,70,70), 1)); p.setBrush(QColor(250,250,250))
        p.drawRoundedRect(rect, 5, 5)
        p.setPen(QColor(20,20,20)); p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        gr = self._groove_rect()
        p.setPen(Qt.NoPen); p.setBrush(QColor(60,60,60)); p.drawRoundedRect(gr, 4, 4)
        lx = self._value_to_pos(self._low); hx = self._value_to_pos(self._high)
        sel = QRect(min(lx, hx), gr.top(), abs(hx - lx), gr.height())
        p.setBrush(QColor(120,180,255)); p.drawRoundedRect(sel, 4, 4)
        for x, active in ((lx, self._pressed == 'low'), (hx, self._pressed == 'high')):
            r = QRect(x-7, gr.center().y()-10, 14, 20)
            p.setBrush(QColor(230,230,230) if active else QColor(200,200,200))
            p.setPen(QPen(QColor(70,70,70), 1)); p.drawRoundedRect(r, 3, 3)
        p.setFont(QFont(self.font().family(), 9))
        low_val  = self._fmt.format(self._map_internal_to_data(self._low))
        high_val = self._fmt.format(self._map_internal_to_data(self._high))
        self._draw_value_bubble(p, lx, low_val,  'low')
        self._draw_value_bubble(p, hx, high_val, 'high')

    def _show_editor(self, which: str):
        rect = self._rect_low_bubble if which == 'low' else self._rect_high_bubble
        if rect.isNull(): return
        self._editing = which
        dv = self._edit.validator(); dv.setBottom(min(self._vmin, self._vmax)); dv.setTop(max(self._vmin, self._vmax))
        current_val = self._map_internal_to_data(self._low if which == 'low' else self._high)
        self._edit.setText(self._fmt.format(current_val))
        r = QRect(rect); r.adjust(2, 2, -2, -2); self._edit.setGeometry(r)
        self._edit.show(); self._edit.setFocus(); self._edit.selectAll()

    def _commit_editor(self):
        if not self._editing: self._edit.hide(); return
        txt = self._edit.text().strip()
        try: val = float(txt)
        except Exception: return self._cancel_editor()
        val = max(min(val, max(self._vmin, self._vmax)), min(self._vmin, self._vmax))
        ival = self._map_data_to_internal(val)
        if self._editing == 'low':
            ival = min(ival, self._high); self.setValues(ival, self._high)
        else:
            ival = max(ival, self._low); self.setValues(self._low, ival)
        self._editing = None; self._edit.hide()

    def _cancel_editor(self):
        self._editing = None; self._edit.hide()

    def mouseDoubleClickEvent(self, e):
        pos = e.position().toPoint()
        if self._rect_low_bubble.contains(pos): self._show_editor('low'); return
        if self._rect_high_bubble.contains(pos): self._show_editor('high'); return
        super().mouseDoubleClickEvent(e)

    def mousePressEvent(self, e):
        lx = self._value_to_pos(self._low); hx = self._value_to_pos(self._high)
        if e.button() != Qt.MouseButton.LeftButton: return super().mousePressEvent(e)
        self._pressed = 'low' if abs(e.position().x() - lx) <= abs(e.position().x() - hx) else 'high'
        self.setCursor(Qt.CursorShape.ClosedHandCursor); self.mouseMoveEvent(e)

    def mouseMoveEvent(self, e):
        if not self._pressed:
            lx = self._value_to_pos(self._low); hx = self._value_to_pos(self._high)
            if min(abs(e.position().x()-lx), abs(e.position().x()-hx)) <= 8:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        val = self._pos_to_value(int(e.position().x()))
        if self._pressed == 'low': self.setValues(val, self._high)
        else: self.setValues(self._low, val)

    def mouseReleaseEvent(self, e):
        self._pressed = None; self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        step = max(1, (self._max - self._min)//100)
        if e.key() in (Qt.Key.Key_A, Qt.Key.Key_Left): self.setValues(self._low - step, self._high)
        elif e.key() in (Qt.Key.Key_D, Qt.Key.Key_Right): self.setValues(self._low, self._high + step)
        elif e.key() in (Qt.Key.Key_W, Qt.Key.Key_Up): self.setValues(self._low + step, self._high)
        elif e.key() in (Qt.Key.Key_S, Qt.Key.Key_Down): self.setValues(self._low, self._high - step)
        else: super().keyPressEvent(e)

    def eventFilter(self, obj, event):
        if obj is self._edit and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape: self._cancel_editor(); return True
        return super().eventFilter(obj, event)

# ------------------------ Preview helpers ------------------------

def pick_reduced_level(series, target_long_edge=3000):
    try:
        levels = series.levels
        if not levels: return 0
        sizes = []
        for i, lvl in enumerate(levels):
            h, w = lvl.shape[:2]; sizes.append((i, max(h, w)))
        under = [i for i, L in sizes if L <= target_long_edge]
        return (max(under, key=lambda i: sizes[i][1]) if under
                else min(range(len(levels)), key=lambda i: sizes[i][1]))
    except Exception:
        return 0

def _read_gdal_nodata_from_page(page):
    tag = page.tags.get(42113) or page.tags.get('GDAL_NODATA')
    if tag is None: return None
    val = tag.value
    if isinstance(val, (bytes, bytearray)): val = val.decode('utf-8', errors='ignore')
    s = str(val).strip(); parts = s.split()
    try: nums = [float(p) for p in parts]
    except Exception:
        try: nums = [float(s)]
        except Exception: return None
    return nums if len(nums) > 1 else nums[0]

def read_tiff_preview_raw(path, target_long_edge=3000):
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]; page0 = tif.pages[0]
        nodata_tag = _read_gdal_nodata_from_page(page0)
        lvl_idx = pick_reduced_level(series, target_long_edge)
        try: arr = series.asarray(level=lvl_idx, maxworkers=1)
        except TypeError:
            arr = (series.asarray(maxworkers=1) if lvl_idx == 0
                   else series.levels[lvl_idx].asarray(maxworkers=1))

    if arr.ndim == 2: arr = arr[:, :, None]
    elif arr.ndim == 3 and arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
        arr = np.moveaxis(arr, 0, -1)

    H, W, C = arr.shape
    valid = np.ones((H, W), dtype=bool)
    if nodata_tag is not None:
        if isinstance(nodata_tag, (list, tuple)) and C >= 1:
            invalid = np.ones((H, W), dtype=bool)
            for b in range(min(C, len(nodata_tag))):
                nd = nodata_tag[b]
                if np.isnan(nd): invalid &= np.isnan(arr[..., b])
                else: invalid &= (arr[..., b] == nd)
            valid = ~invalid
        else:
            nd = float(nodata_tag)
            if np.isnan(nd): valid = ~np.isnan(arr).all(axis=-1)
            else:
                equal_nd = np.all(arr == nd, axis=-1) if C > 1 else (arr[..., 0] == nd)
                valid = ~equal_nd

    if valid.any():
        vmin = float(np.nanmin(arr[valid])); vmax = float(np.nanmax(arr[valid]))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    long_edge = max(H, W)
    if long_edge > target_long_edge:
        scale = target_long_edge / float(long_edge)
        new_w = max(1, int(W*scale)); new_h = max(1, int(H*scale))
        try:
            if HAVE_CV2:
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                valid = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                raise RuntimeError("cv2 not available")
        except Exception:
            step = max(1, int(1/scale)); arr = arr[::step, ::step, :].copy(); valid = valid[::step, ::step].copy()
    return arr, valid, vmin, vmax

def _render_window_to_rgba(arr_raw, valid_mask, vmin, vmax, low_frac, high_frac, cmap_name=None):
    if arr_raw.ndim == 2: arr_raw = arr_raw[:, :, None]
    elif arr_raw.ndim == 3 and arr_raw.shape[0] in (1, 3, 4) and arr_raw.shape[-1] not in (1, 3, 4):
        arr_raw = np.moveaxis(arr_raw, 0, -1)

    if valid_mask.ndim != 2:
        valid_mask = valid_mask[..., 0] if valid_mask.ndim == 3 else valid_mask

    H, W, C = arr_raw.shape
    low = vmin + float(low_frac) * (vmax - vmin)
    high = vmin + float(high_frac) * (vmax - vmin)
    if high <= low: high = low + 1e-6

    arr = arr_raw.astype(np.float64, copy=False)
    scaled = (np.clip(arr, low, high) - low) / (high - low); scaled = np.clip(scaled, 0.0, 1.0)

    if C == 1 and cmap_name:
        scaled1 = scaled[..., 0]
        if HAVE_MPL:
            try: cmap = mpl_cm.colormaps.get_cmap(cmap_name, 256)
            except Exception: cmap = mpl_cm.get_cmap(cmap_name, 256)
            lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255.0 + 0.5).astype(np.uint8)
            idx = (scaled1 * 255.0 + 0.5).astype(np.uint8); rgb8 = lut[idx]
        elif HAVE_CV2:
            idx = (scaled1 * 255.0 + 0.5).astype(np.uint8); code = CV2_CMAPS.get(cmap_name)
            if cmap_name == "gray" or code is None:
                rgb8 = np.repeat(idx[:, :, None], 3, axis=-1)
            else:
                bgr = cv2.applyColorMap(idx, code); rgb8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            g8 = (scaled1 * 255.0 + 0.5).astype(np.uint8); rgb8 = np.repeat(g8[:, :, None], 3, axis=-1)
    else:
        scaled8 = (scaled * 255.0 + 0.5).astype(np.uint8)
        rgb8 = scaled8[..., :3] if C >= 3 else np.repeat(scaled8, 3, axis=-1)

    alpha = (valid_mask.astype(np.uint8) * 255)[..., None]
    rgba8 = np.concatenate([rgb8, alpha], axis=-1)
    return rgba8

def np_to_qimage(arr8):
    h, w, c = arr8.shape
    if c == 3: qimg = QImage(arr8.data, w, h, 3*w, QImage.Format.Format_RGB888)
    elif c == 4: qimg = QImage(arr8.data, w, h, 4*w, QImage.Format.Format_RGBA8888)
    else: raise ValueError("Expected 3 or 4 channels")
    return qimg.copy()

# ------------------------ GeoTIFF/CRS helpers ------------------------

def _snapshot_geo_tags(src_page):
    tg = src_page.tags
    def get_copy(tag_id):
        t = tg.get(tag_id)
        if t is None: return None
        v = t.value
        if isinstance(v, (bytes, bytearray)): return bytes(v)
        if isinstance(v, str): return str(v)
        try: arr = np.array(v); return arr.copy()
        except Exception: return v
    return {33550:get_copy(33550), 33922:get_copy(33922), 34264:get_copy(34264),
            34735:get_copy(34735), 34736:get_copy(34736), 34737:get_copy(34737),
            42112:get_copy(42112), 282:get_copy(282), 283:get_copy(283), 296:get_copy(296)}

def _build_extratags_from_snapshot(snap_dict, nodata_value):
    extratags = []
    def add(tag, tifftype, value):
        if value is None: return
        if tifftype == 's':
            if isinstance(value, (bytes, bytearray)): value = value.decode('utf-8','ignore')
            s = str(value); extratags.append((tag,'s',len(s)+1,s,False))
        elif tifftype in ('H','I','d'):
            dt = {'H':np.uint16,'I':np.uint32,'d':np.float64}[tifftype]
            arr = np.array(value,dtype=dt).ravel(); extratags.append((tag,tifftype,int(arr.size),arr,False))
        elif tifftype == 'r':
            extratags.append((tag,'r',1,(int(value[0]),int(value[1])),False))
    add(33550,'d',snap_dict.get(33550)); add(33922,'d',snap_dict.get(33922))
    add(34264,'d',snap_dict.get(34264)); add(34735,'H',snap_dict.get(34735))
    add(34736,'d',snap_dict.get(34736)); add(34737,'s',snap_dict.get(34737))
    add(42112,'s',snap_dict.get(42112))
    xr=snap_dict.get(282); yr=snap_dict.get(283); ru=snap_dict.get(296)
    if isinstance(xr,(tuple,list)) and len(xr)==2: add(282,'r',xr)
    if isinstance(yr,(tuple,list)) and len(yr)==2: add(283,'r',yr)
    if ru is not None: add(296,'H',ru)
    nds=str(nodata_value); extratags.append((42113,'s',len(nds)+1,nds,False))
    return extratags

def _get_affine_and_wkt_from_raster(path):
    if not HAVE_RASTERIO: return None, None
    try:
        with rasterio.open(path) as src:
            return src.transform, (src.crs.to_wkt() if src.crs else None)
    except Exception:
        return None, None

def _get_affine_from_tifftags(src_page):
    if HAVE_RASTERIO:
        Aff = Affine
    else:
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
    if scale_tag is None or tie_tag is None: return None
    sx, sy = float(scale_tag.value[0]), float(scale_tag.value[1])
    i,j,_k, X,Y,_Z = map(float, tie_tag.value[:6])
    a,b,c = sx, 0.0, X - i*sx
    d,e,f = 0.0, -sy, Y - j*(-sy)
    return Aff(a,b,c,d,e,f)

def _pixel_to_map_xy(affine_like, x_col, y_row):
    if HAVE_RASTERIO and isinstance(affine_like, Affine):
        X, Y = affine_like * (x_col, y_row); return float(X), float(Y)
    else:
        a,b,c,d,e,f = affine_like; X = a*x_col + b*y_row + c; Y = d*x_col + e*y_row + f
        return float(X), float(Y)

def _extract_wkt_from_tifftags(src_page):
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
        s = str(val); start, end = s.find("<SRS>"), s.find("</SRS>")
        if 0 <= start < end:
            w = s[start+5:end].strip()
            if w: return w
    return None

def _write_prj(shp_base, wkt):
    if not wkt: return
    with open(shp_base + ".prj", "w", encoding="utf-8") as f:
        f.write(wkt)

# ------------------------ Interactive image label ------------------------

class ImageLabel(QLabel):
    SNAP_PIX = 12
    polygonFinished = Signal(list)      # list[(x,y)] in preview pixel coords
    pointPicked = Signal(float, float)  # (x_preview, y_preview) for Align

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
        self.picking_point = False
        self.marker_pts_img = []  # preview coords
        self.marker_labels = []

    def set_pixmaps(self, base_pixmap: QPixmap, scaled_pixmap: QPixmap, scale: float, offset_xy):
        self._base_pixmap = base_pixmap; self._scaled_pixmap = scaled_pixmap
        self._scale = scale; self._offset = QPointF(*offset_xy)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._scaled_pixmap is None or not self.marker_pts_img: return
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        for (pt, label) in zip(self.marker_pts_img, self.marker_labels):
            pw = self.img_to_widget(pt); r = 5
            p.setBrush(QBrush(QColor(255, 0, 0, 200)) if label == "pt1" else QBrush(QColor(0, 180, 0, 200)))
            p.setPen(QPen(QColor(0, 0, 0, 220), 1))
            p.drawEllipse(QRectF(pw.x()-r, pw.y()-r, 2*r, 2*r))
            text = label; fm = p.fontMetrics(); tw, th = fm.horizontalAdvance(text), fm.height(); pad = 3
            rect = QRectF(pw.x()+8, pw.y()-th/2-2, tw+2*pad, th+2)
            p.setBrush(QColor(255,255,255,220)); p.setPen(QPen(QColor(0,0,0,180), 1))
            p.drawRoundedRect(rect, 3, 3); p.setPen(QColor(20,20,20))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        p.end()

    def add_marker(self, x_img: float, y_img: float, label: str):
        self.marker_pts_img.append(QPointF(x_img, y_img)); self.marker_labels.append(label); self.update()

    def clear_markers(self):
        self.marker_pts_img.clear(); self.marker_labels.clear(); self.update()

    def start_selection(self):
        self.points_img = []; self.selecting = True
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor)); self.setFocus(); self.update()

    def cancel_selection(self):
        self.selecting = False; self.points_img = []
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor)); self.clearFocus(); self.update()

    def start_point_pick(self):
        self.picking_point = True; self.setCursor(QCursor(Qt.CursorShape.CrossCursor)); self.setFocus(); self.update()

    def cancel_point_pick(self):
        self.picking_point = False; self.setCursor(QCursor(Qt.CursorShape.ArrowCursor)); self.clearFocus(); self.update()

    def img_to_widget(self, p_img: QPointF) -> QPointF:
        return QPointF(self._offset.x() + p_img.x()*self._scale, self._offset.y() + p_img.y()*self._scale)

    def widget_to_img(self, p_w: QPointF) -> QPointF:
        return QPointF((p_w.x()-self._offset.x())/self._scale, (p_w.y()-self._offset.y())/self._scale)

    def mousePressEvent(self, e):
        if self.picking_point and e.button() == Qt.MouseButton.LeftButton and self._scaled_pixmap is not None:
            pos = e.position() if hasattr(e, "position") else e.posF()
            xw, yw = pos.x(), pos.y(); x0, y0 = self._offset.x(), self._offset.y()
            sw, sh = self._scaled_pixmap.width(), self._scaled_pixmap.height()
            if x0 <= xw <= x0+sw and y0 <= yw <= y0+sh:
                p_img = self.widget_to_img(QPointF(xw, yw))  # preview coords
                self.pointPicked.emit(p_img.x(), p_img.y()); return
        return super().mousePressEvent(e)

# ------------------------ Main Window ------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoTIFF Viewer + Crop + Background + Legend/Scale + Align")
        self.resize(1280, 860)

        btn_open = QPushButton("Open"); btn_redraw = QPushButton("Redraw")
        btn_crop = QPushButton("Crop"); btn_bg = QPushButton("Background")
        btn_align = QPushButton("Align"); btn_exit = QPushButton("Exit")
        btn_open.clicked.connect(self.on_open); btn_redraw.clicked.connect(self.on_redraw)
        btn_crop.clicked.connect(self.on_crop); btn_bg.clicked.connect(self.on_background)
        btn_align.clicked.connect(self.on_align); btn_exit.clicked.connect(self.close)

        row_buttons = QHBoxLayout()
        for w in (btn_open, btn_redraw, btn_crop, btn_bg, btn_align, btn_exit): row_buttons.addWidget(w)
        row_buttons.addStretch(1)

        self.cmap_name = "jet"; lbl_cmap = QLabel("Color:")
        self.cmb_cmap = QComboBox(); self.cmb_cmap.addItems(COLORMAPS)
        self.cmb_cmap.setCurrentText(self.cmap_name); self.cmb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        row_buttons.addWidget(lbl_cmap); row_buttons.addWidget(self.cmb_cmap)

        self.lbl_left = QLabel("min"); self.lbl_left.setMinimumWidth(160)
        self.lbl_right = QLabel("max"); self.lbl_right.setMinimumWidth(160)
        self.range = RangeSlider(minimum=0, maximum=1000, low=0, high=1000)
        self.range.valueChanged.connect(self._on_range_changed)

        row_slider = QHBoxLayout()
        row_slider.addWidget(self.lbl_left); row_slider.addWidget(self.range, stretch=1); row_slider.addWidget(self.lbl_right)

        self.image_label = ImageLabel(self)
        self.image_label.pointPicked.connect(self.on_point_picked)
        self.image_label.polygonFinished.connect(self.on_polygon_finished)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setWidget(self.image_label)

        self.legend = QLabel(self.image_label)
        self.legend.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.legend.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.legend.setStyleSheet("background: transparent; border: none;"); self.legend.hide()

        self.align_label = QLabel(self.image_label)
        self.align_label.setStyleSheet("background: rgba(255,255,255,200);border: 1px solid rgba(0,0,0,80);padding: 4px;")
        self.align_label.hide()

        central = QWidget(); lay = QVBoxLayout(central)
        lay.addLayout(row_buttons); lay.addLayout(row_slider); lay.addWidget(self.scroll, stretch=1)
        self.setCentralWidget(central)

        self._pixmap = None; self._scaled = None; self._scale = 1.0; self._offset_xy = (0.0, 0.0)
        self.preview_size = None; self.current_path = None; self.full_shape = None; self.full_dtype = None
        self.series_index = 0

        self.preview_raw = None; self.preview_valid = None; self.preview_min = None; self.preview_max = None
        self.bg_full_rgb = None; self.bg_valid = None
        self._last_bg_path = None   # remember last chosen background

        self.target_affine = None; self.target_wkt = None
        self.m_per_px_x = None

        self.align_pt1_map = None; self.align_pt2_map = None; self._align_pick_count = 0

    # ---------- Internal loader (no file dialog) ----------
    def _load_image_from_path(self, path: str):
        try:
            if not path or not os.path.exists(path): return
            arr_raw, valid, vmin, vmax = read_tiff_preview_raw(path, target_long_edge=3000)
            self.preview_raw, self.preview_valid = arr_raw, valid
            self.preview_min, self.preview_max = vmin, vmax
            self.lbl_left.setText(f"min: {vmin:.6g}"); self.lbl_right.setText(f"max: {vmax:.6g}")
            self.range.setDataRange(vmin, vmax, fmt="{:.6g}"); self.range.setValues(0, 1000, emit=False)
            self.cmap_name = self.cmb_cmap.currentText()
            low_frac  = self.range.lowValue()  / self.range.maximum()
            high_frac = self.range.highValue() / self.range.maximum()
            rgba8 = _render_window_to_rgba(arr_raw, valid, vmin, vmax, low_frac, high_frac, self.cmap_name)
            with tifffile.TiffFile(path) as tif:
                ser = tif.series[self.series_index]; shp = ser.shape
                if len(shp) == 2: H, W = shp; C = 1
                elif len(shp) == 3:
                    if shp[-1] in (1,3,4): H, W, C = shp
                    else: C, H, W = shp
                else: raise RuntimeError(f"Unsupported TIFF shape: {shp}")
                self.full_shape = (H, W, C); self.full_dtype = ser.dtype
            self.current_path = path
            aff, wkt = _get_affine_and_wkt_from_raster(self.current_path)
            if aff is None:
                with tifffile.TiffFile(self.current_path) as _t:
                    _p = _t.pages[0]; aff = _get_affine_from_tifftags(_p)
                    if wkt is None: wkt = _extract_wkt_from_tifftags(_p)
            self.target_affine = aff; self.target_wkt = wkt
            self._compute_m_per_px_x()
            rgba8 = self._composite_with_background(rgba8)
            qimg = np_to_qimage(rgba8); self._pixmap = QPixmap.fromImage(qimg)
            self.preview_size = (rgba8.shape[1], rgba8.shape[0]); self._set_scaled_pixmap()
            self._update_legend(low_frac, high_frac)
            self.setWindowTitle(f"GeoTIFF Viewer — {os.path.basename(path)}")
        except Exception:
            log_exc(); QMessageBox.critical(self, "Error", f"Failed to open image.\nSee log: {os.path.abspath(LOG_FILE)}")

    # ---------- Align box clearer ----------
    def _clear_align_box(self):
        # Hide and clear the lower-right coordinate box and any align state
        self.align_label.clear()
        self.align_label.hide()
        self.align_pt1_map = None
        self.align_pt2_map = None
        self._align_pick_count = 0
        if hasattr(self, "image_label"):
            self.image_label.clear_markers()

    # ---------- Reset state (for switching images cleanly) ----------
    def _reset_view_state(self):
        self._pixmap = QPixmap()
        self._scaled = None
        self.image_label.setPixmap(QPixmap())
        self._clear_align_box()

        self.preview_raw = None
        self.preview_valid = None
        self.preview_min = None
        self.preview_max = None
        self.bg_full_rgb = None
        self.bg_valid = None
        self.current_path = None

        self.image_label.update()
        QApplication.processEvents()

    # ---------- UI plumbing ----------
    def _set_scaled_pixmap(self):
        if self._pixmap is None: self.image_label.setPixmap(QPixmap()); return
        avail = self.scroll.viewport().size()
        scaled = self._pixmap.scaled(QSize(avail.width(), avail.height()),
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        self._scaled = scaled; self.image_label.setPixmap(scaled)
        base_w, base_h = self._pixmap.width(), self._pixmap.height()
        sc = scaled.width() / base_w if base_w else 1.0
        x0 = (self.image_label.width() - scaled.width()) / 2; y0 = (self.image_label.height() - scaled.height()) / 2
        self._scale = sc; self._offset_xy = (x0, y0)
        self.image_label.set_pixmaps(self._pixmap, self._scaled, self._scale, self._offset_xy)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._pixmap is not None:
            self._set_scaled_pixmap()
            if self.preview_raw is not None and self.preview_min is not None:
                low_frac  = self.range.lowValue()  / self.range.maximum()
                high_frac = self.range.highValue() / self.range.maximum()
                self._update_legend(low_frac, high_frac)
        if self.align_label.isVisible(): self._update_align_overlay(self.align_label.text())

    # ---------- Background reprojection helper ----------
    def _reproject_background(self, bg_path: str):
        if not HAVE_RASTERIO:
            QMessageBox.warning(self, "Background", "This feature requires rasterio.")
            return
        if self.full_shape is None or self.target_affine is None:
            return
        H, W, _C = self.full_shape
        with rasterio.open(bg_path) as src:
            src_arr = src.read()
            src_crs = src.crs
            src_transform = src.transform
            if src_arr.ndim == 2:
                src_arr = src_arr[None, ...]
            bands = src_arr.shape[0]
            src_rgb = (src_arr[:3, ...] if bands >= 3 else np.repeat(src_arr[0:1, ...], 3, axis=0)).astype(np.float32)
            dst_rgb = np.zeros((3, H, W), dtype=np.float32)
            if self.target_wkt is None:
                return
            dst_crs = rasterio.crs.CRS.from_wkt(self.target_wkt)
            dst_transform = self.target_affine
            for i in range(3):
                reproject(
                    source=src_rgb[i, ...],
                    destination=dst_rgb[i, ...],
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    dst_nodata=np.nan
                )
            valid = np.isfinite(dst_rgb).all(axis=0)
            bg_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            for i in range(3):
                ch = dst_rgb[i, ...]
                if np.isfinite(ch).any():
                    vmin = np.nanpercentile(ch, 2)
                    vmax = np.nanpercentile(ch, 98)
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                        vmin, vmax = np.nanmin(ch), np.nanmax(ch)
                        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                            vmin, vmax = 0.0, 1.0
                else:
                    vmin, vmax = 0.0, 1.0
                ch8 = ((np.clip(ch, vmin, vmax) - vmin) / (vmax - vmin) * 255.0 + 0.5).astype(np.uint8)
                bg_rgb[..., i] = np.where(np.isfinite(ch8), ch8, 0)
            self.bg_full_rgb = bg_rgb
            self.bg_valid = valid

    # ---------- Buttons ----------
    def on_open(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open TIFF Image", "", "TIFF Images (*.tif *.tiff);;All Files (*)")
            if not path: return
            self._load_image_from_path(path)
        except Exception:
            log_exc(); QMessageBox.critical(self, "Error", f"Failed to open image.\nSee log: {os.path.abspath(LOG_FILE)}")

    def on_redraw(self):
        if self._pixmap is None:
            QMessageBox.information(self, "Redraw", "No image loaded. Click Open first."); return
        if hasattr(self, "cmb_cmap"): self.cmap_name = self.cmb_cmap.currentText()
        try: self._on_range_changed(self.range.lowValue(), self.range.highValue())
        except Exception: log_exc(); QMessageBox.critical(self, "Error", "Redraw failed.")

    def on_crop(self):
        if self._pixmap is None or self.current_path is None:
            QMessageBox.information(self, "Crop", "Open a TIFF first."); return
        self.image_label.start_selection(); self.image_label.setFocus()
        QMessageBox.information(self, "Polygon selection",
                                "Left-click to add vertices.\nRight-click (or Enter) to finish.\nEsc to cancel.\n"
                                "Tip: click near the first vertex to snap-close.")

    def on_background(self):
        if self.full_shape is None or self.target_affine is None:
            QMessageBox.information(self, "Background", "Open a foreground GeoTIFF first."); return
        if not HAVE_RASTERIO:
            QMessageBox.warning(self, "Background", "This feature requires rasterio."); return
        bg_path, _ = QFileDialog.getOpenFileName(self, "Select background GeoTIFF", "", "TIFF Images (*.tif *.tiff);;All Files (*)")
        if not bg_path: return
        self._last_bg_path = bg_path
        try:
            self._reproject_background(bg_path)
            self.on_redraw()
        except Exception:
            log_exc(); QMessageBox.critical(self, "Background", f"Failed to load background.\nSee log: {os.path.abspath(LOG_FILE)}")

    # ---------- Align: pick two points, show & store map coords ----------
    def on_align(self):
        if self._pixmap is None or self.target_affine is None:
            QMessageBox.information(self, "Align", "Open a georeferenced GeoTIFF first."); return
        self.align_pt1_map = None; self.align_pt2_map = None; self._align_pick_count = 0
        self.image_label.clear_markers(); self.align_label.hide(); self.image_label.start_point_pick()

    def on_point_picked(self, x_prev: float, y_prev: float):
        if not self.preview_size or not self.full_shape: return
        prev_w, prev_h = self.preview_size;
        full_h, full_w, _ = self.full_shape
        x_full = x_prev * (full_w / float(prev_w));
        y_full = y_prev * (full_h / float(prev_h))
        try:
            X, Y = _pixel_to_map_xy(self.target_affine, x_full, y_full)
        except Exception:
            X, Y = float('nan'), float('nan')

        # NEW: track whether we opened a new image so we don't redraw the overlay later
        reopened_translated = False

        if self._align_pick_count == 0:
            self.align_pt1_map = (X, Y);
            self._align_pick_count = 1
            self.image_label.add_marker(x_prev, y_prev, "pt1")
            text = f"pt1: {X:.7f}, {Y:.7f}"
        else:
            self.align_pt2_map = (X, Y);
            self._align_pick_count = 2
            self.image_label.add_marker(x_prev, y_prev, "pt2")
            self.image_label.cancel_point_pick()
            p1 = self.align_pt1_map;
            p2 = self.align_pt2_map
            text = f"pt1: {p1[0]:.7f}, {p1[1]:.7f}\npt2: {p2[0]:.7f}, {p2[1]:.7f}"

            reply = QMessageBox.question(
                self, "Translate image?",
                "Translate the image by adjusting georeferencing so pt1 moves to pt2?\n"
                "This writes a new file with suffix _translated.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    from translate_image import translate_image
                    out_path = translate_image(self.current_path, self.align_pt1_map, self.align_pt2_map)
                    QMessageBox.information(self, "Translated", f"Saved:\n{out_path}")
                    reply2 = QMessageBox.question(
                        self, "Open translated image?",
                        "Do you want to open the translated image now?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply2 == QMessageBox.Yes:
                        self._reset_view_state()
                        self._load_image_from_path(out_path)
                        self._clear_align_box()
                        if self._last_bg_path:
                            try:
                                self._reproject_background(self._last_bg_path)
                                self.on_redraw()
                            except Exception:
                                log_exc()
                        # NEW: mark that we reopened, so we skip the overlay update below
                        reopened_translated = True
                except Exception as e:
                    log_exc();
                    QMessageBox.critical(self, "Translate failed", f"{e}")

        # CHANGE: only update overlay if we did NOT reopen a translated image
        if not reopened_translated:
            self._update_align_overlay(text)
        else:
            # ensure hidden just in case
            self._clear_align_box()

    def _update_align_overlay(self, text: str):
        self.align_label.setText(text); self.align_label.adjustSize()
        pad = 12
        x = max(pad, self.image_label.width() - self.align_label.width() - pad)
        y = max(pad, self.image_label.height() - self.align_label.height() - pad)
        self.align_label.move(int(x), int(y)); self.align_label.show(); self.align_label.raise_()

    # ---------- Range slider callback ----------
    def _on_range_changed(self, low, high=None):
        if high is None: low, high = self.range.lowValue(), self.range.highValue()
        if self.preview_raw is None or self.preview_min is None: return
        low_frac  = low  / self.range.maximum(); high_frac = high / self.range.maximum()
        rgba8 = _render_window_to_rgba(self.preview_raw, self.preview_valid,
                                       self.preview_min, self.preview_max,
                                       low_frac, high_frac, self.cmap_name)
        rgba8 = self._composite_with_background(rgba8)
        qimg = np_to_qimage(rgba8); self._pixmap = QPixmap.fromImage(qimg)
        self.preview_size = (rgba8.shape[1], rgba8.shape[0]); self._set_scaled_pixmap()
        self._update_legend(low_frac, high_frac)

    # ---------- Polygon finished ----------
    def on_polygon_finished(self, pts_preview):
        if len(pts_preview) < 3:
            QMessageBox.warning(self, "Crop", "Need at least 3 points."); self.image_label.cancel_selection(); return
        (prev_w, prev_h) = self.preview_size; (full_h, full_w, _c) = self.full_shape
        sx = full_w / float(prev_w); sy = full_h / float(prev_h)
        poly_full = [(x*sx, y*sy) for (x, y) in pts_preview]

        base, _ = os.path.splitext(self.current_path)
        out_tif, _ = QFileDialog.getSaveFileName(self, "Save clipped TIFF as", f"{base}_clipped.tif",
                                                 "TIFF Images (*.tif *.tiff)")
        if not out_tif: self.image_label.cancel_selection(); return
        out_shp, _ = QFileDialog.getSaveFileName(self, "Save polygon shapefile as", f"{base}_poly.shp",
                                                 "ESRI Shapefile (*.shp)")
        if not out_shp: self.image_label.cancel_selection(); return

        try:
            self._apply_polygon_and_save(poly_full, out_tif, out_shp)
            QMessageBox.information(self, "Done", f"Saved:\n{out_tif}\n{out_shp}\n(.dbf/.shx written alongside)")
        except Exception as e:
            log_exc(); QMessageBox.critical(self, "Error", f"Cropping failed:\n{e}")
        finally:
            self.image_label.cancel_selection()

    # ---------- Core masking & saving ----------
    def _apply_polygon_and_save(self, poly_full_xy, out_tif, out_shp):
        if self.current_path is None or self.full_shape is None:
            raise RuntimeError("No image loaded.")
        H, W, C = self.full_shape
        ring = [(float(x), float(y)) for (x, y) in poly_full_xy]
        if ring[0] != ring[-1]: ring.append(ring[0])
        mask_img = PILImage.new('L', (W, H), 0); ImageDraw.Draw(mask_img, 'L').polygon(ring, outline=1, fill=1)
        mask = np.array(mask_img, dtype=np.uint8).astype(bool)

        with tifffile.TiffFile(self.current_path) as tif:
            ser = tif.series[0]; src_page = tif.pages[0]
            data = ser.asarray(maxworkers=1); geo_snapshot = _snapshot_geo_tags(src_page)

        if data.ndim == 2: data = data[:, :, None]
        elif data.ndim == 3 and data.shape[0] in (1,3,4) and data.shape[-1] not in (1,3,4):
            data = np.moveaxis(data, 0, -1)

        data_i16 = data.astype(np.int16, copy=False); m = mask[:, :, None]; data_i16[~m] = NODATA_VALUE
        extratags = _build_extratags_from_snapshot(geo_snapshot, NODATA_VALUE)
        comp = 'deflate' if (imagecodecs is not None) else None
        big = (H * W * data_i16.shape[-1] * 2) > 2_000_000_000
        photometric = 'minisblack' if data_i16.shape[-1] == 1 else 'rgb'

        tifffile.imwrite(out_tif, data_i16 if data_i16.shape[-1] > 1 else data_i16[:, :, 0],
                         bigtiff=big, compression=comp, dtype=np.int16,
                         photometric=photometric, extratags=extratags)

        aff, wkt = _get_affine_and_wkt_from_raster(self.current_path)
        if aff is None:
            with tifffile.TiffFile(self.current_path) as tif:
                src_page = tif.pages[0]
                aff = _get_affine_from_tifftags(src_page)
                if wkt is None: wkt = _extract_wkt_from_tifftags(src_page)

        ring_map = ([_pixel_to_map_xy(aff, x, y) for (x, y) in ring] if aff is not None else ring)
        shp_base, ext = os.path.splitext(out_shp)
        if ext.lower() != '.shp':
            shp_base = out_shp
        w = shapefile.Writer(shp_base, shapeType=shapefile.POLYGON)
        w.field('id', 'N'); w.poly([ring_map]); w.record(1); w.close()
        _write_prj(shp_base, wkt)

    # ---------- Legend & Scale ----------
    def _make_lut(self):
        if HAVE_MPL:
            try: cmap = mpl_cm.colormaps.get_cmap(self.cmap_name, 256)
            except Exception: cmap = mpl_cm.get_cmap(self.cmap_name, 256)
            lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255.0 + 0.5).astype(np.uint8); return lut
        if HAVE_CV2:
            code = CV2_CMAPS.get(self.cmap_name, None); base = np.arange(256, dtype=np.uint8)
            if self.cmap_name == "gray" or code is None: return np.stack([base, base, base], axis=1)
            bgr = cv2.applyColorMap(base, code); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); return rgb.astype(np.uint8)
        base = np.arange(256, dtype=np.uint8); return np.stack([base, base, base], axis=1)

    def _build_legend_pixmap(self, low_val: float, high_val: float, disp_scale: float = None):
        if self.preview_raw is None: return None
        pad = 8; bar_h = 16; tick_h = 6; scale_line_thick = 12
        title_font = QFont(self.font()); title_font.setBold(True); title_font.setPointSize(title_font.pointSize() + 2)
        label_font = QFont(self.font()); label_font.setBold(True); label_font.setPointSize(label_font.pointSize() + 1)
        tmp = QImage(1, 1, QImage.Format.Format_ARGB32); p = QPainter(tmp)
        p.setFont(title_font); fm_title = p.fontMetrics(); p.setFont(label_font); fm_label = p.fontMetrics(); p.end()
        title_h = fm_title.height(); label_h = fm_label.height()
        width = 240; x0, x1 = pad, width - pad; bar_w = x1 - x0
        height = title_h + pad + bar_h + pad + tick_h + label_h + pad + fm_title.height() + scale_line_thick
        img = QImage(width, height, QImage.Format.Format_ARGB32); img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img); p.setRenderHint(QPainter.Antialiasing, True)
        p.setFont(title_font); p.setPen(QColor(20, 20, 20))
        p.drawText(QRect(0, 0, width, title_h), Qt.AlignmentFlag.AlignCenter, "T(°C)")
        y_bar = title_h + pad; lut = self._make_lut()
        strip = np.zeros((bar_h, 256, 3), dtype=np.uint8); strip[:] = lut[np.arange(256)]
        strip_img = QImage(strip.data, 256, bar_h, 3 * 256, QImage.Format.Format_RGB888).copy()
        bar_pix = QPixmap.fromImage(strip_img).scaled(bar_w, bar_h); p.drawPixmap(x0, y_bar, bar_pix)
        p.setPen(QPen(QColor(0, 0, 0), 1)); p.setFont(label_font)
        y_tick = y_bar + bar_h; y_label = y_tick + tick_h + fm_label.ascent()
        def draw_tick(frac):
            xt = int(x0 + frac * bar_w); p.drawLine(xt, y_tick, xt, y_tick + tick_h)
            v = low_val + frac * (high_val - low_val); lbl = f"{int(round(v))}"
            tw = fm_label.horizontalAdvance(lbl); p.drawText(xt - tw // 2, y_label, lbl)
        for f in (0.0, 0.25, 0.5, 0.75, 1.0): draw_tick(f)
        length_text = "N/A"
        if self.m_per_px_x and np.isfinite(self.m_per_px_x) and self.m_per_px_x > 0:
            ds = disp_scale if (disp_scale is not None) else (self._scale or 1.0)
            preview_px = (bar_w / max(ds, 1e-9))
            prev_w = self.preview_size[0] if (self.preview_size and len(self.preview_size) >= 1) else None
            full_w = self.full_shape[1] if (self.full_shape and len(self.full_shape) >= 2) else None
            scale_preview_to_full = (float(full_w) / float(prev_w)) if (prev_w and full_w) else 1.0
            img_pixels = preview_px * scale_preview_to_full; meters = img_pixels * self.m_per_px_x
            length_text = f"{meters:.0f} m"
        p.setFont(title_font); p.setPen(QColor(20, 20, 20))
        extra_gap = int(fm_title.height() * 0.6); y_scale_label = y_label + pad + extra_gap
        p.drawText(QRect(0, y_scale_label, width, fm_title.height()), Qt.AlignmentFlag.AlignCenter, length_text)
        y_scale_line_top = y_scale_label + fm_title.height() + (pad // 2)
        p.setPen(QPen(QColor(0, 0, 0), scale_line_thick, Qt.SolidLine, Qt.FlatCap))
        p.drawLine(x0, y_scale_line_top + scale_line_thick // 2, x0 + bar_w, y_scale_line_top + scale_line_thick // 2)
        p.end(); return QPixmap.fromImage(img)

    def _update_legend(self, low_frac: float, high_frac: float):
        if self.preview_raw is None: self.legend.hide(); return
        low_val = self.preview_min + low_frac * (self.preview_max - self.preview_min)
        high_val = self.preview_min + high_frac * (self.preview_max - self.preview_min)
        if not np.isfinite(low_val) or not np.isfinite(high_val) or high_val <= low_val: self.legend.hide(); return
        pm = self._build_legend_pixmap(low_val, high_val, self._scale or 1.0)
        if pm is None: self.legend.hide(); return
        self.legend.setPixmap(pm); self.legend.adjustSize(); self.legend.move(12, 12)
        self.legend.show(); self.legend.raise_()

    def _compute_m_per_px_x(self):
        try:
            aff, wkt = _get_affine_and_wkt_from_raster(self.current_path)
            if aff is None:
                with tifffile.TiffFile(self.current_path) as tif:
                    src_page = tif.pages[0]; aff = _get_affine_from_tifftags(src_page)
                    if wkt is None: wkt = _extract_wkt_from_tifftags(src_page)
            if aff is None: self.m_per_px_x = None; return
            if HAVE_RASTERIO and hasattr(aff, "a"): a, b, c, d, e, f = aff.a, aff.b, aff.c, aff.d, aff.e, aff.f
            else: a, b, c, d, e, f = aff
            dx_units = float((a*a + d*d) ** 0.5)
            if not np.isfinite(dx_units) or dx_units <= 0: self.m_per_px_x = None; return
            wkt_str = (wkt or ""); is_geographic = ("deg" in wkt_str.lower()) or ("geogcs" in wkt_str.lower()) or ("geogcrs" in wkt_str.lower())
            if is_geographic:
                H, W, _ = self.full_shape; lon, lat = _pixel_to_map_xy(aff, W/2.0, H/2.0)
                meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat)); self.m_per_px_x = dx_units * meters_per_deg_lon
            else: self.m_per_px_x = dx_units
        except Exception:
            log_exc(); self.m_per_px_x = None

    # ---------- Colormap change ----------
    def _on_cmap_changed(self, name: str):
        self.cmap_name = name
        if self.preview_raw is None or self.preview_min is None: return
        low = self.range.lowValue(); high = self.range.highValue()
        low_frac  = low  / self.range.maximum(); high_frac = high / self.range.maximum()
        rgba8 = _render_window_to_rgba(self.preview_raw, self.preview_valid,
                                       self.preview_min, self.preview_max,
                                       low_frac, high_frac, self.cmap_name)
        rgba8 = self._composite_with_background(rgba8)
        qimg = np_to_qimage(rgba8); self._pixmap = QPixmap.fromImage(qimg)
        self.preview_size = (rgba8.shape[1], rgba8.shape[0]); self._set_scaled_pixmap()
        self._update_legend(low_frac, high_frac)

    # ---------- Background compositor ----------
    def _composite_with_background(self, rgba8_fg):
        if self.bg_full_rgb is None or self.bg_valid is None: return rgba8_fg
        ph, pw = rgba8_fg.shape[0], rgba8_fg.shape[1]
        try:
            if HAVE_CV2:
                bg_resized = cv2.resize(self.bg_full_rgb, (pw, ph), interpolation=cv2.INTER_AREA)
                mask_resized = cv2.resize(self.bg_valid.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
            else: raise RuntimeError("cv2 not available")
        except Exception:
            from PIL import Image as _PIL
            bg_resized = np.array(_PIL.fromarray(self.bg_full_rgb).resize((pw, ph), _PIL.BILINEAR))
            mask_resized = np.array(_PIL.fromarray(self.bg_valid.astype(np.uint8)).resize((pw, ph), _PIL.NEAREST)) > 0
        out = rgba8_fg.copy(); fg_alpha = rgba8_fg[..., 3] > 0; use_bg = (~fg_alpha) & mask_resized
        out[use_bg, :3] = bg_resized[use_bg]; out[use_bg, 3] = 255; return out

    # ---------- Align overlay ----------
    def _update_align_overlay(self, text: str):
        self.align_label.setText(text); self.align_label.adjustSize()
        pad = 12
        x = max(pad, self.image_label.width() - self.align_label.width() - pad)
        y = max(pad, self.image_label.height() - self.align_label.height() - pad)
        self.align_label.move(int(x), int(y)); self.align_label.show(); self.align_label.raise_()

    # ---------- Range slider callback ----------
    def _on_range_changed(self, low, high=None):
        if high is None: low, high = self.range.lowValue(), self.range.highValue()
        if self.preview_raw is None or self.preview_min is None: return
        low_frac  = low  / self.range.maximum(); high_frac = high / self.range.maximum()
        rgba8 = _render_window_to_rgba(self.preview_raw, self.preview_valid,
                                       self.preview_min, self.preview_max,
                                       low_frac, high_frac, self.cmap_name)
        rgba8 = self._composite_with_background(rgba8)
        qimg = np_to_qimage(rgba8); self._pixmap = QPixmap.fromImage(qimg)
        self.preview_size = (rgba8.shape[1], rgba8.shape[0]); self._set_scaled_pixmap()
        self._update_legend(low_frac, high_frac)

    # ---------- Polygon finished ----------
    def on_polygon_finished(self, pts_preview):
        if len(pts_preview) < 3:
            QMessageBox.warning(self, "Crop", "Need at least 3 points."); self.image_label.cancel_selection(); return
        (prev_w, prev_h) = self.preview_size; (full_h, full_w, _c) = self.full_shape
        sx = full_w / float(prev_w); sy = full_h / float(prev_h)
        poly_full = [(x*sx, y*sy) for (x, y) in pts_preview]

        base, _ = os.path.splitext(self.current_path)
        out_tif, _ = QFileDialog.getSaveFileName(self, "Save clipped TIFF as", f"{base}_clipped.tif",
                                                 "TIFF Images (*.tif *.tiff)")
        if not out_tif: self.image_label.cancel_selection(); return
        out_shp, _ = QFileDialog.getSaveFileName(self, "Save polygon shapefile as", f"{base}_poly.shp",
                                                 "ESRI Shapefile (*.shp)")
        if not out_shp: self.image_label.cancel_selection(); return

        try:
            self._apply_polygon_and_save(poly_full, out_tif, out_shp)
            QMessageBox.information(self, "Done", f"Saved:\n{out_tif}\n{out_shp}\n(.dbf/.shx written alongside)")
        except Exception as e:
            log_exc(); QMessageBox.critical(self, "Error", f"Cropping failed:\n{e}")
        finally:
            self.image_label.cancel_selection()

    # ---------- Buttons (end) ----------

    # ---------- Core masking & saving ----------
    def _apply_polygon_and_save(self, poly_full_xy, out_tif, out_shp):
        if self.current_path is None or self.full_shape is None:
            raise RuntimeError("No image loaded.")
        H, W, C = self.full_shape
        ring = [(float(x), float(y)) for (x, y) in poly_full_xy]
        if ring[0] != ring[-1]: ring.append(ring[0])
        mask_img = PILImage.new('L', (W, H), 0); ImageDraw.Draw(mask_img, 'L').polygon(ring, outline=1, fill=1)
        mask = np.array(mask_img, dtype=np.uint8).astype(bool)

        with tifffile.TiffFile(self.current_path) as tif:
            ser = tif.series[0]; src_page = tif.pages[0]
            data = ser.asarray(maxworkers=1); geo_snapshot = _snapshot_geo_tags(src_page)

        if data.ndim == 2: data = data[:, :, None]
        elif data.ndim == 3 and data.shape[0] in (1,3,4) and data.shape[-1] not in (1,3,4):
            data = np.moveaxis(data, 0, -1)

        data_i16 = data.astype(np.int16, copy=False); m = mask[:, :, None]; data_i16[~m] = NODATA_VALUE
        extratags = _build_extratags_from_snapshot(geo_snapshot, NODATA_VALUE)
        comp = 'deflate' if (imagecodecs is not None) else None
        big = (H * W * data_i16.shape[-1] * 2) > 2_000_000_000
        photometric = 'minisblack' if data_i16.shape[-1] == 1 else 'rgb'

        tifffile.imwrite(out_tif, data_i16 if data_i16.shape[-1] > 1 else data_i16[:, :, 0],
                         bigtiff=big, compression=comp, dtype=np.int16,
                         photometric=photometric, extratags=extratags)

        aff, wkt = _get_affine_and_wkt_from_raster(self.current_path)
        if aff is None:
            with tifffile.TiffFile(self.current_path) as tif:
                src_page = tif.pages[0]
                aff = _get_affine_from_tifftags(src_page)
                if wkt is None: wkt = _extract_wkt_from_tifftags(src_page)

        ring_map = ([_pixel_to_map_xy(aff, x, y) for (x, y) in ring] if aff is not None else ring)
        shp_base, ext = os.path.splitext(out_shp)
        if ext.lower() != '.shp':
            shp_base = out_shp
        w = shapefile.Writer(shp_base, shapeType=shapefile.POLYGON)
        w.field('id', 'N'); w.poly([ring_map]); w.record(1); w.close()
        _write_prj(shp_base, wkt)

# ------------------------ Entrypoint ------------------------

def main():
    try:
        app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())
    except Exception:
        log_exc(); print("Fatal error. See log:", os.path.abspath(LOG_FILE))

if __name__ == "__main__":
    main()
