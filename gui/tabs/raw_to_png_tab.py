import os
import glob
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QComboBox, 
    QGroupBox, QCheckBox, QSplitter, QSizePolicy, QMessageBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPainter, QPen, QColor, QImage, QPixmap, QIcon

CAMERA_MODELS = {
    "Auto-Detect": (0, 0),
    "FLIR A35 (320x256)": (320, 256),
    "FLIR A65 (640x512)": (640, 512),
    "FLIR A70 (640x480)": (640, 480),
    "FLIR A6705 (640x512)": (640, 512),
    "CALIBIR 320 (320x240)": (320, 240),
    "CALIBIR 640 (640x480)": (640, 480),
    "FLIR A50 (464x348)": (464, 348)
}

class PreviewCanvas(QLabel):
    roi_changed = pyqtSignal(tuple) # (x, y, w, h)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #000; border: 1px solid #444;")
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)  # Receive mouseMoveEvent even when no button pressed
        
        self.rubber_band_start = None
        self.rubber_band_end = None
        self.roi = None # (x, y, w, h) in image coords
        self.image_pixmap = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.img_w = 0
        self.img_h = 0
        
        # Crosshair / pixel probe state
        self._hover_pos = None        # QPoint in widget coords
        self._img_array = None        # numpy RGB array for pixel sampling

        # Zoom / pan state
        self._zoom = 1.0              # current zoom multiplier (1.0 = fit-to-window)
        self._pan_x = 0.0             # pan offset in image pixels (X)
        self._pan_y = 0.0             # pan offset in image pixels (Y)
        self._pan_mode = False        # toggle: True = pan mode active
        self._pan_last = None         # last mouse widget pos for incremental pan

    def set_image(self, img_rgb):
        self.img_h, self.img_w = img_rgb.shape[:2]
        self._img_array = img_rgb.copy()  # keep a numpy copy for pixel sampling
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_pixmap = QPixmap.fromImage(qimg)
        # Reset ROI, zoom and pan on new image load
        self.roi = (0, 0, w, h)
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.roi_changed.emit(self.roi)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.image_pixmap:
            painter = QPainter(self)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Preview Available\nSelect folder to load preview")
            return

        painter = QPainter(self)
        
        widget_w, widget_h = self.width(), self.height()
        # Base fit-scale
        base_scale_x = widget_w / self.img_w
        base_scale_y = widget_h / self.img_h
        base_scale   = min(base_scale_x, base_scale_y)
        
        # Apply user zoom on top of the base fit
        self.scale = base_scale * self._zoom

        draw_w = int(self.img_w * self.scale)
        draw_h = int(self.img_h * self.scale)

        self.offset_x = int((widget_w - draw_w) / 2.0 - self._pan_x * self.scale)
        self.offset_y = int((widget_h - draw_h) / 2.0 - self._pan_y * self.scale)

        # Show cursor hint based on zoom state, handled mostly by events now
        if not self._pan_mode:
            self.setCursor(Qt.CrossCursor if (self.image_pixmap and self._zoom > 1.0) else Qt.ArrowCursor)

        painter.drawPixmap(self.offset_x, self.offset_y, draw_w, draw_h, self.image_pixmap)

        # Draw ROI Rectangle (dimming the outside)
        if self.roi:
            x, y, w, h = self.roi
            rx = int(x * self.scale) + self.offset_x
            ry = int(y * self.scale) + self.offset_y
            rw = int(w * self.scale)
            rh = int(h * self.scale)
            
            # Dim outskirts
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRect(self.offset_x, self.offset_y, draw_w, ry - self.offset_y) # Top
            painter.drawRect(self.offset_x, ry + rh, draw_w, self.offset_y + draw_h - (ry + rh)) # Bottom
            painter.drawRect(self.offset_x, ry, rx - self.offset_x, rh) # Left
            painter.drawRect(rx + rw, ry, self.offset_x + draw_w - (rx + rw), rh) # Right
            
            # Draw green crop border
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rx, ry, rw, rh)

        # Draw current drag rectangle
        if self.rubber_band_start and self.rubber_band_end:
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            rx = min(self.rubber_band_start.x(), self.rubber_band_end.x())
            ry = min(self.rubber_band_start.y(), self.rubber_band_end.y())
            rw = abs(self.rubber_band_start.x() - self.rubber_band_end.x())
            rh = abs(self.rubber_band_start.y() - self.rubber_band_end.y())
            painter.drawRect(rx, ry, rw, rh)

        # Draw crosshair + pixel info on hover
        if self._hover_pos and self.image_pixmap and not self.rubber_band_start:
            hx = self._hover_pos.x()
            hy = self._hover_pos.y()

            # Only draw if cursor is over the actual image area
            img_left   = self.offset_x
            img_top    = self.offset_y
            img_right  = self.offset_x + int(self.img_w * self.scale)
            img_bottom = self.offset_y + int(self.img_h * self.scale)

            if img_left <= hx <= img_right and img_top <= hy <= img_bottom:
                # Crosshair lines (full canvas width/height, cyan)
                painter.setPen(QPen(QColor(0, 220, 220), 1, Qt.SolidLine))
                painter.drawLine(0, hy, self.width(), hy)   # horizontal
                painter.drawLine(hx, 0, hx, self.height())  # vertical

                # Convert widget pos to image pixel coords
                px = int((hx - self.offset_x) / self.scale)
                py = int((hy - self.offset_y) / self.scale)
                px = max(0, min(self.img_w - 1, px))
                py = max(0, min(self.img_h - 1, py))

                # Sample the RGB from our stored numpy array
                if self._img_array is not None:
                    r, g, b = self._img_array[py, px]
                    label = f"{px},{py}  {r},{g},{b}"
                else:
                    label = f"{px},{py}"

                # Draw info label next to cursor (dark background, white text)
                font = painter.font()
                font.setPointSize(9)
                font.setBold(True)
                painter.setFont(font)

                fm = painter.fontMetrics()
                text_w = fm.horizontalAdvance(label) + 10
                text_h = fm.height() + 6

                # Position label to avoid going off-screen
                lx = hx + 12
                ly = hy - text_h - 4
                if lx + text_w > self.width():  lx = hx - text_w - 8
                if ly < 0:                       ly = hy + 8

                # Semi-transparent dark background pill
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 0, 170))
                painter.drawRoundedRect(lx - 2, ly, text_w, text_h, 3, 3)

                # White text
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(lx + 3, ly + text_h - 5, label)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton and self.image_pixmap:
            # Hold middle-click to pan
            self._pan_mode = True
            self._pan_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self.update()
        elif event.button() == Qt.LeftButton and self.image_pixmap:
            self.rubber_band_start = event.pos()
            self.rubber_band_end = event.pos()
            self.update()
        elif event.button() == Qt.RightButton and self.image_pixmap:
            # Right click resets ROI to full image
            self.roi = (0, 0, self.img_w, self.img_h)
            self.roi_changed.emit(self.roi)
            self.update()

    def mouseMoveEvent(self, event):
        self._hover_pos = event.pos()
        if self._pan_mode and self._pan_last is not None:
            # Incremental pan: move by delta since last position
            dx = (event.pos().x() - self._pan_last.x()) / self.scale
            dy = (event.pos().y() - self._pan_last.y()) / self.scale
            self._pan_x -= dx
            self._pan_y -= dy
            self._pan_last = event.pos()
        elif self.rubber_band_start:
            self.rubber_band_end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_mode = False
            self._pan_last = None
            self.setCursor(Qt.CrossCursor if (self.image_pixmap and self._zoom > 1.0) else Qt.ArrowCursor)
            self.update()
            
        elif event.button() == Qt.LeftButton and self.rubber_band_start:
            self.rubber_band_end = event.pos()
            
            # Convert widget coords to image coords
            x1 = (min(self.rubber_band_start.x(), self.rubber_band_end.x()) - self.offset_x) / self.scale
            y1 = (min(self.rubber_band_start.y(), self.rubber_band_end.y()) - self.offset_y) / self.scale
            x2 = (max(self.rubber_band_start.x(), self.rubber_band_end.x()) - self.offset_x) / self.scale
            y2 = (max(self.rubber_band_start.y(), self.rubber_band_end.y()) - self.offset_y) / self.scale
            
            x1 = max(0, min(self.img_w - 1, int(round(x1))))
            y1 = max(0, min(self.img_h - 1, int(round(y1))))
            x2 = max(0, min(self.img_w - 1, int(round(x2))))
            y2 = max(0, min(self.img_h - 1, int(round(y2))))
            
            w = x2 - x1
            h = y2 - y1
            if w > 10 and h > 10:
                self.roi = (x1, y1, w, h)
                self.roi_changed.emit(self.roi)
            
            self.rubber_band_start = None
            self.rubber_band_end = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        """Double-click to fit image back to screen."""
        if event.button() == Qt.LeftButton:
            self.fit_to_screen()

    def fit_to_screen(self):
        """Reset zoom and pan so the image fits the widget."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_mode = False
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def leaveEvent(self, event):
        self._hover_pos = None
        self._pan_last = None
        self.update()

    def wheelEvent(self, event):
        """Scroll to zoom. Zooms toward the exact pixel under the cursor."""
        if not self.image_pixmap or self.img_w == 0:
            return

        hx = event.pos().x()
        hy = event.pos().y()

        # Reconstruct exact floating-point offsets to prevent integer truncation drift
        widget_w, widget_h = self.width(), self.height()
        base_scale = min(widget_w / self.img_w, widget_h / self.img_h)
        cur_scale  = base_scale * self._zoom
        
        cur_draw_w = int(self.img_w * cur_scale)
        cur_draw_h = int(self.img_h * cur_scale)
        base_off_x = (widget_w - cur_draw_w) / 2.0
        base_off_y = (widget_h - cur_draw_h) / 2.0

        exact_offset_x = base_off_x - self._pan_x * cur_scale
        exact_offset_y = base_off_y - self._pan_y * cur_scale

        # Exact floating-point image pixel currently under cursor
        ix = (hx - exact_offset_x) / cur_scale
        iy = (hy - exact_offset_y) / cur_scale

        # Apply zoom
        factor   = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._zoom = max(0.5, min(20.0, self._zoom * factor))

        # --- Compute new geometry ---
        new_scale  = base_scale * self._zoom
        new_draw_w = int(self.img_w * new_scale)
        new_draw_h = int(self.img_h * new_scale)
        new_base_off_x = (widget_w - new_draw_w) / 2.0
        new_base_off_y = (widget_h - new_draw_h) / 2.0

        # Solve exactly for new pan without truncation error
        self._pan_x = (new_base_off_x - hx) / new_scale + ix
        self._pan_y = (new_base_off_y - hy) / new_scale + iy

        self.update()
        event.accept()


class ConverterWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_batch = pyqtSignal()

    def __init__(self, input_dir, output_dir, file_exts, colormap, norm_mode, downscale, camera_model, roi):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_exts = file_exts
        self.colormap = colormap
        self.norm_mode = norm_mode
        self.downscale = downscale
        self.camera_model = camera_model
        self.roi = roi
        self.is_running = True
        
        self.colormap_dict = {
            "Inferno": cv2.COLORMAP_INFERNO,
            "Viridis": cv2.COLORMAP_VIRIDIS,
            "Plasma": cv2.COLORMAP_PLASMA,
            "Magma": cv2.COLORMAP_MAGMA,
            "Jet": cv2.COLORMAP_JET
        }
        
        # Build the exact C++ custom palette as a (433, 3) BGR numpy array
        self._cpp_palette = self._build_cpp_palette()
    
    @staticmethod
    def _build_cpp_palette():
        """Store the exact 433-entry palette from ThermalImageReader.cpp as a
        (433, 3) BGR numpy array for direct pixel-level index lookup."""
        palette_hex = [
            0x00000a, 0x000014, 0x00001e, 0x000025, 0x00002a,
            0x00002e, 0x000032, 0x000036, 0x00003a, 0x00003e,
            0x000042, 0x000046, 0x00004a, 0x00004f, 0x000052,
            0x010055, 0x010057, 0x020059, 0x02005c, 0x03005e,
            0x040061, 0x040063, 0x050065, 0x060067, 0x070069,
            0x08006b, 0x09006e, 0x0a0070, 0x0b0073, 0x0c0074,
            0x0d0075, 0x0d0076, 0x0e0077, 0x100078, 0x120079,
            0x13007b, 0x15007c, 0x17007d, 0x19007e, 0x1b0080,
            0x1c0081, 0x1e0083, 0x200084, 0x220085, 0x240086,
            0x260087, 0x280089, 0x2a0089, 0x2c008a, 0x2e008b,
            0x30008c, 0x32008d, 0x34008e, 0x36008e, 0x38008f,
            0x390090, 0x3b0091, 0x3c0092, 0x3e0093, 0x3f0093,
            0x410094, 0x420095, 0x440095, 0x450096, 0x470096,
            0x490096, 0x4a0096, 0x4c0097, 0x4e0097, 0x4f0097,
            0x510097, 0x520098, 0x540098, 0x560098, 0x580099,
            0x5a0099, 0x5c0099, 0x5d009a, 0x5f009a, 0x61009b,
            0x63009b, 0x64009b, 0x66009b, 0x68009b, 0x6a009b,
            0x6c009c, 0x6d009c, 0x6f009c, 0x70009c, 0x71009d,
            0x73009d, 0x75009d, 0x77009d, 0x78009d, 0x7a009d,
            0x7c009d, 0x7e009d, 0x7f009d, 0x81009d, 0x83009d,
            0x84009d, 0x86009d, 0x87009d, 0x89009d, 0x8a009d,
            0x8b009d, 0x8d009d, 0x8f009c, 0x91009c, 0x93009c,
            0x95009c, 0x96009b, 0x98009b, 0x99009b, 0x9b009b,
            0x9c009b, 0x9d009b, 0x9f009b, 0xa0009b, 0xa2009b,
            0xa3009b, 0xa4009b, 0xa6009a, 0xa7009a, 0xa8009a,
            0xa90099, 0xaa0099, 0xab0099, 0xad0099, 0xae0198,
            0xaf0198, 0xb00198, 0xb00198, 0xb10197, 0xb20197,
            0xb30196, 0xb40296, 0xb50295, 0xb60295, 0xb70395,
            0xb80395, 0xb90495, 0xba0495, 0xba0494, 0xbb0593,
            0xbc0593, 0xbd0593, 0xbe0692, 0xbf0692, 0xbf0692,
            0xc00791, 0xc00791, 0xc10890, 0xc10990, 0xc20a8f,
            0xc30a8e, 0xc30b8e, 0xc40c8d, 0xc50c8c, 0xc60d8b,
            0xc60e8a, 0xc70f89, 0xc81088, 0xc91187, 0xca1286,
            0xca1385, 0xcb1385, 0xcb1484, 0xcc1582, 0xcd1681,
            0xce1780, 0xce187e, 0xcf187c, 0xcf197b, 0xd01a79,
            0xd11b78, 0xd11c76, 0xd21c75, 0xd21d74, 0xd31e72,
            0xd32071, 0xd4216f, 0xd4226e, 0xd5236b, 0xd52469,
            0xd62567, 0xd72665, 0xd82764, 0xd82862, 0xd92a60,
            0xda2b5e, 0xda2c5c, 0xdb2e5a, 0xdb2f57, 0xdc2f54,
            0xdd3051, 0xdd314e, 0xde324a, 0xde3347, 0xdf3444,
            0xdf3541, 0xdf363d, 0xe0373a, 0xe03837, 0xe03933,
            0xe13a30, 0xe23b2d, 0xe23c2a, 0xe33d26, 0xe33e23,
            0xe43f20, 0xe4411d, 0xe4421c, 0xe5431b, 0xe54419,
            0xe54518, 0xe64616, 0xe74715, 0xe74814, 0xe74913,
            0xe84a12, 0xe84c10, 0xe84c0f, 0xe94d0e, 0xe94d0d,
            0xea4e0c, 0xea4f0c, 0xeb500b, 0xeb510a, 0xeb520a,
            0xeb5309, 0xec5409, 0xec5608, 0xec5708, 0xec5808,
            0xed5907, 0xed5a07, 0xed5b06, 0xee5c06, 0xee5c05,
            0xee5d05, 0xee5e05, 0xef5f04, 0xef6004, 0xef6104,
            0xef6204, 0xf06303, 0xf06403, 0xf06503, 0xf16603,
            0xf16603, 0xf16703, 0xf16803, 0xf16902, 0xf16a02,
            0xf16b02, 0xf16b02, 0xf26c01, 0xf26d01, 0xf26e01,
            0xf36f01, 0xf37001, 0xf37101, 0xf37201, 0xf47300,
            0xf47400, 0xf47500, 0xf47600, 0xf47700, 0xf47800,
            0xf47a00, 0xf57b00, 0xf57c00, 0xf57e00, 0xf57f00,
            0xf68000, 0xf68100, 0xf68200, 0xf78300, 0xf78400,
            0xf78500, 0xf78600, 0xf88700, 0xf88800, 0xf88800,
            0xf88900, 0xf88a00, 0xf88b00, 0xf88c00, 0xf98d00,
            0xf98d00, 0xf98e00, 0xf98f00, 0xf99000, 0xf99100,
            0xf99200, 0xf99300, 0xfa9400, 0xfa9500, 0xfa9600,
            0xfb9800, 0xfb9900, 0xfb9a00, 0xfb9c00, 0xfc9d00,
            0xfc9f00, 0xfca000, 0xfca100, 0xfda200, 0xfda300,
            0xfda400, 0xfda600, 0xfda700, 0xfda800, 0xfdaa00,
            0xfdab00, 0xfdac00, 0xfdad00, 0xfdae00, 0xfeaf00,
            0xfeb000, 0xfeb100, 0xfeb200, 0xfeb300, 0xfeb400,
            0xfeb500, 0xfeb600, 0xfeb800, 0xfeb900, 0xfeb900,
            0xfeba00, 0xfebb00, 0xfebc00, 0xfebd00, 0xfebe00,
            0xfec000, 0xfec100, 0xfec200, 0xfec300, 0xfec400,
            0xfec500, 0xfec600, 0xfec700, 0xfec800, 0xfec901,
            0xfeca01, 0xfeca01, 0xfecb01, 0xfecc02, 0xfecd02,
            0xfece03, 0xfecf04, 0xfecf04, 0xfed005, 0xfed106,
            0xfed308, 0xfed409, 0xfed50a, 0xfed60a, 0xfed70b,
            0xfed80c, 0xfed90d, 0xffda0e, 0xffda0e, 0xffdb10,
            0xffdc12, 0xffdc14, 0xffdd16, 0xffde19, 0xffde1b,
            0xffdf1e, 0xffe020, 0xffe122, 0xffe224, 0xffe226,
            0xffe328, 0xffe42b, 0xffe42e, 0xffe531, 0xffe635,
            0xffe638, 0xffe73c, 0xffe83f, 0xffe943, 0xffea46,
            0xffeb49, 0xffeb4d, 0xffec50, 0xffed54, 0xffee57,
            0xffee5b, 0xffee5f, 0xffef63, 0xffef67, 0xfff06a,
            0xfff06e, 0xfff172, 0xfff177, 0xfff17b, 0xfff280,
            0xfff285, 0xfff28a, 0xfff38e, 0xfff492, 0xfff496,
            0xfff49a, 0xfff59e, 0xfff5a2, 0xfff5a6, 0xfff6aa,
            0xfff6af, 0xfff7b3, 0xfff7b6, 0xfff8ba, 0xfff8bd,
            0xfff8c1, 0xfff8c4, 0xfff9c7, 0xfff9ca, 0xfff9cd,
            0xfffad1, 0xfffad4, 0xfffbd8, 0xfffcdb, 0xfffcdf,
            0xfffde2, 0xfffde5, 0xfffde8, 0xfffeeb, 0xfffeee,
            0xfffef1, 0xfffef4, 0xfffff6,
        ]
        # Build (N, 3) BGR array — kept at full 433 entries, no resampling
        palette = np.zeros((len(palette_hex), 3), dtype=np.uint8)
        for i, h in enumerate(palette_hex):
            palette[i] = [h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF]  # BGR
        return palette

    def _apply_cpp_palette_direct(self, img):
        """One-step C++-exact mapping: 16-bit raw → BGR color.
        Replicates CreateVisibleImage() precisely — no 8-bit intermediate,
        no palette resampling, same int() truncation as C++.
        """
        palette = self._cpp_palette
        n = len(palette) - 1  # 432 (the max palette index)

        img_float = img.astype(np.float64)

        # Mask out dead/zero-valued pixels, find min/max among valid ones
        valid = img_float > 0
        if np.any(valid):
            min_rad = float(np.min(img_float[valid]))
            max_rad = float(np.max(img_float[valid]))
        else:
            min_rad, max_rad = 0.0, 1.0

        den = max_rad - min_rad
        if den < 1e-5:
            den = 1.0

        # Compute palette index: int(432 * (pixel - min) / den)  — matches C++ exactly
        norm = (img_float - min_rad) / den           # float in [0.0, 1.0] for in-range pixels
        indices = (n * norm).astype(np.int32)        # int() truncation, same as C++
        indices = np.clip(indices, 0, n)             # safety clamp

        # Bulk palette lookup — result is (H, W, 3) BGR
        result = palette[indices]

        # Pixels below min → pure black (C++ sets these to 0)
        result[img_float < min_rad] = [0, 0, 0]

        # Pixels above max → pure white (C++ sets these to 255)
        result[img_float > max_rad] = [255, 255, 255]

        # Dead/zero pixels → pure black (excluded from mask, so set explicitly)
        result[~valid] = [0, 0, 0]

        return result.astype(np.uint8)

    def process_single_file(self, file_path):
        filename = os.path.basename(file_path)
        ext_lower = filename.lower()
        img = None

        if ext_lower.endswith('.raw'):
            # Process as binary raw thermal data (Little-Endian to fix zebra pattern)
            with open(file_path, 'rb') as f:
                raw_data = np.fromfile(f, dtype='<u2') # 16-bit Little-Endian
            
            w, h = 0, 0
            if self.camera_model == "Auto-Detect":
                total_pixels = len(raw_data)
                for model, (mw, mh) in CAMERA_MODELS.items():
                    if model == "Auto-Detect": continue
                    if mw * mh == total_pixels:
                        w, h = mw, mh
                        break
                if w == 0:
                    raise ValueError(f"Could not auto-detect resolution. Byte count: {len(raw_data)*2}")
            else:
                w, h = CAMERA_MODELS[self.camera_model]
                
            if len(raw_data) != w * h:
                raise ValueError(f"File size mismatch. Expected {w*h*2} bytes, got {len(raw_data)*2}.")
                
            img = raw_data.reshape((h, w))
        else:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Failed to decode image data.")

        # Apply ROI Crop
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            rx = max(0, min(img.shape[1]-1, rx))
            ry = max(0, min(img.shape[0]-1, ry))
            rw = min(img.shape[1]-rx, rw)
            rh = min(img.shape[0]-ry, rh)
            if rw > 0 and rh > 0:
                img = img[ry:ry+rh, rx:rx+rw]

        # Convert to single channel
        if len(img.shape) == 3:
            img = img[:, :, 0] 

        # Normalisation — and C++ Original direct path
        if self.colormap == "C++ Original" and img.dtype != np.uint8:
            # ONE-STEP: skip 8-bit intermediate entirely.
            # Maps 16-bit raw → BGR color directly, using the same int() truncation as C++.
            result_img = self._apply_cpp_palette_direct(img)
        else:
            # Standard two-step path (normalize to 8-bit, then apply colormap)
            if img.dtype == np.uint8:
                img_8bit = img
            else:
                if self.norm_mode == "Original":
                    # Masked min-max, no padding
                    img_float = img.astype(np.float32)
                    mask = img_float > 0
                    if np.any(mask):
                        min_rad = float(np.min(img_float[mask]))
                        max_rad = float(np.max(img_float[mask]))
                    else:
                        min_rad = 0.0
                        max_rad = 1.0
                    den = max_rad - min_rad
                    if den < 1e-5: den = 1.0
                    img_8bit = np.clip((img_float - min_rad) / den * 255.0, 0, 255).astype(np.uint8)
                elif self.norm_mode == "1st-99th Percentile (Robust Contrast)":
                    img_float = img.astype(np.float32)
                    min_val = np.percentile(img_float, 1)
                    max_val = np.percentile(img_float, 99)
                    den = max_val - min_val
                    if den < 1e-5: den = 1.0
                    img_8bit = np.clip((img_float - min_val) / den * 255.0, 0, 255).astype(np.uint8)
                elif self.norm_mode == "Min-Max (Auto Contrast)":
                    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else:
                    img_float = img.astype(np.float32)
                    img_8bit = np.clip((img_float / 65535.0) * 255.0, 0, 255).astype(np.uint8)
            
            # Apply colormap to 8-bit image
            if self.colormap != "None (Grayscale)":
                cmap_flag = self.colormap_dict.get(self.colormap, cv2.COLORMAP_INFERNO)
                result_img = cv2.applyColorMap(img_8bit, cmap_flag)
            else:
                result_img = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
            
        # Downscale
        if self.downscale:
            h_out, w_out = result_img.shape[:2]
            max_dim = 1920
            if max(h_out, w_out) > max_dim:
                scale = max_dim / max(h_out, w_out)
                result_img = cv2.resize(result_img, (int(w_out * scale), int(h_out * scale)), interpolation=cv2.INTER_AREA)

        return result_img

    def run(self):
        files = []
        for ext in self.file_exts:
            files.extend(glob.glob(os.path.join(self.input_dir, f"*.{ext}")))
            files.extend(glob.glob(os.path.join(self.input_dir, f"*.{ext.upper()}")))
        
        files = list(set(files))
        total = len(files)
        
        if total == 0:
            self.log.emit("❌ No matching files found in the directory.")
            self.finished_batch.emit()
            return
            
        self.log.emit(f"🚀 Starting conversion of {total} files...")
        os.makedirs(self.output_dir, exist_ok=True)

        for i, file_path in enumerate(files):
            if not self.is_running:
                self.log.emit("🛑 Conversion canceled by user.")
                break
                
            filename = os.path.basename(file_path)
            name, _ = os.path.splitext(filename)
            out_path = os.path.join(self.output_dir, f"{name}.png")
            
            try:
                result_img = self.process_single_file(file_path)
                cv2.imwrite(out_path, result_img)
                self.log.emit(f"✅ Converted: {filename} -> {name}.png")
            except Exception as e:
                self.log.emit(f"❌ Error processing {filename}: {str(e)}")
                
            progress_pct = int(((i + 1) / total) * 100)
            self.progress.emit(progress_pct)
            
        self.log.emit("🎉 Batch processing complete!")
        self.finished_batch.emit()

    def stop(self):
        self.is_running = False


class RawToPngTab(QWidget):
    conversion_completed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.current_roi = None
        self.preview_file = None
        self._gallery_folder = None
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ===== LEFT PANEL (Config) =====
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)

        title = QLabel("RAW Thermal to PNG")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #8ab4f8; margin-bottom: 10px;")
        left_layout.addWidget(title)

        # Directory Selection
        dir_group = QGroupBox("📁 Directories")
        dir_group.setStyleSheet(self._groupbox_style())
        dir_layout = QVBoxLayout(dir_group)
        
        self.in_dir_input = QLineEdit()
        self.in_dir_input.setPlaceholderText("Select input folder...")
        self.in_dir_input.setStyleSheet(self._input_style())
        btn_in = QPushButton("Browse")
        btn_in.setStyleSheet(self._btn_style())
        btn_in.clicked.connect(self._browse_input)
        
        row1 = QHBoxLayout()
        row1.addWidget(self.in_dir_input)
        row1.addWidget(btn_in)
        dir_layout.addLayout(row1)

        self.out_dir_input = QLineEdit()
        self.out_dir_input.setPlaceholderText("Output folder (auto-generated)")
        self.out_dir_input.setStyleSheet(self._input_style())
        btn_out = QPushButton("Browse")
        btn_out.setStyleSheet(self._btn_style())
        btn_out.clicked.connect(self._browse_output)
        
        row2 = QHBoxLayout()
        row2.addWidget(self.out_dir_input)
        row2.addWidget(btn_out)
        dir_layout.addLayout(row2)
        left_layout.addWidget(dir_group)

        # Settings Group
        settings_group = QGroupBox("⚙️ Conversion Settings")
        settings_group.setStyleSheet(self._groupbox_style())
        settings_layout = QVBoxLayout(settings_group)
        
        ext_layout = QHBoxLayout()
        ext_layout.addWidget(QLabel("Extensions:"))
        self.ext_input = QLineEdit("raw, tif, tiff, png, dcm")
        self.ext_input.setStyleSheet(self._input_style())
        ext_layout.addWidget(self.ext_input)
        settings_layout.addLayout(ext_layout)

        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Camera Model:"))
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(list(CAMERA_MODELS.keys()))
        self.cam_combo.setStyleSheet(self._combo_style())
        self.cam_combo.currentTextChanged.connect(self._refresh_preview)
        cam_layout.addWidget(self.cam_combo)
        settings_layout.addLayout(cam_layout)

        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.setToolTip("The color gradient applied over the grayscale thermal heat.")
        cmap_options = ["C++ Original", "Inferno", "Viridis", "Plasma", "Magma", "Jet", "None (Grayscale)"]
        self.cmap_combo.addItems(cmap_options)
        
        # Add beautiful tooltips to help users
        cmap_tooltips = [
            "C++ Original: The exact custom palette from the legacy COLOUR TOOL. Pixel-perfect match.",
            "Inferno: Dark purple to bright yellow. (Best for general thermal imaging)",
            "Viridis: Dark blue to yellow. (Colorblind friendly, uniform contrast)",
            "Plasma: Rich purple to bright yellow.",
            "Magma: Deep black to white-yellow. (High contrast)",
            "Jet: Classic blue to red rainbow. (Very common, but can falsely emphasize noise)",
            "None (Grayscale): Raw heat intensity plotted purely as white/black."
        ]
        for i, tip in enumerate(cmap_tooltips):
            self.cmap_combo.setItemData(i, tip, Qt.ToolTipRole)
            
        self.cmap_combo.setStyleSheet(self._combo_style())
        self.cmap_combo.currentTextChanged.connect(self._refresh_preview)
        cmap_layout.addWidget(self.cmap_combo)
        settings_layout.addLayout(cmap_layout)

        norm_layout = QHBoxLayout()
        norm_layout.addWidget(QLabel("Normalization:"))
        self.norm_combo = QComboBox()
        self.norm_combo.setToolTip("Mathematical scaling technique used to stretch temperatures into visible light.")
        norm_options = [
            "Original",
            "1st-99th Percentile (Robust Contrast)", 
            "Min-Max (Auto Contrast)", 
            "Absolute (Divide by Max)"
        ]
        self.norm_combo.addItems(norm_options)
        
        # Tooltips summarizing complex math
        norm_tooltips = [
            "Original: Exactly matches your old C++ software. Softens image by artificially padding extremes.",
            "Robust Contrast: Hides broken dead pixels by completely ignoring the top 1% and bottom 1% temperature anomalies.",
            "Auto Contrast: Mathematically stretches the absolute coldest pixel to pure black, and absolute hottest to pure white.",
            "Absolute: Divides every pixel purely by theoretical max. Retains literal heat value but often looks extremely dark."
        ]
        for i, tip in enumerate(norm_tooltips):
            self.norm_combo.setItemData(i, tip, Qt.ToolTipRole)

        self.norm_combo.setStyleSheet(self._combo_style())
        self.norm_combo.currentTextChanged.connect(self._refresh_preview)
        norm_layout.addWidget(self.norm_combo)
        settings_layout.addLayout(norm_layout)

        self.chk_downscale = QCheckBox("Downscale to 1080p limit")
        self.chk_downscale.setChecked(True)
        self.chk_downscale.setStyleSheet("color: #ccc;")
        settings_layout.addWidget(self.chk_downscale)

        # Convert / Cancel buttons — side by side, right below downscale
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ Convert")
        self.btn_run.setFixedHeight(36)
        self.btn_run.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.btn_run.setStyleSheet("""
            QPushButton { background-color: #2a6a4a; color: white; border-radius: 5px; }
            QPushButton:hover { background-color: #3b8a6a; }
            QPushButton:disabled { background-color: #333; color: #666; }
        """)
        self.btn_run.clicked.connect(self._start_conversion)

        self.btn_cancel = QPushButton("⏹ Cancel")
        self.btn_cancel.setFixedHeight(36)
        self.btn_cancel.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("""
            QPushButton { background-color: #5a2a2a; color: white; border-radius: 5px; }
            QPushButton:hover { background-color: #7a3a3a; }
            QPushButton:disabled { background-color: #333; color: #666; }
        """)
        self.btn_cancel.clicked.connect(self._cancel_conversion)

        btn_row.addWidget(self.btn_run, 3)
        btn_row.addWidget(self.btn_cancel, 2)
        settings_layout.addLayout(btn_row)

        left_layout.addWidget(settings_group)

        # ===== PNG Gallery (inside left panel, below settings) =====
        gallery_group = QGroupBox("🖼  PNG Gallery")
        gallery_group.setStyleSheet(self._groupbox_style())
        gallery_layout = QVBoxLayout(gallery_group)
        gallery_layout.setSpacing(6)

        gallery_top_row = QHBoxLayout()
        btn_browse_png = QPushButton("📂 Open PNG Folder")
        btn_browse_png.setToolTip("Load a folder of .png files into the gallery for preview")
        btn_browse_png.setStyleSheet(self._btn_style())
        btn_browse_png.clicked.connect(self._browse_gallery_folder)
        gallery_top_row.addWidget(btn_browse_png)
        gallery_layout.addLayout(gallery_top_row)

        self.gallery_list = QListWidget()
        self.gallery_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.gallery_list.setIconSize(QSize(140, 78))
        self.gallery_list.setSpacing(2)
        self.gallery_list.setFixedHeight(280)
        self.gallery_list.setStyleSheet("""
            QListWidget { background-color: #111; border: 1px solid #333; border-radius: 4px; color: #ccc; font-size: 11px; outline: none; }
            QListWidget::item { padding: 5px 4px; border-bottom: 1px solid #222; }
            QListWidget::item:selected { background-color: #1a3a5a; color: white; }
            QListWidget::item:hover { background-color: #1e1e1e; }
        """)
        self.gallery_list.itemClicked.connect(self._on_gallery_item_clicked)
        # Enable arrow key navigation
        self.gallery_list.currentRowChanged.connect(self._on_gallery_row_changed)
        self.gallery_list.setFocusPolicy(Qt.StrongFocus)
        gallery_layout.addWidget(self.gallery_list)

        gallery_footer = QHBoxLayout()
        self.gallery_count_label = QLabel("0 images")
        self.gallery_count_label.setStyleSheet("color: #666; font-size: 10px;")
        gallery_footer.addWidget(self.gallery_count_label)
        gallery_footer.addStretch()

        self.btn_open_annotate = QPushButton("✏️ Open in Annotate")
        self.btn_open_annotate.setEnabled(False)
        self.btn_open_annotate.setToolTip("Load this gallery folder into the Annotate tab")
        self.btn_open_annotate.setStyleSheet("""
            QPushButton { background-color: #1a3a6a; color: #8ab4f8; border: 1px solid #3a5a9a; border-radius: 4px; padding: 4px 8px; font-size: 10px; }
            QPushButton:hover { background-color: #2a4a8a; color: white; }
            QPushButton:disabled { background-color: #222; color: #555; border-color: #333; }
        """)
        self.btn_open_annotate.clicked.connect(self._open_gallery_in_annotate)
        gallery_footer.addWidget(self.btn_open_annotate)
        gallery_layout.addLayout(gallery_footer)

        left_layout.addWidget(gallery_group)
        left_layout.addStretch()

        # ===== RIGHT PANEL (Preview & Console) =====
        right_panel = QSplitter(Qt.Vertical)
        
        # Preview Group
        preview_group = QGroupBox("🖼️ Real-Time Crop Preview (Draw box to set Region of Interest)")
        preview_group.setStyleSheet(self._groupbox_style())
        preview_layout = QVBoxLayout(preview_group)
        self.preview_canvas = PreviewCanvas()
        self.preview_canvas.roi_changed.connect(self._on_roi_changed)
        preview_layout.addWidget(self.preview_canvas)
        
        # ROI Status and Controls
        roi_layout = QHBoxLayout()
        roi_layout.setContentsMargins(0, 0, 0, 0)
        
        self.roi_status = QLabel("Region of Interest: Full Image (Right-click to reset)")
        self.roi_status.setStyleSheet("color: #888;")
        roi_layout.addWidget(self.roi_status)
        
        roi_layout.addStretch()
        
        self.btn_fit = QPushButton("🔍 Fit to Screen")
        self.btn_fit.setStyleSheet(self._btn_style())
        self.btn_fit.setToolTip("Reset Zoom and Pan (Double-Click image also works)")
        self.btn_fit.clicked.connect(self.preview_canvas.fit_to_screen)
        roi_layout.addWidget(self.btn_fit)
        
        preview_layout.addLayout(roi_layout)
        right_panel.addWidget(preview_group)

        # Progress / Console 
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        console_layout.setContentsMargins(0, 10, 0, 0)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #333; border-radius: 4px; text-align: center; color: white; background-color: #1e1e1e; }
            QProgressBar::chunk { background-color: #8ab4f8; width: 10px; }
        """)
        console_layout.addWidget(self.progress_bar)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("""
            QTextEdit { background-color: #121212; color: #a0a0a0; font-family: Consolas, monospace; font-size: 11px; border: 1px solid #333; border-radius: 4px; padding: 10px; }
        """)
        self.console.append("System ready. Select a directory containing .raw or .tif files.")
        console_layout.addWidget(self.console)
        
        right_panel.addWidget(console_widget)
        right_panel.setSizes([700, 250]) # Give a bit of room back to the text console


        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

    def _build_gallery_panel(self):
        # Legacy stub — gallery is now inside the left panel. Safe to remove but kept for compatibility.
        pass

    def _browse_gallery_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select PNG Folder")
        if folder:
            self.load_png_folder(folder)

    def load_png_folder(self, folder):
        """Populate the gallery with all .png files from *folder*."""
        self._gallery_folder = folder
        self.gallery_list.clear()
        self.btn_open_annotate.setEnabled(False)

        png_files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not png_files:
            self.gallery_count_label.setText("0 images")
            return

        for path in png_files:
            name = os.path.basename(path)
            thumb_img = cv2.imread(path)
            if thumb_img is not None:
                th, tw = thumb_img.shape[:2]
                scale = min(140 / tw, 78 / th)
                tn = cv2.resize(thumb_img, (max(1, int(tw * scale)), max(1, int(th * scale))),
                                interpolation=cv2.INTER_AREA)
                tn_rgb = cv2.cvtColor(tn, cv2.COLOR_BGR2RGB)
                h, w, ch = tn_rgb.shape
                qimg = QImage(tn_rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
                icon = QIcon(QPixmap.fromImage(qimg))
            else:
                icon = QIcon()

            item = QListWidgetItem(icon, name)
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self.gallery_list.addItem(item)

        count = len(png_files)
        self.gallery_count_label.setText(f"{count} image{'s' if count != 1 else ''}")
        self.btn_open_annotate.setEnabled(True)
        # Auto-select and preview the first image
        self.gallery_list.setCurrentRow(0)
        self._on_gallery_item_clicked(self.gallery_list.item(0))

    def _on_gallery_row_changed(self, row):
        """Called when keyboard arrow keys change the selection row."""
        if row >= 0:
            self._on_gallery_item_clicked(self.gallery_list.item(row))

    def _on_gallery_item_clicked(self, item):
        """Load the clicked PNG directly into the preview canvas."""
        if item is None:
            return
        path = item.data(Qt.UserRole)
        if not path or not os.path.isfile(path):
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            self.log_message(f"❌ Could not read: {os.path.basename(path)}")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.preview_canvas.set_image(img_rgb)
        # Show filename in ROI status bar
        self.roi_status.setText(f"📌 {os.path.basename(path)}")

    def _open_gallery_in_annotate(self):
        """Emit the conversion_completed signal to load the gallery folder in the Annotate tab."""
        if self._gallery_folder and os.path.isdir(self._gallery_folder):
            self.conversion_completed.emit(self._gallery_folder)

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Thermal RAW Image Folder")
        if folder:
            self.in_dir_input.setText(folder)
            out = os.path.join(folder, "converted_pngs")
            self.out_dir_input.setText(out)
            # Clear gallery so the raw conversion preview is shown
            self.gallery_list.clear()
            self.gallery_count_label.setText("0 images")
            self.btn_open_annotate.setEnabled(False)
            self._gallery_folder = None
            self.roi_status.setText("Region of Interest: Full Image (Right-click to reset)")
            self._load_first_image_preview(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.out_dir_input.setText(folder)

    def _load_first_image_preview(self, folder):
        exts = [x.strip() for x in self.ext_input.text().split(',')]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
            files.extend(glob.glob(os.path.join(folder, f"*.{ext.upper()}")))
        
        if files:
            self.preview_file = files[0]
            self._refresh_preview()
        else:
            self.log_message("⚠️ No files found for preview.")

    def _refresh_preview(self):
        if not self.preview_file:
            return
            
        try:
            # We construct a mock worker just to process one frame
            temp_worker = ConverterWorker(
                input_dir="", output_dir="", file_exts=[], 
                colormap=self.cmap_combo.currentText(),
                norm_mode=self.norm_combo.currentText(),
                downscale=False, # Don't downscale preview to allow accurate drawing
                camera_model=self.cam_combo.currentText(),
                roi=None # load full image initially to draw crop
            )
            
            result_img = temp_worker.process_single_file(self.preview_file)
            # result_img is BGR, convert to RGB for QImage
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            self.preview_canvas.set_image(result_img_rgb)
            self.log_message(f"👁️ Preview loaded: {os.path.basename(self.preview_file)}")
        except Exception as e:
            self.log_message(f"❌ Failed to load preview: {str(e)}")

    def _on_roi_changed(self, roi):
        self.current_roi = roi
        if roi:
            self.roi_status.setText(f"Region of Interest: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]} (Right-click to reset)")
        else:
            self.roi_status.setText("Region of Interest: Full Image (Right-click to reset)")

    def _start_conversion(self):
        in_dir = self.in_dir_input.text().strip()
        out_dir = self.out_dir_input.text().strip()
        
        if not in_dir or not out_dir:
            self.log_message("⚠️ Please select input and output directories.")
            return
            
        exts = [x.strip() for x in self.ext_input.text().split(',')]
        cmap = self.cmap_combo.currentText()
        norm = self.norm_combo.currentText()
        downscale = self.chk_downscale.isChecked()
        cam = self.cam_combo.currentText()

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.console.clear()
        
        # If the ROI is exactly the full image boundaries, set it to None to save processing
        roi_to_use = self.current_roi
        if roi_to_use and self.preview_canvas.img_w > 0:
            if roi_to_use[0] == 0 and roi_to_use[1] == 0 and roi_to_use[2] == self.preview_canvas.img_w and roi_to_use[3] == self.preview_canvas.img_h:
                roi_to_use = None

        self.worker = ConverterWorker(in_dir, out_dir, exts, cmap, norm, downscale, cam, roi_to_use)
        self.worker.progress.connect(self._update_progress)
        self.worker.log.connect(self.log_message)
        self.worker.finished_batch.connect(self._conversion_finished)
        self.worker.start()

    def _cancel_conversion(self):
        if self.worker:
            self.worker.stop()
            self.btn_cancel.setEnabled(False)

    def _update_progress(self, val):
        self.progress_bar.setValue(val)

    def log_message(self, msg):
        self.console.append(msg)
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _conversion_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.worker = None

        # Auto-load gallery with freshly converted PNGs
        out_dir = self.out_dir_input.text().strip()
        if out_dir and os.path.isdir(out_dir):
            self.load_png_folder(out_dir)

        reply = QMessageBox.question(
            self,
            "Conversion Complete",
            "Batch conversion to PNG is complete!\n\nDo you want to immediately load the output folder into the Annotator window?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply == QMessageBox.Yes:
            self.conversion_completed.emit(out_dir)

    def _groupbox_style(self): return "QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 12px; padding-top: 15px; color: #aaa; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
    def _input_style(self): return "background-color: #1e1e1e; color: white; border: 1px solid #444; border-radius: 4px; padding: 6px;"
    def _btn_style(self): return "background-color: #333; color: white; border: 1px solid #555; border-radius: 4px; padding: 6px 12px;"
    def _combo_style(self): return "background-color: #1e1e1e; color: white; border: 1px solid #444; border-radius: 4px; padding: 6px;"
