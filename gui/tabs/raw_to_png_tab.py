import os
import glob
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QComboBox, 
    QGroupBox, QCheckBox, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPainter, QPen, QColor, QImage, QPixmap

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
        
        self.rubber_band_start = None
        self.rubber_band_end = None
        self.roi = None # (x, y, w, h) in image coords
        self.image_pixmap = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.img_w = 0
        self.img_h = 0

    def set_image(self, img_rgb):
        self.img_h, self.img_w = img_rgb.shape[:2]
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_pixmap = QPixmap.fromImage(qimg)
        # Reset ROI to full image on new image load
        self.roi = (0, 0, w, h) 
        self.roi_changed.emit(self.roi)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.image_pixmap:
            # Draw placeholder
            painter = QPainter(self)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Preview Available\nSelect folder to load preview")
            return

        painter = QPainter(self)
        
        # Calculate scale and offset to fit image
        widget_w, widget_h = self.width(), self.height()
        scale_x = widget_w / self.img_w
        scale_y = widget_h / self.img_h
        self.scale = min(scale_x, scale_y)
        
        draw_w = int(self.img_w * self.scale)
        draw_h = int(self.img_h * self.scale)
        self.offset_x = (widget_w - draw_w) // 2
        self.offset_y = (widget_h - draw_h) // 2

        # Draw image
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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image_pixmap:
            self.rubber_band_start = event.pos()
            self.rubber_band_end = event.pos()
            self.update()
        elif event.button() == Qt.RightButton and self.image_pixmap:
            # Right click resets ROI to full image
            self.roi = (0, 0, self.img_w, self.img_h)
            self.roi_changed.emit(self.roi)
            self.update()

    def mouseMoveEvent(self, event):
        if self.rubber_band_start:
            self.rubber_band_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rubber_band_start:
            self.rubber_band_end = event.pos()
            
            # Convert widget coords to image coords
            x1 = (min(self.rubber_band_start.x(), self.rubber_band_end.x()) - self.offset_x) / self.scale
            y1 = (min(self.rubber_band_start.y(), self.rubber_band_end.y()) - self.offset_y) / self.scale
            x2 = (max(self.rubber_band_start.x(), self.rubber_band_end.x()) - self.offset_x) / self.scale
            y2 = (max(self.rubber_band_start.y(), self.rubber_band_end.y()) - self.offset_y) / self.scale
            
            # Clamp
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

        # Normalisation to 8-bit (0-255)
        if img.dtype == np.uint8:
            img_8bit = img
        else:
            if self.norm_mode == "Original":
                img_float = img.astype(np.float32)
                min_rad = np.min(img_float)
                max_rad = np.max(img_float)
                diff = max_rad - min_rad
                pad_range = 1000.0 if diff > 1000.0 else diff
                min_rad = max(0.0, min_rad - pad_range)
                max_rad += pad_range
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
                # Absolute division
                img_float = img.astype(np.float32)
                img_8bit = np.clip((img_float / 65535.0) * 255.0, 0, 255).astype(np.uint8)
        
        # Apply Colormap
        if self.colormap != "None (Grayscale)":
            cmap_flag = self.colormap_dict.get(self.colormap, cv2.COLORMAP_INFERNO)
            result_img = cv2.applyColorMap(img_8bit, cmap_flag)
        else:
            result_img = cv2.cvtColor(img_8bit, cv2.COLORGRAY2BGR)
            
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.current_roi = None
        self.preview_file = None
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
        cmap_options = ["Inferno", "Viridis", "Plasma", "Magma", "Jet", "None (Grayscale)"]
        self.cmap_combo.addItems(cmap_options)
        
        # Add beautiful tooltips to help users
        cmap_tooltips = [
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

        left_layout.addWidget(settings_group)
        left_layout.addStretch()

        self.btn_run = QPushButton("▶ Convert Folder to PNGs")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_run.setStyleSheet("""
            QPushButton { background-color: #2a6a4a; color: white; border-radius: 6px; }
            QPushButton:hover { background-color: #3b8a6a; }
            QPushButton:disabled { background-color: #333; color: #666; }
        """)
        self.btn_run.clicked.connect(self._start_conversion)
        left_layout.addWidget(self.btn_run)
        
        self.btn_cancel = QPushButton("⏹ Cancel Conversion")
        self.btn_cancel.setFixedHeight(45)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("""
            QPushButton { background-color: #8a3a3a; color: white; border-radius: 6px; }
            QPushButton:hover { background-color: #aa4a4a; }
            QPushButton:disabled { background-color: #333; color: #666; }
        """)
        self.btn_cancel.clicked.connect(self._cancel_conversion)
        left_layout.addWidget(self.btn_cancel)

        # ===== RIGHT PANEL (Preview & Console) =====
        right_panel = QSplitter(Qt.Vertical)
        
        # Preview Group
        preview_group = QGroupBox("🖼️ Real-Time Crop Preview (Draw box to set Region of Interest)")
        preview_group.setStyleSheet(self._groupbox_style())
        preview_layout = QVBoxLayout(preview_group)
        self.preview_canvas = PreviewCanvas()
        self.preview_canvas.roi_changed.connect(self._on_roi_changed)
        preview_layout.addWidget(self.preview_canvas)
        
        # ROI Status Label
        self.roi_status = QLabel("Region of Interest: Full Image (Right-click to reset)")
        self.roi_status.setStyleSheet("color: #888;")
        preview_layout.addWidget(self.roi_status)
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

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Thermal RAW Image Folder")
        if folder:
            self.in_dir_input.setText(folder)
            out = os.path.join(folder, "converted_pngs")
            self.out_dir_input.setText(out)
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

    def _groupbox_style(self): return "QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 12px; padding-top: 15px; color: #aaa; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
    def _input_style(self): return "background-color: #1e1e1e; color: white; border: 1px solid #444; border-radius: 4px; padding: 6px;"
    def _btn_style(self): return "background-color: #333; color: white; border: 1px solid #555; border-radius: 4px; padding: 6px 12px;"
    def _combo_style(self): return "background-color: #1e1e1e; color: white; border: 1px solid #444; border-radius: 4px; padding: 6px;"
