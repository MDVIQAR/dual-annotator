"""
gui/tabs/export_test_tab.py

Export & Test tab — drop a trained version folder to convert best.pt → ONNX,
then run inference on a test image folder and view results in a gallery.

Gallery layout:
  YOLO  → 2 columns: Original | Detected
  UNet  → 3 columns: Original | Mask    | Overlay

Confidence threshold is shown only for YOLO models.
Both .pt and .onnx are accepted for inference.
"""

import os
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
    QFrame, QFileDialog, QScrollArea, QMessageBox,
    QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent


class _NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores mouse-wheel to prevent accidental value changes."""
    def wheelEvent(self, e):
        e.ignore()

_SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mlops", "scripts")
)

# ── Styles ────────────────────────────────────────────────────────────────────

_SECTION = """
    QFrame {
        background-color: #252526;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
    }
"""
_INPUT = """
    QLineEdit {
        background-color: #2d2d2d;
        border: 1px solid #3a3a3a;
        border-radius: 4px;
        color: #ffffff;
        padding: 5px 8px;
        font-size: 12px;
    }
    QLineEdit:focus { border: 1px solid #6a3fc8; }
"""
_BROWSE = """
    QPushButton {
        background-color: #2d2d30;
        border: 1px solid #555;
        border-radius: 4px;
        color: #cccccc;
        padding: 5px 12px;
        font-size: 12px;
    }
    QPushButton:hover { background-color: #3a3a40; border: 1px solid #777; }
"""
_ACTION = """
    QPushButton {
        background-color: #6a3fc8;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 8px;
        font-size: 12px;
        font-weight: bold;
    }
    QPushButton:hover    { background-color: #7a4fd8; }
    QPushButton:pressed  { background-color: #5a2fb8; }
    QPushButton:disabled { background-color: #444444; color: #888888; }
"""
_LOG = """
    QTextEdit {
        background-color: #1e1e1e;
        color: #9cdcfe;
        font-family: Consolas, monospace;
        font-size: 11px;
        border: 1px solid #3a3a3a;
        border-radius: 3px;
    }
"""
_PROGRESS = """
    QProgressBar {
        background-color: #3c3c3c;
        border: none;
        border-radius: 3px;
        max-height: 6px;
    }
    QProgressBar::chunk { background-color: #6a3fc8; border-radius: 3px; }
"""
_DROP_IDLE = """
    QLabel {
        color: #666666;
        font-size: 12px;
        border: 2px dashed #555555;
        border-radius: 6px;
        background-color: #1e1e1e;
        padding: 14px;
    }
"""
_DROP_HOVER = """
    QLabel {
        color: #9cdcfe;
        font-size: 12px;
        border: 2px dashed #6a3fc8;
        border-radius: 6px;
        background-color: #1a1a2e;
        padding: 14px;
    }
"""
_DROP_OK = """
    QLabel {
        color: #4ec9b0;
        font-size: 11px;
        border: 2px solid #4ec9b0;
        border-radius: 6px;
        background-color: #1a2e2a;
        padding: 10px;
    }
"""
_ZOOM_BTN = """
    QPushButton {
        background-color: #2d2d30;
        border: 1px solid #555;
        border-radius: 4px;
        color: #cccccc;
        padding: 2px 10px;
        font-size: 14px;
        font-weight: bold;
        min-width: 28px;
    }
    QPushButton:hover { background-color: #3a3a40; }
    QPushButton:pressed { background-color: #1e1e1e; }
"""

_ZOOM_BASE = 300   # default image height in px (= 100 %)
_ZOOM_STEP = 60
_ZOOM_MIN  = 100
_ZOOM_MAX  = 780


# ── Drop Zone ─────────────────────────────────────────────────────────────────

class _DropZone(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(80)
        self.setAcceptDrops(True)
        self.setWordWrap(True)
        self._reset()

    def reset(self):
        self._reset()

    def _reset(self):
        self.setText("Drop version folder here\n(or click Browse above)")
        self.setStyleSheet(_DROP_IDLE)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(_DROP_HOVER)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(_DROP_IDLE)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            self._reset()
            return
        self._apply(urls[0].toLocalFile())

    def _apply(self, path: str):
        info = _validate_folder(path)
        if info is None:
            self.setText("⚠  Not a valid version folder\n(needs config.json + best.pt)")
            self.setStyleSheet(_DROP_IDLE)
            return
        self.setStyleSheet(_DROP_OK)
        lines = [
            f"📁  {os.path.basename(path)}",
            f"Model : {info['model_type'].upper()}   Arch : {info['arch']}",
            f"Input : {info['in_ch']}ch  ×  {info['w']}×{info['h']}",
            f"ONNX  : {'✓ ready' if info['has_onnx'] else '✗ not yet exported'}",
        ]
        self.setText("\n".join(lines))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_folder(path: str) -> dict | None:
    if not os.path.isdir(path):
        return None
    config_path = os.path.join(path, "config.json")
    pt_path     = os.path.join(path, "best.pt")
    if not os.path.isfile(config_path) or not os.path.isfile(pt_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        hp = cfg.get("hyperparams", {})
        return {
            "model_type":  cfg.get("model_type", "unet"),
            "arch":        hp.get("architecture", "?"),
            "in_ch":       hp.get("in_channels", 3),
            "w":           hp.get("image_width", 320),
            "h":           hp.get("image_height", 240),
            "has_onnx":    os.path.isfile(os.path.join(path, "best.onnx")),
            "config_path": config_path,
            "onnx_path":   os.path.join(path, "best.onnx"),
        }
    except Exception:
        return None


def _lbl(text: str) -> QLabel:
    w = QLabel(text)
    w.setStyleSheet("color:#9cdcfe; font-size:11px;")
    return w


def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color:#3a3a3a;")
    return line


def _header(text: str) -> QLabel:
    w = QLabel(text)
    w.setStyleSheet("color:#cccccc; font-size:13px; font-weight:bold;")
    return w


# ── Gallery Item ──────────────────────────────────────────────────────────────

class _GalleryItem(QFrame):
    """
    Shows 2 panels for YOLO (Original | Detected)
    or 3 panels for UNet  (Original | Mask | Overlay).
    Call set_zoom() to resize images without rebuilding.

    Pixmaps are loaded once at construction and cached; set_zoom() scales
    the cached copies so there are no disk reads during zoom operations.
    """

    def __init__(self, name: str, orig_path: str, annotated_path: str,
                 overlay_path: str, model_type: str,
                 zoom_h: int, gallery_w: int, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame { background:#1e1e1e; border:1px solid #3a3a3a; border-radius:4px; }"
        )
        self._model_type = model_type

        if model_type == "yolo":
            self._panels = [("Original", orig_path), ("Detected", annotated_path)]
        else:
            self._panels = [("Original", orig_path), ("Mask", annotated_path), ("Overlay", overlay_path)]

        # Load pixmaps once — no disk reads during zoom
        self._orig_pixmaps: list[QPixmap] = []
        for _, path in self._panels:
            pix = QPixmap(path) if path and os.path.isfile(path) else QPixmap()
            self._orig_pixmaps.append(pix)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("color:#9cdcfe; font-size:10px; font-family:Consolas;")
        lay.addWidget(name_lbl)

        self._img_row = QHBoxLayout()
        self._img_row.setSpacing(8)

        self._img_labels: list[QLabel] = []
        for caption, _ in self._panels:
            col = QVBoxLayout()
            col.setSpacing(2)
            cap = QLabel(caption)
            cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet("color:#888; font-size:10px;")
            col.addWidget(cap)
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            col.addWidget(img_lbl)
            self._img_labels.append(img_lbl)
            self._img_row.addLayout(col)

        lay.addLayout(self._img_row)
        self.set_zoom(zoom_h, gallery_w)

    def set_zoom(self, zoom_h: int, _gallery_w: int = 0):
        # Scale each cached pixmap to zoom_h, preserving aspect ratio.
        # Height-driven scaling means zoom always has visible effect regardless
        # of viewport width. _gallery_w retained for call-site compatibility.
        for img_lbl, pix in zip(self._img_labels, self._orig_pixmaps):
            if not pix.isNull():
                scaled = pix.scaledToHeight(zoom_h, Qt.SmoothTransformation)
                img_lbl.setPixmap(scaled)
            else:
                img_lbl.clear()


# ── Main Tab ──────────────────────────────────────────────────────────────────

class ExportTestTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._version_folder = None
        self._folder_info    = None
        self._export_worker  = None
        self._infer_worker   = None
        self._gallery_items: list[_GalleryItem] = []
        self._zoom_h         = _ZOOM_BASE

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(0)

        self._v_splitter = QSplitter(Qt.Vertical)
        self._v_splitter.setStyleSheet("QSplitter::handle { background:#3a3a3a; height:5px; }")
        self._v_splitter.setHandleWidth(5)

        self._h_splitter = QSplitter(Qt.Horizontal)
        self._h_splitter.setStyleSheet("QSplitter::handle { background:#3a3a3a; width:5px; }")
        self._h_splitter.setHandleWidth(5)
        self._h_splitter.addWidget(self._build_export_section())
        self._h_splitter.addWidget(self._build_test_section())
        self._h_splitter.setSizes([500, 500])

        self._v_splitter.addWidget(self._h_splitter)
        self._v_splitter.addWidget(self._build_gallery_panel())
        self._v_splitter.setSizes([480, 400])

        root.addWidget(self._v_splitter)

    def _build_export_section(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(_SECTION)
        lay   = QVBoxLayout(frame)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        lay.addWidget(_header("  Export to ONNX"))
        lay.addWidget(_sep())

        browse_row = QHBoxLayout()
        browse_row.setSpacing(6)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Version folder path...")
        self._folder_edit.setReadOnly(True)
        self._folder_edit.setStyleSheet(_INPUT)
        browse_row.addWidget(self._folder_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(_BROWSE)
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_folder)
        browse_row.addWidget(browse_btn)
        lay.addLayout(browse_row)

        self._drop_zone = _DropZone()
        self._drop_zone.mousePressEvent = lambda _: self._browse_folder()
        lay.addWidget(self._drop_zone)

        self._export_btn = QPushButton("Export to ONNX")
        self._export_btn.setStyleSheet(_ACTION)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._run_export)
        lay.addWidget(self._export_btn)

        self._export_progress = QProgressBar()
        self._export_progress.setRange(0, 100)
        self._export_progress.setValue(0)
        self._export_progress.setStyleSheet(_PROGRESS)
        self._export_progress.setTextVisible(False)
        lay.addWidget(self._export_progress)

        lay.addWidget(_lbl("Console"))
        self._export_log = QTextEdit()
        self._export_log.setReadOnly(True)
        self._export_log.setStyleSheet(_LOG)
        lay.addWidget(self._export_log, 1)

        return frame

    def _build_test_section(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(_SECTION)
        lay   = QVBoxLayout(frame)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        lay.addWidget(_header("  Run Inference"))
        lay.addWidget(_sep())

        lay.addWidget(_lbl("Model  (.pt or .onnx)"))
        onnx_row = QHBoxLayout()
        onnx_row.setSpacing(6)
        self._onnx_edit = QLineEdit()
        self._onnx_edit.setPlaceholderText("best.pt or best.onnx")
        self._onnx_edit.setStyleSheet(_INPUT)
        onnx_row.addWidget(self._onnx_edit)
        onnx_browse = QPushButton("Browse")
        onnx_browse.setStyleSheet(_BROWSE)
        onnx_browse.setFixedWidth(70)
        onnx_browse.clicked.connect(self._browse_onnx)
        onnx_row.addWidget(onnx_browse)
        lay.addLayout(onnx_row)

        lay.addWidget(_lbl("Test Images Folder"))
        test_row = QHBoxLayout()
        test_row.setSpacing(6)
        self._test_edit = QLineEdit()
        self._test_edit.setPlaceholderText("Folder containing test images...")
        self._test_edit.setStyleSheet(_INPUT)
        test_row.addWidget(self._test_edit)
        test_browse = QPushButton("Browse")
        test_browse.setStyleSheet(_BROWSE)
        test_browse.setFixedWidth(70)
        test_browse.clicked.connect(self._browse_test)
        test_row.addWidget(test_browse)
        lay.addLayout(test_row)

        lay.addWidget(_lbl("Output Folder"))
        out_row = QHBoxLayout()
        out_row.setSpacing(6)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Where to save result images...")
        self._out_edit.setStyleSheet(_INPUT)
        out_row.addWidget(self._out_edit)
        out_browse = QPushButton("Browse")
        out_browse.setStyleSheet(_BROWSE)
        out_browse.setFixedWidth(70)
        out_browse.clicked.connect(self._browse_out)
        out_row.addWidget(out_browse)
        lay.addLayout(out_row)

        # Confidence threshold — only visible for YOLO
        self._conf_widget = QWidget()
        self._conf_widget.setStyleSheet("background: transparent;")
        conf_row = QHBoxLayout(self._conf_widget)
        conf_row.setContentsMargins(0, 0, 0, 0)
        conf_row.setSpacing(6)
        conf_row.addWidget(_lbl("Confidence Threshold"))
        self._conf_sp = _NoScrollDoubleSpinBox()
        self._conf_sp.setRange(0.001, 1.0)
        self._conf_sp.setDecimals(3)
        self._conf_sp.setSingleStep(0.01)
        self._conf_sp.setValue(0.25)
        self._conf_sp.setStyleSheet(_INPUT)
        conf_row.addWidget(self._conf_sp)
        conf_row.addStretch()
        lay.addWidget(self._conf_widget)
        self._conf_widget.setVisible(False)  # hidden until model type is known

        infer_btn_row = QHBoxLayout()
        infer_btn_row.setSpacing(6)
        self._infer_btn = QPushButton("Run Inference")
        self._infer_btn.setStyleSheet(_ACTION)
        self._infer_btn.setEnabled(False)
        self._infer_btn.clicked.connect(self._run_infer)
        infer_btn_row.addWidget(self._infer_btn)
        infer_clear_btn = QPushButton("Clear")
        infer_clear_btn.setStyleSheet(_BROWSE)
        infer_clear_btn.setFixedWidth(60)
        infer_clear_btn.setToolTip("Clear model, test folder, and output folder fields")
        infer_clear_btn.clicked.connect(self._clear_infer_fields)
        infer_btn_row.addWidget(infer_clear_btn)
        lay.addLayout(infer_btn_row)

        self._infer_progress = QProgressBar()
        self._infer_progress.setRange(0, 100)
        self._infer_progress.setValue(0)
        self._infer_progress.setStyleSheet(_PROGRESS)
        self._infer_progress.setTextVisible(False)
        lay.addWidget(self._infer_progress)

        lay.addWidget(_lbl("Console"))
        self._infer_log = QTextEdit()
        self._infer_log.setReadOnly(True)
        self._infer_log.setStyleSheet(_LOG)
        lay.addWidget(self._infer_log, 1)

        return frame

    def _build_gallery_panel(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(_SECTION)
        lay   = QVBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        # Header row
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(_header("  Results Gallery"))
        hdr_row.addStretch()

        self._gallery_count_lbl = QLabel("0 images")
        self._gallery_count_lbl.setStyleSheet("color:#666; font-size:11px;")
        hdr_row.addWidget(self._gallery_count_lbl)

        # Zoom controls
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setStyleSheet(_ZOOM_BTN)
        zoom_out_btn.setFixedWidth(30)
        zoom_out_btn.clicked.connect(self._zoom_out)

        self._zoom_lbl = QLabel("100%")
        self._zoom_lbl.setStyleSheet("color:#888; font-size:11px; min-width:38px;")
        self._zoom_lbl.setAlignment(Qt.AlignCenter)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setStyleSheet(_ZOOM_BTN)
        zoom_in_btn.setFixedWidth(30)
        zoom_in_btn.clicked.connect(self._zoom_in)

        hdr_row.addWidget(zoom_out_btn)
        hdr_row.addWidget(self._zoom_lbl)
        hdr_row.addWidget(zoom_in_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet(_BROWSE)
        clear_btn.setFixedWidth(55)
        clear_btn.clicked.connect(self._clear_gallery)
        hdr_row.addWidget(clear_btn)

        lay.addLayout(hdr_row)
        lay.addWidget(_sep())

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border:none; background:#1a1a1a; }")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._gallery_widget = QWidget()
        self._gallery_widget.setStyleSheet("background:#1a1a1a;")
        self._gallery_layout = QVBoxLayout(self._gallery_widget)
        self._gallery_layout.setContentsMargins(6, 6, 6, 6)
        self._gallery_layout.setSpacing(8)
        self._gallery_layout.addStretch()

        self._scroll.setWidget(self._gallery_widget)
        lay.addWidget(self._scroll)

        return frame

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def _zoom_in(self):
        self._zoom_h = min(self._zoom_h + _ZOOM_STEP, _ZOOM_MAX)
        self._apply_zoom()

    def _zoom_out(self):
        self._zoom_h = max(self._zoom_h - _ZOOM_STEP, _ZOOM_MIN)
        self._apply_zoom()

    def _apply_zoom(self):
        pct = int(self._zoom_h / _ZOOM_BASE * 100)
        self._zoom_lbl.setText(f"{pct}%")
        gallery_w = self._scroll.viewport().width()
        for item in self._gallery_items:
            item.set_zoom(self._zoom_h, gallery_w)

    # ── Browse actions ────────────────────────────────────────────────────────

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Version Folder", "")
        if path:
            self._load_version_folder(path)

    def _browse_onnx(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "",
            "Model files (*.pt *.onnx);;PyTorch (*.pt);;ONNX (*.onnx)"
        )
        if not path:
            return
        self._onnx_edit.setText(path)
        folder = os.path.dirname(path)
        config_path = os.path.join(folder, "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                hp = cfg.get("hyperparams", {})
                mt = cfg.get("model_type", "unet")
                self._folder_info = {
                    "model_type":  mt,
                    "arch":        hp.get("architecture", "?"),
                    "in_ch":       hp.get("in_channels", 3),
                    "w":           hp.get("image_width", 320),
                    "h":           hp.get("image_height", 240),
                    "has_onnx":    True,
                    "config_path": config_path,
                    "onnx_path":   path,
                }
                self._conf_widget.setVisible(mt == "yolo")
            except Exception:
                pass
        self._update_infer_btn()

    def _browse_test(self):
        path = QFileDialog.getExistingDirectory(self, "Select Test Images Folder", "")
        if path:
            self._test_edit.setText(path)
            if not self._out_edit.text():
                self._out_edit.setText(os.path.join(path, "results"))
            self._update_infer_btn()

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if path:
            self._out_edit.setText(path)

    # ── Version folder loading ─────────────────────────────────────────────────

    def prefill_from_version(self, path: str):
        if not path or not os.path.isdir(path):
            return
        self._load_version_folder(path)
        test_images = os.path.join(path, "dataset", "test", "images")
        if os.path.isdir(test_images):
            self._test_edit.setText(test_images)
            self._out_edit.setText(os.path.join(test_images, "results"))

    def _load_version_folder(self, path: str):
        info = _validate_folder(path)
        if info is None:
            QMessageBox.warning(self, "Invalid Folder",
                "This folder is not a valid version folder.\n"
                "It must contain config.json and best.pt.")
            return

        self._version_folder = path
        self._folder_info    = info
        self._folder_edit.setText(path)
        self._drop_zone._apply(path)

        # Show confidence threshold only for YOLO
        self._conf_widget.setVisible(info["model_type"] == "yolo")

        # Prefer .pt for inference
        pt_path = os.path.join(path, "best.pt")
        self._onnx_edit.setText(pt_path)

        if info["has_onnx"]:
            self._export_btn.setEnabled(False)
            self._export_btn.setText("✓  ONNX already exported")
            self._export_log.append("[INFO] best.onnx already exists — skipping export.")
        else:
            self._export_btn.setEnabled(True)
            self._export_btn.setText("Export to ONNX")

        self._update_infer_btn()

    # ── Export ────────────────────────────────────────────────────────────────

    def _run_export(self):
        if not self._version_folder:
            return

        from mlops.export.onnx_worker import OnnxWorker
        self._export_worker = OnnxWorker(self._version_folder, _SCRIPTS_DIR, parent=self)
        self._export_worker.log.connect(self._on_export_log)
        self._export_worker.progress.connect(self._export_progress.setValue)
        self._export_worker.finished.connect(self._on_export_done)

        self._export_btn.setEnabled(False)
        self._export_btn.setText("Exporting...")
        self._export_log.clear()
        self._export_progress.setValue(0)
        self._export_worker.start()

    @pyqtSlot(str)
    def _on_export_log(self, line: str):
        self._export_log.append(line)

    @pyqtSlot(bool)
    def _on_export_done(self, success: bool):
        if success:
            self._export_btn.setText("✓  Export complete")
            pt_path   = os.path.join(self._version_folder, "best.pt")
            onnx_path = os.path.join(self._version_folder, "best.onnx")
            self._onnx_edit.setText(pt_path if os.path.isfile(pt_path) else onnx_path)
            self._export_log.append("[DONE] best.onnx saved.  Using best.pt for inference.")
            if self._folder_info:
                self._folder_info["has_onnx"] = True
                self._drop_zone._apply(self._version_folder)
        else:
            self._export_btn.setEnabled(True)
            self._export_btn.setText("Export to ONNX")
            self._export_log.append("[ERROR] Export failed — check console above.")
        self._update_infer_btn()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _update_infer_btn(self):
        has_model  = bool(self._onnx_edit.text()) and os.path.isfile(self._onnx_edit.text())
        has_test   = bool(self._test_edit.text()) and os.path.isdir(self._test_edit.text())
        has_config = self._folder_info is not None
        self._infer_btn.setEnabled(has_model and has_test and has_config)

    def _run_infer(self):
        model_path  = self._onnx_edit.text().strip()
        test_folder = self._test_edit.text().strip()
        out_folder  = self._out_edit.text().strip() or os.path.join(test_folder, "results")
        config_path = self._folder_info["config_path"]
        model_type  = self._folder_info["model_type"]

        if not os.path.isfile(model_path):
            QMessageBox.warning(self, "Missing Model", f"Model file not found:\n{model_path}")
            return
        if not os.path.isdir(test_folder):
            QMessageBox.warning(self, "Missing Folder", f"Test folder not found:\n{test_folder}")
            return

        os.makedirs(out_folder, exist_ok=True)
        self._out_edit.setText(out_folder)

        from mlops.export.infer_worker import InferWorker
        self._infer_worker = InferWorker(
            model_type  = model_type,
            onnx_path   = model_path,
            config_path = config_path,
            test_folder = test_folder,
            out_folder  = out_folder,
            scripts_dir = _SCRIPTS_DIR,
            conf        = self._conf_sp.value(),
            parent      = self,
        )
        self._infer_worker.log.connect(self._on_infer_log)
        self._infer_worker.progress.connect(self._infer_progress.setValue)
        self._infer_worker.result.connect(self._on_infer_result)
        self._infer_worker.finished.connect(self._on_infer_done)

        self._infer_btn.setEnabled(False)
        self._infer_btn.setText("Running...")
        self._infer_log.clear()
        self._infer_progress.setValue(0)
        self._infer_worker.start()

    @pyqtSlot(str)
    def _on_infer_log(self, line: str):
        self._infer_log.append(line)

    @pyqtSlot(str, str, str, str)
    def _on_infer_result(self, name: str, orig_path: str,
                         annotated_path: str, overlay_path: str):
        gallery_w  = self._scroll.viewport().width()
        model_type = self._folder_info["model_type"] if self._folder_info else "unet"
        item = _GalleryItem(
            name, orig_path, annotated_path, overlay_path,
            model_type, self._zoom_h, gallery_w,
            parent=self._gallery_widget,
        )
        count = self._gallery_layout.count()
        self._gallery_layout.insertWidget(count - 1, item)
        self._gallery_items.append(item)
        self._gallery_count_lbl.setText(f"{len(self._gallery_items)} image(s)")
        self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        )

    @pyqtSlot(bool, int)
    def _on_infer_done(self, success: bool, total: int):
        self._infer_btn.setEnabled(True)
        self._infer_btn.setText("Run Inference")
        if success:
            self._infer_log.append(f"[DONE] {total} image(s) processed.")
        else:
            self._infer_log.append("[ERROR] Inference failed — check console above.")

    # ── Inference field clear ─────────────────────────────────────────────────

    def _clear_infer_fields(self):
        self._onnx_edit.clear()
        self._test_edit.clear()
        self._out_edit.clear()
        self._folder_info = None
        self._conf_widget.setVisible(False)
        self._update_infer_btn()
        self._infer_log.clear()

    # ── Gallery ───────────────────────────────────────────────────────────────

    def _clear_gallery(self):
        for item in self._gallery_items:
            self._gallery_layout.removeWidget(item)
            item.deleteLater()
        self._gallery_items.clear()
        self._gallery_count_lbl.setText("0 images")
