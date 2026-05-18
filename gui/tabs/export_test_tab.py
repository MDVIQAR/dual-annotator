"""
gui/tabs/export_test_tab.py

Export & Test tab — drop a trained version folder to convert best.pt → ONNX,
then run inference on a test image folder and view results in a gallery.
"""

import os
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
    QFrame, QFileDialog, QScrollArea, QSizePolicy, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent

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

# ── Drop Zone ─────────────────────────────────────────────────────────────────

class _DropZone(QLabel):
    """A QLabel that accepts folder drops and validates version folder contents."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(80)
        self.setAcceptDrops(True)
        self.setWordWrap(True)
        self._reset()

    # public
    def reset(self):
        self._reset()

    def _reset(self):
        self.setText("Drop version folder here\n(or click Browse above)")
        self.setStyleSheet(_DROP_IDLE)

    # drag events
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
        path = urls[0].toLocalFile()
        self._apply(path)

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
    """Return metadata dict if path is a valid version folder, else None."""
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
            "model_type": cfg.get("model_type", "unet"),
            "arch":       hp.get("architecture", "?"),
            "in_ch":      hp.get("in_channels", 3),
            "w":          hp.get("image_width", 320),
            "h":          hp.get("image_height", 240),
            "has_onnx":   os.path.isfile(os.path.join(path, "best.onnx")),
            "config_path": config_path,
            "onnx_path":  os.path.join(path, "best.onnx"),
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
    def __init__(self, name: str, img_path: str, gallery_width: int, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame { background:#1e1e1e; border:1px solid #3a3a3a; border-radius:4px; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("color:#9cdcfe; font-size:10px; font-family:Consolas;")
        lay.addWidget(name_lbl)

        img_lbl = QLabel()
        img_lbl.setAlignment(Qt.AlignCenter)
        pix = QPixmap(img_path)
        if not pix.isNull():
            # Scale to fit gallery width keeping aspect ratio
            max_w = max(gallery_width - 30, 200)
            pix   = pix.scaledToWidth(max_w, Qt.SmoothTransformation)
        img_lbl.setPixmap(pix)
        lay.addWidget(img_lbl)


# ── Main Tab ──────────────────────────────────────────────────────────────────

class ExportTestTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._version_folder = None
        self._folder_info    = None
        self._export_worker  = None
        self._infer_worker   = None
        self._gallery_items  = []

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(0)

        # Outer vertical splitter: [Export | Inference] on top, Gallery on bottom
        self._v_splitter = QSplitter(Qt.Vertical)
        self._v_splitter.setStyleSheet("QSplitter::handle { background:#3a3a3a; height:5px; }")
        self._v_splitter.setHandleWidth(5)

        # Inner horizontal splitter: Export (left) | Inference (right)
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

        # Browse row
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

        # Drop zone
        self._drop_zone = _DropZone()
        self._drop_zone.mousePressEvent = lambda _: self._browse_folder()
        lay.addWidget(self._drop_zone)

        # Export button
        self._export_btn = QPushButton("Export to ONNX")
        self._export_btn.setStyleSheet(_ACTION)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._run_export)
        lay.addWidget(self._export_btn)

        # Progress
        self._export_progress = QProgressBar()
        self._export_progress.setRange(0, 100)
        self._export_progress.setValue(0)
        self._export_progress.setStyleSheet(_PROGRESS)
        self._export_progress.setTextVisible(False)
        lay.addWidget(self._export_progress)

        # Log — stretch=1 so it fills all remaining vertical space
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

        # ONNX path
        lay.addWidget(_lbl("ONNX Model"))
        onnx_row = QHBoxLayout()
        onnx_row.setSpacing(6)
        self._onnx_edit = QLineEdit()
        self._onnx_edit.setPlaceholderText("best.onnx path (auto-filled after export)")
        self._onnx_edit.setStyleSheet(_INPUT)
        onnx_row.addWidget(self._onnx_edit)
        onnx_browse = QPushButton("Browse")
        onnx_browse.setStyleSheet(_BROWSE)
        onnx_browse.setFixedWidth(70)
        onnx_browse.clicked.connect(self._browse_onnx)
        onnx_row.addWidget(onnx_browse)
        lay.addLayout(onnx_row)

        # Test folder
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

        # Output folder
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

        # Run button
        self._infer_btn = QPushButton("Run Inference")
        self._infer_btn.setStyleSheet(_ACTION)
        self._infer_btn.setEnabled(False)
        self._infer_btn.clicked.connect(self._run_infer)
        lay.addWidget(self._infer_btn)

        # Progress
        self._infer_progress = QProgressBar()
        self._infer_progress.setRange(0, 100)
        self._infer_progress.setValue(0)
        self._infer_progress.setStyleSheet(_PROGRESS)
        self._infer_progress.setTextVisible(False)
        lay.addWidget(self._infer_progress)

        # Log — stretch=1 so it fills all remaining vertical space
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
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet(_BROWSE)
        clear_btn.setFixedWidth(55)
        clear_btn.clicked.connect(self._clear_gallery)
        hdr_row.addWidget(clear_btn)
        lay.addLayout(hdr_row)
        lay.addWidget(_sep())

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border:none; background:#1a1a1a; }")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._gallery_widget = QWidget()
        self._gallery_widget.setStyleSheet("background:#1a1a1a;")
        self._gallery_layout = QVBoxLayout(self._gallery_widget)
        self._gallery_layout.setContentsMargins(6, 6, 6, 6)
        self._gallery_layout.setSpacing(8)
        self._gallery_layout.addStretch()

        self._scroll.setWidget(self._gallery_widget)
        lay.addWidget(self._scroll)

        return frame

    # ── Browse actions ────────────────────────────────────────────────────────

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Version Folder", "")
        if path:
            self._load_version_folder(path)

    def _browse_onnx(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX Model", "", "ONNX (*.onnx)")
        if path:
            self._onnx_edit.setText(path)
            self._update_infer_btn()

    def _browse_test(self):
        path = QFileDialog.getExistingDirectory(self, "Select Test Images Folder", "")
        if path:
            self._test_edit.setText(path)
            # Auto-fill output folder as a subfolder
            if not self._out_edit.text():
                self._out_edit.setText(os.path.join(path, "results"))
            self._update_infer_btn()

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if path:
            self._out_edit.setText(path)

    # ── Version folder loading ─────────────────────────────────────────────────

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

        # Enable export only if ONNX not yet present
        if info["has_onnx"]:
            self._export_btn.setEnabled(False)
            self._export_btn.setText("✓  ONNX already exported")
            self._onnx_edit.setText(info["onnx_path"])
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
            onnx_path = os.path.join(self._version_folder, "best.onnx")
            self._onnx_edit.setText(onnx_path)
            self._export_log.append("[DONE] best.onnx saved.")
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
        has_onnx   = bool(self._onnx_edit.text()) and os.path.isfile(self._onnx_edit.text())
        has_test   = bool(self._test_edit.text()) and os.path.isdir(self._test_edit.text())
        has_config = self._folder_info is not None
        self._infer_btn.setEnabled(has_onnx and has_test and has_config)

    def _run_infer(self):
        onnx_path   = self._onnx_edit.text().strip()
        test_folder = self._test_edit.text().strip()
        out_folder  = self._out_edit.text().strip() or os.path.join(test_folder, "results")
        config_path = self._folder_info["config_path"]
        model_type  = self._folder_info["model_type"]

        if not os.path.isfile(onnx_path):
            QMessageBox.warning(self, "Missing ONNX", f"ONNX file not found:\n{onnx_path}")
            return
        if not os.path.isdir(test_folder):
            QMessageBox.warning(self, "Missing Folder", f"Test folder not found:\n{test_folder}")
            return

        os.makedirs(out_folder, exist_ok=True)
        self._out_edit.setText(out_folder)

        from mlops.export.infer_worker import InferWorker
        self._infer_worker = InferWorker(
            model_type  = model_type,
            onnx_path   = onnx_path,
            config_path = config_path,
            test_folder = test_folder,
            out_folder  = out_folder,
            scripts_dir = _SCRIPTS_DIR,
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

    @pyqtSlot(str, str)
    def _on_infer_result(self, name: str, img_path: str):
        if not os.path.isfile(img_path):
            return
        gallery_w = self._scroll.viewport().width()
        item = _GalleryItem(name, img_path, gallery_w, parent=self._gallery_widget)
        # Insert before the trailing stretch
        count = self._gallery_layout.count()
        self._gallery_layout.insertWidget(count - 1, item)
        self._gallery_items.append(item)
        self._gallery_count_lbl.setText(f"{len(self._gallery_items)} image(s)")
        # Auto-scroll to bottom
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

    # ── Gallery ───────────────────────────────────────────────────────────────

    def _clear_gallery(self):
        for item in self._gallery_items:
            self._gallery_layout.removeWidget(item)
            item.deleteLater()
        self._gallery_items.clear()
        self._gallery_count_lbl.setText("0 images")
