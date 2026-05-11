"""
gui/tabs/data_prep_tab.py

Phase 2 — Data Preparation tab for DualAnnotator.
Provides a form-based UI for configuring and running the DataPrepWorker pipeline,
with a live console, progress bar, and a result card on completion.
"""

import os
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QSlider, QTextEdit, QProgressBar, QFrame, QFileDialog,
    QMessageBox, QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont, QColor

from mlops.data_prep.prep_worker import DataPrepWorker
from mlops.registry import RegistrySettings


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

_SETTINGS_DEFAULTS = {
    "task_type":     "unet",
    "project_name":  "",
    "images_folder": "",
    "labels_folder": "",
    "test_folder":   "",
    "output_base":   "",
    "train_pct":     80,
    "class_names":   "",
}


def _settings_path() -> str:
    app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(app_data, "DualAnnotator", "data_prep_settings.json")


def _load_settings() -> dict:
    path = _settings_path()
    if not os.path.isfile(path):
        return dict(_SETTINGS_DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            on_disk = json.load(fh)
        merged = dict(_SETTINGS_DEFAULTS)
        merged.update(on_disk)
        return merged
    except (json.JSONDecodeError, OSError):
        return dict(_SETTINGS_DEFAULTS)


def _save_settings(data: dict):
    path = _settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Styled helpers
# ---------------------------------------------------------------------------

_SECTION_STYLE = """
    QFrame {
        background-color: #252526;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
    }
"""

_INPUT_STYLE = """
    QLineEdit {
        background-color: #2d2d2d;
        border: 1px solid #3a3a3a;
        border-radius: 4px;
        color: #ffffff;
        padding: 5px 8px;
        font-size: 12px;
    }
    QLineEdit:focus {
        border: 1px solid #6a3fc8;
    }
"""

_BROWSE_STYLE = """
    QPushButton {
        background-color: #2d2d30;
        border: 1px solid #555;
        border-radius: 4px;
        color: #cccccc;
        padding: 5px 12px;
        font-size: 12px;
    }
    QPushButton:hover {
        background-color: #3a3a40;
        border: 1px solid #777;
    }
"""

_PREPARE_IDLE_STYLE = """
    QPushButton {
        background-color: #6a3fc8;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #7a4fd8; }
    QPushButton:pressed { background-color: #5a2fb8; }
"""

_PREPARE_RUNNING_STYLE = """
    QPushButton {
        background-color: #444444;
        color: #888888;
        border: none;
        border-radius: 5px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }
"""


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #8ab4f8; font-size: 12px; font-weight: bold; background: transparent; border: none;")
    return lbl


def _make_section(layout_items) -> QFrame:
    frame = QFrame()
    frame.setStyleSheet(_SECTION_STYLE)
    inner = QVBoxLayout(frame)
    inner.setContentsMargins(12, 10, 12, 10)
    inner.setSpacing(8)
    for item in layout_items:
        if isinstance(item, QWidget):
            inner.addWidget(item)
        else:
            inner.addLayout(item)
    return frame


def _folder_row(line_edit: QLineEdit, btn: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(6)
    row.addWidget(line_edit)
    row.addWidget(btn)
    return row


# ---------------------------------------------------------------------------
# DataPrepTab
# ---------------------------------------------------------------------------

class DataPrepTab(QWidget):
    """Phase 2 data preparation tab — left config panel + right console/progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = _load_settings()
        self._worker = None
        self._setup_ui()
        self._restore_settings()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2a2a; }")

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 800])

        root.addWidget(splitter)

    # ── Left panel ──────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(380)
        panel.setStyleSheet("background-color: #1e1e1e;")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: #1e1e1e; border: none; }")
        scroll.setWidget(panel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Section 1 — Task type
        self._radio_yolo = QRadioButton("YOLO (Detection)")
        self._radio_unet = QRadioButton("UNet (Segmentation)")
        self._radio_yolo.setStyleSheet("color: #ffffff; background: transparent; border: none;")
        self._radio_unet.setStyleSheet("color: #ffffff; background: transparent; border: none;")
        self._task_group = QButtonGroup(self)
        self._task_group.addButton(self._radio_yolo)
        self._task_group.addButton(self._radio_unet)
        self._radio_yolo.toggled.connect(self._on_task_changed)
        layout.addWidget(_make_section([_section_label("Task Type"), self._radio_yolo, self._radio_unet]))

        # Section 2 — Project name
        self._project_edit = QLineEdit()
        self._project_edit.setPlaceholderText("e.g. NestleS_cam0")
        self._project_edit.setStyleSheet(_INPUT_STYLE)
        self._project_edit.textChanged.connect(self._autosave)
        layout.addWidget(_make_section([_section_label("Project Name"), self._project_edit]))

        # Section 3 — Source folders
        self._images_edit,  img_btn  = self._make_folder_field("Folder containing your images")
        self._labels_edit,  lbl_btn  = self._make_folder_field("Folder containing .txt labels or .png masks")
        self._test_edit,    test_btn = self._make_folder_field("Optional — held-out test images (no labels needed)")

        img_btn.clicked.connect(lambda: self._browse(self._images_edit, "images_folder"))
        lbl_btn.clicked.connect(lambda: self._browse(self._labels_edit, "labels_folder"))
        test_btn.clicked.connect(lambda: self._browse(self._test_edit, "test_folder"))

        layout.addWidget(_make_section([
            _section_label("Source Folders"),
            QLabel("Images Folder"),
            _folder_row(self._images_edit, img_btn),
            QLabel("Labels / Masks Folder"),
            _folder_row(self._labels_edit, lbl_btn),
            QLabel("Test Images (optional)"),
            _folder_row(self._test_edit, test_btn),
        ]))

        # Section 4 — Output
        self._output_edit, out_btn = self._make_folder_field("Where to save the prepared dataset")
        out_btn.clicked.connect(lambda: self._browse(self._output_edit, "output_base"))
        layout.addWidget(_make_section([
            _section_label("Output"),
            QLabel("Output Base Folder"),
            _folder_row(self._output_edit, out_btn),
        ]))

        # Section 5 — Class names (YOLO only)
        self._class_section = QFrame()
        self._class_section.setStyleSheet(_SECTION_STYLE)
        cs_inner = QVBoxLayout(self._class_section)
        cs_inner.setContentsMargins(12, 10, 12, 10)
        cs_inner.setSpacing(8)
        cs_inner.addWidget(_section_label("Class Names (YOLO only)"))
        cs_inner.addWidget(QLabel("Comma-separated, in order"))
        self._class_edit = QLineEdit()
        self._class_edit.setPlaceholderText("background, defect_A, defect_B")
        self._class_edit.setStyleSheet(_INPUT_STYLE)
        self._class_edit.textChanged.connect(self._autosave)
        cs_inner.addWidget(self._class_edit)
        layout.addWidget(self._class_section)

        # Section 6 — Train/val split
        self._split_label = QLabel("Train: 80%   Val: 20%")
        self._split_label.setStyleSheet("color: #cccccc; background: transparent; border: none;")
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(50, 90)
        self._slider.setValue(80)
        self._slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px; background: #3a3a3a; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #6a3fc8; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #6a3fc8; border-radius: 2px; }
        """)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._slider.sliderReleased.connect(self._autosave)
        layout.addWidget(_make_section([_section_label("Train / Val Split"), self._split_label, self._slider]))

        # Prepare button
        self._prepare_btn = QPushButton("▶  Prepare Dataset")
        self._prepare_btn.setStyleSheet(_PREPARE_IDLE_STYLE)
        self._prepare_btn.clicked.connect(self._on_prepare_clicked)
        layout.addWidget(self._prepare_btn)
        layout.addStretch()

        # The scroll area is the actual widget to return
        outer = QWidget()
        outer.setFixedWidth(380)
        outer.setStyleSheet("background: #1e1e1e;")
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(scroll)
        return outer

    def _make_folder_field(self, placeholder: str):
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setStyleSheet(_INPUT_STYLE)
        edit.textChanged.connect(self._autosave)
        btn = QPushButton("Browse")
        btn.setStyleSheet(_BROWSE_STYLE)
        btn.setFixedWidth(70)
        return edit, btn

    # ── Right panel ──────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background-color: #1e1e1e;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel("Preparation Console")
        title.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.setStyleSheet(_BROWSE_STYLE)
        clear_btn.clicked.connect(lambda: self._console.clear())
        header.addWidget(title)
        header.addStretch()
        header.addWidget(clear_btn)
        layout.addLayout(header)

        # Progress area (hidden until worker starts)
        self._progress_area = QWidget()
        pa_layout = QVBoxLayout(self._progress_area)
        pa_layout.setContentsMargins(0, 0, 0, 0)
        pa_layout.setSpacing(4)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                color: #ffffff;
                text-align: center;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #6a3fc8;
                border-radius: 3px;
            }
        """)
        self._hash_label = QLabel("")
        self._hash_label.setStyleSheet("color: #8ab4f8; font-size: 11px;")
        self._hash_label.hide()
        pa_layout.addWidget(self._progress_bar)
        pa_layout.addWidget(self._hash_label)
        self._progress_area.hide()
        layout.addWidget(self._progress_area)

        # Console
        self._console = QTextEdit()
        self._console.setReadOnly(True)
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.Monospace)
        self._console.setFont(font)
        self._console.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #d4d4d4;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        layout.addWidget(self._console, 1)

        # Result card (hidden until successful run)
        self._result_card = QFrame()
        self._result_card.setStyleSheet("""
            QFrame {
                background-color: #1e2e1e;
                border: 1px solid #3a3a3a;
                border-left: 3px solid #4caf50;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        rc_layout = QVBoxLayout(self._result_card)
        rc_layout.setContentsMargins(12, 8, 12, 8)
        rc_layout.setSpacing(4)
        self._rc_title = QLabel("✓ Dataset ready!")
        self._rc_title.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 13px;")
        self._rc_counts = QLabel("")
        self._rc_counts.setStyleSheet("color: #cccccc; font-size: 12px;")
        self._rc_path = QLabel("")
        self._rc_path.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self._rc_path.setWordWrap(True)
        self._rc_hash = QLabel("")
        self._rc_hash.setStyleSheet("color: #8ab4f8; font-size: 11px;")
        self._rc_gdrive = QLabel("")
        self._rc_gdrive.setStyleSheet("color: #4fc3f7; font-size: 11px;")
        self._rc_gdrive.setWordWrap(True)
        rc_layout.addWidget(self._rc_title)
        rc_layout.addWidget(self._rc_counts)
        rc_layout.addWidget(self._rc_path)
        rc_layout.addWidget(self._rc_hash)
        rc_layout.addWidget(self._rc_gdrive)
        self._result_card.hide()
        layout.addWidget(self._result_card)

        return panel

    # ------------------------------------------------------------------
    # Settings restore
    # ------------------------------------------------------------------

    def _restore_settings(self):
        s = self._settings
        if s.get("task_type", "unet") == "yolo":
            self._radio_yolo.setChecked(True)
        else:
            self._radio_unet.setChecked(True)
        self._on_task_changed()  # show/hide class section

        self._project_edit.setText(s.get("project_name", ""))
        self._images_edit.setText(s.get("images_folder", ""))
        self._labels_edit.setText(s.get("labels_folder", ""))
        self._test_edit.setText(s.get("test_folder", ""))
        self._output_edit.setText(s.get("output_base", ""))
        self._class_edit.setText(s.get("class_names", ""))
        self._slider.setValue(int(s.get("train_pct", 80)))
        self._on_slider_changed(self._slider.value())

    def _autosave(self):
        self._settings["task_type"]     = "yolo" if self._radio_yolo.isChecked() else "unet"
        self._settings["project_name"]  = self._project_edit.text()
        self._settings["images_folder"] = self._images_edit.text()
        self._settings["labels_folder"] = self._labels_edit.text()
        self._settings["test_folder"]   = self._test_edit.text()
        self._settings["output_base"]   = self._output_edit.text()
        self._settings["class_names"]   = self._class_edit.text()
        self._settings["train_pct"]     = self._slider.value()
        _save_settings(self._settings)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_task_changed(self):
        is_yolo = self._radio_yolo.isChecked()
        self._class_section.setVisible(is_yolo)
        self._autosave()

    def _on_slider_changed(self, value: int):
        self._split_label.setText(f"Train: {value}%   Val: {100 - value}%")

    def _browse(self, edit: QLineEdit, key: str):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", edit.text() or "")
        if folder:
            edit.setText(folder)
            self._settings[key] = folder
            _save_settings(self._settings)

    # ------------------------------------------------------------------
    # Prepare action
    # ------------------------------------------------------------------

    def _on_prepare_clicked(self):
        # Validation
        project = self._project_edit.text().strip()
        images  = self._images_edit.text().strip()
        labels  = self._labels_edit.text().strip()
        output  = self._output_edit.text().strip()
        task    = "yolo" if self._radio_yolo.isChecked() else "unet"

        if not project:
            QMessageBox.warning(self, "Validation", "Project Name is required.")
            return
        if not os.path.isdir(images):
            QMessageBox.warning(self, "Validation", "Images Folder does not exist or is not set.")
            return
        if not os.path.isdir(labels):
            QMessageBox.warning(self, "Validation", "Labels / Masks Folder does not exist or is not set.")
            return
        if not os.path.isdir(output):
            QMessageBox.warning(self, "Validation", "Output Base Folder does not exist or is not set.")
            return
        class_names = []
        if task == "yolo":
            raw = self._class_edit.text().strip()
            if not raw:
                QMessageBox.warning(self, "Validation", "Class Names are required for YOLO.")
                return
            class_names = [c.strip() for c in raw.split(",") if c.strip()]

        config = {
            "task_type":     task,
            "images_folder": images,
            "labels_folder": labels,
            "test_folder":   self._test_edit.text().strip(),
            "output_base":   output,
            "project_name":  project,
            "train_pct":     self._slider.value(),
            "class_names":   class_names,
        }

        registry_root = RegistrySettings().get_registry_root()
        self._worker = DataPrepWorker(config, registry_root=registry_root, parent=self)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._progress_bar.setValue)
        self._worker.hash_progress.connect(self._on_hash_progress)
        self._worker.finished.connect(self._on_finished)

        self._result_card.hide()
        self._progress_area.show()
        self._hash_label.hide()
        self._progress_bar.setValue(0)
        self._prepare_btn.setText("Preparing...")
        self._prepare_btn.setStyleSheet(_PREPARE_RUNNING_STYLE)
        self._prepare_btn.setEnabled(False)
        self._console.clear()

        self._worker.start()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _append_log(self, line: str):
        self._console.append(line)
        self._console.verticalScrollBar().setValue(
            self._console.verticalScrollBar().maximum()
        )

    @pyqtSlot(int, int)
    def _on_hash_progress(self, current: int, total: int):
        self._hash_label.show()
        self._hash_label.setText(f"Hashing dataset... {current} / {total} files")

    @pyqtSlot(bool, dict)
    def _on_finished(self, success: bool, info: dict):
        self._prepare_btn.setText("▶  Prepare Dataset")
        self._prepare_btn.setStyleSheet(_PREPARE_IDLE_STYLE)
        self._prepare_btn.setEnabled(True)
        self._progress_area.hide()
        self._hash_label.hide()

        if success:
            train_n = info.get("train_count", 0)
            val_n   = info.get("val_count", 0)
            test_n  = info.get("test_count", 0)
            path    = info.get("output_folder", "")
            sha     = info.get("sha256", "")

            self._rc_counts.setText(
                f"Train: {train_n}   Val: {val_n}   Test: {test_n}"
            )
            max_path = 60
            disp_path = path if len(path) <= max_path else "..." + path[-(max_path - 3):]
            self._rc_path.setText(f"📁 {disp_path}")
            self._rc_hash.setText(f"SHA-256: {sha[:16]}..." if sha else "")
            gdrive_path = info.get("gdrive_dataset_path", "")
            if gdrive_path:
                self._rc_gdrive.setText(f"☁ GDrive: {gdrive_path}")
                self._rc_gdrive.show()
            else:
                self._rc_gdrive.setText("⚠ GDrive copy skipped (registry not configured)")
                self._rc_gdrive.show()
            self._result_card.show()
        else:
            self._append_log("<span style='color:#f87171'>[FAILED] Check the console for details.</span>")
