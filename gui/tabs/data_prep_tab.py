"""
gui/tabs/data_prep_tab.py

Data Preparation tab for DualAnnotator.
Hierarchy:  {gdrive_root}/{model_type}/{project_name}/{project_id}/{variant}/{camera}/
Dataset is staged to %LOCALAPPDATA%/DualAnnotator/staged_dataset/ and picked up by
the Training tab when training starts.
"""

import os
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QComboBox, QSlider, QTextEdit, QProgressBar, QFrame, QFileDialog,
    QMessageBox, QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont

from mlops.data_prep.prep_worker import DataPrepWorker
from mlops.data_prep.preparator import DataPreparator
from mlops.registry import RegistrySettings


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

_SETTINGS_DEFAULTS = {
    "task_type":     "unet",
    "project_name":  "",
    "project_id":    "",
    "variant":       "",
    "camera":        "",
    "images_folder": "",
    "labels_folder": "",
    "test_folder":   "",
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

_COMBO_STYLE = """
    QComboBox {
        background-color: #2d2d2d;
        border: 1px solid #3a3a3a;
        border-radius: 4px;
        color: #ffffff;
        padding: 5px 8px;
        font-size: 12px;
    }
    QComboBox:focus { border: 1px solid #6a3fc8; }
    QComboBox::drop-down { border: none; width: 20px; }
    QComboBox QAbstractItemView {
        background-color: #2d2d2d; color: #ffffff;
        selection-background-color: #6a3fc8;
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

_OPEN_TRAINING_STYLE = """
    QPushButton {
        background-color: #1a3a5c;
        color: #4fc3f7;
        border: 1px solid #4fc3f7;
        border-radius: 5px;
        padding: 8px 14px;
        font-size: 12px;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #1e4a70; }
    QPushButton:pressed { background-color: #163050; }
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


def _editable_combo(placeholder: str) -> QComboBox:
    cb = QComboBox()
    cb.setEditable(True)
    cb.setInsertPolicy(QComboBox.NoInsert)
    cb.setPlaceholderText(placeholder)
    cb.setStyleSheet(_COMBO_STYLE)
    return cb


# ---------------------------------------------------------------------------
# DataPrepTab
# ---------------------------------------------------------------------------

class DataPrepTab(QWidget):
    """Data preparation tab — left config panel + right console/progress."""

    # Emitted when "Open in Training" is clicked; payload = staged_info dict
    open_in_training = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = _load_settings()
        self._worker = None
        self._last_staged_info = {}
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

        # Section 2 — Hierarchy: Project Name / ID / Variant / Camera
        lbl_style = "color: #aaaaaa; font-size: 11px; background: transparent; border: none;"

        self._project_name_cb = _editable_combo("e.g. SnipeX")
        self._project_id_cb   = _editable_combo("e.g. SLXCAP")
        self._variant_cb      = _editable_combo("e.g. CRC")
        self._camera_cb       = _editable_combo("e.g. cam0")

        self._project_name_cb.currentTextChanged.connect(self._on_project_name_changed)
        self._project_id_cb.currentTextChanged.connect(self._on_project_id_changed)
        self._variant_cb.currentTextChanged.connect(self._on_variant_changed)
        self._camera_cb.currentTextChanged.connect(self._autosave)

        pn_lbl  = QLabel("Project Name"); pn_lbl.setStyleSheet(lbl_style)
        pid_lbl = QLabel("Project ID");   pid_lbl.setStyleSheet(lbl_style)
        var_lbl = QLabel("Variant");      var_lbl.setStyleSheet(lbl_style)
        cam_lbl = QLabel("Camera");       cam_lbl.setStyleSheet(lbl_style)

        layout.addWidget(_make_section([
            _section_label("Hierarchy"),
            pn_lbl,  self._project_name_cb,
            pid_lbl, self._project_id_cb,
            var_lbl, self._variant_cb,
            cam_lbl, self._camera_cb,
        ]))

        # Section 3 — Source folders
        self._images_edit, img_btn  = self._make_folder_field("Folder containing your images")
        self._labels_edit, lbl_btn  = self._make_folder_field("Folder containing .txt labels or .png masks")
        self._test_edit,   test_btn = self._make_folder_field("Optional — held-out test images (no labels needed)")

        img_btn.clicked.connect(lambda: self._browse(self._images_edit, "images_folder"))
        lbl_btn.clicked.connect(lambda: self._browse(self._labels_edit, "labels_folder"))
        test_btn.clicked.connect(lambda: self._browse(self._test_edit, "test_folder"))

        src_lbl1 = QLabel("Images Folder");         src_lbl1.setStyleSheet(lbl_style)
        src_lbl2 = QLabel("Labels / Masks Folder"); src_lbl2.setStyleSheet(lbl_style)
        src_lbl3 = QLabel("Test Images (optional)");src_lbl3.setStyleSheet(lbl_style)

        layout.addWidget(_make_section([
            _section_label("Source Folders"),
            src_lbl1, _folder_row(self._images_edit, img_btn),
            src_lbl2, _folder_row(self._labels_edit, lbl_btn),
            src_lbl3, _folder_row(self._test_edit, test_btn),
        ]))

        # Section 4 — Class names (YOLO only)
        self._class_section = QFrame()
        self._class_section.setStyleSheet(_SECTION_STYLE)
        cs_inner = QVBoxLayout(self._class_section)
        cs_inner.setContentsMargins(12, 10, 12, 10)
        cs_inner.setSpacing(8)
        cs_inner.addWidget(_section_label("Class Names (YOLO only)"))
        cls_note = QLabel("Comma-separated, in order")
        cls_note.setStyleSheet(lbl_style)
        cs_inner.addWidget(cls_note)
        self._class_edit = QLineEdit()
        self._class_edit.setPlaceholderText("background, defect_A, defect_B")
        self._class_edit.setStyleSheet(_INPUT_STYLE)
        self._class_edit.textChanged.connect(self._autosave)
        cs_inner.addWidget(self._class_edit)
        layout.addWidget(self._class_section)

        # Section 5 — Train/val split
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

        # Import existing dataset section
        layout.addWidget(self._build_import_section())
        layout.addStretch()

        outer = QWidget()
        outer.setFixedWidth(380)
        outer.setStyleSheet("background: #1e1e1e;")
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(scroll)
        return outer

    def _build_import_section(self) -> QFrame:
        """Collapsible section for importing an already-split dataset."""
        _sep_style = ("color: #555555; font-size: 10px; font-weight: bold; "
                      "background: transparent; border: none; margin-top: 4px;")
        _info_style = "color: #aaaaaa; font-size: 11px; background: transparent; border: none;"
        _ok_style   = "color: #4caf50; font-size: 11px; background: transparent; border: none;"
        _warn_style = "color: #f87171; font-size: 11px; background: transparent; border: none;"

        outer = QFrame()
        outer.setStyleSheet(_SECTION_STYLE)
        vlay = QVBoxLayout(outer)
        vlay.setContentsMargins(12, 10, 12, 10)
        vlay.setSpacing(8)

        hdr = QLabel("▸  IMPORT EXISTING DATASET")
        hdr.setStyleSheet(_sep_style)
        note = QLabel("Already have train/val folders? Generate the missing\n"
                      "dataset_info.json (+ data.yaml for YOLO) in-place.")
        note.setStyleSheet(_info_style)
        note.setWordWrap(True)
        vlay.addWidget(hdr)
        vlay.addWidget(note)

        self._imp_root_edit, imp_browse = self._make_folder_field("Root folder of your existing dataset")
        imp_browse.clicked.connect(self._on_import_browse)
        imp_root_lbl = QLabel("Dataset Root Folder")
        imp_root_lbl.setStyleSheet(_info_style)
        vlay.addWidget(imp_root_lbl)
        vlay.addLayout(_folder_row(self._imp_root_edit, imp_browse))

        _rb = "QRadioButton { color: #cccccc; font-size: 11px; background: transparent; border: none; }"
        self._imp_radio_yolo = QRadioButton("YOLO")
        self._imp_radio_unet = QRadioButton("UNet")
        self._imp_radio_yolo.setStyleSheet(_rb)
        self._imp_radio_unet.setStyleSheet(_rb)
        self._imp_radio_yolo.setChecked(True)
        self._imp_task_grp = QButtonGroup(self)
        self._imp_task_grp.addButton(self._imp_radio_yolo)
        self._imp_task_grp.addButton(self._imp_radio_unet)
        self._imp_radio_yolo.toggled.connect(self._on_imp_task_changed)
        rb_row = QHBoxLayout()
        task_lbl = QLabel("Task Type"); task_lbl.setStyleSheet(_info_style)
        rb_row.addWidget(task_lbl)
        rb_row.addWidget(self._imp_radio_yolo)
        rb_row.addWidget(self._imp_radio_unet)
        rb_row.addStretch()
        vlay.addLayout(rb_row)

        scan_btn = QPushButton("  Scan Dataset")
        scan_btn.setStyleSheet(_BROWSE_STYLE)
        scan_btn.clicked.connect(self._on_import_scan)
        vlay.addWidget(scan_btn)

        self._imp_layout_lbl = QLabel("")
        self._imp_layout_lbl.setStyleSheet(_info_style)
        self._imp_counts_lbl = QLabel("")
        self._imp_counts_lbl.setStyleSheet(_ok_style)
        self._imp_warn_lbl   = QLabel("")
        self._imp_warn_lbl.setStyleSheet(_warn_style)
        self._imp_warn_lbl.setWordWrap(True)
        for lbl in (self._imp_layout_lbl, self._imp_counts_lbl, self._imp_warn_lbl):
            lbl.hide()
            vlay.addWidget(lbl)

        self._imp_class_section = QWidget()
        cs_lay = QVBoxLayout(self._imp_class_section)
        cs_lay.setContentsMargins(0, 0, 0, 0); cs_lay.setSpacing(4)
        cs_lbl = QLabel("Class Names (comma-separated, in order):")
        cs_lbl.setStyleSheet(_info_style)
        self._imp_class_edit = QLineEdit()
        self._imp_class_edit.setPlaceholderText("e.g. background, tablets, capsules")
        self._imp_class_edit.setStyleSheet(_INPUT_STYLE)
        cs_lay.addWidget(cs_lbl)
        cs_lay.addWidget(self._imp_class_edit)
        vlay.addWidget(self._imp_class_section)

        self._imp_gen_btn = QPushButton("  Generate dataset_info.json + data.yaml")
        self._imp_gen_btn.setStyleSheet(_PREPARE_IDLE_STYLE)
        self._imp_gen_btn.setEnabled(False)
        self._imp_gen_btn.clicked.connect(self._on_import_generate)
        vlay.addWidget(self._imp_gen_btn)

        self._on_imp_task_changed()
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

        # Result card
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

        self._rc_title = QLabel("✓ Dataset staged!")
        self._rc_title.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 13px;")
        self._rc_counts = QLabel("")
        self._rc_counts.setStyleSheet("color: #cccccc; font-size: 12px;")
        self._rc_path = QLabel("")
        self._rc_path.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self._rc_path.setWordWrap(True)
        self._rc_hash = QLabel("")
        self._rc_hash.setStyleSheet("color: #8ab4f8; font-size: 11px;")
        self._rc_hierarchy = QLabel("")
        self._rc_hierarchy.setStyleSheet("color: #4fc3f7; font-size: 11px;")
        self._rc_hierarchy.setWordWrap(True)

        self._open_training_btn = QPushButton("→  Open in Training Tab")
        self._open_training_btn.setStyleSheet(_OPEN_TRAINING_STYLE)
        self._open_training_btn.clicked.connect(self._on_open_training_clicked)

        rc_layout.addWidget(self._rc_title)
        rc_layout.addWidget(self._rc_counts)
        rc_layout.addWidget(self._rc_path)
        rc_layout.addWidget(self._rc_hash)
        rc_layout.addWidget(self._rc_hierarchy)
        rc_layout.addWidget(self._open_training_btn)

        self._result_card.hide()
        layout.addWidget(self._result_card)

        return panel

    # ------------------------------------------------------------------
    # Hierarchy scanning
    # ------------------------------------------------------------------

    def _current_model_type(self) -> str:
        return "yolo" if self._radio_yolo.isChecked() else "unet"

    def _refresh_project_names(self):
        settings = RegistrySettings()
        names = settings.scan_project_names(self._current_model_type())
        cur = self._project_name_cb.currentText()
        self._project_name_cb.blockSignals(True)
        self._project_name_cb.clear()
        for n in names:
            self._project_name_cb.addItem(n)
        if cur:
            self._project_name_cb.setCurrentText(cur)
        self._project_name_cb.blockSignals(False)

    def _refresh_project_ids(self):
        settings = RegistrySettings()
        ids = settings.scan_project_ids(
            self._current_model_type(),
            self._project_name_cb.currentText().strip(),
        )
        cur = self._project_id_cb.currentText()
        self._project_id_cb.blockSignals(True)
        self._project_id_cb.clear()
        for i in ids:
            self._project_id_cb.addItem(i)
        if cur:
            self._project_id_cb.setCurrentText(cur)
        self._project_id_cb.blockSignals(False)

    def _refresh_variants(self):
        settings = RegistrySettings()
        variants = settings.scan_variants(
            self._current_model_type(),
            self._project_name_cb.currentText().strip(),
            self._project_id_cb.currentText().strip(),
        )
        cur = self._variant_cb.currentText()
        self._variant_cb.blockSignals(True)
        self._variant_cb.clear()
        for v in variants:
            self._variant_cb.addItem(v)
        if cur:
            self._variant_cb.setCurrentText(cur)
        self._variant_cb.blockSignals(False)

    def _refresh_cameras(self):
        settings = RegistrySettings()
        cameras = settings.scan_cameras(
            self._current_model_type(),
            self._project_name_cb.currentText().strip(),
            self._project_id_cb.currentText().strip(),
            self._variant_cb.currentText().strip(),
        )
        cur = self._camera_cb.currentText()
        self._camera_cb.blockSignals(True)
        self._camera_cb.clear()
        for c in cameras:
            self._camera_cb.addItem(c)
        if cur:
            self._camera_cb.setCurrentText(cur)
        self._camera_cb.blockSignals(False)

    # ------------------------------------------------------------------
    # Settings restore
    # ------------------------------------------------------------------

    def _restore_settings(self):
        s = self._settings
        if s.get("task_type", "unet") == "yolo":
            self._radio_yolo.setChecked(True)
        else:
            self._radio_unet.setChecked(True)
        self._on_task_changed()

        self._refresh_project_names()
        if s.get("project_name"):
            self._project_name_cb.setCurrentText(s["project_name"])
        self._refresh_project_ids()
        if s.get("project_id"):
            self._project_id_cb.setCurrentText(s["project_id"])
        self._refresh_variants()
        if s.get("variant"):
            self._variant_cb.setCurrentText(s["variant"])
        self._refresh_cameras()
        if s.get("camera"):
            self._camera_cb.setCurrentText(s["camera"])

        self._images_edit.setText(s.get("images_folder", ""))
        self._labels_edit.setText(s.get("labels_folder", ""))
        self._test_edit.setText(s.get("test_folder", ""))
        self._class_edit.setText(s.get("class_names", ""))
        self._slider.setValue(int(s.get("train_pct", 80)))
        self._on_slider_changed(self._slider.value())

    def _autosave(self):
        self._settings["task_type"]     = "yolo" if self._radio_yolo.isChecked() else "unet"
        self._settings["project_name"]  = self._project_name_cb.currentText().strip()
        self._settings["project_id"]    = self._project_id_cb.currentText().strip()
        self._settings["variant"]       = self._variant_cb.currentText().strip()
        self._settings["camera"]        = self._camera_cb.currentText().strip()
        self._settings["images_folder"] = self._images_edit.text()
        self._settings["labels_folder"] = self._labels_edit.text()
        self._settings["test_folder"]   = self._test_edit.text()
        self._settings["class_names"]   = self._class_edit.text()
        self._settings["train_pct"]     = self._slider.value()
        _save_settings(self._settings)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_task_changed(self):
        is_yolo = self._radio_yolo.isChecked()
        self._class_section.setVisible(is_yolo)
        self._refresh_project_names()
        self._autosave()

    def _on_project_name_changed(self):
        self._refresh_project_ids()
        self._refresh_variants()
        self._refresh_cameras()
        self._autosave()

    def _on_project_id_changed(self):
        self._refresh_variants()
        self._refresh_cameras()
        self._autosave()

    def _on_variant_changed(self):
        self._refresh_cameras()
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
    # Import existing dataset handlers
    # ------------------------------------------------------------------

    def _on_imp_task_changed(self):
        self._imp_class_section.setVisible(self._imp_radio_yolo.isChecked())

    def _on_import_browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Root Folder",
            self._imp_root_edit.text() or "")
        if not folder:
            return
        self._imp_root_edit.setText(folder)
        self._on_import_scan()

    def _on_import_scan(self):
        root = self._imp_root_edit.text().strip()
        if not root:
            return

        result = DataPreparator.scan_existing_presplit(root)

        if result["task_type"] == "yolo":
            self._imp_radio_yolo.setChecked(True)
        elif result["task_type"] == "unet":
            self._imp_radio_unet.setChecked(True)

        if result["class_names"]:
            self._imp_class_edit.setText(", ".join(result["class_names"]))

        layout_map = {
            "trainval_images": "Layout: train/images/ + val/images/",
            "images_trainval": "Layout: images/train/ + images/val/",
            "flat":            "Layout: flat images/ + labels/ (no split yet — use Prepare above)",
            "unknown":         "Layout: unknown",
        }
        self._imp_layout_lbl.setText(layout_map.get(result["layout"], ""))
        self._imp_layout_lbl.show()

        if result["errors"]:
            self._imp_warn_lbl.setText("  " + "\n".join(result["errors"]))
            self._imp_warn_lbl.show()
            self._imp_counts_lbl.hide()
            self._imp_gen_btn.setEnabled(False)
            return

        self._imp_warn_lbl.hide()

        flags = []
        if result["has_dataset_info"]:
            flags.append("dataset_info.json exists")
        if result["has_data_yaml"]:
            flags.append("data.yaml exists")

        counts_txt = (f"Train: {result['train_count']}  Val: {result['val_count']}  "
                      f"Test: {result['test_count']}")
        if flags:
            counts_txt += f"  ({', '.join(flags)})"
        self._imp_counts_lbl.setText(counts_txt)
        self._imp_counts_lbl.show()

        can_generate = result["layout"] in ("trainval_images", "images_trainval")
        self._imp_gen_btn.setEnabled(can_generate)

        if result["layout"] == "flat":
            self._imp_warn_lbl.setText(
                "  Flat layout detected — use the Prepare Dataset section above "
                "to split this dataset into train/val first.")
            self._imp_warn_lbl.show()
            self._imp_gen_btn.setEnabled(False)

    def _on_import_generate(self):
        root      = self._imp_root_edit.text().strip()
        task_type = "yolo" if self._imp_radio_yolo.isChecked() else "unet"

        if not os.path.isdir(root):
            QMessageBox.warning(self, "Import", "Dataset root folder not found.")
            return

        result = DataPreparator.scan_existing_presplit(root)
        if result["layout"] not in ("trainval_images", "images_trainval"):
            QMessageBox.warning(self, "Import", "Could not detect a valid pre-split layout.")
            return

        class_names = []
        if task_type == "yolo":
            raw = self._imp_class_edit.text().strip()
            if not raw:
                QMessageBox.warning(self, "Import", "Class Names are required for YOLO.")
                return
            class_names = [c.strip() for c in raw.split(",") if c.strip()]

        try:
            DataPreparator.generate_presplit_metadata(
                root, task_type, class_names, result["layout"])
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))
            return

        files = ["dataset_info.json"]
        if task_type == "yolo":
            files.append("data.yaml")
        QMessageBox.information(
            self, "Done",
            f"Generated: {', '.join(files)}\n\nDataset is ready to use in the Training tab.")
        self._on_import_scan()

    # ------------------------------------------------------------------
    # Prepare action
    # ------------------------------------------------------------------

    def _on_prepare_clicked(self):
        task         = "yolo" if self._radio_yolo.isChecked() else "unet"
        project_name = self._project_name_cb.currentText().strip()
        project_id   = self._project_id_cb.currentText().strip()
        variant      = self._variant_cb.currentText().strip()
        camera       = self._camera_cb.currentText().strip()
        images       = self._images_edit.text().strip()
        labels       = self._labels_edit.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Validation", "Project Name is required.")
            return
        if not project_id:
            QMessageBox.warning(self, "Validation", "Project ID is required.")
            return
        if not variant:
            QMessageBox.warning(self, "Validation", "Variant is required.")
            return
        if not camera:
            QMessageBox.warning(self, "Validation", "Camera is required.")
            return
        if not os.path.isdir(images):
            QMessageBox.warning(self, "Validation", "Images Folder does not exist or is not set.")
            return
        if not os.path.isdir(labels):
            QMessageBox.warning(self, "Validation", "Labels / Masks Folder does not exist or is not set.")
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
            "model_type":    task,
            "project_name":  project_name,
            "project_id":    project_id,
            "variant":       variant,
            "camera":        camera,
            "images_folder": images,
            "labels_folder": labels,
            "test_folder":   self._test_edit.text().strip(),
            "train_pct":     self._slider.value(),
            "class_names":   class_names,
        }

        self._worker = DataPrepWorker(config, parent=self)
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
            self._last_staged_info = info

            ds_info  = info.get("dataset_info", {})
            train_n  = ds_info.get("train_count", 0)
            val_n    = ds_info.get("val_count", 0)
            test_n   = ds_info.get("test_count", 0)
            path     = info.get("path", "")
            sha      = ds_info.get("sha256", info.get("sha256", ""))

            self._rc_counts.setText(f"Train: {train_n}   Val: {val_n}   Test: {test_n}")

            max_path = 60
            disp_path = path if len(path) <= max_path else "..." + path[-(max_path - 3):]
            self._rc_path.setText(f"Staged: {disp_path}")
            self._rc_hash.setText(f"SHA-256: {sha[:16]}..." if sha else "")

            hierarchy = (f"{info.get('model_type', '')} / {info.get('project_name', '')} / "
                         f"{info.get('project_id', '')} / {info.get('variant', '')} / "
                         f"{info.get('camera', '')}")
            self._rc_hierarchy.setText(f"→ {hierarchy}")

            self._result_card.show()
        else:
            self._append_log("<span style='color:#f87171'>[FAILED] Check the console for details.</span>")

    @pyqtSlot()
    def _on_open_training_clicked(self):
        if self._last_staged_info:
            self.open_in_training.emit(self._last_staged_info)
