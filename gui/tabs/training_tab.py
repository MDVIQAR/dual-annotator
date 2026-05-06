"""
gui/tabs/training_tab.py

Phase 3 — Training tab for DualAnnotator.
"""

import os
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QProgressBar, QFrame, QFileDialog, QMessageBox, QScrollArea,
    QRadioButton, QButtonGroup,
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont

# Matplotlib integration (wrapped in try/except)
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from mlops.registry import RegistrySettings, RegistryScanner
from mlops.training import TrainWorker, run_preflight

_SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mlops", "scripts")
)

_SETTINGS_DEFAULTS = {
    "task_type":          "unet",
    "annotator_initials": "",
    "project":            "",
    "dataset_folder":     "",
    "architecture":       "Unet",
    "encoder":            "efficientnet-b3",
    "encoder_weights":    "imagenet",
    "yolo_model":         "yolov8n",
    "epochs":             100,
    "batch_size":         4,
    "learning_rate":      0.001,
    "image_width":        320,
    "image_height":       240,
    "in_channels":        3,
    "out_classes":        2,
    "device":             "cpu",
}

def _settings_path() -> str:
    app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(app_data, "DualAnnotator", "training_settings.json")

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

_SECTION_STYLE = """
    QFrame {
        background-color: #252526;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
    }
"""

_INPUT_STYLE = """
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
        background-color: #2d2d2d;
        border: 1px solid #3a3a3a;
        border-radius: 4px;
        color: #ffffff;
        padding: 5px 8px;
        font-size: 12px;
    }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #6a3fc8;
    }
    QComboBox::drop-down { border: none; }
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

_START_STYLE = """
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
    QPushButton:disabled { background-color: #444444; color: #888888; }
"""

_STOP_STYLE = """
    QPushButton {
        background-color: #c0392b;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #d35400; }
    QPushButton:pressed { background-color: #e74c3c; }
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

def _make_row(label_text: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    lbl = QLabel(label_text)
    lbl.setFixedWidth(120)
    row.addWidget(lbl)
    row.addWidget(widget)
    return row

class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = _load_settings()
        self._worker = None
        self._dataset_info = {}
        self._epochs_seen = []
        self._train_losses = []
        self._val_losses = []
        self._registry_root = RegistrySettings().get_registry_root()
        
        self._setup_ui()
        self._restore_settings()
        
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

        # Section 0 - Task Type
        self._radio_yolo = QRadioButton("YOLO (Detection)")
        self._radio_unet = QRadioButton("UNet (Segmentation)")
        self._radio_yolo.setStyleSheet("color: #ffffff; background: transparent; border: none;")
        self._radio_unet.setStyleSheet("color: #ffffff; background: transparent; border: none;")
        self._task_group = QButtonGroup(self)
        self._task_group.addButton(self._radio_yolo)
        self._task_group.addButton(self._radio_unet)
        self._radio_unet.setChecked(True)
        self._radio_yolo.toggled.connect(self._on_task_changed)
        layout.addWidget(_make_section([_section_label("Task Type"), self._radio_yolo, self._radio_unet]))

        # Section 1 - Run Info
        self._annotator_cb = QComboBox()
        self._annotator_cb.setStyleSheet(_INPUT_STYLE)
        for ann in RegistrySettings().get_annotators():
            self._annotator_cb.addItem(f"{ann['name']} ({ann['initials']})", userData=ann)
        self._annotator_cb.currentIndexChanged.connect(self._autosave)

        self._project_cb = QComboBox()
        self._project_cb.setEditable(True)
        self._project_cb.setStyleSheet(_INPUT_STYLE)
        self._project_cb.addItem("— select project —")
        for proj in RegistryScanner(self._registry_root).get_projects():
            self._project_cb.addItem(proj)
        self._project_cb.currentTextChanged.connect(self._autosave)

        self._commit_edit = QLineEdit()
        self._commit_edit.setPlaceholderText("e.g. Added thermal augmentation")
        self._commit_edit.setStyleSheet(_INPUT_STYLE)
        
        layout.addWidget(_make_section([
            _section_label("Run Info"),
            _make_row("Annotator", self._annotator_cb),
            _make_row("Project", self._project_cb),
            _make_row("Commit Msg", self._commit_edit),
        ]))
        
        # Section 2 - Dataset
        self._ds_edit = QLineEdit()
        self._ds_edit.setReadOnly(True)
        self._ds_edit.setStyleSheet(_INPUT_STYLE)
        
        ds_btn = QPushButton("Browse")
        ds_btn.setStyleSheet(_BROWSE_STYLE)
        ds_btn.clicked.connect(self._browse_dataset)
        
        ds_row = QHBoxLayout()
        ds_row.addWidget(self._ds_edit)
        ds_row.addWidget(ds_btn)
        
        self._ds_info_lbl = QLabel("")
        self._ds_info_lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        
        layout.addWidget(_make_section([
            _section_label("Dataset"),
            ds_row,
            self._ds_info_lbl
        ]))
        
        # Section 3 - Architecture
        self._arch_section = QFrame()
        self._arch_section.setStyleSheet(_SECTION_STYLE)
        arch_layout = QVBoxLayout(self._arch_section)
        arch_layout.setContentsMargins(12, 10, 12, 10)
        arch_layout.setSpacing(8)
        arch_layout.addWidget(_section_label("Model Architecture"))
        
        # UNet block
        self._unet_block = QWidget()
        unet_l = QVBoxLayout(self._unet_block)
        unet_l.setContentsMargins(0, 0, 0, 0)
        unet_l.setSpacing(8)
        
        self._unet_arch_cb = QComboBox()
        self._unet_arch_cb.addItems(["Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN"])
        self._unet_arch_cb.setStyleSheet(_INPUT_STYLE)
        self._unet_arch_cb.currentTextChanged.connect(self._autosave)
        
        self._encoder_cb = QComboBox()
        self._encoder_cb.addItems(["efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", "resnet34", "resnet50", "resnet101", "mobilenet_v2"])
        self._encoder_cb.setStyleSheet(_INPUT_STYLE)
        self._encoder_cb.currentTextChanged.connect(self._autosave)
        
        self._weights_cb = QComboBox()
        self._weights_cb.addItems(["imagenet", "none"])
        self._weights_cb.setStyleSheet(_INPUT_STYLE)
        self._weights_cb.currentTextChanged.connect(self._autosave)
        
        unet_l.addLayout(_make_row("Architecture", self._unet_arch_cb))
        unet_l.addLayout(_make_row("Encoder", self._encoder_cb))
        unet_l.addLayout(_make_row("Weights", self._weights_cb))
        
        # YOLO block
        self._yolo_block = QWidget()
        yolo_l = QVBoxLayout(self._yolo_block)
        yolo_l.setContentsMargins(0, 0, 0, 0)
        yolo_l.setSpacing(8)
        
        self._yolo_model_cb = QComboBox()
        self._yolo_model_cb.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self._yolo_model_cb.setStyleSheet(_INPUT_STYLE)
        self._yolo_model_cb.currentTextChanged.connect(self._autosave)
        yolo_l.addLayout(_make_row("Model", self._yolo_model_cb))
        
        arch_layout.addWidget(self._unet_block)
        arch_layout.addWidget(self._yolo_block)
        layout.addWidget(self._arch_section)
        
        # Section 4 - Hyperparameters
        self._epochs_sp = QSpinBox()
        self._epochs_sp.setRange(1, 1000)
        self._epochs_sp.setStyleSheet(_INPUT_STYLE)
        self._epochs_sp.valueChanged.connect(self._autosave)
        
        self._batch_sp = QSpinBox()
        self._batch_sp.setRange(1, 128)
        self._batch_sp.setStyleSheet(_INPUT_STYLE)
        self._batch_sp.valueChanged.connect(self._autosave)
        
        self._lr_sp = QDoubleSpinBox()
        self._lr_sp.setRange(0.000001, 1.0)
        self._lr_sp.setDecimals(6)
        self._lr_sp.setSingleStep(0.0001)
        self._lr_sp.setStyleSheet(_INPUT_STYLE)
        self._lr_sp.valueChanged.connect(self._autosave)
        
        self._width_sp = QSpinBox()
        self._width_sp.setRange(32, 4096)
        self._width_sp.setSingleStep(32)
        self._width_sp.setStyleSheet(_INPUT_STYLE)
        self._width_sp.valueChanged.connect(self._autosave)
        
        self._height_sp = QSpinBox()
        self._height_sp.setRange(32, 4096)
        self._height_sp.setSingleStep(32)
        self._height_sp.setStyleSheet(_INPUT_STYLE)
        self._height_sp.valueChanged.connect(self._autosave)
        
        self._device_cb = QComboBox()
        self._device_cb.addItems(["cpu", "cuda:0", "cuda:1"])
        self._device_cb.setStyleSheet(_INPUT_STYLE)
        self._device_cb.currentTextChanged.connect(self._autosave)

        self._in_channels_sp = QSpinBox()
        self._in_channels_sp.setRange(1, 10)
        self._in_channels_sp.setStyleSheet(_INPUT_STYLE)
        self._in_channels_sp.valueChanged.connect(self._autosave)

        self._out_classes_sp = QSpinBox()
        self._out_classes_sp.setRange(1, 256)
        self._out_classes_sp.setStyleSheet(_INPUT_STYLE)
        self._out_classes_sp.valueChanged.connect(self._autosave)

        layout.addWidget(_make_section([
            _section_label("Hyperparameters"),
            _make_row("Epochs", self._epochs_sp),
            _make_row("Batch Size", self._batch_sp),
            _make_row("Learning Rate", self._lr_sp),
            _make_row("Image Width", self._width_sp),
            _make_row("Image Height", self._height_sp),
            _make_row("In Channels", self._in_channels_sp),
            _make_row("Out Classes", self._out_classes_sp),
            _make_row("Device", self._device_cb),
        ]))
        
        # Section 5 - Pre-flight Checks
        self._checks_frame = QFrame()
        self._checks_frame.setStyleSheet(_SECTION_STYLE)
        self._checks_layout = QVBoxLayout(self._checks_frame)
        self._checks_layout.setContentsMargins(12, 10, 12, 10)
        self._checks_layout.setSpacing(8)
        
        self._checks_layout.addWidget(_section_label("Pre-flight Checks"))
        
        chk_btn = QPushButton("Run Checks")
        chk_btn.setStyleSheet(_BROWSE_STYLE)
        chk_btn.clicked.connect(self._run_checks)
        self._checks_layout.addWidget(chk_btn)
        
        self._checks_list_layout = QVBoxLayout()
        self._checks_list_layout.setSpacing(4)
        self._checks_layout.addLayout(self._checks_list_layout)
        
        layout.addWidget(self._checks_frame)
        
        # Start button
        self._start_btn = QPushButton("▶  Start Training")
        self._start_btn.setStyleSheet(_START_STYLE)
        self._start_btn.clicked.connect(self._on_start_stop)
        layout.addWidget(self._start_btn)
        
        layout.addStretch()
        
        outer = QWidget()
        outer.setFixedWidth(380)
        outer.setStyleSheet("background: #1e1e1e;")
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(scroll)
        return outer
        
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background-color: #1e1e1e;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2a2a; }")
        
        # Top - Graph
        graph_widget = QWidget()
        graph_l = QVBoxLayout(graph_widget)
        graph_l.setContentsMargins(0, 0, 0, 0)
        
        if HAS_MATPLOTLIB:
            self._fig = Figure(facecolor="#1a1a1a")
            self._ax = self._fig.add_subplot(111, facecolor="#1a1a1a")
            self._ax.tick_params(colors="#aaaaaa")
            self._ax.xaxis.label.set_color("#aaaaaa")
            self._ax.yaxis.label.set_color("#aaaaaa")
            self._ax.set_xlabel("Epoch")
            self._ax.set_ylabel("Loss")
            for spine in self._ax.spines.values():
                spine.set_edgecolor("#3a3a3a")
            self._fig.tight_layout()
            
            self._canvas = FigureCanvas(self._fig)
            graph_l.addWidget(self._canvas)
        else:
            lbl = QLabel("Matplotlib not installed. Graph unavailable.")
            lbl.setStyleSheet("color: #aaaaaa; font-style: italic;")
            lbl.setAlignment(Qt.AlignCenter)
            graph_l.addWidget(lbl)
            
        splitter.addWidget(graph_widget)
        
        # Middle - Progress (hidden by default)
        self._prog_frame = QFrame()
        self._prog_frame.hide()
        pl = QVBoxLayout(self._prog_frame)
        pl.setContentsMargins(0, 0, 0, 0)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2d2d2d; border: 1px solid #3a3a3a;
                border-radius: 4px; color: #ffffff; text-align: center; height: 18px;
            }
            QProgressBar::chunk { background-color: #6a3fc8; border-radius: 3px; }
        """)
        pl.addWidget(self._progress_bar)
        
        # Bottom - Console
        console_widget = QWidget()
        console_l = QVBoxLayout(console_widget)
        console_l.setContentsMargins(0, 0, 0, 0)
        
        hdr = QHBoxLayout()
        hdr_lbl = QLabel("Training Console")
        hdr_lbl.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        clr_btn = QPushButton("Clear")
        clr_btn.setFixedWidth(60)
        clr_btn.setStyleSheet(_BROWSE_STYLE)
        clr_btn.clicked.connect(lambda: self._console.clear())
        hdr.addWidget(hdr_lbl)
        hdr.addStretch()
        hdr.addWidget(clr_btn)
        console_l.addLayout(hdr)
        
        console_l.addWidget(self._prog_frame)
        
        self._console = QTextEdit()
        self._console.setReadOnly(True)
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.Monospace)
        self._console.setFont(font)
        self._console.setStyleSheet("""
            QTextEdit { background-color: #1a1a1a; color: #d4d4d4;
            border: 1px solid #2a2a2a; border-radius: 4px; padding: 6px; }
        """)
        console_l.addWidget(self._console)
        
        self._result_card = QFrame()
        self._result_card.setStyleSheet("""
            QFrame { background-color: #1e2e1e; border: 1px solid #3a3a3a;
            border-left: 3px solid #4caf50; border-radius: 4px; padding: 4px; }
        """)
        rc_l = QVBoxLayout(self._result_card)
        rc_l.setContentsMargins(12, 8, 12, 8)
        self._rc_lbl = QLabel("")
        self._rc_lbl.setStyleSheet("color: #4caf50; font-weight: bold;")
        rc_l.addWidget(self._rc_lbl)
        self._result_card.hide()
        console_l.addWidget(self._result_card)
        
        splitter.addWidget(console_widget)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        return panel

    def _restore_settings(self):
        s = self._settings
        # Find annotator index by initials
        initials = s.get("annotator_initials", "")
        if initials:
            for i in range(self._annotator_cb.count()):
                if self._annotator_cb.itemData(i).get("initials") == initials:
                    self._annotator_cb.setCurrentIndex(i)
                    break
                    
        proj = s.get("project", "")
        if proj:
            idx = self._project_cb.findText(proj)
            if idx >= 0:
                self._project_cb.setCurrentIndex(idx)
            else:
                self._project_cb.setCurrentText(proj)
                
        self._unet_arch_cb.setCurrentText(s.get("architecture", "Unet"))
        self._encoder_cb.setCurrentText(s.get("encoder", "efficientnet-b3"))
        self._weights_cb.setCurrentText(s.get("encoder_weights", "imagenet"))
        self._yolo_model_cb.setCurrentText(s.get("yolo_model", "yolov8n"))
        self._epochs_sp.setValue(int(s.get("epochs", 100)))
        self._batch_sp.setValue(int(s.get("batch_size", 4)))
        self._lr_sp.setValue(float(s.get("learning_rate", 0.001)))
        self._width_sp.setValue(int(s.get("image_width", 320)))
        self._height_sp.setValue(int(s.get("image_height", 240)))
        self._in_channels_sp.setValue(int(s.get("in_channels", 3)))
        self._out_classes_sp.setValue(int(s.get("out_classes", 2)))
        self._device_cb.setCurrentText(s.get("device", "cpu"))
        
        if s.get("task_type", "unet") == "yolo":
            self._radio_yolo.setChecked(True)
        else:
            self._radio_unet.setChecked(True)
        self._on_task_changed()

        ds_path = s.get("dataset_folder", "")
        if ds_path and os.path.isdir(ds_path):
            self._load_dataset_info(ds_path)

    def _on_task_changed(self):
        is_yolo = self._radio_yolo.isChecked()
        self._unet_block.setVisible(not is_yolo)
        self._yolo_block.setVisible(is_yolo)
        self._autosave()

    def _autosave(self):
        s = self._settings
        s["task_type"] = "yolo" if self._radio_yolo.isChecked() else "unet"

        ann_data = self._annotator_cb.currentData()
        if ann_data:
            s["annotator_initials"] = ann_data.get("initials", "")

        proj = self._project_cb.currentText()
        s["project"] = proj if proj != "— select project —" else ""

        s["dataset_folder"] = self._ds_edit.text()
        s["architecture"] = self._unet_arch_cb.currentText()
        s["encoder"] = self._encoder_cb.currentText()
        s["encoder_weights"] = self._weights_cb.currentText()
        s["yolo_model"] = self._yolo_model_cb.currentText()
        s["epochs"] = self._epochs_sp.value()
        s["batch_size"] = self._batch_sp.value()
        s["learning_rate"] = self._lr_sp.value()
        s["image_width"] = self._width_sp.value()
        s["image_height"] = self._height_sp.value()
        s["in_channels"] = self._in_channels_sp.value()
        s["out_classes"] = self._out_classes_sp.value()
        s["device"] = self._device_cb.currentText()
        _save_settings(s)

    def _browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", self._ds_edit.text() or "")
        if folder:
            self._load_dataset_info(folder)
            self._autosave()
            
    def _load_dataset_info(self, folder: str):
        self._ds_edit.setText(folder)
        info_path = os.path.join(folder, "dataset_info.json")
        if not os.path.isfile(info_path):
            self._ds_info_lbl.setText("⚠ dataset_info.json not found")
            self._ds_info_lbl.setStyleSheet("color: #f87171; font-size: 11px;")
            self._dataset_info = {}
            return
            
        try:
            with open(info_path, "r", encoding="utf-8") as fh:
                self._dataset_info = json.load(fh)
            t = self._dataset_info.get("task_type", "unet").upper()
            tr = self._dataset_info.get("train_count", 0)
            v = self._dataset_info.get("val_count", 0)
            ts = self._dataset_info.get("test_count", 0)
            self._ds_info_lbl.setText(f"Task: {t}  |  Train: {tr}  Val: {v}  Test: {ts}")
            self._ds_info_lbl.setStyleSheet("color: #4caf50; font-size: 11px;")
            
            # Auto-fill out_classes from dataset (user can still override)
            num_classes = self._dataset_info.get("num_classes", 0)
            if num_classes > 0:
                self._out_classes_sp.setValue(num_classes)

            # Sync radio to dataset task_type (user can still override)
            if t.lower() == "yolo":
                self._radio_yolo.setChecked(True)
            else:
                self._radio_unet.setChecked(True)
        except Exception:
            self._ds_info_lbl.setText("⚠ Failed to parse dataset_info.json")
            self._ds_info_lbl.setStyleSheet("color: #f87171; font-size: 11px;")
            self._dataset_info = {}

    def _get_form_dict(self) -> dict:
        ann_data = self._annotator_cb.currentData() or {}
        proj = self._project_cb.currentText()
        proj = proj if proj != "— select project —" else ""
        
        task_type = "yolo" if self._radio_yolo.isChecked() else "unet"
        arch = self._yolo_model_cb.currentText() if task_type == "yolo" else self._unet_arch_cb.currentText()
        
        return {
            "annotator_name":     ann_data.get("name", ""),
            "annotator_initials": ann_data.get("initials", ""),
            "project":            proj,
            "commit_message":     self._commit_edit.text(),
            "dataset_folder":     self._ds_edit.text(),
            "architecture":       arch,
            "encoder":            self._encoder_cb.currentText(),
            "encoder_weights":    self._weights_cb.currentText(),
            "epochs":             self._epochs_sp.value(),
            "batch_size":         self._batch_sp.value(),
            "learning_rate":      self._lr_sp.value(),
            "image_width":        self._width_sp.value(),
            "image_height":       self._height_sp.value(),
            "in_channels":        self._in_channels_sp.value(),
            "out_classes":        self._out_classes_sp.value(),
            "device":             self._device_cb.currentText(),
        }

    def _run_checks(self, show_dialog_on_failure: bool = False) -> bool:
        # clear old checks
        while self._checks_list_layout.count():
            item = self._checks_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        try:
            from mlops.training import build_training_config
            cfg = build_training_config(self._get_form_dict())
            results = run_preflight(cfg, self._registry_root, _SCRIPTS_DIR)
        except Exception as e:
            lbl = QLabel(f"✗ Configuration error: {e}")
            lbl.setStyleSheet("color: #f87171; font-size: 11px;")
            lbl.setWordWrap(True)
            self._checks_list_layout.addWidget(lbl)
            return False
            
        all_ok = True
        failed_msgs = []
        for r in results:
            if r["ok"]:
                icon = "✓"
                color = "#4caf50"
            elif r["warn"]:
                icon = "⚠"
                color = "#ffc107"
            else:
                icon = "✗"
                color = "#f87171"
                all_ok = False
                failed_msgs.append(r["name"])
                
            lbl = QLabel(f"{icon} {r['name']}: {r['msg']}")
            lbl.setStyleSheet(f"color: {color}; font-size: 11px;")
            lbl.setWordWrap(True)
            self._checks_list_layout.addWidget(lbl)
            
        if not all_ok and show_dialog_on_failure:
            QMessageBox.warning(self, "Pre-flight Checks Failed", "Fix the following issues:\n" + "\n".join(failed_msgs))

        return all_ok

    def _on_start_stop(self):
        if self._worker is not None:
            # currently running, stop it
            self._worker.cancel()
            self._start_btn.setEnabled(False)
            return
            
        if not self._run_checks(show_dialog_on_failure=True):
            return
            
        try:
            from mlops.training import build_training_config
            cfg = build_training_config(self._get_form_dict())
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return
            
        self._result_card.hide()
        self._prog_frame.show()
        self._progress_bar.setValue(0)
        self._console.clear()
        
        self._epochs_seen = []
        self._train_losses = []
        self._val_losses = []
        if HAS_MATPLOTLIB:
            self._ax.cla()
            self._ax.set_facecolor("#1a1a1a")
            self._ax.set_xlabel("Epoch"); self._ax.set_ylabel("Loss")
            self._canvas.draw_idle()
        
        self._start_btn.setText("■  Stop Training")
        self._start_btn.setStyleSheet(_STOP_STYLE)
        
        self._worker = TrainWorker(cfg, self._registry_root, _SCRIPTS_DIR, parent=self)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._progress_bar.setValue)
        self._worker.metric.connect(self._on_metric)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @pyqtSlot(str)
    def _append_log(self, line: str):
        self._console.append(line)
        self._console.verticalScrollBar().setValue(self._console.verticalScrollBar().maximum())
        
    @pyqtSlot(int, float, float)
    def _on_metric(self, epoch: int, train_loss: float, val_loss: float):
        self._epochs_seen.append(epoch)
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        if HAS_MATPLOTLIB:
            self._ax.cla()
            self._ax.set_facecolor("#1a1a1a")
            self._ax.plot(self._epochs_seen, self._train_losses, color="#4a9eff", label="train", linewidth=1.5)
            self._ax.plot(self._epochs_seen, self._val_losses, color="#ff7f50", label="val", linewidth=1.5)
            self._ax.legend(facecolor="#2a2a2a", edgecolor="#3a3a3a", labelcolor="#cccccc")
            self._ax.set_xlabel("Epoch"); self._ax.set_ylabel("Loss")
            self._ax.tick_params(colors="#aaaaaa")
            self._canvas.draw_idle()
            
    @pyqtSlot(bool, str)
    def _on_finished(self, success: bool, version_id: str):
        self._start_btn.setText("▶  Start Training")
        self._start_btn.setStyleSheet(_START_STYLE)
        self._start_btn.setEnabled(True)
        self._prog_frame.hide()
        self._worker = None
        
        if success:
            self._rc_lbl.setText(f"✓ Training Complete — Version ID: {version_id}")
            self._result_card.show()
        else:
            self._append_log("<span style='color:#f87171'>[FAILED] Training failed or was cancelled.</span>")
