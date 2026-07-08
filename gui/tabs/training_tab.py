"""
gui/tabs/training_tab.py

Phase 3 — Training tab for DualAnnotator.
Right panel shows Augmentation Preview before training starts,
then switches to Loss + IoU graphs once training begins.
"""

import os
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QProgressBar, QFrame, QFileDialog, QMessageBox, QScrollArea,
    QRadioButton, QButtonGroup, QCheckBox, QStackedWidget, QDialog,
    QListWidget, QListWidgetItem, QSizePolicy, QSlider, QApplication, QShortcut,
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage, QColor

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import albumentations as _A
    HAS_ALBUMENTATIONS = True
except Exception:
    HAS_ALBUMENTATIONS = False

from mlops.registry import RegistrySettings, projects_config
from mlops.training import TrainWorker, run_preflight
from mlops.export import OnnxWorker
from mlops.engine_manager import needs_setup, EngineInstallWorker
import logging
import traceback


# ---------------------------------------------------------------------------
# Scroll-protected widget subclasses — prevent accidental value changes
# ---------------------------------------------------------------------------

class _NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, e):
        e.ignore()


class _NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e):
        e.ignore()


class _NoScrollComboBox(QComboBox):
    def wheelEvent(self, e):
        e.ignore()

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        if self.isEditable() and not self.view().isVisible():
            self.showPopup()

_SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mlops", "scripts")
)

# ---------------------------------------------------------------------------
# Augmentation definitions: (key, label, category, [(param, label, min, max, default, decimals, step)])
# ---------------------------------------------------------------------------
_AUG_DEFS = [
    # Geometric
    ("horizontal_flip",  "Horizontal Flip",     "geo",   [("p", "Probability", 0.0, 1.0, 0.5, 2, 0.05)]),
    ("vertical_flip",    "Vertical Flip",        "geo",   [("p", "Probability", 0.0, 1.0, 0.5, 2, 0.05)]),
    ("rotation",         "Rotation",             "geo",   [("limit", "Max angle °", 0, 180, 30, 0, 5),
                                                            ("p", "Probability", 0.0, 1.0, 0.4, 2, 0.05)]),
    ("random_crop",      "Random Crop",          "geo",   [("crop_pct", "Crop size %", 50, 95, 80, 0, 5),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("scale",            "Scaling / Resize",     "geo",   [("scale_limit", "Scale range", 0.0, 0.5, 0.1, 2, 0.05),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("shift",            "Translation / Shift",  "geo",   [("shift_limit", "Shift range", 0.0, 0.5, 0.05, 2, 0.01),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("shear",            "Shearing",             "geo",   [("shear_limit", "Shear angle °", 0, 45, 15, 0, 5),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("perspective",      "Perspective Warp",     "geo",   [("scale", "Distortion", 0.01, 0.3, 0.05, 2, 0.01),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("elastic",          "Elastic Distortion",   "geo",   [("alpha", "Alpha", 0.1, 5.0, 1.0, 1, 0.1),
                                                            ("sigma", "Sigma", 10, 100, 50, 0, 5),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("padding",          "Padding / Reflection", "geo",   [("pad_pct", "Pad size %", 1, 30, 10, 0, 1),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("grid_distortion",  "Grid Distortion",      "geo",   [("num_steps", "Steps", 2, 10, 5, 0, 1),
                                                            ("distort_limit", "Distortion", 0.0, 0.5, 0.3, 2, 0.05),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    # Photometric
    ("brightness",       "Brightness Adjust",    "photo", [("limit", "Brightness ±", 0.0, 0.5, 0.2, 2, 0.05),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("contrast",         "Contrast Adjust",      "photo", [("limit", "Contrast ±", 0.0, 0.5, 0.2, 2, 0.05),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("saturation",       "Saturation Adjust",    "photo", [("sat_limit", "Saturation ±", 0, 80, 30, 0, 5),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("hue_shift",        "Hue Shift",            "photo", [("hue_limit", "Hue ±", 0, 50, 20, 0, 5),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("gaussian_noise",   "Gaussian Noise",       "photo", [("var_limit", "Variance", 1.0, 100.0, 10.0, 1, 1.0),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("gaussian_blur",    "Gaussian Blur",        "photo", [("blur_limit", "Max kernel", 3, 21, 7, 0, 2),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("grayscale",        "Grayscale",            "photo", [("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("color_jitter",     "Color Jitter",         "photo", [("brightness", "Brightness", 0.0, 0.5, 0.2, 2, 0.05),
                                                            ("contrast",   "Contrast",   0.0, 0.5, 0.2, 2, 0.05),
                                                            ("saturation", "Saturation", 0.0, 0.5, 0.2, 2, 0.05),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("gamma",            "Gamma Correction",     "photo", [("gamma_low",  "Min gamma", 50, 100, 80, 0, 5),
                                                            ("gamma_high", "Max gamma", 100, 200, 120, 0, 5),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("channel_shuffle",  "Channel Shuffle",      "photo", [("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("cutout",           "Cutout / Erasing",     "photo", [("num_holes", "Num holes", 1, 16, 4, 0, 1),
                                                            ("max_size",  "Max size px", 8, 128, 32, 0, 8),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
    ("jpeg_compression", "JPEG Compression",     "photo", [("quality_lower", "Min quality", 10, 100, 70, 0, 5),
                                                            ("p", "Probability", 0.0, 1.0, 0.3, 2, 0.05)]),
]

_AUG_DESCRIPTIONS = {
    "horizontal_flip":  "Mirrors the image along the vertical axis. Simulates viewing from opposite horizontal directions.",
    "vertical_flip":    "Mirrors the image along the horizontal axis. Useful when vertical orientation is not meaningful.",
    "rotation":         "Rotates by a random angle. Simulates different camera orientations.",
    "random_crop":      "Crops a random region. Forces model to use local context rather than global position.",
    "scale":            "Enlarges or shrinks the image. Simulates zoom in/out.",
    "shift":            "Shifts the image along x/y. Changes apparent object position.",
    "shear":            "Slants image geometry along one axis. Simulates skewed perspectives.",
    "perspective":      "Simulates viewing from a different angle/depth using projective transform.",
    "elastic":          "Applies local deformations via displacement fields. Simulates surface warping.",
    "padding":          "Adds reflected or zero-padded border. Expands the image canvas.",
    "grid_distortion":  "Warps image using a grid of control points. Creates mesh-like distortions.",
    "brightness":       "Adds a constant offset to pixel intensity, making image lighter or darker.",
    "contrast":         "Scales pixel values around their mean, widening or narrowing intensity range.",
    "saturation":       "Changes vividness of colors in the hue-saturation space.",
    "hue_shift":        "Rotates all hue values, shifting the overall color palette.",
    "gaussian_noise":   "Adds random pixel-level noise from a Gaussian distribution.",
    "gaussian_blur":    "Convolves with a Gaussian kernel to soften edges and reduce detail.",
    "grayscale":        "Converts color image to single-channel luminance (3 channels output).",
    "color_jitter":     "Randomly applies brightness, contrast, and saturation together.",
    "gamma":            "Applies power-law transform to pixel intensities (brightening/darkening).",
    "channel_shuffle":  "Randomly permutes R, G, B channel order.",
    "cutout":           "Fills random patches with zeros. Forces model to use global image context.",
    "jpeg_compression": "Applies lossy JPEG encoding to simulate real-world compression artifacts.",
}


def _preview_transform(key: str, vals: dict, img_w: int, img_h: int):
    """Build a single albumentations transform for UI preview (p forced to 1.0)."""
    if not HAS_CV2 or not HAS_ALBUMENTATIONS:
        return None
    try:
        A = _A
        p = 1.0  # Always apply for demo
        if key == "horizontal_flip":
            return A.HorizontalFlip(p=p)
        if key == "vertical_flip":
            return A.VerticalFlip(p=p)
        if key == "rotation":
            return A.Rotate(limit=int(vals.get("limit", 30)), p=p, border_mode=0)
        if key == "random_crop":
            pct = int(vals.get("crop_pct", 80))
            ch = max(32, min(int(img_h * pct / 100), img_h))
            cw = max(32, min(int(img_w * pct / 100), img_w))
            return A.RandomCrop(height=ch, width=cw, p=p)
        if key == "scale":
            return A.RandomScale(scale_limit=float(vals.get("scale_limit", 0.1)), p=p)
        if key == "shift":
            lim = float(vals.get("shift_limit", 0.05))
            return A.Affine(translate_percent={"x": (-lim, lim), "y": (-lim, lim)},
                            scale=1.0, rotate=0, p=p, border_mode=0)
        if key == "shear":
            lim = int(vals.get("shear_limit", 15))
            return A.Affine(shear=(-lim, lim), p=p)
        if key == "perspective":
            sc = max(0.01, float(vals.get("scale", 0.05)))
            return A.Perspective(scale=(0.01, sc), p=p)
        if key == "elastic":
            return A.ElasticTransform(
                alpha=float(vals.get("alpha", 1.0)),
                sigma=int(vals.get("sigma", 50)), p=p)
        if key == "padding":
            pct = int(vals.get("pad_pct", 10))
            ph = max(1, int(img_h * pct / 100))
            pw = max(1, int(img_w * pct / 100))
            return A.PadIfNeeded(min_height=img_h + 2*ph, min_width=img_w + 2*pw,
                                 border_mode=4, p=p)
        if key == "grid_distortion":
            return A.GridDistortion(
                num_steps=int(vals.get("num_steps", 5)),
                distort_limit=float(vals.get("distort_limit", 0.3)), p=p)
        if key == "brightness":
            return A.RandomBrightnessContrast(
                brightness_limit=float(vals.get("limit", 0.2)), contrast_limit=0.0, p=p)
        if key == "contrast":
            return A.RandomBrightnessContrast(
                brightness_limit=0.0, contrast_limit=float(vals.get("limit", 0.2)), p=p)
        if key == "saturation":
            return A.HueSaturationValue(
                hue_shift_limit=0, sat_shift_limit=int(vals.get("sat_limit", 30)),
                val_shift_limit=0, p=p)
        if key == "hue_shift":
            return A.HueSaturationValue(
                hue_shift_limit=int(vals.get("hue_limit", 20)),
                sat_shift_limit=0, val_shift_limit=0, p=p)
        if key == "gaussian_noise":
            v = float(vals.get("var_limit", 10.0))
            return A.GaussNoise(std_range=(max(0.001, v / 1000), v / 500), p=p)
        if key == "gaussian_blur":
            lim = int(vals.get("blur_limit", 7))
            lim = lim if lim % 2 == 1 else lim + 1
            return A.GaussianBlur(blur_limit=(3, lim), p=p)
        if key == "grayscale":
            return A.ToGray(p=p)
        if key == "color_jitter":
            return A.ColorJitter(
                brightness=float(vals.get("brightness", 0.2)),
                contrast=float(vals.get("contrast", 0.2)),
                saturation=float(vals.get("saturation", 0.2)),
                hue=0.0, p=p)
        if key == "gamma":
            return A.RandomGamma(
                gamma_limit=(int(vals.get("gamma_low", 80)), int(vals.get("gamma_high", 120))),
                p=p)
        if key == "channel_shuffle":
            return A.ChannelShuffle(p=p)
        if key == "cutout":
            n = max(1, int(vals.get("num_holes", 4)))
            sz = max(4, int(vals.get("max_size", 32)))
            return A.CoarseDropout(
                num_holes_range=(1, n),
                hole_height_range=(sz, sz),
                hole_width_range=(sz, sz), p=p)
        if key == "jpeg_compression":
            return A.ImageCompression(
                quality_range=(max(1, int(vals.get("quality_lower", 70))), 100), p=p)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

_SETTINGS_DEFAULTS = {
    "task_type":           "unet",
    "annotator_initials":  "",
    "project_name":        "",
    "project_id":          "",
    "variant":             "",
    "camera":              "",
    "dataset_folder":      "",
    "pretrained_weights":  "",
    "architecture":        "Unet",
    "encoder":             "efficientnet-b3",
    "encoder_weights":     "imagenet",
    "yolo_model":          "yolov8n",
    "yolo_task":           "detection",
    "epochs":              100,
    "batch_size":          4,
    "learning_rate":       0.001,
    "image_width":         320,
    "image_height":        240,
    "in_channels":         3,
    "out_classes":              2,
    "early_stopping_patience":  10,
    "extra_metrics":            [],
    "device":                   "cpu",
    "num_workers":              0,
    "repeat_factor":            1,
    "augmentations":       {},
    # UNet — loss / optimizer / scheduler / regularization
    "unet_loss_fn":         "focal",
    "unet_optimizer":       "Adam",
    "unet_scheduler":       "cosine",
    "unet_weight_decay":    0.0,
    "unet_momentum":        0.9,
    "unet_gradient_clip":   0.0,
    "unet_label_smoothing": 0.0,
    # YOLO advanced — optimizer
    "yolo_optimizer":       "AdamW",
    "yolo_lrf":             0.01,
    "yolo_momentum":        0.937,
    "yolo_weight_decay":    0.0005,
    "yolo_warmup_epochs":   5.0,
    "yolo_cos_lr":          False,
    # YOLO advanced — loss weights
    "yolo_box":             7.5,
    "yolo_cls":             3.0,
    "yolo_dfl":             1.5,
    # YOLO seg options
    "yolo_dropout":         0.1,
    "yolo_overlap_mask":    True,
    "yolo_mask_ratio":      4,
    # YOLO augmentations
    "yolo_hsv_h":           0.01,
    "yolo_hsv_s":           0.30,
    "yolo_hsv_v":           0.20,
    "yolo_fliplr":          0.5,
    "yolo_flipud":          0.0,
    "yolo_degrees":         5.0,
    "yolo_translate":       0.05,
    "yolo_scale":           0.25,
    "yolo_mosaic":          0.3,
    "yolo_mixup":           0.0,
    "yolo_copy_paste":      0.1,
    "yolo_close_mosaic":    20,
    # YOLO training utilities
    "yolo_amp":             True,
    "yolo_cache":           False,
    "yolo_save_period":     10,
    "yolo_plots":           True,
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


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

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
    QPushButton:hover { background-color: #3a3a40; border: 1px solid #777; }
"""

_START_STYLE = """
    QPushButton {
        background-color: #6a3fc8; color: #ffffff;
        border: none; border-radius: 5px;
        padding: 10px; font-size: 13px; font-weight: bold;
    }
    QPushButton:hover { background-color: #7a4fd8; }
    QPushButton:pressed { background-color: #5a2fb8; }
    QPushButton:disabled { background-color: #444444; color: #888888; }
"""

_STOP_STYLE = """
    QPushButton {
        background-color: #c0392b; color: #ffffff;
        border: none; border-radius: 5px;
        padding: 10px; font-size: 13px; font-weight: bold;
    }
    QPushButton:hover { background-color: #d35400; }
    QPushButton:pressed { background-color: #e74c3c; }
"""

_SPINBOX_SMALL = """
    QSpinBox, QDoubleSpinBox {
        background: #2d2d2d; border: 1px solid #444; border-radius: 3px;
        color: #ffffff; padding: 2px 4px; font-size: 10px;
    }
"""

_SLIDER_STYLE = """
    QSlider::groove:horizontal {
        background: #3a3a3a; height: 4px; border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #6a3fc8; width: 12px; height: 12px;
        margin: -4px 0; border-radius: 6px;
    }
    QSlider::sub-page:horizontal { background: #6a3fc8; border-radius: 2px; }
"""

_SIDEBAR_TOGGLE_STYLE = """
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #2e2e38, stop:1 #24242c);
        border: 1px solid #4a3a78;
        border-radius: 5px;
        color: #9ab4f8;
        padding: 5px 14px;
        font-size: 11px;
        font-weight: bold;
    }
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #3a3050, stop:1 #2d2a3a);
        border-color: #7a5fe8;
        color: #c0d4ff;
    }
    QPushButton:pressed {
        background: #5a3fa8;
        border-color: #8a6fe8;
        color: #ffffff;
    }
"""

_NAV_BTN_STYLE = """
    QPushButton {
        background: #2a2a32;
        border: 1px solid #3a3a4a;
        border-radius: 4px;
        color: #8ab4f8;
        font-size: 13px;
        font-weight: bold;
        padding: 0px;
    }
    QPushButton:hover { background: #3a3050; border-color: #6a4fc8; color: #c0d4ff; }
    QPushButton:pressed { background: #5a3fa8; color: #ffffff; }
    QPushButton:disabled { color: #3a3a4a; border-color: #2a2a2a; }
"""


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #8ab4f8; font-size: 12px; font-weight: bold; "
                      "background: transparent; border: none;")
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


def _make_collapsible(title: str, content: QWidget, collapsed: bool = False) -> QWidget:
    """Wrap a content widget in a collapsible toggle panel."""
    from PyQt5.QtCore import Qt as _Qt
    outer = QWidget()
    outer.setStyleSheet("background: transparent;")
    vlay = QVBoxLayout(outer)
    vlay.setContentsMargins(0, 0, 0, 0)
    vlay.setSpacing(2)

    btn = QPushButton()
    btn.setCheckable(True)
    btn.setChecked(not collapsed)
    btn.setCursor(_Qt.PointingHandCursor)
    btn.setToolTip("Click to expand / collapse")
    btn.setStyleSheet("""
        QPushButton {
            background-color: #252530;
            border: 1px solid #4a4a6a;
            border-radius: 5px;
            color: #9ab4f8;
            padding: 7px 12px;
            font-size: 12px;
            font-weight: bold;
            text-align: left;
        }
        QPushButton:hover {
            background-color: #2e2e42;
            border-color: #7a6fc8;
            color: #c0d4ff;
        }
        QPushButton:checked {
            background-color: #2a2a3e;
            border: 1px solid #5a5a8a;
            color: #aac4ff;
        }
        QPushButton:!checked {
            background-color: #1e1e28;
            border: 1px dashed #4a4a6a;
            color: #7a8ab8;
        }
    """)

    def _refresh(checked):
        arrow = "▼" if checked else "▶"
        hint  = "" if checked else "  (click to expand)"
        btn.setText(f"  {arrow}  {title}{hint}")
        content.setVisible(checked)

    _refresh(not collapsed)
    btn.toggled.connect(_refresh)

    vlay.addWidget(btn)
    vlay.addWidget(content)
    return outer


# ---------------------------------------------------------------------------
# AI Engine setup dialog
# ---------------------------------------------------------------------------

class _EngineSetupDialog(QDialog):
    """Modal dialog showing AI Engine installation progress."""

    setup_complete = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Engine Setup")
        self.setMinimumSize(600, 400)
        self.setModal(True)

        layout = QVBoxLayout(self)

        header = QLabel(
            "First-time setup — Installing AI Engine\n\n"
            "This downloads and installs PyTorch and ML packages (~2 GB).\n"
            "It only happens once. Please don't close this window."
        )
        header.setWordWrap(True)
        header.setStyleSheet("font-size: 13px; padding: 10px;")
        layout.addWidget(header)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._console = QTextEdit()
        self._console.setReadOnly(True)
        self._console.setStyleSheet(
            "QTextEdit { background: #1a1a1a; color: #cccccc; "
            "font-family: Consolas, monospace; font-size: 11px; }"
        )
        layout.addWidget(self._console)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._cancel_btn)

        self._worker = EngineInstallWorker(parent=self)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_finished)

    def start(self):
        self._worker.start()

    def _append_log(self, line):
        self._console.append(line)
        sb = self._console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, success):
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.accept)
        if success:
            self._progress.setValue(100)
        self.setup_complete.emit(success)

    def _on_cancel(self):
        try:
            self._worker.cancel()
            self.reject()
        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return


# ---------------------------------------------------------------------------
# TrainingTab
# ---------------------------------------------------------------------------

class TrainingTab(QWidget):
    training_completed = pyqtSignal(str)
    onnx_exported = pyqtSignal(str)  # emitted with version folder path after successful ONNX export

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings             = _load_settings()
        self._worker               = None
        self._dataset_info         = {}
        self._epochs_seen          = []
        self._train_losses         = []
        self._val_losses           = []
        self._train_ious           = []
        self._val_ious             = []
        self._registry_root        = RegistrySettings().get_registry_root()
        self._last_version_folder  = ""
        self._onnx_worker          = None
        self._aug_widgets          = {}   # key → {"enabled", "params", "params_widget"}
        self._current_img_path     = None
        self._dataset_paths        = []   # original image paths loaded from dataset
        self._left_collapsed       = False

        self._initialized = False
        self._setup_ui()
        self._restore_settings()
        self._initialized = True

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setHandleWidth(4)
        self._main_splitter.setStyleSheet("QSplitter::handle { background: #2a2a2a; }")
        self._main_splitter.addWidget(self._build_left_panel())
        self._main_splitter.addWidget(self._build_right_panel())
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([380, 800])
        root.addWidget(self._main_splitter)

        # Keyboard shortcuts: Left/Right arrows navigate the image list
        sc_prev = QShortcut(Qt.Key_Left, self)
        sc_prev.setContext(Qt.WidgetWithChildrenShortcut)
        sc_prev.activated.connect(self._nav_prev)
        sc_next = QShortcut(Qt.Key_Right, self)
        sc_next.setContext(Qt.WidgetWithChildrenShortcut)
        sc_next.activated.connect(self._nav_next)

    # ---- Left panel ---------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(0)
        panel.setStyleSheet("background-color: #1e1e1e;")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: #1e1e1e; border: none; }")
        scroll.setWidget(panel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Task Type
        self._radio_yolo = QRadioButton("YOLO (Detection)")
        self._radio_unet = QRadioButton("UNet (Segmentation)")
        for rb in (self._radio_yolo, self._radio_unet):
            rb.setStyleSheet("color: #ffffff; background: transparent; border: none;")
        self._task_group = QButtonGroup(self)
        self._task_group.addButton(self._radio_yolo)
        self._task_group.addButton(self._radio_unet)
        self._radio_unet.setChecked(True)
        self._radio_yolo.toggled.connect(self._on_task_changed)
        layout.addWidget(_make_section([_section_label("Task Type"),
                                        self._radio_yolo, self._radio_unet]))

        # Run Info
        self._annotator_cb = _NoScrollComboBox()
        self._annotator_cb.setStyleSheet(_INPUT_STYLE)
        for ann in RegistrySettings().get_annotators():
            self._annotator_cb.addItem(f"{ann['name']} ({ann['initials']})", userData=ann)
        self._annotator_cb.currentIndexChanged.connect(self._autosave)

        self._project_name_cb = _NoScrollComboBox()
        self._project_name_cb.setEditable(True)
        self._project_name_cb.setInsertPolicy(QComboBox.NoInsert)
        self._project_name_cb.setPlaceholderText("e.g. SnipeX")
        self._project_name_cb.setStyleSheet(_INPUT_STYLE)
        self._project_name_cb.currentTextChanged.connect(self._on_train_project_name_changed)

        self._project_id_cb = _NoScrollComboBox()
        self._project_id_cb.setEditable(True)
        self._project_id_cb.setInsertPolicy(QComboBox.NoInsert)
        self._project_id_cb.setPlaceholderText("e.g. SLXCAP")
        self._project_id_cb.setStyleSheet(_INPUT_STYLE)
        self._project_id_cb.currentTextChanged.connect(self._on_train_project_id_changed)

        self._variant_cb = _NoScrollComboBox()
        self._variant_cb.setEditable(True)
        self._variant_cb.setInsertPolicy(QComboBox.NoInsert)
        self._variant_cb.setPlaceholderText("e.g. CRC")
        self._variant_cb.setStyleSheet(_INPUT_STYLE)
        self._variant_cb.currentTextChanged.connect(self._on_train_variant_changed)

        self._camera_cb = _NoScrollComboBox()
        self._camera_cb.setEditable(True)
        self._camera_cb.setInsertPolicy(QComboBox.NoInsert)
        self._camera_cb.setPlaceholderText("e.g. cam0")
        self._camera_cb.setStyleSheet(_INPUT_STYLE)
        self._camera_cb.currentTextChanged.connect(self._autosave)

        self._commit_edit = QLineEdit()
        self._commit_edit.setPlaceholderText("e.g. Added thermal augmentation")
        self._commit_edit.setStyleSheet(_INPUT_STYLE)

        self._refresh_train_project_names()

        layout.addWidget(_make_section([
            _section_label("Run Info"),
            _make_row("Annotator",    self._annotator_cb),
            _make_row("Project Name", self._project_name_cb),
            _make_row("Project ID",   self._project_id_cb),
            _make_row("Variant",      self._variant_cb),
            _make_row("Camera",       self._camera_cb),
            _make_row("Commit Msg",   self._commit_edit),
        ]))

        # Dataset
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

        self._weights_edit = QLineEdit()
        self._weights_edit.setReadOnly(True)
        self._weights_edit.setPlaceholderText("Optional — .pt or .ckpt for fine-tuning")
        self._weights_edit.setStyleSheet(_INPUT_STYLE)
        w_browse = QPushButton("Browse"); w_browse.setStyleSheet(_BROWSE_STYLE)
        w_browse.clicked.connect(self._browse_weights)
        w_clear = QPushButton("✕ Clear")
        w_clear.setFixedWidth(58)
        w_clear.setToolTip("Remove pre-trained weights path")
        w_clear.setStyleSheet(_BROWSE_STYLE)
        w_clear.clicked.connect(self._clear_weights)
        weights_row = QHBoxLayout()
        weights_row.addWidget(self._weights_edit)
        weights_row.addWidget(w_browse)
        weights_row.addWidget(w_clear)

        layout.addWidget(_make_section([
            _section_label("Dataset"),
            ds_row,
            self._ds_info_lbl,
            QLabel("Pre-trained Weights (fine-tune)"),
            weights_row,
        ]))

        # Augmentations (collapsible) — hidden when YOLO is selected
        self._aug_collapsible = _make_collapsible("Augmentations", self._build_aug_section(), collapsed=True)
        layout.addWidget(self._aug_collapsible)

        # Architecture
        self._arch_section = QFrame()
        self._arch_section.setStyleSheet(_SECTION_STYLE)
        arch_layout = QVBoxLayout(self._arch_section)
        arch_layout.setContentsMargins(12, 10, 12, 10)
        arch_layout.setSpacing(8)

        self._unet_block = QWidget()
        unet_l = QVBoxLayout(self._unet_block)
        unet_l.setContentsMargins(0, 0, 0, 0); unet_l.setSpacing(8)
        self._unet_arch_cb = _NoScrollComboBox()
        self._unet_arch_cb.addItems(["Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN"])
        self._unet_arch_cb.setStyleSheet(_INPUT_STYLE)
        self._unet_arch_cb.currentTextChanged.connect(self._autosave)
        self._encoder_cb = _NoScrollComboBox()
        self._encoder_cb.addItems(["efficientnet-b0", "efficientnet-b1", "efficientnet-b2",
                                    "efficientnet-b3", "efficientnet-b4", "resnet34", "resnet50",
                                    "resnet101", "mobilenet_v2"])
        self._encoder_cb.setStyleSheet(_INPUT_STYLE)
        self._encoder_cb.currentTextChanged.connect(self._autosave)
        self._weights_cb = _NoScrollComboBox()
        self._weights_cb.addItems(["imagenet", "none"])
        self._weights_cb.setStyleSheet(_INPUT_STYLE)
        self._weights_cb.currentTextChanged.connect(self._autosave)
        unet_l.addLayout(_make_row("Architecture", self._unet_arch_cb))
        unet_l.addLayout(_make_row("Encoder",      self._encoder_cb))
        unet_l.addLayout(_make_row("Weights",       self._weights_cb))

        self._yolo_block = QWidget()
        yolo_l = QVBoxLayout(self._yolo_block)
        yolo_l.setContentsMargins(0, 0, 0, 0); yolo_l.setSpacing(8)

        _rb_style = ("QRadioButton { color: #cccccc; font-size: 11px; "
                     "background: transparent; border: none; }")
        self._yolo_detect_rb = QRadioButton("Detection")
        self._yolo_seg_rb    = QRadioButton("Segmentation")
        self._yolo_detect_rb.setStyleSheet(_rb_style)
        self._yolo_seg_rb.setStyleSheet(_rb_style)
        self._yolo_detect_rb.setChecked(True)
        self._yolo_task_grp = QButtonGroup(self)
        self._yolo_task_grp.addButton(self._yolo_detect_rb)
        self._yolo_task_grp.addButton(self._yolo_seg_rb)
        self._yolo_detect_rb.toggled.connect(self._on_yolo_task_changed)
        _rb_row = QHBoxLayout()
        _rb_row.setSpacing(12)
        _rb_row.addWidget(self._yolo_detect_rb)
        _rb_row.addWidget(self._yolo_seg_rb)
        _rb_row.addStretch()
        yolo_l.addLayout(_rb_row)

        self._yolo_model_cb = _NoScrollComboBox()
        self._yolo_model_cb.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self._yolo_model_cb.setStyleSheet(_INPUT_STYLE)
        self._yolo_model_cb.currentTextChanged.connect(self._autosave)
        yolo_l.addLayout(_make_row("Model", self._yolo_model_cb))

        arch_layout.addWidget(self._unet_block)
        arch_layout.addWidget(self._yolo_block)
        layout.addWidget(_make_collapsible("Architecture", self._arch_section))

        # YOLO advanced sections (hidden when UNet is selected)
        self._yolo_adv_widget = self._build_yolo_advanced_sections()
        self._yolo_adv_widget.hide()
        layout.addWidget(self._yolo_adv_widget)

        # Hyperparameters
        self._epochs_sp = _NoScrollSpinBox(); self._epochs_sp.setRange(1, 1000)
        self._batch_sp  = _NoScrollSpinBox(); self._batch_sp.setRange(1, 128)
        self._lr_sp     = _NoScrollDoubleSpinBox()
        self._lr_sp.setRange(0.000001, 1.0); self._lr_sp.setDecimals(6); self._lr_sp.setSingleStep(0.0001)
        self._width_sp  = _NoScrollSpinBox(); self._width_sp.setRange(32, 4096); self._width_sp.setSingleStep(32)
        self._height_sp = _NoScrollSpinBox(); self._height_sp.setRange(32, 4096); self._height_sp.setSingleStep(32)
        self._device_cb = _NoScrollComboBox(); self._device_cb.addItems(["cpu", "cuda:0", "cuda:1"])
        self._in_channels_sp   = _NoScrollSpinBox(); self._in_channels_sp.setRange(1, 10)
        self._out_classes_sp   = _NoScrollSpinBox(); self._out_classes_sp.setRange(1, 256)
        self._patience_sp      = _NoScrollSpinBox(); self._patience_sp.setRange(1, 200); self._patience_sp.setValue(10)
        self._num_workers_sp   = _NoScrollSpinBox(); self._num_workers_sp.setRange(0, 16)
        self._repeat_factor_sp = _NoScrollSpinBox(); self._repeat_factor_sp.setRange(1, 20); self._repeat_factor_sp.setValue(1)
        for w in (self._epochs_sp, self._batch_sp, self._lr_sp, self._width_sp,
                  self._height_sp, self._device_cb, self._in_channels_sp, self._out_classes_sp,
                  self._patience_sp, self._num_workers_sp, self._repeat_factor_sp):
            w.setStyleSheet(_INPUT_STYLE)
            w.currentTextChanged.connect(self._autosave) if isinstance(w, QComboBox) else \
                w.valueChanged.connect(self._autosave)

        # UNet-only hyperparams (hidden for YOLO)
        self._unet_only_hp = QWidget()
        self._unet_only_hp.setStyleSheet("QWidget { background: transparent; }")
        _unet_hp_l = QVBoxLayout(self._unet_only_hp)
        _unet_hp_l.setContentsMargins(0, 0, 0, 0)
        _unet_hp_l.setSpacing(4)
        _unet_hp_sep = QLabel("── UNet Only ──────────────────")
        _unet_hp_sep.setStyleSheet("color: #555555; font-size: 10px; background: transparent; border: none;")
        _unet_hp_l.addWidget(_unet_hp_sep)
        for label, widget in [
            ("In Channels",     self._in_channels_sp),
            ("Out Classes",     self._out_classes_sp),
            ("Repeat Factor",   self._repeat_factor_sp),
        ]:
            _row = _make_row(label, widget)
            _unet_hp_l.addLayout(_row)

        # ── UNet Loss Function ──
        self._unet_loss_cb = _NoScrollComboBox()
        self._unet_loss_cb.addItems([
            "focal",          # FocalLoss (team default)
            "dice_ce",        # Dice + CrossEntropy combo
            "focal_dice",     # Focal + Dice combo
            "dice",           # Dice only
            "ce",             # CrossEntropy only
            "jaccard",        # Jaccard (IoU) loss
            "jaccard_ce",     # Jaccard + CE combo
            "tversky",        # Tversky loss
            "tversky_ce",     # Tversky + CE combo
            "lovasz",         # Lovász loss
        ])
        self._unet_loss_cb.setStyleSheet(_INPUT_STYLE)
        self._unet_loss_cb.currentTextChanged.connect(self._autosave)

        # ── UNet Optimizer ──
        self._unet_optimizer_cb = _NoScrollComboBox()
        self._unet_optimizer_cb.addItems(["Adam", "AdamW", "SGD", "RMSprop", "NAdam", "RAdam"])
        self._unet_optimizer_cb.setStyleSheet(_INPUT_STYLE)
        self._unet_optimizer_cb.currentTextChanged.connect(self._autosave)

        # ── UNet LR Scheduler ──
        self._unet_scheduler_cb = _NoScrollComboBox()
        self._unet_scheduler_cb.addItems([
            "cosine",            # CosineAnnealingLR (team default)
            "reduce_plateau",    # ReduceLROnPlateau
            "cosine_restarts",   # CosineAnnealingWarmRestarts
            "step",              # StepLR
            "exponential",       # ExponentialLR
            "one_cycle",         # OneCycleLR
            "none",              # Constant LR (no scheduler)
        ])
        self._unet_scheduler_cb.setStyleSheet(_INPUT_STYLE)
        self._unet_scheduler_cb.currentTextChanged.connect(self._autosave)

        # ── Weight Decay (L2 regularization) ──
        def _unet_dsb(lo, hi, default, decimals=3, step=0.001):
            w = _NoScrollDoubleSpinBox()
            w.setRange(lo, hi); w.setDecimals(decimals)
            w.setSingleStep(step); w.setValue(default)
            w.setStyleSheet(_INPUT_STYLE)
            w.valueChanged.connect(self._autosave)
            return w

        self._unet_weight_decay_sp = _unet_dsb(0.0, 0.1, 0.0, 6, 0.00001)

        # ── Momentum (only relevant for SGD and RMSprop) ──
        self._unet_momentum_sp = _unet_dsb(0.0, 1.0, 0.9, 2, 0.01)

        # ── Gradient Clipping (0 = disabled) ──
        self._unet_grad_clip_sp = _unet_dsb(0.0, 10.0, 0.0, 1, 0.1)

        # ── Label Smoothing (0 = disabled, used by CE-based losses) ──
        self._unet_label_smoothing_sp = _unet_dsb(0.0, 0.3, 0.0, 2, 0.01)

        # ── Momentum row (needs toggle visibility) ──
        self._unet_momentum_row = QWidget()
        self._unet_momentum_row.setStyleSheet("QWidget { background: transparent; }")
        _mom_l = QHBoxLayout(self._unet_momentum_row)
        _mom_l.setContentsMargins(0, 0, 0, 0)
        _mom_l.setSpacing(0)
        _mom_lbl = QLabel("Momentum")
        _mom_lbl.setFixedWidth(120)
        _mom_l.addWidget(_mom_lbl)
        _mom_l.addWidget(self._unet_momentum_sp)
        self._unet_momentum_row.setVisible(False)  # hidden by default (Adam doesn't use it)

        # ── Show/hide momentum when optimizer changes ──
        self._unet_optimizer_cb.currentTextChanged.connect(self._on_unet_optimizer_changed)

        # ── Build collapsible section ──
        _unet_adv = _make_collapsible("UNet Optimizer & Loss", _make_section([
            _make_row("Loss Function",    self._unet_loss_cb),
            _make_row("Optimizer",        self._unet_optimizer_cb),
            _make_row("LR Scheduler",     self._unet_scheduler_cb),
            _make_row("Weight Decay",     self._unet_weight_decay_sp),
            self._unet_momentum_row,
            _make_row("Gradient Clip",    self._unet_grad_clip_sp),
            _make_row("Label Smoothing",  self._unet_label_smoothing_sp),
        ]))
        _unet_hp_l.addWidget(_unet_adv)

        self._width_row_widget = QWidget()
        self._width_row_widget.setStyleSheet("QWidget { background: transparent; }")
        _wr_l = QHBoxLayout(self._width_row_widget)
        _wr_l.setContentsMargins(0, 0, 0, 0)
        _wr_l.setSpacing(0)
        _wr_lbl = QLabel("Image Width")
        _wr_lbl.setFixedWidth(120)
        _wr_l.addWidget(_wr_lbl)
        _wr_l.addWidget(self._width_sp)

        layout.addWidget(_make_collapsible("Hyperparameters", _make_section([
            _make_row("Epochs",              self._epochs_sp),
            _make_row("Batch Size",          self._batch_sp),
            _make_row("Learning Rate (lr0)", self._lr_sp),
            self._width_row_widget,
            _make_row("Image Height",        self._height_sp),
            _make_row("Early Stop Patience", self._patience_sp),
            _make_row("Num Workers",         self._num_workers_sp),
            _make_row("Device",              self._device_cb),
            self._unet_only_hp,
        ])))

        # Metrics to Track
        _cb_style = ("QCheckBox { color: #cccccc; font-size: 11px; "
                     "background: transparent; border: none; }"
                     "QCheckBox::indicator { width: 13px; height: 13px; }")
        self._metric_checkboxes = {}
        for key, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 / Dice Score")]:
            cb = QCheckBox(label)
            cb.setStyleSheet(_cb_style)
            cb.setChecked(False)
            cb.stateChanged.connect(self._autosave)
            self._metric_checkboxes[key] = cb
        layout.addWidget(_make_collapsible("Metrics to Track", _make_section(
            list(self._metric_checkboxes.values())
        )))

        # Pre-flight Checks
        self._checks_frame = QFrame()
        self._checks_frame.setStyleSheet(_SECTION_STYLE)
        self._checks_layout = QVBoxLayout(self._checks_frame)
        self._checks_layout.setContentsMargins(12, 10, 12, 10)
        self._checks_layout.setSpacing(8)
        self._checks_layout.addWidget(_section_label("Pre-flight Checks"))
        chk_btn = QPushButton("Run Checks"); chk_btn.setStyleSheet(_BROWSE_STYLE)
        chk_btn.clicked.connect(self._run_checks)
        self._checks_layout.addWidget(chk_btn)
        self._checks_list_layout = QVBoxLayout(); self._checks_list_layout.setSpacing(4)
        self._checks_layout.addLayout(self._checks_list_layout)
        self._copy_diag_btn = QPushButton("Copy Diagnostic Info")
        self._copy_diag_btn.setStyleSheet(_BROWSE_STYLE)
        self._copy_diag_btn.clicked.connect(self._copy_diagnostic)
        self._copy_diag_btn.hide()
        self._checks_layout.addWidget(self._copy_diag_btn)
        self._last_preflight_results = []
        layout.addWidget(self._checks_frame)

        # Start button
        self._start_btn = QPushButton("▶  Start Training")
        self._start_btn.setStyleSheet(_START_STYLE)
        self._start_btn.clicked.connect(self._on_start_stop)
        layout.addWidget(self._start_btn)
        layout.addStretch()

        outer = QWidget()
        outer.setMinimumWidth(0)
        outer.setStyleSheet("background: #1e1e1e;")
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(scroll)
        return outer

    # ---- YOLO advanced sections ---------------------------------------

    def _build_yolo_advanced_sections(self) -> QWidget:
        _cb_style = ("QCheckBox { color: #cccccc; font-size: 11px; "
                     "background: transparent; border: none; }"
                     "QCheckBox::indicator { width: 13px; height: 13px; }")

        def _dsb(lo, hi, default, decimals=3, step=0.001):
            w = _NoScrollDoubleSpinBox()
            w.setRange(lo, hi); w.setDecimals(decimals)
            w.setSingleStep(step); w.setValue(default)
            w.setStyleSheet(_INPUT_STYLE)
            w.valueChanged.connect(self._autosave)
            return w

        def _sb(lo, hi, default):
            w = _NoScrollSpinBox()
            w.setRange(lo, hi); w.setValue(default)
            w.setStyleSheet(_INPUT_STYLE)
            w.valueChanged.connect(self._autosave)
            return w

        def _chk(label, default):
            w = QCheckBox(label)
            w.setStyleSheet(_cb_style)
            w.setChecked(default)
            w.stateChanged.connect(self._autosave)
            return w

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)

        # --- Optimizer & LR ---
        self._yolo_optimizer_cb = _NoScrollComboBox()
        self._yolo_optimizer_cb.addItems(["AdamW", "Adam", "SGD", "auto"])
        self._yolo_optimizer_cb.setStyleSheet(_INPUT_STYLE)
        self._yolo_optimizer_cb.currentTextChanged.connect(self._autosave)
        self._yolo_lrf_sp          = _dsb(0.0001, 1.0,   0.01,  4, 0.001)
        self._yolo_momentum_sp     = _dsb(0.0,    1.0,   0.937, 3, 0.001)
        self._yolo_weight_decay_sp = _dsb(0.0,    0.1,   0.0005,5, 0.00001)
        self._yolo_warmup_sp       = _dsb(0.0,    20.0,  5.0,   1, 0.5)
        self._yolo_cos_lr_cb       = _chk("Cosine LR Schedule", False)

        vlay.addWidget(_make_collapsible("YOLO Optimizer & LR", _make_section([
            _make_row("Optimizer",      self._yolo_optimizer_cb),
            _make_row("LR Final (lrf)", self._yolo_lrf_sp),
            _make_row("Momentum",       self._yolo_momentum_sp),
            _make_row("Weight Decay",   self._yolo_weight_decay_sp),
            _make_row("Warmup Epochs",  self._yolo_warmup_sp),
            self._yolo_cos_lr_cb,
        ]), collapsed=True))

        # --- Loss Weights ---
        self._yolo_box_sp = _dsb(0.0, 20.0, 7.5, 1, 0.1)
        self._yolo_cls_sp = _dsb(0.0, 20.0, 3.0, 1, 0.1)
        self._yolo_dfl_sp = _dsb(0.0, 10.0, 1.5, 1, 0.1)

        vlay.addWidget(_make_collapsible("YOLO Loss Weights", _make_section([
            _make_row("Box Loss (box)", self._yolo_box_sp),
            _make_row("Cls Loss (cls)", self._yolo_cls_sp),
            _make_row("DFL Loss (dfl)", self._yolo_dfl_sp),
        ]), collapsed=True))

        # --- YOLO Augmentations ---
        self._yolo_hsv_h_sp       = _dsb(0.0, 1.0,   0.01,  3, 0.01)
        self._yolo_hsv_s_sp       = _dsb(0.0, 1.0,   0.30,  2, 0.01)
        self._yolo_hsv_v_sp       = _dsb(0.0, 1.0,   0.20,  2, 0.01)
        self._yolo_fliplr_sp      = _dsb(0.0, 1.0,   0.5,   2, 0.05)
        self._yolo_flipud_sp      = _dsb(0.0, 1.0,   0.0,   2, 0.05)
        self._yolo_degrees_sp     = _dsb(0.0, 360.0, 5.0,   1, 1.0)
        self._yolo_translate_sp   = _dsb(0.0, 1.0,   0.05,  2, 0.01)
        self._yolo_scale_sp       = _dsb(0.0, 1.0,   0.25,  2, 0.05)
        self._yolo_mosaic_sp      = _dsb(0.0, 1.0,   0.3,   2, 0.05)
        self._yolo_mixup_sp       = _dsb(0.0, 1.0,   0.0,   2, 0.05)
        self._yolo_copy_paste_sp  = _dsb(0.0, 1.0,   0.1,   2, 0.05)
        self._yolo_close_mosaic_sp = _sb(0, 300, 20)

        vlay.addWidget(_make_collapsible("YOLO Augmentations", _make_section([
            _make_row("HSV Hue",        self._yolo_hsv_h_sp),
            _make_row("HSV Saturation", self._yolo_hsv_s_sp),
            _make_row("HSV Value",      self._yolo_hsv_v_sp),
            _make_row("Flip LR (prob)", self._yolo_fliplr_sp),
            _make_row("Flip UD (prob)", self._yolo_flipud_sp),
            _make_row("Rotation °",     self._yolo_degrees_sp),
            _make_row("Translate",      self._yolo_translate_sp),
            _make_row("Scale",          self._yolo_scale_sp),
            _make_row("Mosaic",         self._yolo_mosaic_sp),
            _make_row("Mixup",          self._yolo_mixup_sp),
            _make_row("Copy-Paste",     self._yolo_copy_paste_sp),
            _make_row("Close Mosaic",   self._yolo_close_mosaic_sp),
        ]), collapsed=True))

        # --- Seg Options (hidden in detection mode) ---
        self._yolo_dropout_sp      = _dsb(0.0, 0.9, 0.1, 2, 0.05)
        self._yolo_overlap_mask_cb = _chk("Overlap Mask", True)
        self._yolo_mask_ratio_sp   = _sb(1, 8, 4)

        self._yolo_seg_options_col = _make_collapsible("YOLO Seg Options", _make_section([
            _make_row("Dropout",     self._yolo_dropout_sp),
            _make_row("Mask Ratio",  self._yolo_mask_ratio_sp),
            self._yolo_overlap_mask_cb,
        ]), collapsed=True)
        self._yolo_seg_options_col.hide()
        vlay.addWidget(self._yolo_seg_options_col)

        # --- Training Utilities ---
        self._yolo_amp_cb        = _chk("Mixed Precision (AMP)", True)
        self._yolo_cache_cb      = _chk("Cache Images in RAM",   False)
        self._yolo_plots_cb      = _chk("Generate Plots",        True)
        self._yolo_save_period_sp = _sb(-1, 300, 10)

        vlay.addWidget(_make_collapsible("YOLO Training Utilities", _make_section([
            self._yolo_amp_cb,
            self._yolo_cache_cb,
            self._yolo_plots_cb,
            _make_row("Save Period (epochs)", self._yolo_save_period_sp),
        ]), collapsed=True))

        return container

    # ---- Augmentation section (left panel) ----------------------------

    def _build_aug_section(self) -> QFrame:
        self._aug_section_frame = QFrame()
        self._aug_section_frame.setStyleSheet(_SECTION_STYLE)
        outer = QVBoxLayout(self._aug_section_frame)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)


        self._aug_yolo_note = QLabel("ℹ  YOLO manages augmentations internally.")
        self._aug_yolo_note.setStyleSheet("color: #aaaaaa; font-size: 11px; "
                                          "background: transparent; border: none;")
        self._aug_yolo_note.hide()
        outer.addWidget(self._aug_yolo_note)

        self._aug_controls_widget = QWidget()
        self._aug_controls_widget.setStyleSheet("QWidget { background: transparent; }")
        clay = QVBoxLayout(self._aug_controls_widget)
        clay.setContentsMargins(0, 0, 0, 0)
        clay.setSpacing(2)

        geo_lbl = QLabel("  GEOMETRIC")
        geo_lbl.setStyleSheet("color: #6a9ee8; font-size: 10px; font-weight: bold; "
                               "background: transparent; border: none; margin-top: 4px;")
        clay.addWidget(geo_lbl)
        for key, label, cat, params in _AUG_DEFS:
            if cat == "geo":
                clay.addWidget(self._build_aug_row(key, label, params))

        photo_lbl = QLabel("  PHOTOMETRIC")
        photo_lbl.setStyleSheet("color: #6a9ee8; font-size: 10px; font-weight: bold; "
                                 "background: transparent; border: none; margin-top: 6px;")
        clay.addWidget(photo_lbl)
        for key, label, cat, params in _AUG_DEFS:
            if cat == "photo":
                clay.addWidget(self._build_aug_row(key, label, params))

        outer.addWidget(self._aug_controls_widget)

        # Batch preview controls
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3a3a;")
        outer.addWidget(sep)

        batch_row = QHBoxLayout(); batch_row.setSpacing(6)
        n_lbl = QLabel("Max images:")
        n_lbl.setStyleSheet("color: #888888; font-size: 10px; background: transparent; border: none;")
        self._batch_preview_sp = _NoScrollSpinBox()
        self._batch_preview_sp.setRange(1, 60)
        self._batch_preview_sp.setValue(10)
        self._batch_preview_sp.setFixedWidth(52)
        self._batch_preview_sp.setStyleSheet(_SPINBOX_SMALL)
        run_btn = QPushButton("▶  Run Batch Preview")
        run_btn.setStyleSheet(_BROWSE_STYLE)
        run_btn.clicked.connect(self._run_batch_preview)
        batch_row.addWidget(n_lbl)
        batch_row.addWidget(self._batch_preview_sp)
        batch_row.addStretch()
        batch_row.addWidget(run_btn)
        outer.addLayout(batch_row)

        return self._aug_section_frame

    def _build_aug_row(self, key: str, label: str, params: list) -> QWidget:
        container = QWidget()
        container.setStyleSheet("QWidget { background: transparent; }")
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(4, 1, 4, 1)
        vlay.setSpacing(1)

        # Header row
        hlay = QHBoxLayout(); hlay.setSpacing(4)
        cb = QCheckBox(label)
        cb.setStyleSheet("QCheckBox { color: #cccccc; font-size: 11px; "
                         "background: transparent; border: none; }")
        q_btn = QPushButton("?")
        q_btn.setFixedSize(18, 18)
        q_btn.setToolTip(_AUG_DESCRIPTIONS.get(key, label))
        q_btn.setStyleSheet("""
            QPushButton {
                background: #2d2d2d; color: #8ab4f8;
                border: 1px solid #555; border-radius: 9px;
                font-size: 10px; font-weight: bold;
            }
            QPushButton:hover { background: #3a3a3a; }
        """)
        hlay.addWidget(cb)
        hlay.addStretch()
        hlay.addWidget(q_btn)
        vlay.addLayout(hlay)

        # Parameter controls (hidden when unchecked)
        params_widget = QWidget()
        params_widget.setStyleSheet("QWidget { background: transparent; }")
        play = QVBoxLayout(params_widget)
        play.setContentsMargins(18, 0, 4, 4)
        play.setSpacing(3)

        param_data = {}
        for (pname, plabel, pmin, pmax, pdefault, pdecimals, pstep) in params:
            row = QHBoxLayout(); row.setSpacing(4)
            lbl = QLabel(f"{plabel}:")
            lbl.setFixedWidth(82)
            lbl.setStyleSheet("color: #888888; font-size: 10px; "
                              "background: transparent; border: none;")

            if pdecimals > 0:
                scale = pstep
                smin  = round(pmin / scale)
                smax  = round(pmax / scale)
                sval  = round(pdefault / scale)
            else:
                scale = 1
                smin, smax, sval = int(pmin), int(pmax), int(pdefault)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(smin, smax)
            slider.setValue(sval)
            slider.setStyleSheet(_SLIDER_STYLE)

            fmt = f"{pdefault:.{pdecimals}f}" if pdecimals > 0 else str(int(pdefault))
            val_lbl = QLabel(fmt)
            val_lbl.setFixedWidth(36)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet("color: #cccccc; font-size: 10px; "
                                  "background: transparent; border: none;")

            def _on_slide(v, vl=val_lbl, sc=scale, dec=pdecimals):
                actual = v * sc
                vl.setText(f"{actual:.{dec}f}" if dec > 0 else str(int(actual)))

            slider.valueChanged.connect(_on_slide)
            slider.sliderReleased.connect(self._on_aug_changed)
            
            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(val_lbl)
            play.addLayout(row)
            param_data[pname] = {"slider": slider, "scale": scale, "decimals": pdecimals}

        params_widget.hide()
        cb.toggled.connect(params_widget.setVisible)
        cb.toggled.connect(self._on_aug_changed)
        vlay.addWidget(params_widget)

        self._aug_widgets[key] = {
            "enabled":       cb,
            "params":        param_data,
            "params_widget": params_widget,
        }
        return container

    # ---- Right panel --------------------------------------------------

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background-color: #1e1e1e;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Collapse toggle for left settings panel
        self._toggle_btn = QPushButton("«  Hide Settings")
        self._toggle_btn.setFixedHeight(30)
        self._toggle_btn.setMinimumWidth(130)
        self._toggle_btn.setMaximumWidth(160)
        self._toggle_btn.setToolTip("Collapse / expand settings panel  (or drag the divider)")
        self._toggle_btn.setStyleSheet(_SIDEBAR_TOGGLE_STYLE)
        self._toggle_btn.clicked.connect(self._toggle_left_panel)
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.addWidget(self._toggle_btn)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2a2a; }")

        # Top: stacked widget (aug preview ↔ metrics)
        self._right_stack = QStackedWidget()
        self._right_stack.addWidget(self._build_aug_preview_page())   # index 0
        self._right_stack.addWidget(self._build_metrics_page())        # index 1
        self._right_stack.setCurrentIndex(0)
        splitter.addWidget(self._right_stack)

        # Bottom: console
        console_widget = QWidget()
        console_l = QVBoxLayout(console_widget)
        console_l.setContentsMargins(0, 0, 0, 0)

        hdr = QHBoxLayout()
        hdr_lbl = QLabel("Training Console")
        hdr_lbl.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        clr_btn = QPushButton("Clear"); clr_btn.setFixedWidth(60); clr_btn.setStyleSheet(_BROWSE_STYLE)
        clr_btn.clicked.connect(lambda: self._console.clear())
        hdr.addWidget(hdr_lbl); hdr.addStretch(); hdr.addWidget(clr_btn)
        console_l.addLayout(hdr)

        self._prog_frame = QFrame(); self._prog_frame.hide()
        pl = QVBoxLayout(self._prog_frame); pl.setContentsMargins(0, 0, 0, 0)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setStyleSheet("""
            QProgressBar { background-color: #2d2d2d; border: 1px solid #3a3a3a;
                border-radius: 4px; color: #ffffff; text-align: center; height: 18px; }
            QProgressBar::chunk { background-color: #6a3fc8; border-radius: 3px; }
        """)
        pl.addWidget(self._progress_bar)
        console_l.addWidget(self._prog_frame)

        self._console = QTextEdit()
        self._console.setReadOnly(True)
        font = QFont("Consolas", 9); font.setStyleHint(QFont.Monospace)
        self._console.setFont(font)
        self._console.setStyleSheet("""
            QTextEdit { background-color: #1a1a1a; color: #d4d4d4;
                border: 1px solid #2a2a2a; border-radius: 4px; padding: 6px; }
        """)
        console_l.addWidget(self._console)

        # Result card
        self._result_card = QFrame()
        self._result_card.setStyleSheet("""
            QFrame { background-color: #1e2e1e; border: 1px solid #3a3a3a;
                border-left: 3px solid #4caf50; border-radius: 4px; padding: 4px; }
        """)
        rc_l = QVBoxLayout(self._result_card)
        rc_l.setContentsMargins(12, 8, 12, 8); rc_l.setSpacing(6)
        self._rc_lbl = QLabel(""); self._rc_lbl.setStyleSheet("color: #4caf50; font-weight: bold;")
        self._rc_local_lbl = QLabel("")
        self._rc_local_lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self._rc_local_lbl.setWordWrap(True)
        self._rc_onnx_btn = QPushButton("⚙  Export to ONNX")
        self._rc_onnx_btn.setStyleSheet("""
            QPushButton { background-color: #1a3a5c; color: #4fc3f7;
                border: 1px solid #4fc3f7; border-radius: 4px;
                padding: 6px 12px; font-size: 12px; font-weight: bold; }
            QPushButton:hover { background-color: #1e4a72; }
            QPushButton:pressed { background-color: #153050; }
            QPushButton:disabled { background-color: #2a2a2a; color: #555555; border-color: #444444; }
        """)
        self._rc_onnx_btn.clicked.connect(self._on_export_onnx_clicked)
        self._rc_onnx_btn.hide()
        self._rc_onnx_status = QLabel("")
        self._rc_onnx_status.setStyleSheet("color: #4fc3f7; font-size: 11px;")
        self._rc_onnx_status.hide()
        rc_l.addWidget(self._rc_lbl)
        rc_l.addWidget(self._rc_local_lbl)
        rc_l.addWidget(self._rc_onnx_btn)
        rc_l.addWidget(self._rc_onnx_status)
        self._result_card.hide()
        console_l.addWidget(self._result_card)

        splitter.addWidget(console_widget)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)
        return panel

    # ---- Augmentation preview page (stack index 0) --------------------

    def _build_aug_preview_page(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: #1e1e1e;")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(6)

        title = QLabel("Augmentation Preview")
        title.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: bold;")
        lay.addWidget(title)

        # 3-column horizontal splitter
        h_split = QSplitter(Qt.Horizontal)
        h_split.setStyleSheet("QSplitter::handle { background: #2a2a2a; }")

        # Column 1 — Dataset thumbnails
        browser = QWidget()
        browser.setStyleSheet("background: #181818; border: 1px solid #2a2a2a; border-radius: 4px;")
        blay = QVBoxLayout(browser); blay.setContentsMargins(6, 6, 6, 6); blay.setSpacing(4)

        # Header: title + prev/next buttons
        b_hdr = QHBoxLayout(); b_hdr.setSpacing(4)
        b_title = QLabel("Dataset Images")
        b_title.setStyleSheet("color: #8ab4f8; font-size: 11px; font-weight: bold; "
                               "background: transparent; border: none;")
        b_hdr.addWidget(b_title)
        b_hdr.addStretch()
        self._nav_prev_btn = QPushButton("‹")
        self._nav_prev_btn.setFixedSize(24, 24)
        self._nav_prev_btn.setStyleSheet(_NAV_BTN_STYLE)
        self._nav_prev_btn.setToolTip("Previous image  (Left arrow)")
        self._nav_prev_btn.clicked.connect(self._nav_prev)
        self._nav_next_btn = QPushButton("›")
        self._nav_next_btn.setFixedSize(24, 24)
        self._nav_next_btn.setStyleSheet(_NAV_BTN_STYLE)
        self._nav_next_btn.setToolTip("Next image  (Right arrow)")
        self._nav_next_btn.clicked.connect(self._nav_next)
        b_hdr.addWidget(self._nav_prev_btn)
        b_hdr.addWidget(self._nav_next_btn)
        blay.addLayout(b_hdr)

        self._img_list = QListWidget()
        self._img_list.setViewMode(QListWidget.ListMode)
        self._img_list.setIconSize(QSize(56, 56))
        self._img_list.setSpacing(1)
        self._img_list.setStyleSheet(
            "QListWidget { background: #181818; border: none; color: #aaaaaa; font-size: 10px; }"
            "QListWidget::item { padding: 2px 4px; }"
            "QListWidget::item:selected { background: #6a3fc8; color: #ffffff; }"
        )
        self._img_list.currentItemChanged.connect(self._on_thumbnail_selected)
        blay.addWidget(self._img_list, 1)

        # Footer count label
        self._img_count_lbl = QLabel("Images: 0")
        self._img_count_lbl.setStyleSheet(
            "color: #666666; font-size: 9px; background: transparent; border: none;")
        self._img_count_lbl.setAlignment(Qt.AlignCenter)
        blay.addWidget(self._img_count_lbl)

        h_split.addWidget(browser)

        # Column 2 — Original
        orig_w = QWidget()
        orig_w.setStyleSheet("background: #181818; border: 1px solid #2a2a2a; border-radius: 4px;")
        olay = QVBoxLayout(orig_w); olay.setContentsMargins(6, 6, 6, 6); olay.setSpacing(4)
        o_title = QLabel("Original")
        o_title.setStyleSheet("color: #8ab4f8; font-size: 11px; font-weight: bold; "
                               "background: transparent; border: none;")
        olay.addWidget(o_title)
        self._orig_lbl = QLabel("Select an image →")
        self._orig_lbl.setAlignment(Qt.AlignCenter)
        self._orig_lbl.setStyleSheet("color: #444; background: #111;")
        self._orig_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        olay.addWidget(self._orig_lbl, 1)
        h_split.addWidget(orig_w)

        # Column 3 — Augmented full-size view
        aug_w = QWidget()
        aug_w.setStyleSheet("background: #181818; border: 1px solid #2a2a2a; border-radius: 4px;")
        alay = QVBoxLayout(aug_w); alay.setContentsMargins(6, 6, 6, 6); alay.setSpacing(4)
        a_title = QLabel("Augmented")
        a_title.setStyleSheet("color: #8ab4f8; font-size: 11px; font-weight: bold; "
                               "background: transparent; border: none;")
        alay.addWidget(a_title)
        self._aug_lbl = QLabel("Select augmentations →\nthen click  ▶ Run Batch Preview")
        self._aug_lbl.setAlignment(Qt.AlignCenter)
        self._aug_lbl.setStyleSheet("color: #444; font-size: 11px; background: #111; border: none;")
        self._aug_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        alay.addWidget(self._aug_lbl, 1)
        h_split.addWidget(aug_w)

        h_split.setSizes([180, 280, 280])
        lay.addWidget(h_split, 1)

        return page

    # ---- Metrics page (stack index 1) ---------------------------------

    def _build_metrics_page(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: #1e1e1e;")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        title = QLabel("Training Metrics")
        title.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: bold;")
        lay.addWidget(title)

        if HAS_MATPLOTLIB:
            self._fig = Figure(facecolor="#1a1a1a")
            self._ax_loss = self._fig.add_subplot(121, facecolor="#1a1a1a")
            self._ax_iou  = self._fig.add_subplot(122, facecolor="#1a1a1a")
            for ax, ylabel in [(self._ax_loss, "Loss"), (self._ax_iou, "IoU / mAP50")]:
                ax.tick_params(colors="#aaaaaa")
                ax.xaxis.label.set_color("#aaaaaa")
                ax.yaxis.label.set_color("#aaaaaa")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(ylabel)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#3a3a3a")
            try:
                self._fig.tight_layout(pad=2.5)
            except Exception:
                pass
            self._canvas = FigureCanvas(self._fig)
            lay.addWidget(self._canvas, 1)
        else:
            lbl = QLabel("Matplotlib not installed — graphs unavailable.")
            lbl.setStyleSheet("color: #aaaaaa; font-style: italic;")
            lbl.setAlignment(Qt.AlignCenter)
            lay.addWidget(lbl, 1)

        return page

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _get_aug_config(self) -> dict:
        config = {}
        for key, label, cat, params in _AUG_DEFS:
            if key not in self._aug_widgets:
                continue
            w = self._aug_widgets[key]
            entry = {"enabled": w["enabled"].isChecked()}
            for pname, pdata in w["params"].items():
                entry[pname] = pdata["slider"].value() * pdata["scale"]
            config[key] = entry
        return config

    def _on_aug_changed(self):
        try:
            if self._aug_widgets:
                s = self._settings
                s["augmentations"] = self._get_aug_config()
                _save_settings(s)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _restore_aug_settings(self):
        saved = self._settings.get("augmentations", {})
        for key, vals in saved.items():
            if key not in self._aug_widgets:
                continue
            w = self._aug_widgets[key]
            checked = bool(vals.get("enabled", False))
            w["enabled"].blockSignals(True)
            w["enabled"].setChecked(checked)
            w["enabled"].blockSignals(False)
            w["params_widget"].setVisible(checked)
            for pname, pdata in w["params"].items():
                if pname in vals:
                    pdata["slider"].blockSignals(True)
                    scale = pdata["scale"]
                    pdata["slider"].setValue(round(vals[pname] / scale))
                    pdata["slider"].blockSignals(False)

    def _load_dataset_thumbnails(self, folder: str):
        self._img_list.clear()
        self._dataset_paths = []
        if not HAS_CV2:
            return
        images_dir = os.path.join(folder, "train", "images")
        if not os.path.isdir(images_dir):
            images_dir = os.path.join(folder, "images")
        if not os.path.isdir(images_dir):
            return

        valid_exts = {".png", ".jpg", ".jpeg"}
        paths = []
        for f in sorted(os.listdir(images_dir)):
            if os.path.splitext(f)[1].lower() in valid_exts:
                paths.append(os.path.join(images_dir, f))
            if len(paths) >= 60:
                break

        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._dataset_paths.append(path)
            h, w = img_rgb.shape[:2]
            scale = 56 / max(h, w)
            tw, th = max(1, int(w * scale)), max(1, int(h * scale))
            thumb = np.ascontiguousarray(cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_AREA))
            qimg = QImage(thumb.data, tw, th, tw * 3, QImage.Format_RGB888)
            item = QListWidgetItem(QIcon(QPixmap.fromImage(qimg)), os.path.basename(path))
            item.setData(Qt.UserRole, path)
            item.setData(Qt.UserRole + 1, None)         # augmented image (filled by batch preview)
            item.setData(Qt.UserRole + 2, "original")   # row type
            item.setSizeHint(QSize(-1, 66))
            self._img_list.addItem(item)
        self._update_img_count_lbl()

    def _on_thumbnail_selected(self, current, _previous):
        if current is None:
            return
        path = current.data(Qt.UserRole)
        self._current_img_path = path
        if HAS_CV2 and path:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self._display_image(self._orig_lbl, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        aug_img = current.data(Qt.UserRole + 1)
        if aug_img is not None:
            self._display_image(self._aug_lbl, aug_img)
        else:
            self._aug_lbl.setPixmap(QPixmap())
            self._aug_lbl.setText("No augmented result.\nClick ▶ Run Batch Preview.")
            self._aug_lbl.setStyleSheet("color: #444; font-size: 11px; background: #111; border: none;")

    def _run_batch_preview(self):
        try:
            if not HAS_CV2:
                return
            aug_cfg = self._get_aug_config()
            any_enabled = any(v.get("enabled") for v in aug_cfg.values())

            if not any_enabled:
                self._aug_lbl.setPixmap(QPixmap())
                self._aug_lbl.setText("No augmentations selected.\nEnable at least one above.")
                return

            # Collect original paths from list (skip augmented sub-rows from prior run)
            original_paths = []
            for i in range(self._img_list.count()):
                item = self._img_list.item(i)
                if item and item.data(Qt.UserRole + 2) == "original":
                    original_paths.append(item.data(Qt.UserRole))

            if not original_paths:
                self._aug_lbl.setPixmap(QPixmap())
                self._aug_lbl.setText("No dataset images loaded.\nBrowse a dataset first.")
                return

            self._aug_lbl.setPixmap(QPixmap())
            self._aug_lbl.setText("Processing…")
            QApplication.processEvents()

            if not HAS_ALBUMENTATIONS:
                self._aug_lbl.setText("albumentations not available\n(check torch/DLL installation).")
                return

            A = _A
            n = self._batch_preview_sp.value()
            to_process = set(original_paths[:n])
            aug_results = {}  # path → aug numpy array

            for path in original_paths[:n]:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_transforms = []
                for key, vals in aug_cfg.items():
                    if not vals.get("enabled"):
                        continue
                    t = _preview_transform(key, vals, img_rgb.shape[1], img_rgb.shape[0])
                    if t is not None:
                        img_transforms.append(t)
                aug_img = img_rgb
                if img_transforms:
                    try:
                        result = A.Compose(img_transforms)(image=img_rgb)
                        aug_img = result["image"]
                        if aug_img.ndim == 2:
                            aug_img = np.stack([aug_img] * 3, axis=2)
                        if aug_img.shape[:2] != img_rgb.shape[:2]:
                            aug_img = cv2.resize(aug_img, (img_rgb.shape[1], img_rgb.shape[0]),
                                                 interpolation=cv2.INTER_AREA)
                    except Exception:
                        aug_img = img_rgb
                aug_results[path] = aug_img.copy()

            # Rebuild list: original row + augmented sub-row for each processed image
            self._img_list.clear()
            for path in original_paths:
                fname = os.path.basename(path)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                aug_img = aug_results.get(path)  # None if not in batch

                # — Original row —
                orig_item = QListWidgetItem(fname)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img_rgb.shape[:2]
                    sc = 56 / max(h, w)
                    tw, th = max(1, int(w * sc)), max(1, int(h * sc))
                    thumb = np.ascontiguousarray(cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_AREA))
                    qimg = QImage(thumb.data, tw, th, tw * 3, QImage.Format_RGB888)
                    orig_item.setIcon(QIcon(QPixmap.fromImage(qimg)))
                orig_item.setData(Qt.UserRole, path)
                orig_item.setData(Qt.UserRole + 1, aug_img)
                orig_item.setData(Qt.UserRole + 2, "original")
                orig_item.setSizeHint(QSize(-1, 66))
                self._img_list.addItem(orig_item)

                # — Augmented sub-row (only if in processed batch) —
                if aug_img is not None:
                    aug_item = QListWidgetItem(f"  ↳ augmented")
                    h, w = aug_img.shape[:2]
                    sc = 56 / max(h, w)
                    tw, th = max(1, int(w * sc)), max(1, int(h * sc))
                    thumb = np.ascontiguousarray(cv2.resize(aug_img, (tw, th), interpolation=cv2.INTER_AREA))
                    qimg = QImage(thumb.data, tw, th, tw * 3, QImage.Format_RGB888)
                    aug_item.setIcon(QIcon(QPixmap.fromImage(qimg)))
                    aug_item.setData(Qt.UserRole, path)
                    aug_item.setData(Qt.UserRole + 1, aug_img)
                    aug_item.setData(Qt.UserRole + 2, "augmented")
                    aug_item.setBackground(QColor("#222222"))
                    aug_item.setForeground(QColor("#888888"))
                    aug_item.setSizeHint(QSize(-1, 66))
                    self._img_list.addItem(aug_item)

            self._update_img_count_lbl()
            # Select first item and show its augmented result
            if self._img_list.count() > 0:
                self._img_list.setCurrentRow(0)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _nav_prev(self):
        try:
            n = self._img_list.count()
            if n == 0:
                return
            cur = self._img_list.currentRow()
            self._img_list.setCurrentRow(max(0, cur - 1) if cur > 0 else n - 1)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _nav_next(self):
        try:
            n = self._img_list.count()
            if n == 0:
                return
            cur = self._img_list.currentRow()
            self._img_list.setCurrentRow((cur + 1) % n)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _update_img_count_lbl(self):
        total = 0
        aug_count = 0
        for i in range(self._img_list.count()):
            item = self._img_list.item(i)
            if not item:
                continue
            if item.data(Qt.UserRole + 2) == "original":
                total += 1
            elif item.data(Qt.UserRole + 2) == "augmented":
                aug_count += 1
        if aug_count:
            self._img_count_lbl.setText(f"Images: {total}   |   Augmented: {aug_count}")
        else:
            self._img_count_lbl.setText(f"Images: {total}")

    def _toggle_left_panel(self):
        try:
            self._left_collapsed = not self._left_collapsed
            if self._left_collapsed:
                self._main_splitter.setSizes([0, 10000])
                self._toggle_btn.setText("»  Show Settings")
            else:
                self._main_splitter.setSizes([380, 10000])
                self._toggle_btn.setText("«  Hide Settings")

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _display_image(self, label: QLabel, img):
        if img is None or not HAS_CV2:
            return
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2)
        h, w = img.shape[:2]
        lw = max(label.width(), 80)
        lh = max(label.height(), 80)
        scale = min(lw / w, lh / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        display = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        display = np.ascontiguousarray(display)
        qimg = QImage(display.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------
    # Settings restore
    # ------------------------------------------------------------------

    def _restore_settings(self):
        s = self._settings
        _block = [
            self._annotator_cb,
            self._project_name_cb, self._project_id_cb, self._variant_cb, self._camera_cb,
            self._unet_arch_cb, self._encoder_cb, self._weights_cb,
            self._yolo_model_cb, self._epochs_sp, self._batch_sp,
            self._lr_sp, self._width_sp, self._height_sp,
            self._in_channels_sp, self._out_classes_sp, self._device_cb,
            self._radio_yolo, self._radio_unet,
        ]
        for w in _block:
            w.blockSignals(True)

        initials = s.get("annotator_initials", "")
        if initials:
            for i in range(self._annotator_cb.count()):
                if self._annotator_cb.itemData(i) and \
                        self._annotator_cb.itemData(i).get("initials") == initials:
                    self._annotator_cb.setCurrentIndex(i); break

        self._refresh_train_project_names()
        if s.get("project_name"):
            self._project_name_cb.setCurrentText(s["project_name"])
        self._refresh_train_project_ids()
        if s.get("project_id"):
            self._project_id_cb.setCurrentText(s["project_id"])
        self._refresh_train_variants()
        if s.get("variant"):
            self._variant_cb.setCurrentText(s["variant"])
        self._refresh_train_cameras()
        if s.get("camera"):
            self._camera_cb.setCurrentText(s["camera"])

        pw = s.get("pretrained_weights", "")
        if pw:
            self._weights_edit.setText(pw)

        self._unet_arch_cb.setCurrentText(s.get("architecture", "Unet"))
        self._encoder_cb.setCurrentText(s.get("encoder", "efficientnet-b3"))
        self._weights_cb.setCurrentText(s.get("encoder_weights", "imagenet"))
        if s.get("yolo_task", "detection") == "segmentation":
            self._yolo_seg_rb.setChecked(True)
        else:
            self._yolo_detect_rb.setChecked(True)
        self._on_yolo_task_changed()
        self._yolo_model_cb.setCurrentText(s.get("yolo_model", "yolov8n"))
        self._epochs_sp.setValue(int(s.get("epochs", 100)))
        self._batch_sp.setValue(int(s.get("batch_size", 4)))
        self._lr_sp.setValue(float(s.get("learning_rate", 0.001)))
        self._width_sp.setValue(int(s.get("image_width", 320)))
        self._height_sp.setValue(int(s.get("image_height", 240)))
        self._in_channels_sp.setValue(int(s.get("in_channels", 3)))
        self._out_classes_sp.setValue(int(s.get("out_classes", 2)))
        self._patience_sp.setValue(int(s.get("early_stopping_patience", 10)))
        self._num_workers_sp.setValue(int(s.get("num_workers", 0)))
        self._repeat_factor_sp.setValue(int(s.get("repeat_factor", 1)))
        _enabled_metrics = set(s.get("extra_metrics", ["precision", "recall", "f1"]))
        for key, cb in self._metric_checkboxes.items():
            cb.setChecked(key in _enabled_metrics)
        self._device_cb.setCurrentText(s.get("device", "cpu"))
        # UNet — loss / optimizer / scheduler / regularization
        self._unet_loss_cb.setCurrentText(s.get("unet_loss_fn", "focal"))
        self._unet_optimizer_cb.setCurrentText(s.get("unet_optimizer", "Adam"))
        self._unet_scheduler_cb.setCurrentText(s.get("unet_scheduler", "cosine"))
        self._unet_weight_decay_sp.setValue(float(s.get("unet_weight_decay", 0.0)))
        self._unet_momentum_sp.setValue(float(s.get("unet_momentum", 0.9)))
        self._unet_grad_clip_sp.setValue(float(s.get("unet_gradient_clip", 0.0)))
        self._unet_label_smoothing_sp.setValue(float(s.get("unet_label_smoothing", 0.0)))
        self._on_unet_optimizer_changed(self._unet_optimizer_cb.currentText())
        # YOLO advanced
        self._yolo_optimizer_cb.setCurrentText(s.get("yolo_optimizer", "AdamW"))
        self._yolo_lrf_sp.setValue(float(s.get("yolo_lrf", 0.01)))
        self._yolo_momentum_sp.setValue(float(s.get("yolo_momentum", 0.937)))
        self._yolo_weight_decay_sp.setValue(float(s.get("yolo_weight_decay", 0.0005)))
        self._yolo_warmup_sp.setValue(float(s.get("yolo_warmup_epochs", 5.0)))
        self._yolo_cos_lr_cb.setChecked(bool(s.get("yolo_cos_lr", False)))
        self._yolo_box_sp.setValue(float(s.get("yolo_box", 7.5)))
        self._yolo_cls_sp.setValue(float(s.get("yolo_cls", 3.0)))
        self._yolo_dfl_sp.setValue(float(s.get("yolo_dfl", 1.5)))
        self._yolo_dropout_sp.setValue(float(s.get("yolo_dropout", 0.1)))
        self._yolo_overlap_mask_cb.setChecked(bool(s.get("yolo_overlap_mask", True)))
        self._yolo_mask_ratio_sp.setValue(int(s.get("yolo_mask_ratio", 4)))
        self._yolo_hsv_h_sp.setValue(float(s.get("yolo_hsv_h", 0.01)))
        self._yolo_hsv_s_sp.setValue(float(s.get("yolo_hsv_s", 0.30)))
        self._yolo_hsv_v_sp.setValue(float(s.get("yolo_hsv_v", 0.20)))
        self._yolo_fliplr_sp.setValue(float(s.get("yolo_fliplr", 0.5)))
        self._yolo_flipud_sp.setValue(float(s.get("yolo_flipud", 0.0)))
        self._yolo_degrees_sp.setValue(float(s.get("yolo_degrees", 5.0)))
        self._yolo_translate_sp.setValue(float(s.get("yolo_translate", 0.05)))
        self._yolo_scale_sp.setValue(float(s.get("yolo_scale", 0.25)))
        self._yolo_mosaic_sp.setValue(float(s.get("yolo_mosaic", 0.3)))
        self._yolo_mixup_sp.setValue(float(s.get("yolo_mixup", 0.0)))
        self._yolo_copy_paste_sp.setValue(float(s.get("yolo_copy_paste", 0.1)))
        self._yolo_close_mosaic_sp.setValue(int(s.get("yolo_close_mosaic", 20)))
        self._yolo_amp_cb.setChecked(bool(s.get("yolo_amp", True)))
        self._yolo_cache_cb.setChecked(bool(s.get("yolo_cache", False)))
        self._yolo_save_period_sp.setValue(int(s.get("yolo_save_period", 10)))
        self._yolo_plots_cb.setChecked(bool(s.get("yolo_plots", True)))

        if s.get("task_type", "unet") == "yolo":
            self._radio_yolo.setChecked(True)
        else:
            self._radio_unet.setChecked(True)

        for w in _block:
            w.blockSignals(False)

        self._on_task_changed()
        self._restore_aug_settings()

        ds_path = s.get("dataset_folder", "")
        if ds_path and os.path.isdir(ds_path):
            self._load_dataset_info(ds_path)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_task_changed(self):
        try:
            is_yolo = self._radio_yolo.isChecked()
            self._unet_block.setVisible(not is_yolo)
            self._yolo_block.setVisible(is_yolo)
            if hasattr(self, "_yolo_adv_widget"):
                self._yolo_adv_widget.setVisible(is_yolo)
            if hasattr(self, "_aug_collapsible"):
                self._aug_collapsible.setVisible(not is_yolo)
            if hasattr(self, "_unet_only_hp"):
                self._unet_only_hp.setVisible(not is_yolo)
            if hasattr(self, "_width_row_widget"):
                self._width_row_widget.setVisible(not is_yolo)
            if hasattr(self, "_aug_yolo_note"):
                self._aug_yolo_note.setVisible(is_yolo)
                self._aug_controls_widget.setVisible(not is_yolo)
            if hasattr(self, "_project_name_cb"):
                self._refresh_train_project_names()
            if getattr(self, '_initialized', False):
                if is_yolo:
                    self._epochs_sp.setValue(100)
                    self._batch_sp.setValue(4)
                    self._lr_sp.setValue(0.002)
                    self._height_sp.setValue(640)
                    self._patience_sp.setValue(40)
                else:
                    self._epochs_sp.setValue(100)
                    self._batch_sp.setValue(4)
                    self._lr_sp.setValue(0.001)
                    self._width_sp.setValue(320)
                    self._height_sp.setValue(240)
                    self._patience_sp.setValue(10)
            self._autosave()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_yolo_task_changed(self):
        try:
            is_det = self._yolo_detect_rb.isChecked()
            current = self._yolo_model_cb.currentText()
            self._yolo_model_cb.blockSignals(True)
            self._yolo_model_cb.clear()
            if is_det:
                models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
            else:
                models = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"]
            self._yolo_model_cb.addItems(models)
            base = current.replace("-seg", "")
            target = base if is_det else base + "-seg"
            idx = self._yolo_model_cb.findText(target)
            if idx >= 0:
                self._yolo_model_cb.setCurrentIndex(idx)
            self._yolo_model_cb.blockSignals(False)
            if hasattr(self, "_yolo_seg_options_col"):
                self._yolo_seg_options_col.setVisible(not is_det)
            self._autosave()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_unet_optimizer_changed(self, text):
        try:
            """Show momentum row only for SGD and RMSprop."""
            self._unet_momentum_row.setVisible(text in ("SGD", "RMSprop"))

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _current_train_model_type(self) -> str:
        return "yolo" if self._radio_yolo.isChecked() else "unet"

    def _refresh_train_project_names(self):
        mt  = self._current_train_model_type()
        reg = RegistrySettings().scan_project_names(mt)
        cus = projects_config.get_custom_project_names(mt)
        names = sorted(set(reg) | set(cus))
        cur = self._project_name_cb.currentText() if hasattr(self, "_project_name_cb") else ""
        self._project_name_cb.blockSignals(True)
        self._project_name_cb.clear()
        for n in names:
            self._project_name_cb.addItem(n)
        if cur:
            self._project_name_cb.setCurrentText(cur)
        self._project_name_cb.blockSignals(False)

    def _refresh_train_project_ids(self):
        mt  = self._current_train_model_type()
        pn  = self._project_name_cb.currentText().strip()
        cur = self._project_id_cb.currentText()
        self._project_id_cb.blockSignals(True)
        self._project_id_cb.clear()
        if pn:
            reg = RegistrySettings().scan_project_ids(mt, pn)
            cus = projects_config.get_custom_project_ids(mt, pn)
            for i in sorted(set(reg) | set(cus)):
                self._project_id_cb.addItem(i)
            if cur:
                self._project_id_cb.setCurrentText(cur)
        self._project_id_cb.blockSignals(False)

    def _refresh_train_variants(self):
        mt  = self._current_train_model_type()
        pn  = self._project_name_cb.currentText().strip()
        pid = self._project_id_cb.currentText().strip()
        cur = self._variant_cb.currentText()
        self._variant_cb.blockSignals(True)
        self._variant_cb.clear()
        if pn and pid:
            reg = RegistrySettings().scan_variants(mt, pn, pid)
            cus = projects_config.get_custom_variants(mt, pn, pid)
            for v in sorted(set(reg) | set(cus)):
                self._variant_cb.addItem(v)
            if cur:
                self._variant_cb.setCurrentText(cur)
        self._variant_cb.blockSignals(False)

    def _refresh_train_cameras(self):
        mt  = self._current_train_model_type()
        pn  = self._project_name_cb.currentText().strip()
        pid = self._project_id_cb.currentText().strip()
        var = self._variant_cb.currentText().strip()
        cur = self._camera_cb.currentText()
        self._camera_cb.blockSignals(True)
        self._camera_cb.clear()
        if pn and pid and var:
            reg = RegistrySettings().scan_cameras(mt, pn, pid, var)
            cus = projects_config.get_custom_cameras(mt, pn, pid, var)
            for c in sorted(set(reg) | set(cus)):
                self._camera_cb.addItem(c)
            if cur:
                self._camera_cb.setCurrentText(cur)
        self._camera_cb.blockSignals(False)

    def _on_train_project_name_changed(self):
        try:
            self._refresh_train_project_ids()
            self._refresh_train_variants()
            self._refresh_train_cameras()
            self._autosave()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_train_project_id_changed(self):
        try:
            self._refresh_train_variants()
            self._refresh_train_cameras()
            self._autosave()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_train_variant_changed(self):
        try:
            self._refresh_train_cameras()
            self._autosave()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def prefill_from_staged(self, staged_info: dict):
        """Called from Data Prep tab — pre-fills hierarchy and dataset path."""
        model_type = staged_info.get("model_type", "unet").lower()
        if model_type == "yolo":
            self._radio_yolo.setChecked(True)
        else:
            self._radio_unet.setChecked(True)
        self._on_task_changed()

        self._project_name_cb.setCurrentText(staged_info.get("project_name", ""))
        self._refresh_train_project_ids()
        self._project_id_cb.setCurrentText(staged_info.get("project_id", ""))
        self._refresh_train_variants()
        self._variant_cb.setCurrentText(staged_info.get("variant", ""))
        self._refresh_train_cameras()
        self._camera_cb.setCurrentText(staged_info.get("camera", ""))

        path = staged_info.get("path", "")
        if path and os.path.isdir(path):
            self._load_dataset_info(path)

        self._autosave()

    def _autosave(self):
        try:
            if not getattr(self, '_initialized', False):
                return
            s = self._settings
            s["task_type"] = "yolo" if self._radio_yolo.isChecked() else "unet"
            ann_data = self._annotator_cb.currentData()
            if ann_data:
                s["annotator_initials"] = ann_data.get("initials", "")
            s["project_name"] = self._project_name_cb.currentText().strip()
            s["project_id"]   = self._project_id_cb.currentText().strip()
            s["variant"]      = self._variant_cb.currentText().strip()
            s["camera"]       = self._camera_cb.currentText().strip()
            s["dataset_folder"]     = self._ds_edit.text()
            s["pretrained_weights"] = self._weights_edit.text()
            s["architecture"]       = self._unet_arch_cb.currentText()
            s["encoder"]            = self._encoder_cb.currentText()
            s["encoder_weights"]    = self._weights_cb.currentText()
            s["yolo_model"]         = self._yolo_model_cb.currentText()
            s["yolo_task"]          = "segmentation" if self._yolo_seg_rb.isChecked() else "detection"
            s["epochs"]             = self._epochs_sp.value()
            s["batch_size"]         = self._batch_sp.value()
            s["learning_rate"]      = self._lr_sp.value()
            s["image_width"]        = self._width_sp.value()
            s["image_height"]       = self._height_sp.value()
            s["in_channels"]               = self._in_channels_sp.value()
            s["out_classes"]               = self._out_classes_sp.value()
            s["early_stopping_patience"]   = self._patience_sp.value()
            s["extra_metrics"]             = [k for k, cb in self._metric_checkboxes.items() if cb.isChecked()]
            s["device"]                    = self._device_cb.currentText()
            s["num_workers"]               = self._num_workers_sp.value()
            s["repeat_factor"]             = self._repeat_factor_sp.value()
            s["unet_loss_fn"]         = self._unet_loss_cb.currentText()
            s["unet_optimizer"]       = self._unet_optimizer_cb.currentText()
            s["unet_scheduler"]       = self._unet_scheduler_cb.currentText()
            s["unet_weight_decay"]    = self._unet_weight_decay_sp.value()
            s["unet_momentum"]        = self._unet_momentum_sp.value()
            s["unet_gradient_clip"]   = self._unet_grad_clip_sp.value()
            s["unet_label_smoothing"] = self._unet_label_smoothing_sp.value()
            if self._aug_widgets:
                s["augmentations"]  = self._get_aug_config()
            # YOLO advanced
            s["yolo_optimizer"]      = self._yolo_optimizer_cb.currentText()
            s["yolo_lrf"]            = self._yolo_lrf_sp.value()
            s["yolo_momentum"]       = self._yolo_momentum_sp.value()
            s["yolo_weight_decay"]   = self._yolo_weight_decay_sp.value()
            s["yolo_warmup_epochs"]  = self._yolo_warmup_sp.value()
            s["yolo_cos_lr"]         = self._yolo_cos_lr_cb.isChecked()
            s["yolo_box"]            = self._yolo_box_sp.value()
            s["yolo_cls"]            = self._yolo_cls_sp.value()
            s["yolo_dfl"]            = self._yolo_dfl_sp.value()
            s["yolo_dropout"]        = self._yolo_dropout_sp.value()
            s["yolo_overlap_mask"]   = self._yolo_overlap_mask_cb.isChecked()
            s["yolo_mask_ratio"]     = self._yolo_mask_ratio_sp.value()
            s["yolo_hsv_h"]          = self._yolo_hsv_h_sp.value()
            s["yolo_hsv_s"]          = self._yolo_hsv_s_sp.value()
            s["yolo_hsv_v"]          = self._yolo_hsv_v_sp.value()
            s["yolo_fliplr"]         = self._yolo_fliplr_sp.value()
            s["yolo_flipud"]         = self._yolo_flipud_sp.value()
            s["yolo_degrees"]        = self._yolo_degrees_sp.value()
            s["yolo_translate"]      = self._yolo_translate_sp.value()
            s["yolo_scale"]          = self._yolo_scale_sp.value()
            s["yolo_mosaic"]         = self._yolo_mosaic_sp.value()
            s["yolo_mixup"]          = self._yolo_mixup_sp.value()
            s["yolo_copy_paste"]     = self._yolo_copy_paste_sp.value()
            s["yolo_close_mosaic"]   = self._yolo_close_mosaic_sp.value()
            s["yolo_amp"]            = self._yolo_amp_cb.isChecked()
            s["yolo_cache"]          = self._yolo_cache_cb.isChecked()
            s["yolo_save_period"]    = self._yolo_save_period_sp.value()
            s["yolo_plots"]          = self._yolo_plots_cb.isChecked()
            _save_settings(s)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _browse_dataset(self):
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder",
                                                       self._ds_edit.text() or "")
            if folder:
                self._load_dataset_info(folder)
                self._autosave()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _browse_weights(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Weights File", "",
                "Weights (*.pt *.ckpt *.pth);;All Files (*)")
            if path:
                self._weights_edit.setText(path)
                self._autosave()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _clear_weights(self):
        try:
            self._weights_edit.clear()
            self._autosave()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
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
            t  = self._dataset_info.get("task_type", "unet").upper()
            tr = self._dataset_info.get("train_count", 0)
            v  = self._dataset_info.get("val_count", 0)
            ts = self._dataset_info.get("test_count", 0)
            self._ds_info_lbl.setText(f"Task: {t}  |  Train: {tr}  Val: {v}  Test: {ts}")
            self._ds_info_lbl.setStyleSheet("color: #4caf50; font-size: 11px;")
            nc = self._dataset_info.get("num_classes", 0)
            if nc > 0:
                self._out_classes_sp.setValue(nc)
            if t.lower() == "yolo":
                self._radio_yolo.setChecked(True)
            else:
                self._radio_unet.setChecked(True)
        except Exception:
            self._ds_info_lbl.setText("⚠ Failed to parse dataset_info.json")
            self._ds_info_lbl.setStyleSheet("color: #f87171; font-size: 11px;")
            self._dataset_info = {}
        self._load_dataset_thumbnails(folder)

    def _get_form_dict(self) -> dict:
        ann_data = self._annotator_cb.currentData() or {}
        task_type = "yolo" if self._radio_yolo.isChecked() else "unet"
        arch = (self._yolo_model_cb.currentText() if task_type == "yolo"
                else self._unet_arch_cb.currentText())
        return {
            "annotator_name":     ann_data.get("name", ""),
            "annotator_initials": ann_data.get("initials", ""),
            "project_name":       self._project_name_cb.currentText().strip(),
            "project_id":         self._project_id_cb.currentText().strip(),
            "variant":            self._variant_cb.currentText().strip(),
            "camera":             self._camera_cb.currentText().strip(),
            "pretrained_weights": self._weights_edit.text().strip(),
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
            "out_classes":               self._out_classes_sp.value(),
            "early_stopping_patience":   self._patience_sp.value(),
            "extra_metrics":             [k for k, cb in self._metric_checkboxes.items() if cb.isChecked()],
            "device":                    self._device_cb.currentText(),
            "num_workers":               self._num_workers_sp.value(),
            "repeat_factor":             self._repeat_factor_sp.value(),
            "augmentations":      self._get_aug_config() if not self._radio_yolo.isChecked() else {},
            # UNet — loss / optimizer / scheduler / regularization
            "unet_loss_fn":         self._unet_loss_cb.currentText(),
            "unet_optimizer":       self._unet_optimizer_cb.currentText(),
            "unet_scheduler":       self._unet_scheduler_cb.currentText(),
            "unet_weight_decay":    self._unet_weight_decay_sp.value(),
            "unet_momentum":        self._unet_momentum_sp.value(),
            "unet_gradient_clip":   self._unet_grad_clip_sp.value(),
            "unet_label_smoothing": self._unet_label_smoothing_sp.value(),
            # YOLO advanced
            "yolo_optimizer":     self._yolo_optimizer_cb.currentText(),
            "yolo_lrf":           self._yolo_lrf_sp.value(),
            "yolo_momentum":      self._yolo_momentum_sp.value(),
            "yolo_weight_decay":  self._yolo_weight_decay_sp.value(),
            "yolo_warmup_epochs": self._yolo_warmup_sp.value(),
            "yolo_cos_lr":        self._yolo_cos_lr_cb.isChecked(),
            "yolo_box":           self._yolo_box_sp.value(),
            "yolo_cls":           self._yolo_cls_sp.value(),
            "yolo_dfl":           self._yolo_dfl_sp.value(),
            "yolo_dropout":       self._yolo_dropout_sp.value(),
            "yolo_overlap_mask":  self._yolo_overlap_mask_cb.isChecked(),
            "yolo_mask_ratio":    self._yolo_mask_ratio_sp.value(),
            "yolo_hsv_h":         self._yolo_hsv_h_sp.value(),
            "yolo_hsv_s":         self._yolo_hsv_s_sp.value(),
            "yolo_hsv_v":         self._yolo_hsv_v_sp.value(),
            "yolo_fliplr":        self._yolo_fliplr_sp.value(),
            "yolo_flipud":        self._yolo_flipud_sp.value(),
            "yolo_degrees":       self._yolo_degrees_sp.value(),
            "yolo_translate":     self._yolo_translate_sp.value(),
            "yolo_scale":         self._yolo_scale_sp.value(),
            "yolo_mosaic":        self._yolo_mosaic_sp.value(),
            "yolo_mixup":         self._yolo_mixup_sp.value(),
            "yolo_copy_paste":    self._yolo_copy_paste_sp.value(),
            "yolo_close_mosaic":  self._yolo_close_mosaic_sp.value(),
            "yolo_amp":           self._yolo_amp_cb.isChecked(),
            "yolo_cache":         self._yolo_cache_cb.isChecked(),
            "yolo_save_period":   self._yolo_save_period_sp.value(),
            "yolo_plots":         self._yolo_plots_cb.isChecked(),
        }

    def _run_checks(self, show_dialog_on_failure: bool = False) -> bool:
        try:
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
            self._last_preflight_results = results
            all_ok = True
            failed_msgs = []
            for r in results:
                if r["ok"]:
                    icon, color = "✓", "#4caf50"
                elif r["warn"]:
                    icon, color = "⚠", "#ffc107"
                else:
                    icon, color = "✗", "#f87171"
                    all_ok = False
                    failed_msgs.append(r["name"])
                lbl = QLabel(f"{icon} {r['name']}: {r['msg']}")
                lbl.setStyleSheet(f"color: {color}; font-size: 11px;")
                lbl.setWordWrap(True)
                self._checks_list_layout.addWidget(lbl)
            self._copy_diag_btn.show()
            if not all_ok and show_dialog_on_failure:
                QMessageBox.warning(self, "Pre-flight Checks Failed",
                                    "Fix the following issues:\n" + "\n".join(failed_msgs))
            return all_ok

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _copy_diagnostic(self):
        try:
            import sys, platform
            lines = [
                "=== DualAnnotator Diagnostic Info ===",
                f"OS         : {platform.platform()}",
                f"Python     : {sys.version}",
                f"Scripts dir: {_SCRIPTS_DIR}",
                f"Registry   : {self._registry_root}",
                "",
                "--- Pre-flight Results ---",
            ]
            for r in self._last_preflight_results:
                if r["ok"]:
                    status = "OK  "
                elif r["warn"]:
                    status = "WARN"
                else:
                    status = "FAIL"
                lines.append(f"[{status}] {r['name']}: {r['msg']}")
            QApplication.clipboard().setText("\n".join(lines))
            self._copy_diag_btn.setText("Copied!")
            QTimer.singleShot(2000, lambda: self._copy_diag_btn.setText("Copy Diagnostic Info"))

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_start_stop(self):
        try:
            if self._worker is not None:
                self._worker.cancel()
                self._start_btn.setEnabled(False)
                return

            if not self._run_checks(show_dialog_on_failure=True):
                return

            # ── Check if AI Engine needs setup ──
            if needs_setup():
                reply = QMessageBox.question(
                    self,
                    "AI Engine Not Found",
                    "Training requires the AI Engine (PyTorch + ML packages).\n\n"
                    "This is a one-time download (~2 GB) that takes 5-10 minutes.\n\n"
                    "Would you like to install it now?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    dialog = _EngineSetupDialog(self)
                    dialog.setup_complete.connect(self._on_engine_setup_done)
                    dialog.start()
                    dialog.exec_()
                return

            try:
                from mlops.training import build_training_config
                cfg = build_training_config(self._get_form_dict())
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
                return

            self._result_card.hide()
            self._rc_onnx_btn.hide()
            self._rc_onnx_status.hide()
            self._last_version_folder = ""
            self._prog_frame.show()
            self._progress_bar.setValue(0)
            self._console.clear()

            self._epochs_seen  = []
            self._train_losses = []
            self._val_losses   = []
            self._train_ious   = []
            self._val_ious     = []

            if HAS_MATPLOTLIB:
                for ax, ylabel in [(self._ax_loss, "Loss"), (self._ax_iou, "IoU / mAP50")]:
                    ax.cla()
                    ax.set_facecolor("#1a1a1a")
                    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
                    ax.tick_params(colors="#aaaaaa")
                    ax.xaxis.label.set_color("#aaaaaa")
                    ax.yaxis.label.set_color("#aaaaaa")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#3a3a3a")
                self._canvas.draw_idle()

            # Switch to metrics view
            self._right_stack.setCurrentIndex(1)

            self._start_btn.setText("■  Stop Training")
            self._start_btn.setStyleSheet(_STOP_STYLE)

            self._worker = TrainWorker(cfg, self._registry_root, _SCRIPTS_DIR, parent=self)
            self._worker.log.connect(self._append_log)
            self._worker.progress.connect(self._progress_bar.setValue)
            self._worker.metric.connect(self._on_metric)
            self._worker.finished.connect(self._on_finished)
            self._worker.start()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_engine_setup_done(self, success: bool):
        if success:
            QMessageBox.information(
                self,
                "Setup Complete",
                "AI Engine installed successfully!\n\n"
                "Click 'Start Training' again to begin.",
            )
        else:
            QMessageBox.warning(
                self,
                "Setup Failed",
                "AI Engine installation failed.\n\n"
                "Check the log above for details.\n\n"
                "For step-by-step help, open the docs and go to:\n"
                "Train → Troubleshooting",
            )

    @pyqtSlot(str)
    def _append_log(self, line: str):
        sb = self._console.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self._console.append(line)
        if at_bottom:
            sb.setValue(sb.maximum())

    @pyqtSlot(int, float, float, float, float, float, float)
    def _on_metric(self, epoch: int, train_loss: float, val_loss: float,
                   train_iou: float = 0.0, val_iou: float = 0.0,
                   _train_per_iou: float = 0.0, _val_per_iou: float = 0.0):
        self._epochs_seen.append(epoch)
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        self._train_ious.append(train_iou)
        self._val_ious.append(val_iou)

        if HAS_MATPLOTLIB:
            _style = dict(facecolor="#2a2a2a", edgecolor="#3a3a3a", labelcolor="#cccccc")

            self._ax_loss.cla()
            self._ax_loss.set_facecolor("#1a1a1a")
            self._ax_loss.plot(self._epochs_seen, self._train_losses,
                               color="#4a9eff", label="train", linewidth=1.5)
            self._ax_loss.plot(self._epochs_seen, self._val_losses,
                               color="#ff7f50", label="val", linewidth=1.5)
            self._ax_loss.legend(**_style)
            self._ax_loss.set_xlabel("Epoch"); self._ax_loss.set_ylabel("Loss")
            self._ax_loss.tick_params(colors="#aaaaaa")
            self._ax_loss.xaxis.label.set_color("#aaaaaa")
            self._ax_loss.yaxis.label.set_color("#aaaaaa")
            for sp in self._ax_loss.spines.values():
                sp.set_edgecolor("#3a3a3a")

            self._ax_iou.cla()
            self._ax_iou.set_facecolor("#1a1a1a")
            self._ax_iou.plot(self._epochs_seen, self._train_ious,
                              color="#66bb6a", label="train", linewidth=1.5)
            self._ax_iou.plot(self._epochs_seen, self._val_ious,
                              color="#ffa726", label="val", linewidth=1.5)
            self._ax_iou.legend(**_style)
            self._ax_iou.set_xlabel("Epoch"); self._ax_iou.set_ylabel("IoU / mAP50")
            self._ax_iou.tick_params(colors="#aaaaaa")
            self._ax_iou.xaxis.label.set_color("#aaaaaa")
            self._ax_iou.yaxis.label.set_color("#aaaaaa")
            for sp in self._ax_iou.spines.values():
                sp.set_edgecolor("#3a3a3a")

            try:
                self._fig.tight_layout(pad=2.5)
            except Exception:
                pass
            self._canvas.draw_idle()

    @pyqtSlot(bool, str, str, bool)
    def _on_finished(self, success: bool, version_id: str, local_run_dir: str, is_partial: bool):
        self._start_btn.setText("▶  Start Training")
        self._start_btn.setStyleSheet(_START_STYLE)
        self._start_btn.setEnabled(True)
        self._prog_frame.hide()
        self._worker = None

        if success:
            self.training_completed.emit(local_run_dir or "")
            projects_config.ensure_hierarchy(
                self._current_train_model_type(),
                self._project_name_cb.currentText().strip(),
                self._project_id_cb.currentText().strip(),
                self._variant_cb.currentText().strip(),
                self._camera_cb.currentText().strip(),
            )
            self._refresh_train_project_names()

            if is_partial:
                self._rc_lbl.setText(f"⚠ Stopped early (partial) — Version ID: {version_id}")
                self._rc_lbl.setStyleSheet("color: #ffc107; font-weight: bold;")
            else:
                self._rc_lbl.setText(f"✓ Training Complete — Version ID: {version_id}")
                self._rc_lbl.setStyleSheet("color: #4caf50; font-weight: bold;")

            if local_run_dir:
                self._rc_local_lbl.setText(f"📁 Local: {local_run_dir}")
                self._rc_local_lbl.show()
                self._last_version_folder = local_run_dir
            else:
                self._rc_local_lbl.hide()
                self._last_version_folder = ""

            has_weights = os.path.isfile(os.path.join(self._last_version_folder, "best.pt"))
            if has_weights:
                self._rc_onnx_btn.setEnabled(True)
                self._rc_onnx_btn.setText("⚙  Export to ONNX")
                self._rc_onnx_btn.show()
            self._rc_onnx_status.hide()
            self._result_card.show()
        else:
            self._append_log("<span style='color:#f87171'>[FAILED] Training failed or was cancelled.</span>")

    def _on_export_onnx_clicked(self):
        try:
            if not self._last_version_folder or not os.path.isdir(self._last_version_folder):
                self._rc_onnx_status.setText("✗ Version folder not found.")
                self._rc_onnx_status.setStyleSheet("color: #f87171; font-size: 11px;")
                self._rc_onnx_status.show()
                return
            weights = os.path.join(self._last_version_folder, "best.pt")
            if not os.path.isfile(weights):
                self._rc_onnx_status.setText("✗ best.pt not found in version folder.")
                self._rc_onnx_status.setStyleSheet("color: #f87171; font-size: 11px;")
                self._rc_onnx_status.show()
                return
            self._rc_onnx_btn.setEnabled(False)
            self._rc_onnx_btn.setText("Exporting...")
            self._rc_onnx_status.setText("Converting best.pt → ONNX...")
            self._rc_onnx_status.setStyleSheet("color: #4fc3f7; font-size: 11px;")
            self._rc_onnx_status.show()
            self._prog_frame.show()
            self._progress_bar.setValue(0)
            self._onnx_worker = OnnxWorker(self._last_version_folder, _SCRIPTS_DIR, parent=self)
            self._onnx_worker.log.connect(self._append_log)
            self._onnx_worker.progress.connect(self._progress_bar.setValue)
            self._onnx_worker.finished.connect(self._on_onnx_finished)
            self._onnx_worker.start()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return

    @pyqtSlot(bool)
    def _on_onnx_finished(self, success: bool):
        self._prog_frame.hide()
        self._onnx_worker = None
        if success:
            self._rc_onnx_btn.setText("✓  ONNX Exported")
            self._rc_onnx_btn.setEnabled(False)
            self._rc_onnx_status.setText(f"best.onnx saved to: {self._last_version_folder}")
            self._rc_onnx_status.setStyleSheet("color: #4caf50; font-size: 11px;")
            if self._last_version_folder:
                self.onnx_exported.emit(self._last_version_folder)
        else:
            self._rc_onnx_btn.setText("⚙  Export to ONNX")
            self._rc_onnx_btn.setEnabled(True)
            self._rc_onnx_status.setText("✗ ONNX export failed — see console.")
            self._rc_onnx_status.setStyleSheet("color: #f87171; font-size: 11px;")
        self._rc_onnx_status.show()

    # ------------------------------------------------------------------
    # Clone & Re-train (called from registry browser)
    # ------------------------------------------------------------------

    def load_from_manifest(self, manifest: dict):
        hp         = manifest.get("hyperparams", {})
        model_type = manifest.get("model_type", "unet").lower()

        if model_type == "yolo":
            self._radio_yolo.setChecked(True)
        else:
            self._radio_unet.setChecked(True)
        self._on_task_changed()

        if model_type == "unet":
            arch = hp.get("architecture", "Unet")
            if self._unet_arch_cb.findText(arch) >= 0:
                self._unet_arch_cb.setCurrentText(arch)
            enc = hp.get("encoder", "efficientnet-b3")
            if self._encoder_cb.findText(enc) >= 0:
                self._encoder_cb.setCurrentText(enc)
            w = hp.get("encoder_weights", "imagenet")
            if self._weights_cb.findText(w) >= 0:
                self._weights_cb.setCurrentText(w)
        else:
            arch = hp.get("architecture", "yolov8n")
            if arch.endswith("-seg"):
                self._yolo_seg_rb.setChecked(True)
            else:
                self._yolo_detect_rb.setChecked(True)
            self._on_yolo_task_changed()
            if self._yolo_model_cb.findText(arch) >= 0:
                self._yolo_model_cb.setCurrentText(arch)

        self._epochs_sp.setValue(int(hp.get("epochs", 100)))
        self._batch_sp.setValue(int(hp.get("batch_size", 4)))
        self._lr_sp.setValue(float(hp.get("learning_rate", 0.001)))
        self._width_sp.setValue(int(hp.get("image_width", 320)))
        self._height_sp.setValue(int(hp.get("image_height", 240)))
        self._in_channels_sp.setValue(int(hp.get("in_channels", 3)))
        self._out_classes_sp.setValue(int(hp.get("out_classes", 2)))
        self._patience_sp.setValue(int(hp.get("early_stopping_patience", 15)))
        _enabled_metrics = set(hp.get("extra_metrics", ["precision", "recall", "f1"]))
        for key, cb in self._metric_checkboxes.items():
            cb.setChecked(key in _enabled_metrics)
        dev = hp.get("device", "cpu")
        if self._device_cb.findText(dev) >= 0:
            self._device_cb.setCurrentText(dev)

        if manifest.get("project_name"):
            self._project_name_cb.setCurrentText(manifest["project_name"])
        if manifest.get("project_id"):
            self._project_id_cb.setCurrentText(manifest["project_id"])
        if manifest.get("variant"):
            self._variant_cb.setCurrentText(manifest["variant"])
        if manifest.get("camera"):
            self._camera_cb.setCurrentText(manifest["camera"])

        version_id = manifest.get("version_id", "")
        self._commit_edit.setText(f"Clone of {version_id}")

        vf = manifest.get("version_folder", "")
        if vf:
            from mlops.registry.manifest import ManifestReader
            ds_path = ManifestReader(vf).get_dataset_path()
            if ds_path and os.path.isdir(ds_path):
                self._load_dataset_info(ds_path)

        self._autosave()
