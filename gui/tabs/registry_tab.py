"""
gui/tabs/registry_tab.py

Registry Browser tab — cascading filter + compare mode.
"""
import os
import sys

from mlops.registry.csv_exporter import refresh_registry_csv

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QComboBox, QScrollArea, QFrame,
    QSpacerItem, QSizePolicy, QApplication, QMessageBox,
    QStackedWidget, QGridLayout, QProgressBar,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from mlops.registry import RegistrySettings, RegistryScanner
import logging
import traceback

_SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mlops", "scripts")
)

# ── Palette ───────────────────────────────────────────────────────────────────
_BG        = "#1e1e1e"
_PANEL_BG  = "#252525"
_CARD_BG   = "#2d2d2d"
_BORDER    = "#3a3a3a"
_TEXT      = "#e0e0e0"
_MUTED     = "#888888"
_ACCENT    = "#4fc3f7"
_ACCENT_B  = "#ffa040"
_CARD_SEL_A = "#1a3a5c"
_CARD_SEL_B = "#3a2000"
_DIFF_COLOR   = "#ffc107"
_BETTER_COLOR = "#4caf50"
_WORSE_COLOR  = "#f87171"

_STATUS_COLORS = {
    "completed":  ("#1b5e20", "#4caf50"),
    "training":   ("#1a3a00", "#8bc34a"),
    "crashed":    ("#7f0000", "#ef9a9a"),
    "cancelled":  ("#333333", "#9e9e9e"),
    "partial":    ("#2a1a00", "#ffa040"),
    "syncing":    ("#0d2a3a", "#4fc3f7"),
    "unknown":    ("#333333", "#9e9e9e"),
}

_COMBO_STYLE = f"""
    QComboBox {{
        background-color: {_CARD_BG}; border: 1px solid {_BORDER};
        border-radius: 4px; color: {_TEXT}; padding: 4px;
    }}
    QComboBox::drop-down {{ border: none; }}
    QComboBox QAbstractItemView {{
        background-color: {_CARD_BG}; color: {_TEXT};
        selection-background-color: {_BORDER};
    }}
"""

_BTN_STYLE = f"""
    QPushButton {{
        background-color: {_PANEL_BG}; color: {_TEXT};
        border: 1px solid {_BORDER}; border-radius: 4px;
        padding: 8px; font-size: 11px;
    }}
    QPushButton:hover {{ border-color: {_ACCENT}; }}
    QPushButton:disabled {{ color: {_MUTED}; }}
"""

_TOGGLE_ON  = (f"background-color:{_ACCENT};color:#000;border-radius:4px;"
               f"padding:4px 14px;font-weight:bold;border:none;")
_TOGGLE_OFF = (f"background-color:{_CARD_BG};color:{_TEXT};border-radius:4px;"
               f"padding:4px 14px;border:1px solid {_BORDER};")
_CMP_ON     = (f"background-color:{_ACCENT_B};color:#000;border-radius:4px;"
               f"padding:3px 8px;font-weight:bold;border:none;font-size:11px;")
_CMP_OFF    = (f"background-color:{_CARD_BG};color:{_TEXT};border-radius:4px;"
               f"padding:3px 8px;border:1px solid {_BORDER};font-size:11px;")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _make_status_badge(status: str) -> QLabel:
    lbl = QLabel(status.upper())
    bg, fg = _STATUS_COLORS.get(status.lower(), _STATUS_COLORS["unknown"])
    lbl.setStyleSheet(
        f"background-color:{bg};color:{fg};border-radius:4px;"
        f"padding:2px 8px;font-weight:bold;font-size:10px;"
    )
    return lbl

def _fmt_bytes(n: int) -> str:
    if n >= 1e9: return f"{n/1e9:.1f} GB"
    if n >= 1e6: return f"{n/1e6:.1f} MB"
    if n >= 1e3: return f"{n/1e3:.1f} KB"
    return f"{n} B"

def _fmt_loss(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

def _fmt_date(s: str) -> str:
    try:
        if not s or "T" not in s: return "—"
        dp, tp = s.split("T")
        y, m, d = dp.split("-")
        months = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
        return f"{months[int(m)-1]} {int(d)}, {y}  {tp[:5]}"
    except Exception:
        return "—"

def _section_header(title: str) -> QWidget:
    w = QWidget()
    l = QVBoxLayout(w)
    l.setContentsMargins(0, 10, 0, 0)
    l.setSpacing(2)
    lbl = QLabel(title.upper())
    lbl.setStyleSheet(f"font-weight:bold;font-size:11px;color:{_ACCENT};")
    l.addWidget(lbl)
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"background-color:{_BORDER};")
    l.addWidget(line)
    return w

def _filter_row(label: str, combo: QComboBox) -> QHBoxLayout:
    row = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setFixedWidth(52)
    lbl.setStyleSheet(f"color:{_MUTED};font-size:11px;")
    row.addWidget(lbl)
    row.addWidget(combo, 1)
    return row


# ── VersionCard ────────────────────────────────────────────────────────────────
class VersionCard(QFrame):
    def __init__(self, summary: dict, on_click, parent=None):
        super().__init__(parent)
        self.summary  = summary
        self._on_click = on_click
        self.setCursor(Qt.PointingHandCursor)
        self.set_selected(None)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        r1 = QHBoxLayout(); r1.setContentsMargins(0, 0, 0, 0)
        vid = QLabel(summary.get("version_id", ""))
        vid.setStyleSheet("font-weight:bold;font-size:11px;")
        r1.addWidget(vid); r1.addStretch()
        r1.addWidget(_make_status_badge(summary.get("status", "unknown")))
        layout.addLayout(r1)

        arch = summary.get("architecture", "")
        if summary.get("model_type", "") == "unet":
            enc = summary.get("encoder", "")
            arch_str = f"{arch} ({enc})" if enc else arch
        else:
            arch_str = arch
        r2 = QLabel(arch_str)
        r2.setStyleSheet(f"color:{_MUTED};font-size:10px;")
        layout.addWidget(r2)

        r3 = QLabel(f"Best IoU: {_fmt_loss(summary.get('best_val_iou'))}  "
                    f"{_fmt_date(summary.get('started_at', ''))}")
        r3.setStyleSheet(f"color:{_MUTED};font-size:10px;")
        layout.addWidget(r3)

    def set_selected(self, slot):
        if slot == 'a':
            self.setStyleSheet(
                f"VersionCard{{background-color:{_CARD_SEL_A};"
                f"border:2px solid {_ACCENT};border-radius:6px;margin:2px;}}"
            )
        elif slot == 'b':
            self.setStyleSheet(
                f"VersionCard{{background-color:{_CARD_SEL_B};"
                f"border:2px solid {_ACCENT_B};border-radius:6px;margin:2px;}}"
            )
        else:
            self.setStyleSheet(f"""
                VersionCard{{background-color:{_CARD_BG};
                    border:1px solid {_BORDER};border-radius:6px;margin:2px;}}
                VersionCard:hover{{border:1px solid #555555;}}
            """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._on_click(self.summary)
        super().mousePressEvent(event)


# ── ComparePanel ───────────────────────────────────────────────────────────────
class ComparePanel(QScrollArea):
    """Side-by-side comparison of two version manifests with diff highlighting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self._container = QWidget()
        self._root_layout = QVBoxLayout(self._container)
        self._root_layout.setContentsMargins(16, 16, 16, 16)
        self._root_layout.setSpacing(0)
        self.setWidget(self._container)

    def show_comparison(self, manifest_a: dict, manifest_b: dict):
        while self._root_layout.count():
            item = self._root_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self._row = 0

        def _add_headers():
            for col, manifest, bg, accent in (
                (0, manifest_a, _CARD_SEL_A, _ACCENT),
                (1, manifest_b, _CARD_SEL_B, _ACCENT_B),
            ):
                slot = "A" if col == 0 else "B"
                h = QWidget()
                h.setStyleSheet(f"background:{bg};border-radius:6px;")
                hl = QHBoxLayout(h)
                hl.setContentsMargins(8, 6, 8, 6)
                s = QLabel(slot)
                s.setStyleSheet(f"color:{accent};font-weight:bold;font-size:13px;background:transparent;")
                v = QLabel(manifest.get("version_id", "—"))
                v.setStyleSheet("font-weight:bold;font-size:12px;background:transparent;")
                hl.addWidget(s); hl.addWidget(v); hl.addStretch()
                hl.addWidget(_make_status_badge(manifest.get("status", "unknown")))
                grid.addWidget(h, self._row, col)
            self._row += 1

        def _add_section(title: str):
            for col in range(2):
                grid.addWidget(_section_header(title), self._row, col)
            self._row += 1

        def _add_row(val_a: str, val_b: str,
                     num_a=None, num_b=None, lower_is_better: bool = True):
            lbl_a = QLabel(val_a); lbl_a.setWordWrap(True)
            lbl_b = QLabel(val_b); lbl_b.setWordWrap(True)
            base = "font-size:11px;padding:2px 0px;"
            if num_a is not None and num_b is not None:
                try:
                    fa, fb = float(num_a), float(num_b)
                    if lower_is_better:
                        a_wins = fa < fb
                    else:
                        a_wins = fa > fb
                    if abs(fa - fb) < 1e-9:
                        lbl_a.setStyleSheet(f"color:{_TEXT};{base}")
                        lbl_b.setStyleSheet(f"color:{_TEXT};{base}")
                    elif a_wins:
                        lbl_a.setStyleSheet(f"color:{_BETTER_COLOR};font-weight:bold;{base}")
                        lbl_b.setStyleSheet(f"color:{_WORSE_COLOR};{base}")
                    else:
                        lbl_a.setStyleSheet(f"color:{_WORSE_COLOR};{base}")
                        lbl_b.setStyleSheet(f"color:{_BETTER_COLOR};font-weight:bold;{base}")
                except (TypeError, ValueError):
                    lbl_a.setStyleSheet(f"color:{_TEXT};{base}")
                    lbl_b.setStyleSheet(f"color:{_TEXT};{base}")
            elif val_a == val_b:
                lbl_a.setStyleSheet(f"color:{_TEXT};{base}")
                lbl_b.setStyleSheet(f"color:{_TEXT};{base}")
            else:
                lbl_a.setStyleSheet(f"color:{_DIFF_COLOR};{base}")
                lbl_b.setStyleSheet(f"color:{_DIFF_COLOR};{base}")
            grid.addWidget(lbl_a, self._row, 0)
            grid.addWidget(lbl_b, self._row, 1)
            self._row += 1

        _add_headers()

        # Commit messages
        cm_a = manifest_a.get("commit_message", "")
        cm_b = manifest_b.get("commit_message", "")
        if cm_a or cm_b:
            _add_row(cm_a or "—", cm_b or "—")

        # ── Dataset ──
        _add_section("Dataset")
        ds_a = manifest_a.get("dataset", {})
        ds_b = manifest_b.get("dataset", {})
        _add_row(
            f"Train: {ds_a.get('train_count','—')}  Val: {ds_a.get('val_count','—')}  Test: {ds_a.get('test_count','—')}",
            f"Train: {ds_b.get('train_count','—')}  Val: {ds_b.get('val_count','—')}  Test: {ds_b.get('test_count','—')}",
        )
        sha_a = ds_a.get("sha256", "")
        sha_b = ds_b.get("sha256", "")
        _add_row(
            f"SHA: {sha_a[:14]}..." if sha_a else "SHA: —",
            f"SHA: {sha_b[:14]}..." if sha_b else "SHA: —",
        )

        # ── Hyperparameters ──
        _add_section("Hyperparameters")
        hp_a = manifest_a.get("hyperparams", {})
        hp_b = manifest_b.get("hyperparams", {})
        _add_row(
            f"Arch: {hp_a.get('architecture','—')}   Encoder: {hp_a.get('encoder','—')}",
            f"Arch: {hp_b.get('architecture','—')}   Encoder: {hp_b.get('encoder','—')}",
        )
        _add_row(
            f"Epochs: {hp_a.get('epochs','—')}   Batch: {hp_a.get('batch_size','—')}   LR: {hp_a.get('learning_rate','—')}",
            f"Epochs: {hp_b.get('epochs','—')}   Batch: {hp_b.get('batch_size','—')}   LR: {hp_b.get('learning_rate','—')}",
        )
        _add_row(
            f"Size: {hp_a.get('image_width','—')}×{hp_a.get('image_height','—')}   Ch: {hp_a.get('in_channels','—')}   Classes: {hp_a.get('out_classes','—')}",
            f"Size: {hp_b.get('image_width','—')}×{hp_b.get('image_height','—')}   Ch: {hp_b.get('in_channels','—')}   Classes: {hp_b.get('out_classes','—')}",
        )
        _add_row(
            f"Device: {hp_a.get('device','—')}   Enc weights: {hp_a.get('encoder_weights','—')}",
            f"Device: {hp_b.get('device','—')}   Enc weights: {hp_b.get('encoder_weights','—')}",
        )

        # ── Metrics ──
        _add_section("Metrics")
        mx_a = manifest_a.get("metrics", {})
        mx_b = manifest_b.get("metrics", {})
        bvl_a  = mx_a.get("best_val_loss");     bvl_b  = mx_b.get("best_val_loss")
        bep_a  = mx_a.get("best_epoch", "—");   bep_b  = mx_b.get("best_epoch", "—")
        bvi_a  = mx_a.get("best_val_iou");      bvi_b  = mx_b.get("best_val_iou")
        ftl_a  = mx_a.get("final_train_loss");  ftl_b  = mx_b.get("final_train_loss")
        fvl_a  = mx_a.get("final_val_loss");    fvl_b  = mx_b.get("final_val_loss")
        fti_a  = mx_a.get("final_train_iou");   fti_b  = mx_b.get("final_train_iou")
        fvi_a  = mx_a.get("final_val_iou");     fvi_b  = mx_b.get("final_val_iou")
        _add_row(
            f"Best Val Loss: {_fmt_loss(bvl_a)}   @ Epoch {bep_a}",
            f"Best Val Loss: {_fmt_loss(bvl_b)}   @ Epoch {bep_b}",
            num_a=bvl_a, num_b=bvl_b, lower_is_better=True,
        )
        if bvi_a is not None or bvi_b is not None:
            _add_row(
                f"Best Val IoU: {_fmt_loss(bvi_a)}",
                f"Best Val IoU: {_fmt_loss(bvi_b)}",
                num_a=bvi_a, num_b=bvi_b, lower_is_better=False,
            )
        _add_row(
            f"Final Train Loss: {_fmt_loss(ftl_a)}",
            f"Final Train Loss: {_fmt_loss(ftl_b)}",
            num_a=ftl_a, num_b=ftl_b, lower_is_better=True,
        )
        if fvl_a is not None or fvl_b is not None:
            _add_row(
                f"Final Val Loss: {_fmt_loss(fvl_a)}",
                f"Final Val Loss: {_fmt_loss(fvl_b)}",
                num_a=fvl_a, num_b=fvl_b, lower_is_better=True,
            )
        if fti_a is not None or fti_b is not None:
            _add_row(
                f"Final Train IoU: {_fmt_loss(fti_a)}",
                f"Final Train IoU: {_fmt_loss(fti_b)}",
                num_a=fti_a, num_b=fti_b, lower_is_better=False,
            )
        if fvi_a is not None or fvi_b is not None:
            _add_row(
                f"Final Val IoU: {_fmt_loss(fvi_a)}",
                f"Final Val IoU: {_fmt_loss(fvi_b)}",
                num_a=fvi_a, num_b=fvi_b, lower_is_better=False,
            )

        # ── Artifacts ──
        _add_section("Artifacts")
        art_a = manifest_a.get("artifacts", {})
        art_b = manifest_b.get("artifacts", {})
        ox_a  = manifest_a.get("onnx_config", {})
        ox_b  = manifest_b.get("onnx_config", {})
        ts_a  = manifest_a.get("torchscript_config", {})
        ts_b  = manifest_b.get("torchscript_config", {})
        _add_row(
            f"Weights: {'best.pt ✓' if art_a.get('weights') else '—'}",
            f"Weights: {'best.pt ✓' if art_b.get('weights') else '—'}",
        )
        _add_row(
            f"ONNX: {'exported ✓' if ox_a.get('exported') else 'not exported'}",
            f"ONNX: {'exported ✓' if ox_b.get('exported') else 'not exported'}",
        )
        if manifest_a.get("model_type") == "yolo" or manifest_b.get("model_type") == "yolo":
            _add_row(
                f"TorchScript: {'exported ✓' if ts_a.get('exported') else 'not exported'}",
                f"TorchScript: {'exported ✓' if ts_b.get('exported') else 'not exported'}",
            )

        grid.setRowStretch(self._row, 1)
        wrapper = QWidget()
        wrapper.setLayout(grid)
        self._root_layout.addWidget(wrapper)
        self._root_layout.addStretch()


# ── RegistryTab ────────────────────────────────────────────────────────────────
class RegistryTab(QWidget):
    retrain_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._registry_root    = RegistrySettings().get_effective_root()
        self._current_summary  = None
        self._cards            = []

        # Compare state
        self._compare_mode   = False
        self._compare_a      = None   # summary dict
        self._compare_b      = None   # summary dict
        self._last_slot      = 'b'    # next click replaces this slot

        # Cascading filter state
        self._active_model_type = ""

        self._setup_ui()
        self._refresh()
        self.refresh_csv()

    # ── UI Construction ────────────────────────────────────────────────────────
    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(f"QSplitter::handle{{background:{_BORDER};}}")
        root.addWidget(splitter)

        # ── LEFT PANEL ──────────────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(270)
        left.setMaximumWidth(320)
        left.setStyleSheet(f"background-color:{_BG};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.setSpacing(8)

        # Header
        h_row = QHBoxLayout()
        h_lbl = QLabel("Registry Browser")
        h_lbl.setStyleSheet("font-weight:bold;font-size:14px;")
        h_row.addWidget(h_lbl)
        h_row.addStretch()

        self._compare_btn = QPushButton("Compare")
        self._compare_btn.setStyleSheet(_CMP_OFF)
        self._compare_btn.setCheckable(True)
        self._compare_btn.clicked.connect(self._toggle_compare)
        h_row.addWidget(self._compare_btn)

        self._refresh_btn = QPushButton("🔄")
        self._refresh_btn.setFixedSize(28, 28)
        self._refresh_btn.setFlat(True)
        self._refresh_btn.setStyleSheet(
            f"QPushButton:hover{{background-color:{_PANEL_BG};border-radius:4px;}}"
        )
        self._refresh_btn.clicked.connect(self._refresh)
        h_row.addWidget(self._refresh_btn)
        ll.addLayout(h_row)

        # Model type toggles
        mt_row = QHBoxLayout()
        mt_row.setSpacing(6)
        self._btn_unet = QPushButton("UNet")
        self._btn_yolo = QPushButton("YOLO")
        self._btn_unet.setStyleSheet(_TOGGLE_OFF)
        self._btn_yolo.setStyleSheet(_TOGGLE_OFF)
        self._btn_unet.clicked.connect(lambda: self._on_model_type_btn("unet"))
        self._btn_yolo.clicked.connect(lambda: self._on_model_type_btn("yolo"))
        mt_row.addWidget(self._btn_unet)
        mt_row.addWidget(self._btn_yolo)
        mt_row.addStretch()
        ll.addLayout(mt_row)

        # Cascading dropdowns
        self._project_cb = QComboBox(); self._project_cb.setStyleSheet(_COMBO_STYLE)
        self._id_cb      = QComboBox(); self._id_cb.setStyleSheet(_COMBO_STYLE)
        self._variant_cb = QComboBox(); self._variant_cb.setStyleSheet(_COMBO_STYLE)
        self._camera_cb  = QComboBox(); self._camera_cb.setStyleSheet(_COMBO_STYLE)

        self._project_row = _filter_row("Project", self._project_cb)
        self._id_row      = _filter_row("ID",      self._id_cb)
        self._variant_row = _filter_row("Variant", self._variant_cb)
        self._camera_row  = _filter_row("Camera",  self._camera_cb)

        ll.addLayout(self._project_row)
        ll.addLayout(self._id_row)
        ll.addLayout(self._variant_row)
        ll.addLayout(self._camera_row)

        self._project_cb.currentTextChanged.connect(self._on_project_changed)
        self._id_cb.currentTextChanged.connect(self._on_id_changed)
        self._variant_cb.currentTextChanged.connect(self._on_variant_changed)
        self._camera_cb.currentTextChanged.connect(self._on_camera_changed)

        # Warnings
        self._warning_lbl = QLabel("⚠ Registry not configured.\nSet a Google Drive or Local root in Settings.")
        self._warning_lbl.setAlignment(Qt.AlignCenter)
        self._warning_lbl.setStyleSheet(f"color:{_MUTED};")
        self._warning_lbl.hide()
        ll.addWidget(self._warning_lbl)

        self._conflict_lbl = QLabel()
        self._conflict_lbl.setStyleSheet(
            f"color:#ffc107;font-size:10px;background:#2a2000;"
            f"border:1px solid #ffc107;border-radius:4px;padding:4px;"
        )
        self._conflict_lbl.setWordWrap(True)
        self._conflict_lbl.hide()
        ll.addWidget(self._conflict_lbl)

        # Compare hint label
        self._compare_hint = QLabel("Click a card for  [A],  click another for  [B]")
        self._compare_hint.setStyleSheet(f"color:{_MUTED};font-size:10px;font-style:italic;")
        self._compare_hint.setWordWrap(True)
        self._compare_hint.hide()
        ll.addWidget(self._compare_hint)

        # Cards scroll
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet(f"QScrollArea{{background:transparent;border:none;}}")
        cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(cards_widget)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(2)
        self._cards_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        self._scroll.setWidget(cards_widget)
        ll.addWidget(self._scroll, 1)

        self._storage_lbl = QLabel("")
        self._storage_lbl.setStyleSheet(f"color:{_MUTED};font-size:10px;")
        ll.addWidget(self._storage_lbl)

        self._csv_btn = QPushButton("📊  Refresh Models CSV")
        self._csv_btn.setStyleSheet(f"""
            QPushButton{{background-color:{_PANEL_BG};color:{_TEXT};
                border:1px solid {_BORDER};border-radius:4px;padding:6px;font-size:11px;}}
            QPushButton:hover{{border-color:{_ACCENT};}}
            QPushButton:disabled{{color:{_MUTED};}}
        """)
        self._csv_btn.clicked.connect(self._export_csv)
        ll.addWidget(self._csv_btn)

        splitter.addWidget(left)

        # ── RIGHT PANEL ─────────────────────────────────────────────────────────
        right = QWidget()
        right.setStyleSheet(f"background-color:{_BG};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self._right_stack = QStackedWidget()
        rl.addWidget(self._right_stack)

        # Page 0 — empty
        self._empty_lbl = QLabel("← Select a version to view details")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet(f"color:{_MUTED};font-size:13px;")
        self._right_stack.addWidget(self._empty_lbl)

        # Page 1 — single detail view
        self._detail_scroll = QScrollArea()
        self._detail_scroll.setWidgetResizable(True)
        self._detail_scroll.setFrameShape(QFrame.NoFrame)
        self._detail_widget = QWidget()
        dl = QVBoxLayout(self._detail_widget)
        dl.setContentsMargins(20, 20, 20, 20)
        dl.setSpacing(6)
        self._build_detail_panel(dl)
        self._detail_scroll.setWidget(self._detail_widget)
        self._right_stack.addWidget(self._detail_scroll)

        # Page 2 — compare view
        self._compare_panel = ComparePanel()
        self._right_stack.addWidget(self._compare_panel)

        self._right_stack.setCurrentIndex(0)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _build_detail_panel(self, dl: QVBoxLayout):
        dh_row = QHBoxLayout()
        self._d_vid = QLabel()
        self._d_vid.setStyleSheet("font-weight:bold;font-size:14px;")
        dh_row.addWidget(self._d_vid); dh_row.addStretch()
        self._d_status_layout = QVBoxLayout()
        dh_row.addLayout(self._d_status_layout)
        dl.addLayout(dh_row)

        self._d_commit = QLabel()
        self._d_commit.setStyleSheet(f"color:{_MUTED};font-style:italic;font-size:11px;")
        self._d_commit.setWordWrap(True)
        dl.addWidget(self._d_commit)

        dl.addWidget(_section_header("Dataset"))
        self._d_ds_counts  = QLabel(); dl.addWidget(self._d_ds_counts)
        self._d_ds_classes = QLabel(); dl.addWidget(self._d_ds_classes)
        self._d_ds_sha     = QLabel(); dl.addWidget(self._d_ds_sha)
        self._d_ds_path    = QLabel()
        self._d_ds_path.setStyleSheet(f"color:{_MUTED};font-size:10px;")
        self._d_ds_path.setWordWrap(True)
        dl.addWidget(self._d_ds_path)
        self._d_sha_warn = QLabel()
        self._d_sha_warn.setStyleSheet("color:#ffc107;font-size:10px;background:transparent;")
        self._d_sha_warn.setWordWrap(True)
        dl.addWidget(self._d_sha_warn)

        dl.addWidget(_section_header("Hyperparameters"))
        self._d_hp_l1 = QLabel(); dl.addWidget(self._d_hp_l1)
        self._d_hp_l2 = QLabel(); dl.addWidget(self._d_hp_l2)
        self._d_hp_l3 = QLabel(); dl.addWidget(self._d_hp_l3)
        self._d_hp_l4 = QLabel(); dl.addWidget(self._d_hp_l4)

        dl.addWidget(_section_header("Metrics"))
        self._d_mx_best  = QLabel(); dl.addWidget(self._d_mx_best)
        self._d_mx_final = QLabel(); dl.addWidget(self._d_mx_final)

        dl.addWidget(_section_header("Artifacts"))
        self._d_art_weights     = QLabel(); dl.addWidget(self._d_art_weights)
        self._d_art_onnx        = QLabel(); dl.addWidget(self._d_art_onnx)
        self._d_art_torchscript = QLabel(); dl.addWidget(self._d_art_torchscript)

        self._crash_header    = _section_header("Crash Info")
        self._d_crash_reason  = QLabel(); self._d_crash_reason.setWordWrap(True)
        self._d_crash_hint    = QLabel(); self._d_crash_hint.setWordWrap(True)
        dl.addWidget(self._crash_header)
        dl.addWidget(self._d_crash_reason)
        dl.addWidget(self._d_crash_hint)

        dl.addStretch()

        btn_layout = QHBoxLayout()
        self._open_btn    = QPushButton("📂  Open Folder");       self._open_btn.setStyleSheet(_BTN_STYLE)
        self._open_ds_btn = QPushButton("🗄  Open Dataset Folder"); self._open_ds_btn.setStyleSheet(_BTN_STYLE)
        self._copy_btn    = QPushButton("📋  Copy Weights Path");  self._copy_btn.setStyleSheet(_BTN_STYLE)
        self._export_btn  = QPushButton("⚙  Export Model");       self._export_btn.setStyleSheet(_BTN_STYLE)
        self._retrain_btn = QPushButton("🔁  Clone & Re-train");   self._retrain_btn.setStyleSheet(_BTN_STYLE)
        self._copy_status_lbl = QLabel("")
        self._copy_status_lbl.setStyleSheet(f"color:#4caf50;font-size:10px;")

        self._export_btn.setEnabled(False)
        self._retrain_btn.setEnabled(False)

        self._open_btn.clicked.connect(self._open_folder)
        self._open_ds_btn.clicked.connect(self._open_dataset_folder)
        self._copy_btn.clicked.connect(self._copy_weights_path)
        self._export_btn.clicked.connect(self._on_export_onnx)
        self._retrain_btn.clicked.connect(self._on_retrain_clicked)

        for w in (self._open_btn, self._open_ds_btn, self._copy_btn,
                  self._export_btn, self._retrain_btn, self._copy_status_lbl):
            btn_layout.addWidget(w)
        btn_layout.addStretch()
        dl.addLayout(btn_layout)

        self._export_progress = QProgressBar()
        self._export_progress.setRange(0, 100)
        self._export_progress.setFixedHeight(6)
        self._export_progress.setTextVisible(False)
        self._export_progress.hide()
        self._export_progress.setStyleSheet("""
            QProgressBar{background:#333;border-radius:3px;border:none;}
            QProgressBar::chunk{background:#4fc3f7;border-radius:3px;}
        """)
        dl.addWidget(self._export_progress)
        dl.addStretch()

    # ── Refresh & Cascading Filter ─────────────────────────────────────────────
    def _refresh(self):
        try:
            self._registry_root = RegistrySettings().get_effective_root()
            if not self._registry_root or not os.path.isdir(self._registry_root):
                self._warning_lbl.show()
                return
            self._warning_lbl.hide()
            self.refresh_csv()

            # Reset model type if current type no longer exists
            scanner = RegistryScanner(self._registry_root)
            available = scanner.get_model_types()
            if self._active_model_type and self._active_model_type not in available:
                self._active_model_type = ""

            # Update toggle button states to match available types and active selection
            self._btn_unet.setStyleSheet(
                _TOGGLE_ON if self._active_model_type == "unet" else _TOGGLE_OFF
            )
            self._btn_yolo.setStyleSheet(
                _TOGGLE_ON if self._active_model_type == "yolo" else _TOGGLE_OFF
            )

            if self._active_model_type:
                self._repopulate_projects(self._active_model_type)
            else:
                self._clear_all_dropdowns()
                self._clear_cards()
                self._show_empty()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_model_type_btn(self, mt: str):
        if self._active_model_type == mt:
            # Deselect
            self._active_model_type = ""
            self._btn_unet.setStyleSheet(_TOGGLE_OFF)
            self._btn_yolo.setStyleSheet(_TOGGLE_OFF)
            self._clear_all_dropdowns()
            self._clear_cards()
            self._show_empty()
        else:
            self._active_model_type = mt
            self._btn_unet.setStyleSheet(_TOGGLE_ON if mt == "unet" else _TOGGLE_OFF)
            self._btn_yolo.setStyleSheet(_TOGGLE_ON if mt == "yolo" else _TOGGLE_OFF)
            self._repopulate_projects(mt)

    def _repopulate_projects(self, mt: str):
        scanner = RegistryScanner(self._registry_root)
        names   = scanner.get_project_names(mt)
        self._project_cb.blockSignals(True)
        self._project_cb.clear()
        self._project_cb.addItem("— select —")
        for n in names:
            self._project_cb.addItem(n)
        self._project_cb.blockSignals(False)
        self._clear_dropdowns_from_id()
        self._clear_cards()
        self._show_empty()

    def _on_project_changed(self, pn: str):
        try:
            self._clear_dropdowns_from_id()
            self._clear_cards()
            self._show_empty()
            if not pn or pn == "— select —":
                return
            scanner = RegistryScanner(self._registry_root)
            pids = scanner.get_project_ids(self._active_model_type, pn)
            self._id_cb.blockSignals(True)
            self._id_cb.clear()
            if len(pids) == 1:
                self._id_cb.addItem(pids[0])
                self._id_cb.blockSignals(False)
                self._on_id_changed(pids[0])
            else:
                self._id_cb.addItem("— select —")
                for pid in pids:
                    self._id_cb.addItem(pid)
                self._id_cb.blockSignals(False)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_id_changed(self, pid: str):
        try:
            self._clear_dropdowns_from_variant()
            self._clear_cards()
            self._show_empty()
            if not pid or pid == "— select —":
                return
            scanner  = RegistryScanner(self._registry_root)
            pn       = self._project_cb.currentText()
            variants = scanner.get_variants(self._active_model_type, pn, pid)
            self._variant_cb.blockSignals(True)
            self._variant_cb.clear()
            if len(variants) == 1:
                self._variant_cb.addItem(variants[0])
                self._variant_cb.blockSignals(False)
                self._on_variant_changed(variants[0])
            else:
                self._variant_cb.addItem("— select —")
                for v in variants:
                    self._variant_cb.addItem(v)
                self._variant_cb.blockSignals(False)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_variant_changed(self, variant: str):
        try:
            self._clear_dropdowns_from_camera()
            self._clear_cards()
            self._show_empty()
            if not variant or variant == "— select —":
                return
            scanner = RegistryScanner(self._registry_root)
            pn  = self._project_cb.currentText()
            pid = self._id_cb.currentText()
            cameras = scanner.get_cameras(self._active_model_type, pn, pid, variant)
            self._camera_cb.blockSignals(True)
            self._camera_cb.clear()
            if len(cameras) == 1:
                self._camera_cb.addItem(cameras[0])
                self._camera_cb.blockSignals(False)
                self._on_camera_changed(cameras[0])
            else:
                self._camera_cb.addItem("— select —")
                for c in cameras:
                    self._camera_cb.addItem(c)
                self._camera_cb.blockSignals(False)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_camera_changed(self, camera: str):
        try:
            self._clear_cards()
            self._show_empty()
            if not camera or camera == "— select —":
                return
            pn      = self._project_cb.currentText()
            pid     = self._id_cb.currentText()
            variant = self._variant_cb.currentText()
            if any(x == "— select —" or not x for x in (pn, pid, variant)):
                return
            project = f"{self._active_model_type}/{pn}/{pid}/{variant}/{camera}"
            scanner  = RegistryScanner(self._registry_root)
            versions = scanner.get_versions(project)
            for summary in versions:
                card = VersionCard(summary, self._on_card_clicked)
                self._cards.append(card)
                self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
            storage = scanner.get_storage_stats(project)
            self._storage_lbl.setText(
                f"Storage: {_fmt_bytes(storage.get('total_bytes', 0))}"
            )
            conflicts = scanner.detect_conflict_files(project)
            if conflicts:
                names = [os.path.basename(c) for c in conflicts[:3]]
                extra = f" (+{len(conflicts)-3} more)" if len(conflicts) > 3 else ""
                self._conflict_lbl.setText(
                    f"⚠ {len(conflicts)} GDrive conflict file(s) detected:\n"
                    + "\n".join(names) + extra
                    + "\nDelete them manually in Explorer."
                )
                self._conflict_lbl.show()
            else:
                self._conflict_lbl.hide()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _current_project_key(self) -> str:
        pn      = self._project_cb.currentText()
        pid     = self._id_cb.currentText()
        variant = self._variant_cb.currentText()
        camera  = self._camera_cb.currentText()
        if any(not x or x == "— select —" for x in (pn, pid, variant, camera)):
            return ""
        return f"{self._active_model_type}/{pn}/{pid}/{variant}/{camera}"

    # ── Clear helpers ──────────────────────────────────────────────────────────
    def _clear_all_dropdowns(self):
        for cb in (self._project_cb, self._id_cb, self._variant_cb, self._camera_cb):
            cb.blockSignals(True); cb.clear(); cb.blockSignals(False)

    def _clear_dropdowns_from_id(self):
        for cb in (self._id_cb, self._variant_cb, self._camera_cb):
            cb.blockSignals(True); cb.clear(); cb.blockSignals(False)

    def _clear_dropdowns_from_variant(self):
        for cb in (self._variant_cb, self._camera_cb):
            cb.blockSignals(True); cb.clear(); cb.blockSignals(False)

    def _clear_dropdowns_from_camera(self):
        self._camera_cb.blockSignals(True)
        self._camera_cb.clear()
        self._camera_cb.blockSignals(False)

    def _clear_cards(self):
        while self._cards_layout.count() > 1:
            item = self._cards_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._cards.clear()
        self._storage_lbl.setText("")
        self._conflict_lbl.hide()

    # ── Compare mode ───────────────────────────────────────────────────────────
    def _toggle_compare(self):
        try:
            self._compare_mode = self._compare_btn.isChecked()
            self._compare_btn.setStyleSheet(_CMP_ON if self._compare_mode else _CMP_OFF)
            self._compare_hint.setVisible(self._compare_mode)
            if not self._compare_mode:
                self._compare_a = None
                self._compare_b = None
                self._last_slot = 'b'
                for card in self._cards:
                    card.set_selected(None)
                self._show_empty()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_card_clicked(self, summary: dict):
        if self._compare_mode:
            self._handle_compare_click(summary)
        else:
            self._current_summary = summary
            for card in self._cards:
                card.set_selected(
                    None if card.summary.get("version_id") != summary.get("version_id")
                    else 'a'
                )
            proj = self._current_project_key()
            if proj:
                scanner  = RegistryScanner(self._registry_root)
                manifest = scanner.get_version(proj, summary.get("version_id", ""))
                self._show_detail(manifest)

    def _handle_compare_click(self, summary: dict):
        vid = summary.get("version_id", "")
        # If already selected as A or B, deselect
        if self._compare_a and self._compare_a.get("version_id") == vid:
            self._compare_a = None
        elif self._compare_b and self._compare_b.get("version_id") == vid:
            self._compare_b = None
        else:
            # Assign to whichever slot was last used (alternate)
            if self._last_slot == 'b':
                self._compare_a = summary
                self._last_slot = 'a'
            else:
                self._compare_b = summary
                self._last_slot = 'b'

        # Update card visuals
        for card in self._cards:
            cvid = card.summary.get("version_id")
            if self._compare_a and cvid == self._compare_a.get("version_id"):
                card.set_selected('a')
            elif self._compare_b and cvid == self._compare_b.get("version_id"):
                card.set_selected('b')
            else:
                card.set_selected(None)

        # Show compare view if both slots filled
        if self._compare_a and self._compare_b:
            proj    = self._current_project_key()
            scanner = RegistryScanner(self._registry_root)
            ma = scanner.get_version(proj, self._compare_a.get("version_id", ""))
            mb = scanner.get_version(proj, self._compare_b.get("version_id", ""))
            if ma and mb:
                self._compare_panel.show_comparison(ma, mb)
                self._right_stack.setCurrentIndex(2)
        else:
            self._show_empty()

    # ── Detail view ────────────────────────────────────────────────────────────
    def _show_empty(self):
        self._right_stack.setCurrentIndex(0)
        self._export_btn.setEnabled(False)
        self._export_btn.setText("⚙  Export Model")
        self._open_ds_btn.setEnabled(False)
        self._retrain_btn.setEnabled(False)

    def _show_detail(self, manifest: dict | None):
        if not manifest:
            self._show_empty()
            return

        self._right_stack.setCurrentIndex(1)

        self._d_vid.setText(manifest.get("version_id", ""))
        while self._d_status_layout.count():
            w = self._d_status_layout.takeAt(0).widget()
            if w: w.deleteLater()
        self._d_status_layout.addWidget(_make_status_badge(manifest.get("status", "unknown")))
        self._d_commit.setText(manifest.get("commit_message", ""))

        ds = manifest.get("dataset", {})
        tr, va, te = ds.get("train_count", 0), ds.get("val_count", 0), ds.get("test_count", 0)
        self._d_ds_counts.setText(f"Train: {tr}  Val: {va}  Test: {te}")
        cls_names = ds.get("classes", [])
        cls_str   = ", ".join(cls_names)
        if len(cls_str) > 60: cls_str = cls_str[:57] + "..."
        self._d_ds_classes.setText(f"Classes: {cls_str} ({ds.get('num_classes', 0)})")
        sha = ds.get("sha256", "")
        self._d_ds_sha.setText(f"SHA-256: {sha[:16]}..." if sha else "SHA-256: —")

        vf = self._current_summary.get("version_folder", "") if self._current_summary else ""
        ds_path = ""
        if vf:
            from mlops.registry.manifest import ManifestReader
            ds_path = ManifestReader(vf).get_dataset_path()
        if ds_path and os.path.isdir(ds_path):
            self._d_ds_path.setText(f"📁 {ds_path}")
            self._open_ds_btn.setEnabled(True)
        else:
            self._d_ds_path.setText("Dataset not found on this machine")
            self._open_ds_btn.setEnabled(False)

        self._d_sha_warn.setText("")
        if sha and ds_path and os.path.isdir(ds_path):
            info_path = os.path.join(ds_path, "dataset_info.json")
            if os.path.isfile(info_path):
                try:
                    import json as _json
                    with open(info_path, "r", encoding="utf-8") as fh:
                        current_sha = _json.load(fh).get("sha256", "")
                    if current_sha and current_sha != sha:
                        self._d_sha_warn.setText(
                            f"⚠ Dataset changed since training.\n"
                            f"Model SHA: {sha[:16]}...  Current: {current_sha[:16]}..."
                        )
                except Exception:
                    pass

        hp = manifest.get("hyperparams", {})
        arch, enc = hp.get("architecture", ""), hp.get("encoder", "")
        if manifest.get("model_type", "") == "unet" and enc:
            self._d_hp_l1.setText(f"Arch: {arch}   Encoder: {enc}")
        else:
            self._d_hp_l1.setText(f"Arch: {arch}")
        self._d_hp_l2.setText(
            f"Epochs: {hp.get('epochs','—')}   Batch: {hp.get('batch_size','—')}   LR: {hp.get('learning_rate','—')}"
        )
        self._d_hp_l3.setText(
            f"Size: {hp.get('image_width','—')}×{hp.get('image_height','—')}   "
            f"Channels: {hp.get('in_channels','—')}   Out classes: {hp.get('out_classes','—')}"
        )
        self._d_hp_l4.setText(
            f"Device: {hp.get('device','—')}   Encoder weights: {hp.get('encoder_weights','—')}"
        )

        mx  = manifest.get("metrics", {})
        bvl = mx.get("best_val_loss")
        bvi = mx.get("best_val_iou")
        bep = mx.get("best_epoch")
        ftl = mx.get("final_train_loss")
        fvi = mx.get("final_val_iou")
        if bvl is None and ftl is None:
            self._d_mx_best.setText("No metrics recorded")
            self._d_mx_best.setStyleSheet(f"color:{_MUTED};")
            self._d_mx_final.setText("")
        else:
            self._d_mx_best.setStyleSheet(f"color:{_TEXT};")
            self._d_mx_best.setText(
                f"Best Val Loss: {_fmt_loss(bvl)}   Best Val IoU: {_fmt_loss(bvi)}   @ Epoch {bep or '—'}"
            )
            self._d_mx_final.setText(
                f"Final Train Loss: {_fmt_loss(ftl)}   Final Val IoU: {_fmt_loss(fvi)}"
            )

        art     = manifest.get("artifacts", {})
        weights = art.get("weights")
        w_tick  = ""
        if weights and self._current_summary:
            wp = os.path.join(self._current_summary.get("version_folder", ""), weights)
            if os.path.isfile(wp): w_tick = "  ✓"
        self._d_art_weights.setText(f"Weights: {weights or '—'}{w_tick}")

        model_type = manifest.get("model_type", "")

        ox = manifest.get("onnx_config", {})
        if ox.get("exported"):
            self._d_art_onnx.setText(
                f"ONNX: exported (opset {ox.get('opset_version','—')}, {_fmt_date(ox.get('exported_at',''))})"
            )
        else:
            self._d_art_onnx.setText("ONNX: not exported")

        ts = manifest.get("torchscript_config", {})
        if model_type == "yolo":
            self._d_art_torchscript.show()
            if ts.get("exported"):
                self._d_art_torchscript.setText(
                    f"TorchScript: exported ({_fmt_date(ts.get('exported_at',''))})"
                )
            else:
                self._d_art_torchscript.setText("TorchScript: not exported")
        else:
            self._d_art_torchscript.hide()

        cinfo = manifest.get("crash_info")
        if manifest.get("status") == "crashed" and cinfo:
            self._crash_header.show()
            self._d_crash_reason.show(); self._d_crash_hint.show()
            self._d_crash_reason.setText(f"Reason: {cinfo.get('reason','')}")
            self._d_crash_hint.setText(f"Hint: {cinfo.get('hint','')}")
        else:
            self._crash_header.hide()
            self._d_crash_reason.hide(); self._d_crash_hint.hide()

        if model_type == "yolo":
            already_exported = ox.get("exported", False) and ts.get("exported", False)
        else:
            already_exported = ox.get("exported", False)

        can_export = (
            manifest.get("status") == "completed"
            and bool(art.get("weights"))
            and not already_exported
        )
        self._export_btn.setEnabled(can_export)
        self._export_btn.setText(
            "✓ Export Ready" if already_exported else "⚙  Export Model"
        )
        self._retrain_btn.setEnabled(
            manifest.get("status") in ("completed", "crashed", "cancelled", "partial")
        )

    # ── Action handlers ────────────────────────────────────────────────────────
    def _on_retrain_clicked(self):
        try:
            if not self._current_summary: return
            vf = self._current_summary.get("version_folder", "")
            if not vf: return
            proj    = self._current_project_key()
            scanner = RegistryScanner(self._registry_root)
            manifest = scanner.get_version(proj, self._current_summary.get("version_id", ""))
            if not manifest: return
            manifest["version_folder"] = vf
            self.retrain_requested.emit(manifest)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _open_folder(self):
        try:
            if not self._current_summary: return
            vf = self._current_summary.get("version_folder", "")
            if sys.platform == "win32" and os.path.isdir(vf):
                try: os.startfile(vf)
                except Exception: pass

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _open_dataset_folder(self):
        try:
            if not self._current_summary: return
            vf = self._current_summary.get("version_folder", "")
            if not vf: return
            from mlops.registry.manifest import ManifestReader
            ds_path = ManifestReader(vf).get_dataset_path()
            if ds_path and os.path.isdir(ds_path) and sys.platform == "win32":
                try: os.startfile(ds_path)
                except Exception: pass

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _copy_weights_path(self):
        try:
            if not self._current_summary: return
            vf = self._current_summary.get("version_folder", "")
            w  = "best.pt"
            try:
                proj = self._current_project_key()
                m    = RegistryScanner(self._registry_root).get_version(
                    proj, self._current_summary.get("version_id", "")
                )
                w = m.get("artifacts", {}).get("weights") or w
            except Exception:
                pass
            QApplication.clipboard().setText(os.path.join(vf, w))
            self._copy_status_lbl.setText("✓ Copied")

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_export_onnx(self):
        try:
            if not self._current_summary: return
            vf = self._current_summary.get("version_folder", "")
            if not vf or not os.path.isdir(vf): return
            from mlops.export import OnnxWorker
            self._export_btn.setEnabled(False)
            self._export_btn.setText("Exporting...")
            self._export_progress.setValue(0)
            self._export_progress.show()
            self._copy_status_lbl.setText("")
            self._onnx_worker = OnnxWorker(
                version_folder=vf, scripts_dir=_SCRIPTS_DIR, parent=self
            )
            self._onnx_worker.log.connect(lambda msg: self._copy_status_lbl.setText(msg[-60:]))
            self._onnx_worker.progress.connect(self._export_progress.setValue)
            self._onnx_worker.finished.connect(self._on_export_finished)
            self._onnx_worker.start()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_export_finished(self, success: bool):
        self._export_progress.hide()
        if success:
            self._copy_status_lbl.setText("✓ Export complete")
            if self._current_summary:
                proj     = self._current_project_key()
                scanner  = RegistryScanner(self._registry_root)
                manifest = scanner.get_version(
                    proj, self._current_summary.get("version_id", "")
                )
                self._show_detail(manifest)
        else:
            self._export_btn.setEnabled(True)
            self._export_btn.setText("⚙  Export Model")
            self._copy_status_lbl.setText("✗ Export failed — see log")

    def _export_csv(self):
        try:
            self.refresh_csv(notify=True)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def refresh_csv(self, notify: bool = False):
        if not self._registry_root or not os.path.isdir(self._registry_root):
            if notify:
                QMessageBox.warning(self, "Registry Not Configured",
                                    "Set a Google Drive or Local root in Settings before exporting.")
            return
        count = refresh_registry_csv(self._registry_root)
        if notify:
            if count == 0:
                QMessageBox.information(self, "No Models",
                                        "No model versions found in the registry.")
            else:
                import os as _os
                csv_path = _os.path.join(self._registry_root, "models_registry.csv")
                QMessageBox.information(self, "CSV Updated",
                                        f"Saved {count} model(s) to:\n{csv_path}")
