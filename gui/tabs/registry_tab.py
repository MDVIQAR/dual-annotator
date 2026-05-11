"""
gui/tabs/registry_tab.py

Phase 6 — Registry Browser tab for DualAnnotator.
Provides a read-only view of trained models in the Google Drive registry.
"""
import os
import sys

import csv

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QComboBox, QScrollArea, QFrame,
    QSpacerItem, QSizePolicy, QApplication, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from mlops.registry import RegistrySettings, RegistryScanner

_SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mlops", "scripts")
)



_BG        = "#1e1e1e"
_PANEL_BG  = "#252525"
_CARD_BG   = "#2d2d2d"
_CARD_SEL  = "#3a3a3a"
_BORDER    = "#3a3a3a"
_TEXT      = "#e0e0e0"
_MUTED     = "#888888"
_ACCENT    = "#4fc3f7"

_STATUS_COLORS = {
    "completed":  ("#1b5e20", "#4caf50"),   # (bg, fg)
    "training":   ("#1a3a00", "#8bc34a"),
    "crashed":    ("#7f0000", "#ef9a9a"),
    "cancelled":  ("#333333", "#9e9e9e"),
    "syncing":    ("#0d2a3a", "#4fc3f7"),
    "unknown":    ("#333333", "#9e9e9e"),
}


def _make_status_badge(status: str) -> QLabel:
    lbl = QLabel(status.upper())
    bg, fg = _STATUS_COLORS.get(status.lower(), _STATUS_COLORS["unknown"])
    lbl.setStyleSheet(f"""
        background-color: {bg};
        color: {fg};
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: bold;
        font-size: 10px;
    """)
    return lbl

def _fmt_bytes(n: int) -> str:
    if n >= 1e9: return f"{n / 1e9:.1f} GB"
    if n >= 1e6: return f"{n / 1e6:.1f} MB"
    if n >= 1e3: return f"{n / 1e3:.1f} KB"
    return f"{n} B"

def _fmt_loss(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return "—"

def _fmt_date(s: str) -> str:
    try:
        if not s or not isinstance(s, str): return "—"
        if "T" not in s: return "—"
        date_part, time_part = s.split("T")
        # simple parsing assuming YYYY-MM-DD
        y, m, d = date_part.split("-")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        m_str = months[int(m)-1]
        time_part = time_part[:5] # HH:MM
        return f"{m_str} {int(d)}, {y}  {time_part}"
    except Exception:
        return "—"


class VersionCard(QFrame):
    def __init__(self, summary: dict, on_click, parent=None):
        super().__init__(parent)
        self.summary = summary
        self._on_click = on_click
        
        self.setCursor(Qt.PointingHandCursor)
        self.set_selected(False)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)
        
        # Row 1
        r1 = QHBoxLayout()
        r1.setContentsMargins(0, 0, 0, 0)
        vid = QLabel(summary.get("version_id", ""))
        vid.setStyleSheet("font-weight: bold; font-size: 11px;")
        r1.addWidget(vid)
        r1.addStretch()
        r1.addWidget(_make_status_badge(summary.get("status", "unknown")))
        layout.addLayout(r1)
        
        # Row 2
        arch = summary.get("architecture", "")
        if summary.get("model_type", "") == "unet":
            encoder = summary.get("encoder", "")
            arch_str = f"{arch} ({encoder})" if encoder else arch
        else:
            arch_str = arch
        
        r2 = QLabel(arch_str)
        r2.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        layout.addWidget(r2)
        
        # Row 3
        best_loss = _fmt_loss(summary.get("best_val_loss"))
        started = _fmt_date(summary.get("started_at", ""))
        r3 = QLabel(f"Best loss: {best_loss}  {started}")
        r3.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        layout.addWidget(r3)

    def set_selected(self, selected: bool):
        if selected:
            self.setStyleSheet(f"""
                VersionCard {{
                    background-color: {_CARD_SEL};
                    border: 1px solid {_ACCENT};
                    border-radius: 6px;
                    margin: 2px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                VersionCard {{
                    background-color: {_CARD_BG};
                    border: 1px solid {_BORDER};
                    border-radius: 6px;
                    margin: 2px;
                }}
                VersionCard:hover {{
                    border: 1px solid #555555;
                }}
            """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._on_click(self.summary)
        super().mousePressEvent(event)


class RegistryTab(QWidget):
    retrain_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._registry_root = RegistrySettings().get_registry_root()
        self._current_summary = None
        self._cards = []
        self._setup_ui()
        self._refresh()

    def _make_section_header(self, title: str) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 10, 0, 0)
        l.setSpacing(2)
        
        lbl = QLabel(title.upper())
        lbl.setStyleSheet(f"font-weight: bold; font-size: 11px; color: {_ACCENT};")
        l.addWidget(lbl)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet(f"background-color: {_BORDER};")
        l.addWidget(line)
        return w

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {_BORDER}; }}")
        root.addWidget(splitter)

        # ---------------- LEFT PANEL ----------------
        left_panel = QWidget()
        left_panel.setMinimumWidth(270)
        left_panel.setMaximumWidth(320)
        left_panel.setStyleSheet(f"background-color: {_BG};")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        # Header row
        h_row = QHBoxLayout()
        h_lbl = QLabel("Registry Browser")
        h_lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        h_row.addWidget(h_lbl)
        h_row.addStretch()
        self._refresh_btn = QPushButton("🔄")
        self._refresh_btn.setFixedSize(28, 28)
        self._refresh_btn.setFlat(True)
        self._refresh_btn.setStyleSheet(f"QPushButton:hover {{ background-color: {_PANEL_BG}; border-radius: 4px; }}")
        self._refresh_btn.clicked.connect(self._refresh)
        h_row.addWidget(self._refresh_btn)
        left_layout.addLayout(h_row)

        # Project selector
        p_row = QHBoxLayout()
        p_lbl = QLabel("Project")
        self._project_cb = QComboBox()
        self._project_cb.setStyleSheet(f"""
            QComboBox {{
                background-color: {_CARD_BG}; border: 1px solid {_BORDER};
                border-radius: 4px; color: {_TEXT}; padding: 4px;
            }}
        """)
        self._project_cb.currentTextChanged.connect(self._on_project_changed)
        p_row.addWidget(p_lbl)
        p_row.addWidget(self._project_cb, 1)
        left_layout.addLayout(p_row)

        # Warning label
        self._warning_lbl = QLabel("⚠ Registry not configured.\nSet GDrive root in Settings.")
        self._warning_lbl.setAlignment(Qt.AlignCenter)
        self._warning_lbl.setStyleSheet(f"color: {_MUTED};")
        self._warning_lbl.hide()
        left_layout.addWidget(self._warning_lbl)

        # Conflict file warning label
        self._conflict_lbl = QLabel()
        self._conflict_lbl.setStyleSheet(
            "color: #ffc107; font-size: 10px; background: #2a2000; "
            "border: 1px solid #ffc107; border-radius: 4px; padding: 4px;"
        )
        self._conflict_lbl.setWordWrap(True)
        self._conflict_lbl.hide()
        left_layout.addWidget(self._conflict_lbl)

        # Scroll area for cards
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet(f"QScrollArea {{ background: transparent; border: none; }}")
        
        cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(cards_widget)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(2)
        self._cards_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        self._scroll.setWidget(cards_widget)
        left_layout.addWidget(self._scroll, 1)

        # Storage
        self._storage_lbl = QLabel("")
        self._storage_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        left_layout.addWidget(self._storage_lbl)

        # Export CSV button
        self._csv_btn = QPushButton("📊  Export All Models to CSV")
        self._csv_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {_PANEL_BG}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 4px; padding: 6px;
                font-size: 11px;
            }}
            QPushButton:hover {{ border-color: {_ACCENT}; }}
            QPushButton:disabled {{ color: {_MUTED}; }}
        """)
        self._csv_btn.clicked.connect(self._export_csv)
        left_layout.addWidget(self._csv_btn)

        splitter.addWidget(left_panel)

        # ---------------- RIGHT PANEL ----------------
        right_panel = QWidget()
        right_panel.setStyleSheet(f"background-color: {_BG};")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._empty_lbl = QLabel("← Select a version to view details")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 13px;")
        right_layout.addWidget(self._empty_lbl)

        # Detail Widget
        self._detail_scroll = QScrollArea()
        self._detail_scroll.setWidgetResizable(True)
        self._detail_scroll.setFrameShape(QFrame.NoFrame)
        
        self._detail_widget = QWidget()
        dl = QVBoxLayout(self._detail_widget)
        dl.setContentsMargins(20, 20, 20, 20)
        dl.setSpacing(6)
        
        # Header section
        dh_row = QHBoxLayout()
        self._d_vid = QLabel()
        self._d_vid.setStyleSheet("font-weight: bold; font-size: 14px;")
        dh_row.addWidget(self._d_vid)
        dh_row.addStretch()
        self._d_status_layout = QVBoxLayout() # placeholder for status badge
        dh_row.addLayout(self._d_status_layout)
        dl.addLayout(dh_row)
        
        self._d_commit = QLabel()
        self._d_commit.setStyleSheet(f"color: {_MUTED}; font-style: italic; font-size: 11px;")
        self._d_commit.setWordWrap(True)
        dl.addWidget(self._d_commit)
        
        # Dataset section
        dl.addWidget(self._make_section_header("DATASET"))
        self._d_ds_counts = QLabel()
        self._d_ds_classes = QLabel()
        self._d_ds_sha = QLabel()
        for w in (self._d_ds_counts, self._d_ds_classes, self._d_ds_sha):
            dl.addWidget(w)
        self._d_ds_path = QLabel()
        self._d_ds_path.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        self._d_ds_path.setWordWrap(True)
        dl.addWidget(self._d_ds_path)

        self._d_sha_warn = QLabel()
        self._d_sha_warn.setStyleSheet("color: #ffc107; font-size: 10px; background: transparent;")
        self._d_sha_warn.setWordWrap(True)
        dl.addWidget(self._d_sha_warn)

        # Hyperparameters section
        dl.addWidget(self._make_section_header("HYPERPARAMETERS"))
        self._d_hp_l1 = QLabel()
        self._d_hp_l2 = QLabel()
        self._d_hp_l3 = QLabel()
        self._d_hp_l4 = QLabel()
        for w in (self._d_hp_l1, self._d_hp_l2, self._d_hp_l3, self._d_hp_l4):
            dl.addWidget(w)
            
        # Metrics section
        dl.addWidget(self._make_section_header("METRICS"))
        self._d_mx_best = QLabel()
        self._d_mx_final = QLabel()
        dl.addWidget(self._d_mx_best)
        dl.addWidget(self._d_mx_final)
        
        # Artifacts section
        dl.addWidget(self._make_section_header("ARTIFACTS"))
        self._d_art_weights = QLabel()
        self._d_art_onnx = QLabel()
        dl.addWidget(self._d_art_weights)
        dl.addWidget(self._d_art_onnx)
        
        # Crash section
        self._crash_header = self._make_section_header("CRASH INFO")
        dl.addWidget(self._crash_header)
        self._d_crash_reason = QLabel()
        self._d_crash_hint = QLabel()
        self._d_crash_reason.setWordWrap(True)
        self._d_crash_hint.setWordWrap(True)
        dl.addWidget(self._d_crash_reason)
        dl.addWidget(self._d_crash_hint)
        
        dl.addStretch()
        
        # Action buttons
        btn_layout = QHBoxLayout()
        self._open_btn = QPushButton("📂  Open Folder")
        self._open_btn.setStyleSheet(f"background-color: {_PANEL_BG}; padding: 8px; border-radius: 4px; border: 1px solid {_BORDER};")
        self._open_btn.clicked.connect(self._open_folder)

        self._open_ds_btn = QPushButton("🗄  Open Dataset Folder")
        self._open_ds_btn.setStyleSheet(f"background-color: {_PANEL_BG}; padding: 8px; border-radius: 4px; border: 1px solid {_BORDER};")
        self._open_ds_btn.clicked.connect(self._open_dataset_folder)

        self._copy_btn = QPushButton("📋  Copy Weights Path")
        self._copy_btn.setStyleSheet(f"background-color: {_PANEL_BG}; padding: 8px; border-radius: 4px; border: 1px solid {_BORDER};")
        self._copy_btn.clicked.connect(self._copy_weights_path)
        
        self._export_btn = QPushButton("⚙  Export ONNX")
        self._export_btn.setStyleSheet(
            f"background-color: {_PANEL_BG}; padding: 8px; "
            f"border-radius: 4px; border: 1px solid {_BORDER};"
        )
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export_onnx)

        self._retrain_btn = QPushButton("🔁  Clone & Re-train")
        self._retrain_btn.setStyleSheet(
            f"background-color: {_PANEL_BG}; padding: 8px; "
            f"border-radius: 4px; border: 1px solid {_BORDER};"
        )
        self._retrain_btn.setEnabled(False)
        self._retrain_btn.clicked.connect(self._on_retrain_clicked)

        self._copy_status_lbl = QLabel("")
        self._copy_status_lbl.setStyleSheet(f"color: #4caf50; font-size: 10px;")

        btn_layout.addWidget(self._open_btn)
        btn_layout.addWidget(self._open_ds_btn)
        btn_layout.addWidget(self._copy_btn)
        btn_layout.addWidget(self._export_btn)
        btn_layout.addWidget(self._retrain_btn)
        btn_layout.addWidget(self._copy_status_lbl)
        btn_layout.addStretch()
        dl.addLayout(btn_layout)
        
        from PyQt5.QtWidgets import QProgressBar
        self._export_progress = QProgressBar()
        self._export_progress.setRange(0, 100)
        self._export_progress.setValue(0)
        self._export_progress.setFixedHeight(6)
        self._export_progress.setTextVisible(False)
        self._export_progress.hide()
        self._export_progress.setStyleSheet("""
            QProgressBar { background: #333; border-radius: 3px; border: none; }
            QProgressBar::chunk { background: #4fc3f7; border-radius: 3px; }
        """)
        dl.addWidget(self._export_progress)
        
        dl.addStretch()
        
        self._detail_scroll.setWidget(self._detail_widget)
        right_layout.addWidget(self._detail_scroll)
        
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _refresh(self):
        self._registry_root = RegistrySettings().get_registry_root()
        if not self._registry_root or not os.path.isdir(self._registry_root):
            self._warning_lbl.show()
            self._project_cb.blockSignals(True)
            self._project_cb.clear()
            self._project_cb.blockSignals(False)
            self._on_project_changed()
            return
            
        self._warning_lbl.hide()
        scanner = RegistryScanner(self._registry_root)
        
        self._project_cb.blockSignals(True)
        curr = self._project_cb.currentText()
        self._project_cb.clear()
        for p in scanner.get_projects():
            self._project_cb.addItem(p)
        if curr:
            self._project_cb.setCurrentText(curr)
        self._project_cb.blockSignals(False)
        
        self._on_project_changed()

    def _on_project_changed(self):
        # Clear existing cards
        while self._cards_layout.count() > 1: # keeping spacer
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()
        
        proj = self._project_cb.currentText()
        if not proj:
            self._storage_lbl.setText("")
            self._show_detail(None)
            return
            
        scanner = RegistryScanner(self._registry_root)
        versions = scanner.get_versions(proj)
        
        for summary in versions:
            card = VersionCard(summary, self._on_version_clicked)
            self._cards.append(card)
            # Insert before spacer
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
            
        storage = scanner.get_storage_stats(proj)
        self._storage_lbl.setText(f"Storage: {_fmt_bytes(storage.get('total_bytes', 0))}")

        conflicts = scanner.detect_conflict_files(proj)
        if conflicts:
            names = [os.path.basename(c) for c in conflicts[:3]]
            extra = f" (+{len(conflicts)-3} more)" if len(conflicts) > 3 else ""
            self._conflict_lbl.setText(
                f"⚠ {len(conflicts)} GDrive conflict file(s) detected:\n" +
                "\n".join(names) + extra +
                "\nDelete them manually in Explorer."
            )
            self._conflict_lbl.show()
        else:
            self._conflict_lbl.hide()

        self._show_detail(None)

    def _on_version_clicked(self, summary: dict):
        for card in self._cards:
            card.set_selected(card.summary.get("version_id") == summary.get("version_id"))
            
        self._current_summary = summary
        scanner = RegistryScanner(self._registry_root)
        manifest = scanner.get_version(summary.get("project", ""), summary.get("version_id", ""))
        self._show_detail(manifest)

    def _show_detail(self, manifest: dict | None):
        if not manifest:
            self._detail_scroll.hide()
            self._empty_lbl.show()
            self._export_btn.setEnabled(False)
            self._export_btn.setText("⚙  Export ONNX")
            self._d_ds_path.setText("")
            self._d_sha_warn.setText("")
            self._open_ds_btn.setEnabled(False)
            self._retrain_btn.setEnabled(False)
            return
            
        self._empty_lbl.hide()
        self._detail_scroll.show()
        
        self._d_vid.setText(manifest.get("version_id", ""))
        
        # clear and re-add status badge
        while self._d_status_layout.count():
            w = self._d_status_layout.takeAt(0).widget()
            if w: w.deleteLater()
        self._d_status_layout.addWidget(_make_status_badge(manifest.get("status", "unknown")))
        
        self._d_commit.setText(manifest.get("commit_message", ""))
        
        # Dataset
        ds = manifest.get("dataset", {})
        tr, va, te = ds.get("train_count", 0), ds.get("val_count", 0), ds.get("test_count", 0)
        self._d_ds_counts.setText(f"Train: {tr}  Val: {va}  Test: {te}")
        
        cls_names = ds.get("classes", [])
        cls_str = ", ".join(cls_names)
        if len(cls_str) > 60: cls_str = cls_str[:57] + "..."
        nc = ds.get("num_classes", 0)
        self._d_ds_classes.setText(f"Classes: {cls_str} ({nc})")
        
        sha = ds.get("sha256", "")
        self._d_ds_sha.setText(f"SHA-256: {sha[:16]}..." if sha else "SHA-256: —")

        vf = self._current_summary.get("version_folder", "") if self._current_summary else ""
        if vf:
            from mlops.registry.manifest import ManifestReader
            ds_path = ManifestReader(vf).get_dataset_path()
        else:
            ds_path = ""
        if ds_path and os.path.isdir(ds_path):
            self._d_ds_path.setText(f"📁 {ds_path}")
            self._open_ds_btn.setEnabled(True)
        else:
            self._d_ds_path.setText("Dataset not found on this machine")
            self._open_ds_btn.setEnabled(False)

        # SHA mismatch check
        manifest_sha = ds.get("sha256", "")
        self._d_sha_warn.setText("")
        if manifest_sha and ds_path and os.path.isdir(ds_path):
            info_path = os.path.join(ds_path, "dataset_info.json")
            if os.path.isfile(info_path):
                try:
                    import json as _json
                    with open(info_path, "r", encoding="utf-8") as _f:
                        current_info = _json.load(_f)
                    current_sha = current_info.get("sha256", "")
                    if current_sha and current_sha != manifest_sha:
                        self._d_sha_warn.setText(
                            f"⚠ Dataset has changed since this model was trained.\n"
                            f"Model SHA: {manifest_sha[:16]}...   Current: {current_sha[:16]}...\n"
                            f"Consider retraining."
                        )
                except Exception:
                    pass

        # Hyperparameters
        hp = manifest.get("hyperparams", {})
        arch = hp.get("architecture", "")
        enc = hp.get("encoder", "")
        if manifest.get("model_type", "") == "unet" and enc:
            self._d_hp_l1.setText(f"Arch: {arch}   Encoder: {enc}")
        else:
            self._d_hp_l1.setText(f"Arch: {arch}")
            
        ep, bs, lr = hp.get("epochs", "—"), hp.get("batch_size", "—"), hp.get("learning_rate", "—")
        self._d_hp_l2.setText(f"Epochs: {ep}   Batch: {bs}   LR: {lr}")
        
        iw, ih = hp.get("image_width", "—"), hp.get("image_height", "—")
        inc, ouc = hp.get("in_channels", "—"), hp.get("out_classes", "—")
        self._d_hp_l3.setText(f"Size: {iw}×{ih}   Channels: {inc}   Out classes: {ouc}")
        
        dev, enc_w = hp.get("device", "—"), hp.get("encoder_weights", "—")
        self._d_hp_l4.setText(f"Device: {dev}   Encoder weights: {enc_w}")
        
        # Metrics
        mx = manifest.get("metrics", {})
        bvl, bep = mx.get("best_val_loss"), mx.get("best_epoch")
        ftl = mx.get("final_train_loss")
        
        if bvl is None and ftl is None:
            self._d_mx_best.setText("No metrics recorded")
            self._d_mx_best.setStyleSheet(f"color: {_MUTED};")
            self._d_mx_final.setText("")
        else:
            self._d_mx_best.setStyleSheet(f"color: {_TEXT};")
            self._d_mx_best.setText(f"Best Val Loss: {_fmt_loss(bvl)}   @ Epoch {bep if bep else '—'}")
            self._d_mx_final.setText(f"Final Train Loss: {_fmt_loss(ftl)}")
            
        # Artifacts
        art = manifest.get("artifacts", {})
        weights = art.get("weights")
        w_exists = ""
        if weights and self._current_summary:
            w_path = os.path.join(self._current_summary.get("version_folder", ""), weights)
            if os.path.isfile(w_path): w_exists = "  ✓"
        self._d_art_weights.setText(f"Weights: {weights or '—'}{w_exists}")
        
        ox = manifest.get("onnx_config", {})
        if ox.get("exported"):
            opv, ext = ox.get("opset_version", "—"), _fmt_date(ox.get("exported_at", ""))
            self._d_art_onnx.setText(f"ONNX: exported (opset {opv}, {ext})")
        else:
            self._d_art_onnx.setText("ONNX: not exported")
            
        # Crash info
        cinfo = manifest.get("crash_info")
        if manifest.get("status") == "crashed" and cinfo:
            self._crash_header.show()
            self._d_crash_reason.show()
            self._d_crash_hint.show()
            self._d_crash_reason.setText(f"Reason: {cinfo.get('reason', '')}")
            self._d_crash_hint.setText(f"Hint: {cinfo.get('hint', '')}")
        else:
            self._crash_header.hide()
            self._d_crash_reason.hide()
            self._d_crash_hint.hide()

        can_export = (
            manifest.get("status") == "completed"
            and bool(manifest.get("artifacts", {}).get("weights"))
            and not manifest.get("onnx_config", {}).get("exported", False)
        )
        self._export_btn.setEnabled(can_export)
        self._export_btn.setText("⚙  Export ONNX" if can_export else ("✓ ONNX Ready" if manifest.get("onnx_config", {}).get("exported") else "⚙  Export ONNX"))

        self._retrain_btn.setEnabled(manifest.get("status") in ("completed", "crashed", "cancelled"))

    def _on_retrain_clicked(self):
        if not self._current_summary:
            return
        vf = self._current_summary.get("version_folder", "")
        if not vf:
            return
        scanner = RegistryScanner(self._registry_root)
        manifest = scanner.get_version(
            self._current_summary.get("project", ""),
            self._current_summary.get("version_id", ""),
        )
        if not manifest:
            return
        manifest["version_folder"] = vf
        self.retrain_requested.emit(manifest)

    def _export_csv(self):
        if not self._registry_root or not os.path.isdir(self._registry_root):
            QMessageBox.warning(self, "Registry Not Configured",
                                "Set the GDrive registry root before exporting.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Models CSV", "models_registry.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        scanner = RegistryScanner(self._registry_root)
        projects = scanner.get_projects()

        columns = [
            "version_id", "project", "model_type", "status", "annotator",
            "commit_message", "started_at", "finished_at",
            "architecture", "encoder", "encoder_weights",
            "in_channels", "out_classes", "epochs", "batch_size",
            "learning_rate", "image_width", "image_height", "device",
            "best_val_loss", "best_epoch", "final_train_loss",
            "train_count", "val_count", "test_count", "num_classes",
            "class_names", "sha256", "has_weights", "has_onnx",
            "version_folder",
        ]

        rows = []
        for proj in projects:
            full_manifests = [
                scanner.get_version(proj, s["version_id"])
                for s in scanner.get_versions(proj)
            ]
            for m in full_manifests:
                if not m:
                    continue
                hp  = m.get("hyperparams", {})
                mx  = m.get("metrics", {})
                ds  = m.get("dataset", {})
                art = m.get("artifacts", {})
                ox  = m.get("onnx_config", {})
                rows.append({
                    "version_id":       m.get("version_id", ""),
                    "project":          m.get("project", ""),
                    "model_type":       m.get("model_type", ""),
                    "status":           m.get("status", ""),
                    "annotator":        m.get("annotator", ""),
                    "commit_message":   m.get("commit_message", ""),
                    "started_at":       m.get("started_at", ""),
                    "finished_at":      m.get("finished_at", ""),
                    "architecture":     hp.get("architecture", ""),
                    "encoder":          hp.get("encoder", ""),
                    "encoder_weights":  hp.get("encoder_weights", ""),
                    "in_channels":      hp.get("in_channels", ""),
                    "out_classes":      hp.get("out_classes", ""),
                    "epochs":           hp.get("epochs", ""),
                    "batch_size":       hp.get("batch_size", ""),
                    "learning_rate":    hp.get("learning_rate", ""),
                    "image_width":      hp.get("image_width", ""),
                    "image_height":     hp.get("image_height", ""),
                    "device":           hp.get("device", ""),
                    "best_val_loss":    mx.get("best_val_loss", ""),
                    "best_epoch":       mx.get("best_epoch", ""),
                    "final_train_loss": mx.get("final_train_loss", ""),
                    "train_count":      ds.get("train_count", ""),
                    "val_count":        ds.get("val_count", ""),
                    "test_count":       ds.get("test_count", ""),
                    "num_classes":      ds.get("num_classes", ""),
                    "class_names":      "|".join(ds.get("classes", [])),
                    "sha256":           ds.get("sha256", ""),
                    "has_weights":      "yes" if art.get("weights") else "no",
                    "has_onnx":         "yes" if ox.get("exported") else "no",
                    "version_folder":   m.get("version_id", ""),
                })

        if not rows:
            QMessageBox.information(self, "No Models", "No model versions found in the registry.")
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        QMessageBox.information(
            self, "CSV Exported",
            f"Exported {len(rows)} model(s) to:\n{path}"
        )

    def _open_folder(self):
        if not self._current_summary: return
        vf = self._current_summary.get("version_folder", "")
        if sys.platform == "win32" and os.path.isdir(vf):
            try: os.startfile(vf)
            except Exception: pass

    def _open_dataset_folder(self):
        if not self._current_summary:
            return
        vf = self._current_summary.get("version_folder", "")
        if not vf:
            return
        from mlops.registry.manifest import ManifestReader
        ds_path = ManifestReader(vf).get_dataset_path()
        if ds_path and os.path.isdir(ds_path) and sys.platform == "win32":
            try:
                os.startfile(ds_path)
            except Exception:
                pass

    def _copy_weights_path(self):
        if not self._current_summary: return
        vf = self._current_summary.get("version_folder", "")
        # Assuming manifest is the one backing current summary
        # We don't store manifest, but we can assume best.pt as default if we don't fetch it again
        # Actually it's cleaner to read it from scanner if needed, or assume best.pt
        w = "best.pt"
        try:
            m = RegistryScanner(self._registry_root).get_version(
                self._current_summary.get("project", ""),
                self._current_summary.get("version_id", "")
            )
            w = m.get("artifacts", {}).get("weights") or w
        except Exception:
            pass
            
        p = os.path.join(vf, w)
        QApplication.clipboard().setText(p)
        self._copy_status_lbl.setText("✓ Copied")

    def _on_export_onnx(self):
        if not self._current_summary:
            return

        from mlops.export import OnnxWorker

        vf = self._current_summary.get("version_folder", "")
        if not vf or not os.path.isdir(vf):
            return

        self._export_btn.setEnabled(False)
        self._export_btn.setText("Exporting...")
        self._export_progress.setValue(0)
        self._export_progress.show()
        self._copy_status_lbl.setText("")

        self._onnx_worker = OnnxWorker(
            version_folder = vf,
            scripts_dir    = _SCRIPTS_DIR,
            parent         = self,
        )
        self._onnx_worker.log.connect(lambda msg: self._copy_status_lbl.setText(msg[-60:]))
        self._onnx_worker.progress.connect(self._export_progress.setValue)
        self._onnx_worker.finished.connect(self._on_export_finished)
        self._onnx_worker.start()

    def _on_export_finished(self, success: bool):
        self._export_progress.hide()
        if success:
            self._copy_status_lbl.setText("✓ ONNX exported")
            if self._current_summary:
                scanner  = RegistryScanner(self._registry_root)
                manifest = scanner.get_version(
                    self._current_summary.get("project", ""),
                    self._current_summary.get("version_id", ""),
                )
                self._show_detail(manifest)
        else:
            self._export_btn.setEnabled(True)
            self._export_btn.setText("⚙  Export ONNX")
            self._copy_status_lbl.setText("✗ Export failed — see log")
