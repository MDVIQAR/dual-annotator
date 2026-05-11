"""
gui/tabs/settings_tab.py

Settings tab — configure GDrive root and manage annotators.
All changes are persisted immediately via RegistrySettings.
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QScrollArea, QSizePolicy, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt

from mlops.registry import RegistrySettings

_BG       = "#1e1e1e"
_PANEL_BG = "#252525"
_CARD_BG  = "#2d2d2d"
_BORDER   = "#3a3a3a"
_TEXT     = "#e0e0e0"
_MUTED    = "#888888"
_ACCENT   = "#4fc3f7"
_GREEN    = "#4caf50"
_RED      = "#f87171"

_SECTION_STYLE = f"""
    QFrame {{
        background-color: {_PANEL_BG};
        border: 1px solid {_BORDER};
        border-radius: 6px;
    }}
"""

_INPUT_STYLE = f"""
    QLineEdit {{
        background-color: {_CARD_BG};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        color: {_TEXT};
        padding: 6px 10px;
        font-size: 12px;
    }}
    QLineEdit:focus {{ border: 1px solid {_ACCENT}; }}
"""

_BTN_STYLE = f"""
    QPushButton {{
        background-color: {_CARD_BG};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        color: {_TEXT};
        padding: 6px 14px;
        font-size: 12px;
    }}
    QPushButton:hover {{ border-color: {_ACCENT}; }}
    QPushButton:pressed {{ background-color: {_PANEL_BG}; }}
"""

_ADD_BTN_STYLE = f"""
    QPushButton {{
        background-color: #1b5e20;
        border: 1px solid #2e7d32;
        border-radius: 4px;
        color: #ffffff;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: bold;
    }}
    QPushButton:hover {{ background-color: #2e7d32; }}
"""

_REMOVE_BTN_STYLE = f"""
    QPushButton {{
        background-color: #7f0000;
        border: 1px solid #b71c1c;
        border-radius: 4px;
        color: #ffffff;
        padding: 4px 10px;
        font-size: 11px;
    }}
    QPushButton:hover {{ background-color: #b71c1c; }}
"""


def _section_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {_ACCENT}; font-size: 12px; font-weight: bold; background: transparent; border: none;")
    return lbl


def _make_section(title, content_widget):
    frame = QFrame()
    frame.setStyleSheet(_SECTION_STYLE)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 12, 16, 16)
    lay.setSpacing(10)
    lay.addWidget(_section_label(title))
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"background-color: {_BORDER}; border: none;")
    line.setFixedHeight(1)
    lay.addWidget(line)
    lay.addWidget(content_widget)
    return frame


class AnnotatorRow(QFrame):
    def __init__(self, name, initials, on_remove, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame {{ background-color: {_CARD_BG}; border: 1px solid {_BORDER}; border-radius: 4px; }}")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 6, 6)
        row.setSpacing(8)

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(f"color: {_TEXT}; font-size: 12px; background: transparent; border: none;")
        init_lbl = QLabel(f"[{initials}]")
        init_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 11px; background: transparent; border: none;")
        remove_btn = QPushButton("Remove")
        remove_btn.setStyleSheet(_REMOVE_BTN_STYLE)
        remove_btn.setFixedWidth(70)
        remove_btn.clicked.connect(lambda: on_remove(initials))

        row.addWidget(name_lbl)
        row.addWidget(init_lbl)
        row.addStretch()
        row.addWidget(remove_btn)


class SettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = RegistrySettings()
        self._setup_ui()
        self._load_current()

    def _setup_ui(self):
        self.setStyleSheet(f"background-color: {_BG};")
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background: {_BG}; border: none; }}")

        container = QWidget()
        container.setStyleSheet(f"background-color: {_BG};")
        main = QVBoxLayout(container)
        main.setContentsMargins(30, 24, 30, 24)
        main.setSpacing(20)

        # Title
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        main.addWidget(title)

        # ── GDrive section ──────────────────────────────────────────────
        gdrive_widget = QWidget()
        gdrive_widget.setStyleSheet("background: transparent;")
        gdrive_lay = QVBoxLayout(gdrive_widget)
        gdrive_lay.setContentsMargins(0, 0, 0, 0)
        gdrive_lay.setSpacing(10)

        # GDrive root row
        root_lbl = QLabel("Google Drive Root Folder")
        root_lbl.setStyleSheet(f"color: {_TEXT}; font-size: 12px; background: transparent; border: none;")
        gdrive_lay.addWidget(root_lbl)

        root_row = QHBoxLayout()
        self._gdrive_edit = QLineEdit()
        self._gdrive_edit.setPlaceholderText("e.g. C:\\Users\\You\\Google Drive\\My Drive")
        self._gdrive_edit.setStyleSheet(_INPUT_STYLE)
        self._gdrive_edit.textChanged.connect(self._on_gdrive_changed)
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(_BTN_STYLE)
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_gdrive)
        root_row.addWidget(self._gdrive_edit)
        root_row.addWidget(browse_btn)
        gdrive_lay.addLayout(root_row)

        # Computed registry root
        self._registry_status = QLabel()
        self._registry_status.setStyleSheet(f"font-size: 11px; background: transparent; border: none;")
        self._registry_status.setWordWrap(True)
        gdrive_lay.addWidget(self._registry_status)

        main.addWidget(_make_section("GOOGLE DRIVE CONFIGURATION", gdrive_widget))

        # ── Annotators section ──────────────────────────────────────────
        ann_widget = QWidget()
        ann_widget.setStyleSheet("background: transparent;")
        ann_lay = QVBoxLayout(ann_widget)
        ann_lay.setContentsMargins(0, 0, 0, 0)
        ann_lay.setSpacing(8)

        info_lbl = QLabel("Each team member must add themselves here. Initials appear in model version IDs.")
        info_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 11px; background: transparent; border: none;")
        info_lbl.setWordWrap(True)
        ann_lay.addWidget(info_lbl)

        # Annotator rows container
        self._ann_rows_widget = QWidget()
        self._ann_rows_widget.setStyleSheet("background: transparent;")
        self._ann_rows_lay = QVBoxLayout(self._ann_rows_widget)
        self._ann_rows_lay.setContentsMargins(0, 0, 0, 0)
        self._ann_rows_lay.setSpacing(4)
        ann_lay.addWidget(self._ann_rows_widget)

        # Add new annotator form
        add_frame = QFrame()
        add_frame.setStyleSheet(f"QFrame {{ background-color: {_BG}; border: 1px dashed {_BORDER}; border-radius: 4px; }}")
        add_lay = QHBoxLayout(add_frame)
        add_lay.setContentsMargins(10, 8, 10, 8)
        add_lay.setSpacing(8)

        self._new_name_edit = QLineEdit()
        self._new_name_edit.setPlaceholderText("Full name  (e.g. Mohammed Viqar)")
        self._new_name_edit.setStyleSheet(_INPUT_STYLE)

        self._new_initials_edit = QLineEdit()
        self._new_initials_edit.setPlaceholderText("Initials  (e.g. MV)")
        self._new_initials_edit.setFixedWidth(100)
        self._new_initials_edit.setStyleSheet(_INPUT_STYLE)

        add_btn = QPushButton("+ Add")
        add_btn.setStyleSheet(_ADD_BTN_STYLE)
        add_btn.setFixedWidth(70)
        add_btn.clicked.connect(self._add_annotator)

        add_lay.addWidget(self._new_name_edit)
        add_lay.addWidget(self._new_initials_edit)
        add_lay.addWidget(add_btn)
        ann_lay.addWidget(add_frame)

        main.addWidget(_make_section("ANNOTATORS", ann_widget))
        main.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

    def _load_current(self):
        self._settings = RegistrySettings()
        gdrive = self._settings._data.get("gdrive_root", "")
        self._gdrive_edit.setText(gdrive)
        self._update_registry_status()
        self._rebuild_annotator_rows()

    def _browse_gdrive(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Google Drive Root", self._gdrive_edit.text() or "")
        if folder:
            self._gdrive_edit.setText(folder)

    def _on_gdrive_changed(self, text):
        self._settings.set_gdrive_root(text.strip())
        self._update_registry_status()

    def _update_registry_status(self):
        root = self._settings.get_registry_root()
        if not root:
            self._registry_status.setText(f"<span style='color:{_RED}'>&#9888; Google Drive root is not set.</span>")
            return
        if os.path.isdir(root):
            self._registry_status.setText(f"<span style='color:{_GREEN}'>&#10003; Registry root: {root}</span>")
        else:
            self._registry_status.setText(
                f"<span style='color:#ffc107'>&#9888; Registry root: {root}<br>"
                f"Folder does not exist yet — it will be created on first training run.</span>"
            )

    def _rebuild_annotator_rows(self):
        while self._ann_rows_lay.count():
            item = self._ann_rows_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        annotators = self._settings.get_annotators()
        if not annotators:
            empty = QLabel("No annotators yet. Add yourself below.")
            empty.setStyleSheet(f"color: {_MUTED}; font-size: 11px; background: transparent; border: none;")
            self._ann_rows_lay.addWidget(empty)
        else:
            for ann in annotators:
                row = AnnotatorRow(ann["name"], ann["initials"], self._remove_annotator)
                self._ann_rows_lay.addWidget(row)

    def _add_annotator(self):
        name     = self._new_name_edit.text().strip()
        initials = self._new_initials_edit.text().strip()
        if not name or not initials:
            QMessageBox.warning(self, "Validation", "Both Name and Initials are required.")
            return
        self._settings.add_annotator(name, initials)
        self._new_name_edit.clear()
        self._new_initials_edit.clear()
        self._rebuild_annotator_rows()

    def _remove_annotator(self, initials: str):
        self._settings.remove_annotator(initials)
        self._rebuild_annotator_rows()
