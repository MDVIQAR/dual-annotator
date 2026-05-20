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

        # ── Registry Paths section ──────────────────────────────────────
        paths_widget = QWidget()
        paths_widget.setStyleSheet("background: transparent;")
        paths_lay = QVBoxLayout(paths_widget)
        paths_lay.setContentsMargins(0, 0, 0, 0)
        paths_lay.setSpacing(12)

        # Explanation note
        note_lbl = QLabel(
            "Set both paths to save every trained version in two places.\n"
            "Training runs in the Local path (fast); the version folder is also\n"
            "copied to Google Drive automatically when training completes."
        )
        note_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 11px; background: transparent; border: none;")
        note_lbl.setWordWrap(True)
        paths_lay.addWidget(note_lbl)

        # GDrive root row
        gdrive_lbl = QLabel("Google Drive Root Folder")
        gdrive_lbl.setStyleSheet(f"color: {_TEXT}; font-size: 12px; background: transparent; border: none;")
        paths_lay.addWidget(gdrive_lbl)

        gdrive_row = QHBoxLayout()
        self._gdrive_edit = QLineEdit()
        self._gdrive_edit.setPlaceholderText("e.g. C:\\Users\\You\\Google Drive\\My Drive")
        self._gdrive_edit.setStyleSheet(_INPUT_STYLE)
        self._gdrive_edit.textChanged.connect(self._on_gdrive_changed)
        gdrive_browse = QPushButton("Browse")
        gdrive_browse.setStyleSheet(_BTN_STYLE)
        gdrive_browse.setFixedWidth(80)
        gdrive_browse.clicked.connect(self._browse_gdrive)
        gdrive_row.addWidget(self._gdrive_edit)
        gdrive_row.addWidget(gdrive_browse)
        paths_lay.addLayout(gdrive_row)

        self._gdrive_status = QLabel()
        self._gdrive_status.setStyleSheet(f"font-size: 11px; background: transparent; border: none;")
        self._gdrive_status.setWordWrap(True)
        paths_lay.addWidget(self._gdrive_status)

        # Local root row
        local_lbl = QLabel("Local Registry Root Folder")
        local_lbl.setStyleSheet(f"color: {_TEXT}; font-size: 12px; background: transparent; border: none;")
        paths_lay.addWidget(local_lbl)

        local_row = QHBoxLayout()
        self._local_edit = QLineEdit()
        self._local_edit.setPlaceholderText("e.g. D:\\Models\\Registry")
        self._local_edit.setStyleSheet(_INPUT_STYLE)
        self._local_edit.textChanged.connect(self._on_local_changed)
        local_browse = QPushButton("Browse")
        local_browse.setStyleSheet(_BTN_STYLE)
        local_browse.setFixedWidth(80)
        local_browse.clicked.connect(self._browse_local)
        local_row.addWidget(self._local_edit)
        local_row.addWidget(local_browse)
        paths_lay.addLayout(local_row)

        self._local_status = QLabel()
        self._local_status.setStyleSheet(f"font-size: 11px; background: transparent; border: none;")
        self._local_status.setWordWrap(True)
        paths_lay.addWidget(self._local_status)

        # Keep legacy alias so _update_registry_status() still works
        self._registry_status = self._gdrive_status

        main.addWidget(_make_section("REGISTRY PATHS", paths_widget))

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
        self._gdrive_edit.setText(self._settings.get_registry_root())
        self._local_edit.setText(self._settings.get_local_root())
        self._update_gdrive_status()
        self._update_local_status()
        self._rebuild_annotator_rows()

    # GDrive handlers
    def _browse_gdrive(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Google Drive Root", self._gdrive_edit.text() or "")
        if folder:
            self._gdrive_edit.setText(folder)

    def _on_gdrive_changed(self, text):
        self._settings.set_gdrive_root(text.strip())
        self._update_gdrive_status()

    def _update_gdrive_status(self):
        root = self._settings.get_registry_root()
        if not root:
            self._gdrive_status.setText(f"<span style='color:{_RED}'>&#9888; Google Drive root is not set.</span>")
        elif os.path.isdir(root):
            self._gdrive_status.setText(f"<span style='color:{_GREEN}'>&#10003; {root}</span>")
        else:
            self._gdrive_status.setText(
                f"<span style='color:#ffc107'>&#9888; {root} — folder doesn't exist yet, will be created.</span>"
            )

    # Keep legacy name so any external callers still work
    def _update_registry_status(self):
        self._update_gdrive_status()

    # Local root handlers
    def _browse_local(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Local Registry Root", self._local_edit.text() or "")
        if folder:
            self._local_edit.setText(folder)

    def _on_local_changed(self, text):
        self._settings.set_local_root(text.strip())
        self._update_local_status()

    def _update_local_status(self):
        root = self._settings.get_local_root()
        if not root:
            self._local_status.setText(f"<span style='color:{_MUTED}'>Not set — training will only save to Google Drive.</span>")
        elif os.path.isdir(root):
            self._local_status.setText(f"<span style='color:{_GREEN}'>&#10003; {root}</span>")
        else:
            self._local_status.setText(
                f"<span style='color:#ffc107'>&#9888; {root} — folder doesn't exist yet, will be created.</span>"
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
