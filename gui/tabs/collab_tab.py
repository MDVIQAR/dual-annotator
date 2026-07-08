# gui/tabs/collab_tab.py
"""
Simultaneous Annotation – Collaboration Tab
Allows the Admin to host a session and Annotators to join via a Project ID.
Uses only Python stdlib networking — no extra pip packages needed.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QTextEdit, QListWidget, QListWidgetItem,
    QFrame, QSplitter, QMessageBox, QFileDialog, QGridLayout, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor

# ── Style helpers ─────────────────────────────────────────────────────────────

CARD_STYLE = """
    QFrame {
        background-color: #1a1a2e;
        border: 1px solid #2e3a58;
        border-radius: 12px;
    }
"""
BTN_PRIMARY = """
    QPushButton {
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 #4a6bff, stop:1 #7b4fff);
        color: white; border: none; border-radius: 8px;
        padding: 12px 20px; font-size: 14px; font-weight: bold;
    }
    QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #5a7bff, stop:1 #8b5fff); }
    QPushButton:disabled { background: #333; color: #666; }
"""
BTN_DANGER = """
    QPushButton {
        background-color: #7f1d1d; color: #fca5a5;
        border: 1px solid #991b1b; border-radius: 8px;
        padding: 10px 18px; font-size: 13px; font-weight: bold;
    }
    QPushButton:hover { background-color: #991b1b; }
"""
BTN_SUCCESS = """
    QPushButton {
        background-color: #14532d; color: #86efac;
        border: 1px solid #166534; border-radius: 8px;
        padding: 12px 20px; font-size: 14px; font-weight: bold;
    }
    QPushButton:hover { background-color: #166534; }
"""
INPUT_STYLE = """
    QLineEdit, QSpinBox {
        background-color: #0f0f1a; color: #e2e8f0;
        border: 1px solid #2e3a58; border-radius: 6px;
        padding: 8px 12px; font-size: 13px;
    }
    QLineEdit:focus, QSpinBox:focus { border: 1px solid #4a6bff; }
"""
LOG_STYLE = """
    QTextEdit {
        background-color: #0a0a14; color: #94a3b8;
        border: 1px solid #1e2840; border-radius: 6px;
        font-family: 'Courier New', monospace; font-size: 11px;
        padding: 6px;
    }
"""
LIST_STYLE = """
    QListWidget {
        background-color: #0f0f1a; color: #e2e8f0;
        border: 1px solid #1e2840; border-radius: 6px;
        font-size: 12px;
    }
    QListWidget::item { padding: 6px 10px; border-bottom: 1px solid #1e2840; }
    QListWidget::item:selected { background-color: #1e3a5f; color: #93c5fd; }
"""


def _label(text, size=13, bold=False, color="#e2e8f0"):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{color}; font-size:{size}px;"
        + ("font-weight:bold;" if bold else "")
        + "background:transparent; border:none;"
    )
    return lbl


def _sep():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("background:#2e3a58; max-height:1px; border:none;")
    return line


def _card(layout_cls=QVBoxLayout, margins=(20, 20, 20, 20), spacing=12):
    frame = QFrame()
    frame.setStyleSheet(CARD_STYLE)
    lay = layout_cls(frame)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return frame, lay


# ── Main Tab Widget ───────────────────────────────────────────────────────────

class CollabTab(QWidget):
    """
    Signals emitted to MainWindow:
      session_hosted(project_id, port)   – admin started server
      session_joined(project_data, client, image_folder_name)
      session_stopped()
    """
    session_hosted  = pyqtSignal(str, int)   # project_id, port
    session_joined  = pyqtSignal(dict, object)  # project_data, CollabClient
    session_stopped = pyqtSignal()

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._server     = None   # CollabServer instance
        self._client     = None   # CollabClient instance
        self._mode       = "offline"  # "offline" | "host" | "client"
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(500)  # 0.5 seconds for instant updates
        self._sync_timer.timeout.connect(self._poll_sync)

        self.setStyleSheet("QWidget { background-color: #0d0d1a; }")
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 24, 30, 24)
        root.setSpacing(20)

        # Header
        hdr = QHBoxLayout()
        title = _label("👥  Simultaneous Annotation", 22, bold=True, color="#93c5fd")
        sub   = _label("Host a session or join one — everyone annotates the same project in real time.",
                        12, color="#64748b")
        hdr.addWidget(title)
        hdr.addStretch()
        self._status_dot = _label("⚫ Offline", 12, color="#64748b")
        hdr.addWidget(self._status_dot)
        root.addLayout(hdr)
        root.addWidget(sub)
        root.addWidget(_sep())

        # ── Stacked pages ─────────────────────────────────────────────────
        # Page 0: offline (two cards)
        self._page_offline = self._build_offline_page()
        root.addWidget(self._page_offline)

        # Page 1: hosting dashboard
        self._page_hosting = self._build_hosting_page()
        self._page_hosting.hide()
        root.addWidget(self._page_hosting)

        # Page 2: client connected
        self._page_client = self._build_client_page()
        self._page_client.hide()
        root.addWidget(self._page_client)

        root.addStretch()

    # ── Offline page ─────────────────────────────────────────────────────────

    def _build_offline_page(self):
        page = QWidget()
        lay  = QHBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(24)
        lay.addWidget(self._build_host_card())
        lay.addWidget(self._build_join_card())
        return page

    def _build_host_card(self):
        card, lay = _card()
        lay.addWidget(_label("🖥️  Host a Session", 16, bold=True, color="#93c5fd"))
        lay.addWidget(_label(
            "Open a local image folder, start a server, and share the Project ID\n"
            "with your team. All annotations save directly to your machine.",
            11, color="#64748b"))
        lay.addWidget(_sep())

        # Folder selector
        lay.addWidget(_label("Image Folder", 12, bold=True))
        row = QHBoxLayout()
        self._host_folder_edit = QLineEdit()
        self._host_folder_edit.setPlaceholderText("Select image folder…")
        self._host_folder_edit.setReadOnly(True)
        self._host_folder_edit.setStyleSheet(INPUT_STYLE)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.setStyleSheet("""
            QPushButton { background:#1e3a5f; color:#93c5fd; border:1px solid #2e5a8f;
                          border-radius:6px; padding:8px; font-size:12px; }
            QPushButton:hover { background:#2e4a6f; }
        """)
        browse_btn.clicked.connect(self._browse_host_folder)
        row.addWidget(self._host_folder_edit)
        row.addWidget(browse_btn)
        lay.addLayout(row)

        # Port
        port_row = QHBoxLayout()
        port_row.addWidget(_label("Port", 12, bold=True))
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(8765)
        self._port_spin.setFixedWidth(100)
        self._port_spin.setStyleSheet(INPUT_STYLE)
        port_row.addWidget(self._port_spin)
        port_row.addStretch()
        lay.addLayout(port_row)

        lay.addStretch()
        self._start_host_btn = QPushButton("🚀  Start Hosting")
        self._start_host_btn.setStyleSheet(BTN_PRIMARY)
        self._start_host_btn.clicked.connect(self._start_hosting)
        lay.addWidget(self._start_host_btn)
        return card

    def _build_join_card(self):
        card, lay = _card()
        lay.addWidget(_label("🔗  Join a Session", 16, bold=True, color="#86efac"))
        lay.addWidget(_label(
            "Enter the Project ID shared by your Admin and your name.\n"
            "Images stream on-demand — no folder selection needed.",
            11, color="#64748b"))
        lay.addWidget(_sep())

        lay.addWidget(_label("Project ID", 12, bold=True))
        self._project_id_edit = QLineEdit()
        self._project_id_edit.setPlaceholderText("e.g.  DA-ORSXG5DTORUW45DP…")
        self._project_id_edit.setStyleSheet(INPUT_STYLE)
        lay.addWidget(self._project_id_edit)

        lay.addWidget(_label("Your Name", 12, bold=True))
        self._username_edit = QLineEdit()
        self._username_edit.setPlaceholderText("e.g.  Alice")
        self._username_edit.setStyleSheet(INPUT_STYLE)
        lay.addWidget(self._username_edit)

        lay.addStretch()
        self._join_btn = QPushButton("🔗  Connect to Session")
        self._join_btn.setStyleSheet(BTN_SUCCESS)
        self._join_btn.clicked.connect(self._join_session)
        lay.addWidget(self._join_btn)
        return card

    # ── Hosting dashboard ─────────────────────────────────────────────────────

    def _build_hosting_page(self):
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(20)

        # Project ID display
        id_card, id_lay = _card()
        id_lay.addWidget(_label("Your Project ID  (share this with annotators)", 12, color="#94a3b8"))
        self._project_id_display = QLineEdit()
        self._project_id_display.setReadOnly(True)
        self._project_id_display.setAlignment(Qt.AlignCenter)
        self._project_id_display.setStyleSheet("""
            QLineEdit {
                background:#0a0a14; color:#7dd3fc; border:2px solid #0ea5e9;
                border-radius:8px; padding:12px; font-size:18px;
                font-weight:bold; font-family:'Courier New';
                letter-spacing:2px;
            }
        """)
        id_lay.addWidget(self._project_id_display)
        copy_btn = QPushButton("📋  Copy to Clipboard")
        copy_btn.setStyleSheet("""
            QPushButton { background:#0c4a6e; color:#7dd3fc; border:1px solid #0369a1;
                          border-radius:6px; padding:8px; font-size:12px; }
            QPushButton:hover { background:#075985; }
        """)
        copy_btn.clicked.connect(self._copy_project_id)
        id_lay.addWidget(copy_btn)
        lay.addWidget(id_card)

        # Two columns: users + log
        cols = QHBoxLayout()

        # Connected users
        users_card, users_lay = _card()
        users_lay.addWidget(_label("👤  Connected Annotators", 13, bold=True))
        self._users_list = QListWidget()
        self._users_list.setStyleSheet(LIST_STYLE)
        self._users_list.setMinimumHeight(180)
        users_lay.addWidget(self._users_list)
        cols.addWidget(users_card)

        # Activity log
        log_card, log_lay = _card()
        log_lay.addWidget(_label("📋  Live Activity", 13, bold=True))
        self._activity_log = QTextEdit()
        self._activity_log.setReadOnly(True)
        self._activity_log.setStyleSheet(LOG_STYLE)
        self._activity_log.setMinimumHeight(180)
        log_lay.addWidget(self._activity_log)
        cols.addWidget(log_card)

        lay.addLayout(cols)

        # Stop button
        stop_btn = QPushButton("⏹  Stop Server & Return Offline")
        stop_btn.setStyleSheet(BTN_DANGER)
        stop_btn.clicked.connect(self._stop_hosting)
        lay.addWidget(stop_btn)
        return page

    # ── Client connected page ─────────────────────────────────────────────────

    def _build_client_page(self):
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(20)

        info_card, info_lay = _card()
        info_lay.addWidget(_label("✅  Connected to Session", 16, bold=True, color="#86efac"))
        self._client_info_label = _label("", 12, color="#94a3b8")
        info_lay.addWidget(self._client_info_label)
        info_lay.addWidget(_sep())
        info_lay.addWidget(_label(
            "Switch to the  ✏️ Annotate  tab to start annotating.\n"
            "Images load on demand from the host. Image locking prevents conflicts.",
            12, color="#64748b"))
        lay.addWidget(info_card)

        # Users + locks
        cols = QHBoxLayout()

        users_card, users_lay = _card()
        users_lay.addWidget(_label("👥  Active Annotators", 13, bold=True))
        self._client_users_list = QListWidget()
        self._client_users_list.setStyleSheet(LIST_STYLE)
        self._client_users_list.setMinimumHeight(160)
        users_lay.addWidget(self._client_users_list)
        cols.addWidget(users_card)

        locks_card, locks_lay = _card()
        locks_lay.addWidget(_label("🔒  Current Locks", 13, bold=True))
        self._locks_list = QListWidget()
        self._locks_list.setStyleSheet(LIST_STYLE)
        self._locks_list.setMinimumHeight(160)
        locks_lay.addWidget(self._locks_list)
        cols.addWidget(locks_card)

        lay.addLayout(cols)

        disc_btn = QPushButton("🔌  Disconnect")
        disc_btn.setStyleSheet(BTN_DANGER)
        disc_btn.clicked.connect(self._disconnect)
        lay.addWidget(disc_btn)
        return page

    # ── Actions: Host ─────────────────────────────────────────────────────────

    def _browse_host_folder(self):
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
            if folder:
                self._host_folder_edit.setText(folder)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _start_hosting(self):
        try:
            folder = self._host_folder_edit.text().strip()
            if not folder or not os.path.isdir(folder):
                QMessageBox.warning(self, "No Folder", "Please select a valid image folder first.")
                return

            # Load the folder into main window's project manager
            mw = self.main_window
            mw.open_image_folder(folder_path=folder)

            from core.collab_server import CollabServer, encode_project_id, get_local_ip
            port = self._port_spin.value()
            self._server = CollabServer(mw.project_manager, port=port)
            self._server.started_ok.connect(self._on_server_started)
            self._server.start_error.connect(self._on_server_error)
            self._server.start()

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _on_server_started(self, ip: str, port: int):
        from core.collab_server import encode_project_id
        pid = encode_project_id(ip, port)
        self._project_id_display.setText(pid)
        self._mode = "host"
        self._status_dot.setText("🟢  Hosting")
        self._status_dot.setStyleSheet("color:#4ade80; font-size:12px; background:transparent; border:none;")
        self._page_offline.hide()
        self._page_hosting.show()
        self._sync_timer.start()
        self.session_hosted.emit(pid, port)

    def _on_server_error(self, msg: str):
        QMessageBox.critical(self, "Server Error", f"Failed to start server:\n{msg}")

    def _copy_project_id(self):
        try:
            from PyQt5.QtWidgets import QApplication
            QApplication.clipboard().setText(self._project_id_display.text())
            self.main_window.status_bar.showMessage("Project ID copied to clipboard!", 2000)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _stop_hosting(self):
        try:
            self._sync_timer.stop()
            if self._server:
                self._server.stop()
                self._server = None
            self._mode = "offline"
            self._status_dot.setText("⚫ Offline")
            self._status_dot.setStyleSheet("color:#64748b; font-size:12px; background:transparent; border:none;")
            self._page_hosting.hide()
            self._page_offline.show()
            self._users_list.clear()
            self._activity_log.clear()
            self.session_stopped.emit()

        # ── Actions: Join ─────────────────────────────────────────────────────────

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _join_session(self):
        try:
            pid      = self._project_id_edit.text().strip()
            username = self._username_edit.text().strip()
            if not pid:
                QMessageBox.warning(self, "Missing Info", "Please enter the Project ID.")
                return
            if not username:
                QMessageBox.warning(self, "Missing Info", "Please enter your name.")
                return

            self._join_btn.setEnabled(False)
            self._join_btn.setText("Connecting…")

            from core.collab_client import CollabClient
            client = CollabClient()
            ok, result = client.connect(pid, username)

            self._join_btn.setEnabled(True)
            self._join_btn.setText("🔗  Connect to Session")

            if not ok:
                QMessageBox.critical(self, "Connection Failed", str(result))
                return

            self._client = client
            self._mode   = "client"
            project_data = result

            info = f"Connected as  {username}  ·  Server: {client.base_url}"
            self._client_info_label.setText(info)
            self._status_dot.setText(f"🟣  Connected as {username}")
            self._status_dot.setStyleSheet("color:#c084fc; font-size:12px; background:transparent; border:none;")

            self._page_offline.hide()
            self._page_client.show()

            # Start background polling
            client.start_sync_poller(self._on_client_sync)

            # Tell main window to enter client mode
            self.session_joined.emit(project_data, client)

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _disconnect(self):
        try:
            if self._client:
                self._client.disconnect()
                self._client.clean_cache()
                self._client = None
            self._mode = "offline"
            self._status_dot.setText("⚫ Offline")
            self._status_dot.setStyleSheet("color:#64748b; font-size:12px; background:transparent; border:none;")
            self._page_client.hide()
            self._page_offline.show()
            self._client_users_list.clear()
            self._locks_list.clear()
            self.session_stopped.emit()

        # ── Sync callbacks ────────────────────────────────────────────────────────

        except Exception as e:
            logging.error(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Operation Failed",
                f"This operation couldn't complete.\n\n{str(e)}"
            )
            return
    def _poll_sync(self):
        """Called by QTimer when hosting — refreshes users and activity log."""
        if self._mode != "host" or not self._server:
            return
        snap = self._server.get_snapshot()
        self._refresh_host_ui(snap)
        
        # Propagate lock/status info to main window file list for the host too!
        if hasattr(self.main_window, '_update_collab_badges'):
            self.main_window._update_collab_badges(
                snap.get("locks", {}),
                snap.get("statuses", {})
            )

    def _refresh_host_ui(self, snap: dict):
        self._users_list.clear()
        for u in snap.get("users", []):
            item = QListWidgetItem(f"👤  {u}")
            item.setForeground(QColor("#86efac"))
            self._users_list.addItem(item)

        log_lines = []
        for ts, msg in snap.get("activity", []):
            log_lines.append(f"[{ts}]  {msg}")
        self._activity_log.setPlainText("\n".join(log_lines[-40:]))
        sb = self._activity_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_client_sync(self, snap: dict):
        """Called by SyncPoller every 2.5 s when in client mode."""
        # Update users list
        self._client_users_list.clear()
        for u in snap.get("users", []):
            item = QListWidgetItem(f"👤  {u}")
            item.setForeground(QColor("#93c5fd"))
            self._client_users_list.addItem(item)

        # Update locks list
        self._locks_list.clear()
        for filename, locker in snap.get("locks", {}).items():
            item = QListWidgetItem(f"🔒  {filename}  ←  {locker}")
            item.setForeground(QColor("#fbbf24"))
            self._locks_list.addItem(item)

        # Propagate lock/status info to main window file list
        if hasattr(self.main_window, '_update_collab_badges'):
            self.main_window._update_collab_badges(
                snap.get("locks", {}),
                snap.get("statuses", {})
            )


import os
import logging
import traceback
