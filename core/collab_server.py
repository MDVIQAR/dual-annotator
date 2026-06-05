# core/collab_server.py
"""
Lightweight HTTP server that runs in a background QThread on the Admin's machine.
Serves images and annotations to connected annotator clients over local Wi-Fi.
Requires zero extra dependencies — uses only Python stdlib.
"""

import os
import json
import time
import socket
import base64
import threading
import http.server
import socketserver
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal


# ── Utility functions ─────────────────────────────────────────────────────────

def get_local_ip() -> str:
    """Return the machine's primary LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def encode_project_id(ip: str, port: int) -> str:
    """Encode IP:port into a short human-readable Project ID."""
    raw = f"{ip}:{port}".encode()
    encoded = base64.b32encode(raw).decode().rstrip("=")
    return f"DA-{encoded}"


def decode_project_id(project_id: str):
    """Decode a Project ID back to (ip, port). Raises ValueError on bad input."""
    if not project_id.startswith("DA-"):
        raise ValueError("Invalid Project ID — must start with DA-")
    encoded = project_id[3:]
    pad = (8 - len(encoded) % 8) % 8
    encoded += "=" * pad
    raw = base64.b32decode(encoded.upper()).decode()
    ip, port_str = raw.rsplit(":", 1)
    return ip, int(port_str)


# ── Shared server state (passed to every request handler) ────────────────────

class _ServerState:
    def __init__(self, project_manager, on_join=None, on_leave=None, on_classes_updated=None):
        self.pm                  = project_manager
        self.locks               = {}
        self.users               = {}
        self.activity            = []
        self._mu                 = threading.Lock()
        self._on_join            = on_join
        self._on_leave           = on_leave
        self._on_classes_updated = on_classes_updated

    def log(self, msg: str):
        self.activity.append((datetime.now().strftime("%H:%M:%S"), msg))
        if len(self.activity) > 200:
            self.activity = self.activity[-200:]

    def register_user(self, username: str):
        with self._mu:
            self.users[username] = time.time()
            self.log(f"{username} joined")
        if self._on_join:
            self._on_join(username)

    def unregister_user(self, username: str):
        with self._mu:
            self.users.pop(username, None)
            # Release any locks held by this user
            stale = [f for f, u in self.locks.items() if u == username]
            for f in stale:
                self.locks.pop(f, None)
            self.log(f"{username} left")
        if self._on_leave:
            self._on_leave(username)

    def acquire_lock(self, filename: str, username: str) -> bool:
        with self._mu:
            if filename not in self.locks:
                self.locks[filename] = username
                self.log(f"{username} locked {filename}")
                return True
            return self.locks[filename] == username  # already owned

    def release_lock(self, filename: str, username: str):
        with self._mu:
            if self.locks.get(filename) == username:
                self.locks.pop(filename, None)
                self.log(f"{username} released {filename}")

    def merge_classes(self, incoming: list, username: str):
        """Merge incoming classes into project_data; persist and notify if anything changed."""
        if not self.pm.project_data:
            return
        with self._mu:
            existing = {c["id"]: c for c in self.pm.project_data.get("classes", [])}
            changed = False
            for c in incoming:
                if c["id"] not in existing:
                    existing[c["id"]] = c
                    self.log(f"{username} added class '{c['name']}'")
                    changed = True
                else:
                    prev = existing[c["id"]]
                    if prev.get("name") != c.get("name") or prev.get("color") != c.get("color"):
                        existing[c["id"]] = c
                        self.log(f"{username} updated class '{c['name']}'")
                        changed = True
            if changed:
                merged = list(existing.values())
                self.pm.project_data["classes"] = merged
                self.pm.update_project_classes(merged)
                classes_snapshot = list(merged)
        if changed and self._on_classes_updated:
            self._on_classes_updated(classes_snapshot)

    def snapshot(self) -> dict:
        with self._mu:
            statuses = {}
            classes  = []
            if self.pm.project_data:
                statuses = dict(self.pm.project_data.get("image_statuses", {}))
                classes  = list(self.pm.project_data.get("classes", []))
            return {
                "locks":    dict(self.locks),
                "statuses": statuses,
                "users":    list(self.users.keys()),
                "activity": self.activity[-30:],
                "classes":  classes,
            }


# ── Request handler ───────────────────────────────────────────────────────────

def _make_handler(state: _ServerState):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

    class Handler(http.server.BaseHTTPRequestHandler):

        def log_message(self, fmt, *args):
            pass  # suppress console noise

        # ── helpers ──────────────────────────────────────────────────────────

        def _send_json(self, data: dict, code: int = 200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, data: bytes, mime: str = "application/octet-stream"):
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length:
                return json.loads(self.rfile.read(length))
            return {}

        def _path(self) -> str:
            from urllib.parse import urlparse, unquote
            return unquote(urlparse(self.path).path)

        # ── GET ──────────────────────────────────────────────────────────────

        def do_GET(self):
            p = self._path()

            if p == "/api/project":
                data = state.pm.project_data or {}
                self._send_json(data)

            elif p == "/api/images":
                folder = state.pm.image_folder or ""
                try:
                    files = sorted([
                        f for f in os.listdir(folder)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                    ])
                except Exception:
                    files = []
                self._send_json({"images": files})

            elif p.startswith("/api/image/"):
                filename = p[len("/api/image/"):]
                path = os.path.join(state.pm.image_folder or "", filename)
                if os.path.isfile(path):
                    with open(path, "rb") as fh:
                        data = fh.read()
                    ext = os.path.splitext(filename)[1].lower()
                    mime = {"png": "image/png", "jpg": "image/jpeg",
                            "jpeg": "image/jpeg", "bmp": "image/bmp",
                            "tiff": "image/tiff"}.get(ext.lstrip("."), "application/octet-stream")
                    self._send_bytes(data, mime)
                else:
                    self._send_json({"error": "not found"}, 404)

            elif p.startswith("/api/annotations/"):
                filename = p[len("/api/annotations/"):]
                ann = state.pm.load_image_annotations(filename)
                self._send_json(ann or {})

            elif p == "/api/sync":
                self._send_json(state.snapshot())

            else:
                self._send_json({"error": "unknown endpoint"}, 404)

        # ── POST ─────────────────────────────────────────────────────────────

        def do_POST(self):
            p    = self._path()
            body = self._read_body()

            if p.startswith("/api/annotations/"):
                filename = p[len("/api/annotations/"):]
                ann_dir  = state.pm.annotations_dir
                if ann_dir:
                    out = os.path.join(ann_dir, f"{filename}.json")
                    state.pm._write_json_atomic(out, body)
                    # Sync statuses back into project_data
                    status = body.get("status", "in_progress")
                    state.pm.set_image_status(filename, status, write_json=False)
                    state.log(f"Annotations saved for {filename}")
                self._send_json({"ok": True})

            elif p.startswith("/api/lock/"):
                filename = p[len("/api/lock/"):]
                username = body.get("user", "unknown")
                granted  = state.acquire_lock(filename, username)
                locked_by = state.locks.get(filename, "")
                self._send_json({"granted": granted, "locked_by": locked_by})

            elif p.startswith("/api/unlock/"):
                filename = p[len("/api/unlock/"):]
                username = body.get("user", "unknown")
                state.release_lock(filename, username)
                self._send_json({"ok": True})

            elif p == "/api/register":
                username = body.get("user", "unknown")
                state.register_user(username)
                self._send_json({"ok": True})

            elif p == "/api/unregister":
                username = body.get("user", "unknown")
                state.unregister_user(username)
                self._send_json({"ok": True})

            elif p == "/api/classes":
                username = body.get("user", "unknown")
                classes  = body.get("classes", [])
                state.merge_classes(classes, username)
                self._send_json({"ok": True})

            else:
                self._send_json({"error": "unknown endpoint"}, 404)

    return Handler


# ── ThreadingHTTPServer ───────────────────────────────────────────────────────

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ── QThread wrapper ───────────────────────────────────────────────────────────

class CollabServer(QThread):
    """Runs the collaboration HTTP server in a background thread."""

    started_ok       = pyqtSignal(str, int)   # local_ip, port
    start_error      = pyqtSignal(str)
    user_joined      = pyqtSignal(str)        # username
    user_left        = pyqtSignal(str)        # username
    activity_msg     = pyqtSignal(str)
    classes_updated  = pyqtSignal(list)       # full merged classes list

    def __init__(self, project_manager, port: int = 8765, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.port            = port
        self._server         = None
        self._state          = None

    @property
    def state(self) -> _ServerState:
        return self._state

    def run(self):
        try:
            self._state  = _ServerState(
                self.project_manager,
                on_join=self.user_joined.emit,
                on_leave=self.user_left.emit,
                on_classes_updated=self.classes_updated.emit,
            )
            handler_cls  = _make_handler(self._state)
            self._server = _ThreadedHTTPServer(("0.0.0.0", self.port), handler_cls)
            local_ip     = get_local_ip()
            self.started_ok.emit(local_ip, self.port)
        except OSError as e:
            self.start_error.emit(str(e))
            return

        self._server.serve_forever()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None

    def get_snapshot(self) -> dict:
        if self._state:
            return self._state.snapshot()
        return {}
