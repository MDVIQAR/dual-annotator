# core/collab_client.py
"""
Client-side networking for Dual Annotator collaboration mode.
All HTTP calls use Python stdlib urllib — zero extra dependencies.
"""

import os
import json
import time
import tempfile
import urllib.request
import urllib.error
from urllib.parse import quote

from PyQt5.QtCore import QThread, pyqtSignal, QTimer

from core.collab_server import decode_project_id   # reuse utility


# ── Low-level HTTP helpers ────────────────────────────────────────────────────

def _get(base_url: str, path: str, timeout: int = 8):
    url = f"{base_url}{path}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _get_bytes(base_url: str, path: str, timeout: int = 15) -> bytes:
    url = f"{base_url}{path}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _post(base_url: str, path: str, payload: dict, timeout: int = 10):
    url  = f"{base_url}{path}"
    data = json.dumps(payload, ensure_ascii=False).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ── Sync poller (runs in background thread) ───────────────────────────────────

class SyncPoller(QThread):
    """Polls /api/sync every 500ms and emits the result."""
    synced      = pyqtSignal(dict)   # full sync snapshot
    error       = pyqtSignal(str)
    user_joined = pyqtSignal(str)    # username that just connected
    user_left   = pyqtSignal(str)    # username that just disconnected

    def __init__(self, base_url: str, interval_ms: int = 500, parent=None):
        super().__init__(parent)
        self.base_url    = base_url
        self.interval_ms = interval_ms
        self._running    = False
        self._known_users = None   # None = not seeded yet (first poll is silent)

    def run(self):
        self._running = True
        while self._running:
            try:
                data = _get(self.base_url, "/api/sync", timeout=4)
                self.synced.emit(data)
                current = set(data.get("users", []))
                if self._known_users is None:
                    self._known_users = current   # seed silently on first poll
                else:
                    for u in current - self._known_users:
                        self.user_joined.emit(u)
                    for u in self._known_users - current:
                        self.user_left.emit(u)
                    self._known_users = current
            except Exception as e:
                self.error.emit(str(e))
            time.sleep(self.interval_ms / 1000.0)

    def stop(self):
        self._running = False
        self.quit()
        self.wait(3000)


# ── Main client class ─────────────────────────────────────────────────────────

class CollabClient:
    """
    Wraps all server communication for an annotator client.

    Usage
    -----
    client = CollabClient()
    ok, msg = client.connect("DA-XXXXX", "Alice")
    images  = client.get_image_list()
    path    = client.fetch_image("img001.png")      # returns local cache path
    ann     = client.load_annotations("img001.png")
    client.acquire_lock("img001.png")
    client.save_annotations("img001.png", ann_dict)
    client.release_lock("img001.png")
    client.disconnect()
    """

    def __init__(self):
        self.base_url     = ""
        self.username     = ""
        self._connected   = False
        self._cache_dir   = tempfile.mkdtemp(prefix="da_collab_")
        self._poller      = None
        # Callbacks (set by owner)
        self.on_sync      = None   # callable(snapshot_dict)

    # ── connection ──────────────────────────────────────────────────────────

    def connect(self, project_id: str, username: str):
        """Return (True, project_data) or (False, error_str)."""
        try:
            ip, port = decode_project_id(project_id.strip())
        except Exception as e:
            return False, f"Bad Project ID: {e}"

        self.base_url  = f"http://{ip}:{port}"
        self.username  = username.strip() or "Annotator"

        try:
            _post(self.base_url, "/api/register", {"user": self.username})
            project_data = _get(self.base_url, "/api/project")
            self._connected = True
            return True, project_data
        except Exception as e:
            self.base_url   = ""
            self._connected = False
            return False, f"Cannot reach server: {e}"

    def disconnect(self):
        if self._poller:
            self._poller.stop()
            self._poller = None
        if self._connected:
            try:
                _post(self.base_url, "/api/unregister", {"user": self.username})
            except Exception:
                pass
        self._connected = False
        self.base_url   = ""

    def start_sync_poller(self, callback):
        """Start background polling. callback(snapshot_dict) called every 2 s."""
        self.on_sync = callback
        self._poller = SyncPoller(self.base_url)
        self._poller.synced.connect(callback)
        self._poller.start()

    def stop_sync_poller(self):
        if self._poller:
            self._poller.stop()
            self._poller = None

    # ── data access ─────────────────────────────────────────────────────────

    def get_image_list(self):
        """Returns list of image filenames from server."""
        try:
            data = _get(self.base_url, "/api/images")
            return data.get("images", [])
        except Exception as e:
            return []

    def fetch_image(self, filename: str) -> str:
        """
        Download the image from server and cache locally.
        Returns the local file path (for canvas.load_image).
        """
        cache_path = os.path.join(self._cache_dir, filename)
        if os.path.exists(cache_path):
            return cache_path   # already cached
        try:
            raw = _get_bytes(self.base_url, f"/api/image/{quote(filename)}", timeout=20)
            with open(cache_path, "wb") as fh:
                fh.write(raw)
            return cache_path
        except Exception as e:
            raise RuntimeError(f"Failed to fetch image '{filename}': {e}")

    def load_annotations(self, filename: str) -> dict:
        """Fetch annotation JSON for a given image."""
        try:
            return _get(self.base_url, f"/api/annotations/{quote(filename)}")
        except Exception:
            return {}

    def save_annotations(self, filename: str, ann_dict: dict) -> bool:
        """POST annotation JSON back to server. Returns True on success."""
        try:
            _post(self.base_url, f"/api/annotations/{quote(filename)}", ann_dict)
            return True
        except Exception as e:
            print(f"[CollabClient] save error: {e}")
            return False

    def acquire_lock(self, filename: str) -> tuple:
        """
        Try to acquire an exclusive lock for this image.
        Returns (granted: bool, locked_by: str).
        """
        try:
            r = _post(self.base_url, f"/api/lock/{quote(filename)}",
                      {"user": self.username})
            return r.get("granted", False), r.get("locked_by", "")
        except Exception:
            return False, ""

    def release_lock(self, filename: str):
        """Release the lock on an image."""
        try:
            _post(self.base_url, f"/api/unlock/{quote(filename)}",
                  {"user": self.username})
        except Exception:
            pass

    def get_sync(self) -> dict:
        """Manual one-shot sync."""
        try:
            return _get(self.base_url, "/api/sync", timeout=4)
        except Exception:
            return {}

    @property
    def connected(self) -> bool:
        return self._connected

    def clean_cache(self):
        """Remove temp image cache."""
        import shutil
        try:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
        except Exception:
            pass
