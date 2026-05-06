"""
mlops/registry/scanner.py

Scans the Google Drive registry folder structure to enumerate projects and model versions.
Uses ManifestReader to read each version's metadata without loading any UI components.
"""

import os
import re

from .manifest import ManifestReader


class RegistryScanner:
    """Reads the GDrive registry folder and returns structured version summaries."""

    def __init__(self, registry_root: str):
        self._root = registry_root

    def get_projects(self) -> list:
        """Return sorted project names (top-level subdirs). Returns [] if root missing."""
        if not self._root or not os.path.isdir(self._root):
            return []
        try:
            return sorted(e.name for e in os.scandir(self._root) if e.is_dir())
        except OSError:
            return []

    def get_versions(self, project: str) -> list:
        """
        Scan {registry_root}/{project}/ for version subfolders.
        Skips folders with no manifest.json.
        Overrides status to 'syncing' when sync_complete=False.
        Returns results sorted by started_at descending (newest first).
        """
        project_dir = os.path.join(self._root, project)
        if not os.path.isdir(project_dir):
            return []
        try:
            entries = [e for e in os.scandir(project_dir) if e.is_dir()]
        except OSError:
            return []

        results = []
        for entry in entries:
            reader = ManifestReader(entry.path)
            summary = reader.summary()
            if not summary.get("version_id"):
                continue
            if not summary.get("sync_complete", False):
                summary["status"] = "syncing"
            results.append(summary)

        results.sort(key=lambda s: s.get("started_at") or "", reverse=True)
        return results

    def get_version(self, project: str, version_id: str) -> dict | None:
        """Return the full manifest dict for a specific version, or None if not found."""
        version_dir = os.path.join(self._root, project, version_id)
        if not os.path.isdir(version_dir):
            return None
        return ManifestReader(version_dir).read()

    _CONFLICT_PATTERN = re.compile(r".+ \(\d+\)\..+")

    def detect_conflict_files(self, project: str) -> list:
        """Return list of absolute paths to Google Drive conflict copies inside a project."""
        project_dir = os.path.join(self._root, project)
        if not os.path.isdir(project_dir):
            return []
        conflicts = []
        for dirpath, _, filenames in os.walk(project_dir):
            for fname in filenames:
                if self._CONFLICT_PATTERN.match(fname):
                    conflicts.append(os.path.join(dirpath, fname))
        return conflicts

    def get_storage_stats(self, project: str) -> dict:
        """
        Return storage usage for all versions inside a project.
        Schema: {"total_bytes": int, "versions": [{"version_id", "bytes", "has_last_pt"}]}
        Does not follow symlinks.
        """
        project_dir = os.path.join(self._root, project)
        if not os.path.isdir(project_dir):
            return {"total_bytes": 0, "versions": []}
        try:
            entries = [e for e in os.scandir(project_dir) if e.is_dir()]
        except OSError:
            return {"total_bytes": 0, "versions": []}

        total_bytes = 0
        versions = []
        for entry in entries:
            folder_bytes = 0
            has_last_pt = False
            for dirpath, _, filenames in os.walk(entry.path, followlinks=False):
                for fname in filenames:
                    try:
                        folder_bytes += os.path.getsize(os.path.join(dirpath, fname))
                    except OSError:
                        pass
                    if fname.lower() == "last.pt":
                        has_last_pt = True
            total_bytes += folder_bytes
            versions.append({"version_id": entry.name, "bytes": folder_bytes, "has_last_pt": has_last_pt})

        return {"total_bytes": total_bytes, "versions": versions}
