"""
mlops/registry/scanner.py

Scans the new 5-level Google Drive registry hierarchy:
  {gdrive_root}/{model_type}/{project_name}/{project_id}/{variant}/{camera}/{version_id}/

The "project" key used throughout the API is a forward-slash path:
  e.g.  "unet/SnipeX/SLXCAP/CRC/cam0"

This matches what is stored in manifest["project"] so that
  scanner.get_version(summary["project"], summary["version_id"])
always resolves correctly.
"""

import os
import re

from .manifest import ManifestReader


def _subdirs(path: str) -> list:
    if not path or not os.path.isdir(path):
        return []
    try:
        return sorted(
            e.name for e in os.scandir(path)
            if e.is_dir() and not e.name.startswith("_") and not e.name.startswith(".")
        )
    except OSError:
        return []


def _has_manifest(folder: str) -> bool:
    return os.path.isfile(os.path.join(folder, "manifest.json"))


class RegistryScanner:
    """
    Reads the GDrive registry folder and returns structured version summaries.

    All methods accept `project` as a forward-slash key like "unet/SnipeX/SLXCAP/CRC/cam0".
    """

    def __init__(self, gdrive_root: str):
        self._root = gdrive_root

    # ------------------------------------------------------------------
    # Hierarchy navigation
    # ------------------------------------------------------------------

    def get_model_types(self) -> list:
        return _subdirs(self._root)

    def get_project_names(self, model_type: str) -> list:
        return _subdirs(os.path.join(self._root, model_type))

    def get_project_ids(self, model_type: str, project_name: str) -> list:
        return _subdirs(os.path.join(self._root, model_type, project_name))

    def get_variants(self, model_type: str, project_name: str, project_id: str) -> list:
        return _subdirs(os.path.join(self._root, model_type, project_name, project_id))

    def get_cameras(self, model_type: str, project_name: str,
                    project_id: str, variant: str) -> list:
        return _subdirs(os.path.join(self._root, model_type, project_name, project_id, variant))

    # ------------------------------------------------------------------
    # Project-level (camera-path) API
    # ------------------------------------------------------------------

    def get_projects(self) -> list:
        """
        Return all camera-level project keys (forward-slash paths) that contain
        at least one version folder with a manifest.json.
        e.g. ["unet/SnipeX/SLXCAP/CRC/cam0", "yolo/CX/CXPRO/Standard/cam0"]
        """
        if not self._root or not os.path.isdir(self._root):
            return []
        paths = []
        for mt in _subdirs(self._root):
            for pn in _subdirs(os.path.join(self._root, mt)):
                for pid in _subdirs(os.path.join(self._root, mt, pn)):
                    for var in _subdirs(os.path.join(self._root, mt, pn, pid)):
                        for cam in _subdirs(os.path.join(self._root, mt, pn, pid, var)):
                            cam_abs = os.path.join(self._root, mt, pn, pid, var, cam)
                            if any(_has_manifest(os.path.join(cam_abs, v))
                                   for v in _subdirs(cam_abs)):
                                paths.append(f"{mt}/{pn}/{pid}/{var}/{cam}")
        return sorted(paths)

    def get_versions(self, project: str) -> list:
        """
        Scan {gdrive_root}/{project}/ for version subfolders.
        project is a forward-slash key like "unet/SnipeX/SLXCAP/CRC/cam0".
        Returns summaries sorted by started_at descending (newest first).
        """
        project_dir = os.path.join(self._root, *project.split("/"))
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
        """Return the full manifest dict for a specific version, or None."""
        version_dir = os.path.join(self._root, *project.split("/"), version_id)
        if not os.path.isdir(version_dir):
            return None
        return ManifestReader(version_dir).read()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    _CONFLICT_PATTERN = re.compile(r".+ \(\d+\)\..+")

    def detect_conflict_files(self, project: str) -> list:
        project_dir = os.path.join(self._root, *project.split("/"))
        if not os.path.isdir(project_dir):
            return []
        conflicts = []
        for dirpath, _, filenames in os.walk(project_dir):
            for fname in filenames:
                if self._CONFLICT_PATTERN.match(fname):
                    conflicts.append(os.path.join(dirpath, fname))
        return conflicts

    def get_storage_stats(self, project: str) -> dict:
        """Storage usage for all versions inside a camera-level project folder."""
        project_dir = os.path.join(self._root, *project.split("/"))
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
            versions.append({
                "version_id": entry.name,
                "bytes": folder_bytes,
                "has_last_pt": has_last_pt,
            })

        return {"total_bytes": total_bytes, "versions": versions}
