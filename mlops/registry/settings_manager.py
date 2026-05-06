"""
mlops/registry/settings_manager.py

Manages persistent registry settings stored at:
  %LOCALAPPDATA%\\DualAnnotator\\registry_settings.json

Intentionally uses a separate file from the legacy mlops_settings.json
to avoid any conflict with the existing MLOps system.
"""

import os
import json
import copy


_DEFAULTS = {
    "gdrive_root": "",
    "datasets_root": "",
    "annotators": [
        {"name": "Mohammed Viqar", "initials": "mviqar"},
    ],
    "last_project": "",
    "registry_folder_name": "DualAnnotator_Registry",
}


class RegistrySettings:
    """Loads, mutates, and persists registry configuration as JSON on disk."""

    def __init__(self):
        app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        self._dir = os.path.join(app_data, "DualAnnotator")
        self._path = os.path.join(self._dir, "registry_settings.json")
        self._data = self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> dict:
        """Read settings from disk. Returns defaults if the file is missing or malformed."""
        if not os.path.isfile(self._path):
            return copy.deepcopy(_DEFAULTS)
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                on_disk = json.load(fh)
            # Merge with defaults so any new keys are always present
            merged = copy.deepcopy(_DEFAULTS)
            merged.update(on_disk)
            return merged
        except (json.JSONDecodeError, OSError):
            return copy.deepcopy(_DEFAULTS)

    def save(self):
        """Write current settings to disk, creating the directory if needed."""
        os.makedirs(self._dir, exist_ok=True)
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)
        os.replace(tmp_path, self._path)

    # ------------------------------------------------------------------
    # Root paths
    # ------------------------------------------------------------------

    def get_registry_root(self) -> str:
        """Return the full path to the GDrive registry folder, or '' if unconfigured."""
        gdrive = self._data.get("gdrive_root", "")
        folder = self._data.get("registry_folder_name", "DualAnnotator_Registry")
        if not gdrive:
            return ""
        return os.path.join(gdrive, folder)

    def get_datasets_root(self) -> str:
        """Return the datasets root path, or '' if unconfigured."""
        return self._data.get("datasets_root", "")

    def set_gdrive_root(self, path: str):
        """Set the Google Drive root path and persist."""
        self._data["gdrive_root"] = path
        self.save()

    def set_datasets_root(self, path: str):
        """Set the team datasets root path and persist."""
        self._data["datasets_root"] = path
        self.save()

    def set_last_project(self, project: str):
        """Remember the last active project and persist."""
        self._data["last_project"] = project
        self.save()

    # ------------------------------------------------------------------
    # Annotator management
    # ------------------------------------------------------------------

    def get_annotators(self) -> list:
        """Return the list of annotator dicts (name, initials)."""
        return list(self._data.get("annotators", []))

    def add_annotator(self, name: str, initials: str):
        """Add a new annotator if initials are not already registered, then persist."""
        existing_initials = {a["initials"] for a in self._data.get("annotators", [])}
        if initials in existing_initials:
            return
        self._data.setdefault("annotators", []).append({"name": name, "initials": initials})
        self.save()

    def remove_annotator(self, initials: str):
        """Remove the annotator with the given initials, then persist."""
        self._data["annotators"] = [
            a for a in self._data.get("annotators", []) if a["initials"] != initials
        ]
        self.save()

    # ------------------------------------------------------------------
    # Project discovery
    # ------------------------------------------------------------------

    def get_projects(self) -> list:
        """Return sorted names of top-level project folders inside the registry root."""
        root = self.get_registry_root()
        if not root or not os.path.isdir(root):
            return []
        try:
            names = [
                entry.name
                for entry in os.scandir(root)
                if entry.is_dir()
            ]
            return sorted(names)
        except OSError:
            return []

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_configured(self) -> dict:
        """Return dict indicating whether GDrive and datasets roots are set and exist on disk."""
        gdrive = self._data.get("gdrive_root", "")
        datasets = self._data.get("datasets_root", "")
        return {
            "gdrive": bool(gdrive) and os.path.isdir(gdrive),
            "datasets": bool(datasets) and os.path.isdir(datasets),
        }
