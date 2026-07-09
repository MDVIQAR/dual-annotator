"""
mlops/registry/settings_manager.py

Manages persistent registry settings stored at:
  %LOCALAPPDATA%\\DualAnnotator\\registry_settings.json

Structure under gdrive_root:
  {gdrive_root}/{model_type}/{project_name}/{project_id}/{variant}/{camera}/{version_id}/
"""

import os
import json
import copy


_DEFAULTS = {
    "gdrive_root":  "",
    "local_root":   "",
    "annotators": [
        {"name": "Mohammed Viqar", "initials": "mviqar"},
    ],
    # Staged dataset persisted between Data Prep and Training tabs.
    # Keys: path, model_type, project_name, project_id, variant, camera, dataset_info
    "staged_dataset": {},
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
        if not os.path.isfile(self._path):
            return copy.deepcopy(_DEFAULTS)
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                on_disk = json.load(fh)
            merged = copy.deepcopy(_DEFAULTS)
            merged.update(on_disk)
            return merged
        except (json.JSONDecodeError, OSError):
            return copy.deepcopy(_DEFAULTS)

    def save(self):
        os.makedirs(self._dir, exist_ok=True)
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)
        os.replace(tmp_path, self._path)

    # ------------------------------------------------------------------
    # GDrive root
    # ------------------------------------------------------------------

    def get_registry_root(self) -> str:
        """Return the Google Drive root path (the base for all hierarchy folders)."""
        return self._data.get("gdrive_root", "")

    def set_gdrive_root(self, path: str):
        self._data["gdrive_root"] = path
        self.save()

    # ------------------------------------------------------------------
    # Local root (primary fast storage, mirrors GDrive hierarchy)
    # ------------------------------------------------------------------

    def get_local_root(self) -> str:
        """Return the local registry root path, or '' if not configured."""
        return self._data.get("local_root", "")

    def set_local_root(self, path: str):
        self._data["local_root"] = path
        self.save()

    # ------------------------------------------------------------------
    # Effective root — gdrive_root if configured, else local_root.
    # This is the single source of truth for where the registry actually
    # lives; Google Drive is preferred when both are set (for the mirror
    # copy behavior in TrainWorker), but local-only setups work the same
    # way everywhere that reads the registry.
    # ------------------------------------------------------------------

    def get_effective_root(self) -> str:
        gdrive = self._data.get("gdrive_root", "")
        if gdrive:
            return gdrive
        return self._data.get("local_root", "")

    # ------------------------------------------------------------------
    # Hierarchy path builders
    # ------------------------------------------------------------------

    def get_camera_path(self, model_type: str, project_name: str,
                        project_id: str, variant: str, camera: str) -> str:
        """Return absolute path to the camera-level folder (parent of all versions)."""
        root = self.get_effective_root()
        if not root:
            return ""
        return os.path.join(root, model_type, project_name, project_id, variant, camera)

    def build_project_key(self, model_type: str, project_name: str,
                          project_id: str, variant: str, camera: str) -> str:
        """Return the slash-separated project key used in manifests and scanner."""
        return "/".join([model_type, project_name, project_id, variant, camera])

    # ------------------------------------------------------------------
    # Staged dataset (persisted between Data Prep and Training tabs)
    # ------------------------------------------------------------------

    def get_staged_dataset(self) -> dict:
        """Return the staged dataset info dict, or {} if none prepared."""
        return dict(self._data.get("staged_dataset", {}))

    def set_staged_dataset(self, info: dict):
        """Persist staged dataset info. Pass {} to clear."""
        self._data["staged_dataset"] = dict(info)
        self.save()

    def get_staged_dataset_path(self) -> str:
        """Return the local staging path, or '' if not set."""
        return self._data.get("staged_dataset", {}).get("path", "")

    # ------------------------------------------------------------------
    # Hierarchy scanning (reads live from disk)
    # ------------------------------------------------------------------

    @staticmethod
    def _list_subdirs(path: str) -> list:
        if not path or not os.path.isdir(path):
            return []
        try:
            return sorted(e.name for e in os.scandir(path) if e.is_dir()
                          and not e.name.startswith("_") and not e.name.startswith("."))
        except OSError:
            return []

    def scan_model_types(self) -> list:
        root = self.get_effective_root()
        return self._list_subdirs(root)

    def scan_project_names(self, model_type: str) -> list:
        root = self.get_effective_root()
        return self._list_subdirs(os.path.join(root, model_type))

    def scan_project_ids(self, model_type: str, project_name: str) -> list:
        root = self.get_effective_root()
        return self._list_subdirs(os.path.join(root, model_type, project_name))

    def scan_variants(self, model_type: str, project_name: str, project_id: str) -> list:
        root = self.get_effective_root()
        return self._list_subdirs(os.path.join(root, model_type, project_name, project_id))

    def scan_cameras(self, model_type: str, project_name: str,
                     project_id: str, variant: str) -> list:
        root = self.get_effective_root()
        return self._list_subdirs(
            os.path.join(root, model_type, project_name, project_id, variant))

    # ------------------------------------------------------------------
    # Annotator management
    # ------------------------------------------------------------------

    def get_annotators(self) -> list:
        return list(self._data.get("annotators", []))

    def add_annotator(self, name: str, initials: str):
        existing = {a["initials"] for a in self._data.get("annotators", [])}
        if initials in existing:
            return
        self._data.setdefault("annotators", []).append({"name": name, "initials": initials})
        self.save()

    def remove_annotator(self, initials: str):
        self._data["annotators"] = [
            a for a in self._data.get("annotators", []) if a["initials"] != initials
        ]
        self.save()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_configured(self) -> dict:
        root = self.get_effective_root()
        return {
            "gdrive": bool(root) and os.path.isdir(root),
        }

    # ------------------------------------------------------------------
    # Legacy compat — kept so existing registry_tab / training_tab don't crash
    # ------------------------------------------------------------------

    def get_datasets_root(self) -> str:
        return ""

    def get_projects(self) -> list:
        """Legacy: return top-level dirs of all model types combined."""
        root = self.get_effective_root()
        if not root or not os.path.isdir(root):
            return []
        projects = set()
        for mt in self._list_subdirs(root):
            for pn in self._list_subdirs(os.path.join(root, mt)):
                projects.add(pn)
        return sorted(projects)
