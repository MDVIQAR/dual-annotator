"""
mlops/registry/manifest.py

Handles creation and mutation of manifest.json files stored inside each model version folder.
The manifest is the single source of truth for a trained model's metadata, status, and artifacts.

All writes are atomic: data is written to a .tmp file first, then moved with os.replace()
to prevent corruption if the process crashes mid-write.
"""

import os
import json
from datetime import datetime


def generate_version_id(annotator_initials: str) -> str:
    """
    Return a unique version ID string in the format:
      YYYY-MM-DD_{annotator_initials}_HHMMSS

    Example: '2026-05-06_mviqar_143022'
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")
    return f"{date_str}_{annotator_initials}_{time_str}"


# ---------------------------------------------------------------------------
# Default manifest skeleton
# ---------------------------------------------------------------------------

def _build_default_manifest() -> dict:
    return {
        "version_id": "",
        "model_type": "",
        "status": "training",
        "sync_complete": False,

        "annotator": "",
        # project = full forward-slash camera key: "unet/SnipeX/SLXCAP/CRC/cam0"
        "project": "",
        "project_name": "",
        "project_id": "",
        "variant": "",
        "camera": "",
        "commit_message": "",
        "started_at": None,
        "finished_at": None,

        "dataset": {
            "absolute_path": "",
            "relative_path": "",
            "sha256": "",
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "classes": [],
            "num_classes": 0,
        },

        "hyperparams": {
            "architecture": "",
            "encoder": "",
            "encoder_weights": "imagenet",
            "in_channels": 1,
            "out_classes": 1,
            "epochs": 100,
            "batch_size": 8,
            "image_width": 640,
            "image_height": 512,
            "learning_rate": 0.001,
            "device": "cpu",
        },

        "metrics": {
            "best_val_loss": None,
            "best_epoch": None,
            "final_train_loss": None,
        },

        "artifacts": {
            "weights": None,
            "onnx": None,
            "curves_plot": None,
            "sample_preds": None,
        },

        "onnx_config": {
            "exported": False,
            "opset_version": 11,
            "input_shape": None,
            "exported_at": None,
        },

        "crash_info": None,
        "augmentations": {},
    }


# ---------------------------------------------------------------------------
# ManifestWriter
# ---------------------------------------------------------------------------

class ManifestWriter:
    """Creates and mutates the manifest.json inside a single model version folder."""

    def __init__(self, version_folder: str):
        self._folder = version_folder
        self._path = os.path.join(version_folder, "manifest.json")

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        version_id: str,
        model_type: str,
        annotator: str,
        project: str,
        project_name: str = "",
        project_id: str = "",
        variant: str = "",
        camera: str = "",
        commit_message: str,
        dataset_info: dict,
        hyperparams: dict,
        augmentations: dict = None,
    ) -> dict:
        """
        Write the initial manifest.json at the start of training.
        status is set to 'training', sync_complete to False.
        Returns the full manifest dict.
        """
        manifest = _build_default_manifest()
        manifest["version_id"] = version_id
        manifest["model_type"] = model_type
        manifest["annotator"] = annotator
        manifest["project"] = project
        manifest["project_name"] = project_name
        manifest["project_id"] = project_id
        manifest["variant"] = variant
        manifest["camera"] = camera
        manifest["commit_message"] = commit_message
        manifest["started_at"] = datetime.now().isoformat(timespec="seconds")

        # Merge caller-supplied dataset and hyperparams fields
        manifest["dataset"].update(dataset_info)
        manifest["hyperparams"].update(hyperparams)
        if augmentations:
            manifest["augmentations"] = augmentations

        self._write(manifest)
        return manifest

    # ------------------------------------------------------------------
    # Status updates
    # ------------------------------------------------------------------

    def update_status(self, status: str):
        """Set manifest status. Valid: 'preparing', 'training', 'completed', 'crashed', 'cancelled'."""
        data = self._read()
        data["status"] = status
        self._write(data)

    def update_metrics(self, best_val_loss: float, best_epoch: int, final_train_loss: float):
        """Fill the metrics block in the manifest."""
        data = self._read()
        data["metrics"]["best_val_loss"] = best_val_loss
        data["metrics"]["best_epoch"] = best_epoch
        data["metrics"]["final_train_loss"] = final_train_loss
        self._write(data)

    def update_artifacts(self, weights=None, onnx=None, curves_plot=None, sample_preds=None):
        """Fill non-None artifact fields in the manifest."""
        data = self._read()
        if weights is not None:
            data["artifacts"]["weights"] = weights
        if onnx is not None:
            data["artifacts"]["onnx"] = onnx
        if curves_plot is not None:
            data["artifacts"]["curves_plot"] = curves_plot
        if sample_preds is not None:
            data["artifacts"]["sample_preds"] = sample_preds
        self._write(data)

    def update_onnx(self, input_shape: list, opset_version: int = 11):
        """Mark ONNX export as complete in the manifest."""
        data = self._read()
        data["onnx_config"]["exported"] = True
        data["onnx_config"]["opset_version"] = opset_version
        data["onnx_config"]["input_shape"] = input_shape
        data["onnx_config"]["exported_at"] = datetime.now().isoformat(timespec="seconds")
        data["artifacts"]["onnx"] = "best.onnx"
        self._write(data)

    # ------------------------------------------------------------------
    # Terminal state transitions
    # ------------------------------------------------------------------

    def finalize(self):
        """
        Called as the absolute last step after successful training.
        Sets status='completed', records finished_at, then sets sync_complete=True
        as the very final write so GDrive-syncing teammates only see fully synced versions.
        """
        data = self._read()
        data["status"] = "completed"
        data["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self._write(data)

        # sync_complete is set in a separate write so it is the last field to land on GDrive
        data["sync_complete"] = True
        self._write(data)

    def mark_crashed(self, reason: str, hint: str = ""):
        """
        Record a crash. sync_complete is intentionally left False.
        Crashed versions appear in the registry with a crash badge but are not valid for deployment.
        """
        data = self._read()
        data["status"] = "crashed"
        data["finished_at"] = datetime.now().isoformat(timespec="seconds")
        data["crash_info"] = {"reason": reason, "hint": hint}
        self._write(data)

    def mark_cancelled(self):
        """Record a user-initiated cancellation."""
        data = self._read()
        data["status"] = "cancelled"
        data["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self._write(data)

    def mark_partial(self):
        """
        Record a graceful early stop where at least one epoch completed and
        best.pt was saved. The run is usable for fine-tuning.
        """
        data = self._read()
        data["status"] = "partial"
        data["finished_at"] = datetime.now().isoformat(timespec="seconds")
        data["sync_complete"] = True
        self._write(data)

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _read(self) -> dict:
        """Read and parse manifest.json. Raises FileNotFoundError if missing."""
        if not os.path.isfile(self._path):
            raise FileNotFoundError(f"manifest.json not found in: {self._folder}")
        with open(self._path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, data: dict):
        """Atomically write data to manifest.json via a .tmp intermediary."""
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, self._path)


# ---------------------------------------------------------------------------
# ManifestReader
# ---------------------------------------------------------------------------

class ManifestReader:
    """Read-only access to a manifest.json. Used by the registry scanner and UI."""

    def __init__(self, version_folder: str):
        self._folder = version_folder
        self._path = os.path.join(version_folder, "manifest.json")

    def read(self) -> dict | None:
        """Return the full manifest dict, or None if the file does not exist."""
        if not os.path.isfile(self._path):
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

    def is_sync_complete(self) -> bool:
        """Return True only if sync_complete is explicitly set to True in the manifest."""
        data = self.read()
        if data is None:
            return False
        return bool(data.get("sync_complete", False))

    def get_status(self) -> str:
        """Return the manifest status string, or 'unknown' if unreadable."""
        data = self.read()
        if data is None:
            return "unknown"
        return data.get("status", "unknown")

    def get_model_type(self) -> str:
        """Return the model_type field, or 'unknown' if unreadable."""
        data = self.read()
        if data is None:
            return "unknown"
        return data.get("model_type", "unknown")

    def get_dataset_path(self, datasets_root: str = "") -> str:
        """
        Resolve the dataset path with the following priority:
          1. absolute_path if it exists on the current machine
          2. {project_folder}/dataset/ — the GDrive copy inside the registry
          3. {datasets_root}/{relative_path} — legacy fallback
          4. absolute_path as-is (caller handles missing)
        """
        data = self.read()
        if data is None:
            return ""

        absolute_path = data.get("dataset", {}).get("absolute_path", "")
        relative_path = data.get("dataset", {}).get("relative_path", "")

        if absolute_path and os.path.isdir(absolute_path):
            return absolute_path

        # GDrive copy: dataset/ lives next to the version folder inside the project folder
        project_folder  = os.path.dirname(self._folder)
        gdrive_dataset  = os.path.join(project_folder, "dataset")
        if os.path.isdir(gdrive_dataset):
            return gdrive_dataset

        if datasets_root and relative_path:
            candidate = os.path.join(datasets_root, relative_path)
            if os.path.isdir(candidate):
                return candidate

        return absolute_path

    def get_onnx_path(self) -> str:
        """
        Return the absolute path to best.onnx if ONNX export is marked complete
        and the file actually exists on disk. Otherwise return ''.
        """
        data = self.read()
        if data is None:
            return ""

        if not data.get("onnx_config", {}).get("exported", False):
            return ""

        candidate = os.path.join(self._folder, "best.onnx")
        return candidate if os.path.isfile(candidate) else ""

    def get_weights_path(self) -> str:
        """
        Return the full path to the weights file if artifacts.weights is set
        and the file actually exists on disk. Otherwise return ''.
        """
        data = self.read()
        if data is None:
            return ""

        weights = data.get("artifacts", {}).get("weights", None)
        if not weights:
            return ""

        # weights may be stored as an absolute path or a filename relative to the folder
        candidate = weights if os.path.isabs(weights) else os.path.join(self._folder, weights)
        return candidate if os.path.isfile(candidate) else ""

    def summary(self) -> dict:
        """
        Return a lightweight dict for the registry browser list.
        All keys are always present; missing values default to None / False / ''.
        Never raises KeyError.
        """
        data = self.read() or {}
        hyperparams = data.get("hyperparams", {})
        metrics = data.get("metrics", {})
        onnx_cfg = data.get("onnx_config", {})

        return {
            "version_id":     data.get("version_id", ""),
            "version_folder": self._folder,
            "model_type":     data.get("model_type", ""),
            "status":         data.get("status", "unknown"),
            "sync_complete":  bool(data.get("sync_complete", False)),
            "annotator":      data.get("annotator", ""),
            "project":        data.get("project", ""),
            "project_name":   data.get("project_name", ""),
            "project_id":     data.get("project_id", ""),
            "variant":        data.get("variant", ""),
            "camera":         data.get("camera", ""),
            "commit_message": data.get("commit_message", ""),
            "started_at":     data.get("started_at", ""),
            "architecture":   hyperparams.get("architecture", ""),
            "encoder":        hyperparams.get("encoder", ""),
            "epochs":         hyperparams.get("epochs", None),
            "best_val_loss":  metrics.get("best_val_loss", None),
            "best_epoch":     metrics.get("best_epoch", None),
            "has_onnx":       bool(onnx_cfg.get("exported", False)),
            "has_weights":    bool(data.get("artifacts", {}).get("weights", None)),
            "crash_info":     data.get("crash_info", None),
        }
