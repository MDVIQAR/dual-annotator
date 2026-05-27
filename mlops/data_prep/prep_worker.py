"""
mlops/data_prep/prep_worker.py

QThread background worker that orchestrates the full data preparation pipeline.

New behaviour:
  - Exports the dataset to a LOCAL staging folder in %LOCALAPPDATA%\DualAnnotator\staged_dataset\
  - Does NOT copy to GDrive at this stage — that happens when training starts
  - Persists the staging path + hierarchy info in RegistrySettings.staged_dataset
    so the Training tab can pre-fill itself automatically

Config keys expected:
  task_type, images_folder, labels_folder, test_folder (optional),
  train_pct, class_names (for YOLO),
  model_type, project_name, project_id, variant, camera
"""

import os
import shutil

from PyQt5.QtCore import QThread, pyqtSignal

from mlops.data_prep.preparator import DataPreparator
from mlops.registry.utils import get_dataset_hash_cached
from mlops.registry import RegistrySettings

_SEP = "=" * 52


def _staging_path() -> str:
    app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(app_data, "DualAnnotator", "staged_dataset")


class DataPrepWorker(QThread):
    """Runs the preparation pipeline in a background thread."""

    log           = pyqtSignal(str)        # one line to the console
    progress      = pyqtSignal(int)        # 0–100 overall progress
    hash_progress = pyqtSignal(int, int)   # (current_file, total_files)
    finished      = pyqtSignal(bool, dict) # (success, staged_info)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self):
        try:
            info = self._execute()
            self.finished.emit(True, info)
        except Exception as exc:
            self.log.emit(f"[ERROR] {exc}")
            self.finished.emit(False, {})

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _execute(self) -> dict:
        cfg = self._config
        task_type     = cfg["task_type"]
        images_folder = cfg["images_folder"]
        labels_folder = cfg["labels_folder"]
        test_folder   = cfg.get("test_folder", "") or None
        if test_folder and not os.path.isdir(test_folder):
            test_folder = None
        train_pct    = int(cfg.get("train_pct", 80))
        class_names  = cfg.get("class_names", [])

        # Hierarchy fields
        model_type   = cfg.get("model_type", task_type)
        project_name = cfg.get("project_name", "")
        project_id   = cfg.get("project_id", "")
        variant      = cfg.get("variant", "")
        camera       = cfg.get("camera", "")

        # ── Step 1: Banner ──────────────────────────────────────────────
        self.progress.emit(2)
        self.log.emit(_SEP)
        self.log.emit("  DualAnnotator — Data Preparation")
        self.log.emit(f"  Model    : {model_type.upper()}")
        self.log.emit(f"  Project  : {project_name} / {project_id}")
        self.log.emit(f"  Variant  : {variant}  Camera: {camera}")
        self.log.emit(f"  Task     : {task_type.upper()}")
        self.log.emit(_SEP)

        # ── Step 2: Scan pairs ──────────────────────────────────────────
        self.progress.emit(5)
        self.log.emit("Scanning image/label pairs...")
        scan = DataPreparator.scan_pairs(images_folder, labels_folder, task_type)
        self.log.emit(f"  Found {scan['total']} matched pairs")
        if scan["unpaired_images"]:
            self.log.emit(f"  ⚠ {len(scan['unpaired_images'])} images have no matching label")
        if scan["unpaired_labels"]:
            self.log.emit(f"  ⚠ {len(scan['unpaired_labels'])} labels have no matching image")
        if scan["total"] == 0:
            raise ValueError("No matched image/label pairs found. Aborting.")

        # ── Step 3: Split ───────────────────────────────────────────────
        self.progress.emit(15)
        self.log.emit(f"Splitting dataset ({train_pct}% train / {100 - train_pct}% val)...")
        split = DataPreparator.split(scan["pairs"], train_pct)
        self.log.emit(f"  Train: {len(split['train'])} pairs")
        self.log.emit(f"  Val  : {len(split['val'])} pairs")

        # ── Step 4: Clear staging area and export ───────────────────────
        self.progress.emit(20)
        staging = _staging_path()
        self.log.emit(f"Preparing staging area: {staging}")
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(staging)

        # ── Step 5: Export dataset layout ──────────────────────────────
        self.progress.emit(25)
        self.log.emit(f"Exporting {'YOLO' if task_type == 'yolo' else 'UNet'} layout...")
        self.progress.emit(35)
        self.log.emit("  Copying train images...")
        self.progress.emit(45)
        self.log.emit("  Copying val images...")
        self.progress.emit(55)
        self.log.emit("  Copying labels/masks...")
        self.progress.emit(65)
        if test_folder:
            self.log.emit("  Copying test images...")

        if task_type == "yolo":
            self.log.emit("  Writing data.yaml...")
            self.progress.emit(68)
            info = DataPreparator.export_yolo(
                split["train"], split["val"], test_folder, staging, class_names
            )
        else:
            info = DataPreparator.export_unet(
                split["train"], split["val"], test_folder, staging
            )

        info["project_name"] = project_name
        info["project_id"]   = project_id
        info["variant"]      = variant
        info["camera"]       = camera
        info["model_type"]   = model_type
        self.progress.emit(70)

        # ── Step 6: Hash ────────────────────────────────────────────────
        self.log.emit("Computing dataset hash...")

        def _hash_cb(current, total):
            self.hash_progress.emit(current, total)

        sha256 = get_dataset_hash_cached(staging, progress_callback=_hash_cb)
        self.progress.emit(95)

        # ── Step 7: Write dataset_info.json ────────────────────────────
        info["sha256"]       = sha256
        info["path"]         = staging
        info["class_names"]  = class_names
        info["num_classes"]  = len(class_names)
        DataPreparator.write_dataset_info(staging, info)
        self.log.emit("  dataset_info.json written")

        # ── Step 8: Persist to settings ────────────────────────────────
        staged_info = {
            "path":         staging,
            "model_type":   model_type,
            "project_name": project_name,
            "project_id":   project_id,
            "variant":      variant,
            "camera":       camera,
            "dataset_info": info,
        }
        RegistrySettings().set_staged_dataset(staged_info)

        # ── Step 9: Done ────────────────────────────────────────────────
        self.progress.emit(100)
        self.log.emit(_SEP)
        self.log.emit(f"  ✓ Dataset staged at: {staging}")
        self.log.emit(f"  SHA-256 : {sha256[:16]}...")
        self.log.emit("  Ready to use in the Training tab.")
        self.log.emit(_SEP)

        return staged_info
