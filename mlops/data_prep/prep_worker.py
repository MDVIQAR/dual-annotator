"""
mlops/data_prep/prep_worker.py

QThread background worker that orchestrates the full data preparation pipeline.
Emits log/progress signals so the UI stays responsive during long operations.
"""

import os
import shutil

from PyQt5.QtCore import QThread, pyqtSignal

from mlops.data_prep.preparator import DataPreparator
from mlops.registry.utils import get_dataset_hash_cached

_SEP = "=" * 52


class DataPrepWorker(QThread):
    """Runs the preparation pipeline in a background thread."""

    log           = pyqtSignal(str)        # one line to the console
    progress      = pyqtSignal(int)        # 0–100 overall progress
    hash_progress = pyqtSignal(int, int)   # (current_file, total_files)
    finished      = pyqtSignal(bool, dict) # (success, dataset_info)

    def __init__(self, config: dict, registry_root: str = "", parent=None):
        super().__init__(parent)
        self._config = config
        self._registry_root = registry_root

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
        task_type    = cfg["task_type"]
        images_folder = cfg["images_folder"]
        labels_folder = cfg["labels_folder"]
        test_folder   = cfg.get("test_folder", "") or None
        if test_folder and not os.path.isdir(test_folder):
            test_folder = None
        output_base  = cfg["output_base"]
        project_name = cfg["project_name"]
        category     = cfg.get("category", "").strip()
        train_pct    = int(cfg.get("train_pct", 80))
        class_names  = cfg.get("class_names", [])

        # ── Step 1: Banner ─────────────────────────────────────────────
        self.progress.emit(2)
        self.log.emit(_SEP)
        self.log.emit(f"  DualAnnotator — Data Preparation")
        self.log.emit(f"  Project  : {project_name}")
        if category:
            self.log.emit(f"  Category : {category}")
        self.log.emit(f"  Task     : {task_type.upper()}")
        self.log.emit(_SEP)

        # ── Step 2: Scan pairs ─────────────────────────────────────────
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

        # ── Step 3: Split ──────────────────────────────────────────────
        self.progress.emit(15)
        self.log.emit(f"Splitting dataset ({train_pct}% train / {100 - train_pct}% val)...")
        split = DataPreparator.split(scan["pairs"], train_pct)
        self.log.emit(f"  Train: {len(split['train'])} pairs")
        self.log.emit(f"  Val  : {len(split['val'])} pairs")

        # ── Step 4: Output folder ──────────────────────────────────────
        self.progress.emit(20)
        self.log.emit("Creating output folder...")
        output_folder = DataPreparator.make_output_folder(output_base, project_name)
        self.log.emit(f"  Output: {output_folder}")

        # ── Step 5: Export ─────────────────────────────────────────────
        self.progress.emit(25)
        self.log.emit(f"Exporting {'YOLO' if task_type == 'yolo' else 'UNet'} layout...")

        self.log.emit("  Copying train images...")
        self.progress.emit(35)
        self.log.emit("  Copying val images...")
        self.progress.emit(45)
        self.log.emit("  Copying labels/masks...")
        self.progress.emit(55)
        if test_folder:
            self.log.emit("  Copying test images...")
        self.progress.emit(65)

        if task_type == "yolo":
            self.log.emit("  Writing data.yaml...")
            self.progress.emit(68)
            info = DataPreparator.export_yolo(
                split["train"], split["val"], test_folder, output_folder, class_names
            )
        else:
            info = DataPreparator.export_unet(
                split["train"], split["val"], test_folder, output_folder
            )

        info["project"]  = project_name
        info["category"] = category
        self.progress.emit(70)

        # ── Step 6: Hash ───────────────────────────────────────────────
        self.log.emit("Computing dataset hash...")

        def _hash_cb(current, total):
            self.hash_progress.emit(current, total)

        sha256 = get_dataset_hash_cached(output_folder, progress_callback=_hash_cb)
        self.progress.emit(95)

        # ── Step 7: Write dataset_info.json ────────────────────────────
        info["sha256"] = sha256
        DataPreparator.write_dataset_info(output_folder, info)
        self.log.emit("  dataset_info.json written")

        # ── Step 8: Copy to GDrive ─────────────────────────────────────────────
        gdrive_dataset_path = ""
        if self._registry_root:
            project_folder = os.path.join(self._registry_root, project_name)
            gdrive_dest    = os.path.join(project_folder, "dataset")
            try:
                os.makedirs(project_folder, exist_ok=True)
                self.log.emit("Copying dataset to Google Drive...")

                # Collect all files to copy
                all_files = []
                for dirpath, _, filenames in os.walk(output_folder):
                    for fname in filenames:
                        all_files.append(os.path.join(dirpath, fname))

                total_files = len(all_files)

                # Issue 13 fix: Atomic copy via .tmp intermediary
                # Copy to a temp folder first, then rename — so an interrupted
                # copy never leaves the destination in a broken state.
                tmp_dest = gdrive_dest + ".tmp"
                if os.path.exists(tmp_dest):
                    shutil.rmtree(tmp_dest)
                os.makedirs(tmp_dest)

                for i, src in enumerate(all_files):
                    rel = os.path.relpath(src, output_folder)
                    dst = os.path.join(tmp_dest, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    pct = 85 + int((i + 1) / total_files * 12) if total_files else 97
                    self.progress.emit(pct)

                # Rewrite dataset_info.json in GDrive copy with GDrive path
                info_gdrive = dict(info)
                info_gdrive["output_folder"] = gdrive_dest.replace("\\", "/")
                DataPreparator.write_dataset_info(tmp_dest, info_gdrive)

                # For YOLO: rewrite data.yaml with GDrive path
                if task_type == "yolo":
                    yaml_path = os.path.join(tmp_dest, "data.yaml")
                    if os.path.isfile(yaml_path):
                        with open(yaml_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        gdrive_abs = gdrive_dest.replace("\\", "/")
                        new_lines = []
                        for line in content.splitlines():
                            new_lines.append(f"path: {gdrive_abs}" if line.startswith("path:") else line)
                        with open(yaml_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(new_lines) + "\n")

                # Issue 13 fix: Atomic swap — remove old, rename tmp to final
                if os.path.isdir(gdrive_dest):
                    shutil.rmtree(gdrive_dest)
                os.rename(tmp_dest, gdrive_dest)

                gdrive_dataset_path = gdrive_dest
                self.log.emit(f"  ✓ GDrive copy: {gdrive_dest}")
            except Exception as copy_err:
                self.log.emit(f"  ⚠ GDrive copy failed: {copy_err}")

        info["gdrive_dataset_path"] = gdrive_dataset_path

        # ── Step 9: Done ───────────────────────────────────────────────
        self.progress.emit(100)
        self.log.emit(_SEP)
        self.log.emit(f"  ✓ Dataset ready at: {output_folder}")
        self.log.emit(f"  SHA-256 : {sha256[:16]}...")
        self.log.emit(_SEP)

        return info
