# mlops/data_aggregator.py
# Phase 1: Raw Pool Ingestion, Duplicate Handling, Train/Val Split, YOLO + UNet Structure Builder
# Standalone module — no imports from gui/ or core/. Pure logic layer.
#
# KEY DESIGN DECISIONS (based on team workflow):
#   - Labeled pool  → shuffled → split into TRAIN + VAL only
#   - Test images   → come from a SEPARATE dedicated folder (no labels needed, held blind)
#   - Test set never has a labels/ or masks/ folder — only images/test/
#   - Supports both FLAT format (images + .txt in same folder) and
#     STRUCTURED format (images/ + labels/ or masks/ as subfolders)

import os
import re
import json
import shutil
import hashlib
import random
import uuid
from datetime import datetime
from PyQt5.QtCore import pyqtSignal, QThread


# ---------------------------------------------------------------------------
# Registry Folder Structure:
#
#   <Registry Root>/
#   ├── YOLO/
#   │   └── <ProjectName>/
#   │       └── v1/                       ← auto-incremented version
#   │           ├── hyperparams.json
#   │           ├── best.pt               ← written by trainer (Phase 2)
#   │           ├── best.onnx             ← written by trainer (Phase 2)
#   │           ├── results.png           ← written by trainer (Phase 2)
#   │           └── data/
#   │               ├── images/
#   │               │   ├── train/
#   │               │   ├── val/
#   │               │   └── test/         ← IMAGES ONLY — no labels folder
#   │               ├── labels/
#   │               │   ├── train/
#   │               │   └── val/          ← no test labels (held blind)
#   │               └── data.yaml
#   └── UNet/
#       └── <ProjectName>/
#           └── v1/
#               ├── hyperparams.json
#               └── data/
#                   ├── images/
#                   │   ├── train/
#                   │   ├── val/
#                   │   └── test/         ← IMAGES ONLY — no masks folder
#                   └── masks/
#                       ├── train/
#                       └── val/          ← no test masks (held blind)
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_image(filename: str) -> bool:
    """Return True if the file extension is a recognised image format."""
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


def compute_dataset_hash(folder_path: str) -> str:
    """
    Fast 'Quick Fingerprint': sha256 of the sorted list of (relative_path + filesize).
    Instant for thousands of files. Provides 99% of the integrity-checking we need.
    """
    entries = []
    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                size = 0
            rel = os.path.relpath(fpath, folder_path)
            entries.append(f"{rel}::{size}")
    payload = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def detect_format(labels_pool_path: str) -> str:
    """
    Search the specified labels pool folder to detect if it contains YOLO (.txt)
    or UNet (.png masks).

    Returns: "yolo" | "unet" | "mixed" | "unknown"
    """
    has_txt  = False
    has_mask = False

    for root, dirs, files in os.walk(labels_pool_path):
        folder_name = os.path.basename(root).lower()

        for f in files:
            fl = f.lower()
            if fl.endswith(".txt"):
                has_txt = True
            # For UNet, ANY image in the dedicated labels folder is assumed to be a mask
            if _is_image(f):
                has_mask = True
                
            if has_txt and has_mask:
                break

    if has_txt and has_mask:
        return "mixed"
    if has_txt:
        return "yolo"
    if has_mask:
        return "unet"
    return "unknown"


def next_version_folder(task_type: str, project_name: str, registry_root: str) -> str:
    """
    Returns the next auto-incremented version folder path without creating it.
    Example: G:/Models/YOLO/MyProject/v3
    """
    task_folder     = "YOLO" if task_type.lower() == "yolo" else "UNet"
    project_folder  = os.path.join(registry_root, task_folder, project_name)
    os.makedirs(project_folder, exist_ok=True)

    existing     = [
        d for d in os.listdir(project_folder)
        if os.path.isdir(os.path.join(project_folder, d)) and re.match(r"^v\d+$", d)
    ]
    existing_nums = [int(d[1:]) for d in existing]
    next_num      = max(existing_nums, default=0) + 1
    return os.path.join(project_folder, f"v{next_num}")


# ---------------------------------------------------------------------------
# Pair Collection — matches across separate Image and Label folders
# ---------------------------------------------------------------------------

def _collect_image_label_pairs_yolo(images_path: str, labels_path: str, log_fn=None) -> list:
    """
    Collect (image_path, label_path) pairs for YOLO.
    Looks for matching file stems between the disjoint images and labels folders.
    """
    pairs = []
    
    # 1. Build a map of all image stems inside images_path
    image_map = {}
    for root, dirs, files in os.walk(images_path):
        for f in files:
            if _is_image(f):
                stem = os.path.splitext(f)[0]
                image_map[stem] = os.path.join(root, f)

    # 2. Iterate through all labels, looking for a matching image stem
    for root, dirs, files in os.walk(labels_path):
        for f in sorted(files):
            if f.lower().endswith(".txt"):
                stem = os.path.splitext(f)[0]
                if stem in image_map:
                    pairs.append((image_map[stem], os.path.join(root, f)))
                elif log_fn:
                    log_fn(f"   [skip] No source image found for label: {f}")

    return pairs


def _collect_image_mask_pairs_unet(images_path: str, labels_path: str, log_fn=None) -> list:
    """
    Collect (image_path, mask_path) pairs for UNet.
    Searches for image stems matching across the two separated folders.
    Handles masks with or without a '_mask' appended to their stem.
    """
    pairs = []
    
    # 1. Build image map
    image_map = {}
    for root, dirs, files in os.walk(images_path):
        for f in files:
            if _is_image(f):
                stem = os.path.splitext(f)[0]
                image_map[stem] = os.path.join(root, f)

    # 2. Match masks
    for root, dirs, files in os.walk(labels_path):
        for f in sorted(files):
            if _is_image(f):
                stem = os.path.splitext(f)[0]
                mask_path = os.path.join(root, f)
                
                # Check direct match first
                if stem in image_map:
                    pairs.append((image_map[stem], mask_path))
                # Check if mask stem has '_mask' suffix appended to the base image name
                elif stem.endswith("_mask") and stem[:-5] in image_map:
                    pairs.append((image_map[stem[:-5]], mask_path))
                elif log_fn:
                    log_fn(f"   [skip] No source image found for mask: {f}")

    return pairs


def _collect_test_images(test_images_path: str, log_fn=None) -> list:
    """
    Collect bare image paths from the dedicated test images folder.
    No labels or masks needed — test set is held blind.

    Returns: list of (image_path, new_stem) tuples  (new_stem = original filename stem)
    """
    images = []
    seen_stems = {}

    for root, dirs, files in os.walk(test_images_path):
        dirs.sort()
        for fname in sorted(files):
            if not _is_image(fname):
                continue
            stem     = os.path.splitext(fname)[0]
            img_path = os.path.join(root, fname)

            if stem in seen_stems:
                short_id = str(uuid.uuid4())[:8]
                new_stem = f"{stem}_{short_id}"
                if log_fn:
                    log_fn(f"   [test] Duplicate renamed: {fname} -> {new_stem}")
            else:
                new_stem = stem
                seen_stems[stem] = True

            images.append((img_path, new_stem))

    return images


def _resolve_duplicates(pairs: list, log_fn=None) -> tuple:
    """
    Detect duplicate image filenames in the labeled pairs and rename by appending
    a short UUID suffix.  Returns (resolved_triples, rename_count)
    where each triple is (img_path, label_path, unique_stem).
    """
    seen_stems   = {}
    resolved     = []
    rename_count = 0

    for img_path, label_path in pairs:
        img_filename = os.path.basename(img_path)
        stem, img_ext = os.path.splitext(img_filename)

        if stem in seen_stems:
            short_id = str(uuid.uuid4())[:8]
            new_stem = f"{stem}_{short_id}"
            rename_count += 1
            if log_fn:
                log_fn(f"   [dup] '{img_filename}' -> '{new_stem}{img_ext}'")
            resolved.append((img_path, label_path, new_stem))
        else:
            seen_stems[stem] = True
            resolved.append((img_path, label_path, stem))

    return resolved, rename_count


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class AggregationResult:
    """Lightweight result object returned after aggregation completes."""
    def __init__(self):
        self.success: bool        = False
        self.error: str           = ""
        self.format_detected: str = ""
        self.total_labeled: int   = 0
        self.duplicates_renamed: int = 0
        self.train_count: int     = 0
        self.val_count: int       = 0
        self.test_count: int      = 0      # from separate test folder (images only)
        self.dataset_hash: str    = ""
        self.version_folder: str  = ""
        self.data_folder: str     = ""
        self.yaml_path: str       = ""


# ---------------------------------------------------------------------------
# QThread Worker
# ---------------------------------------------------------------------------

class AggregatorWorker(QThread):
    """
    Runs the full aggregation pipeline in a background thread.
    Emits real-time log messages and progress.
    """
    log      = pyqtSignal(str)     # human-readable log line
    progress = pyqtSignal(int)     # 0–100
    finished = pyqtSignal(object)  # AggregationResult

    def __init__(self,
                 raw_images_path:  str,
                 raw_labels_path:  str,
                 test_images_path: str,      # "" if not provided
                 registry_root:    str,
                 project_name:     str,
                 task_type:        str,      # "yolo" | "unet"
                 class_names:      list,     # list[str] — for data.yaml
                 train_pct:        int,
                 val_pct:          int,
                 parent=None):
        super().__init__(parent)
        self.raw_images_path  = raw_images_path
        self.raw_labels_path  = raw_labels_path
        self.test_images_path = test_images_path.strip() if test_images_path else ""
        self.registry_root    = registry_root
        self.project_name     = project_name
        self.task_type        = task_type
        self.class_names      = class_names
        self.train_pct        = train_pct
        self.val_pct          = val_pct
        self._running         = True

    # -----------------------------------------------------------------------
    def run(self):
        result = AggregationResult()
        try:
            self._run_pipeline(result)
        except Exception as exc:
            result.success = False
            result.error   = str(exc)
            self.log.emit(f"[ERROR] {exc}")
        finally:
            self.finished.emit(result)

    def stop(self):
        self._running = False

    # -----------------------------------------------------------------------
    def _run_pipeline(self, result: AggregationResult):
        self.log.emit("=" * 52)
        self.log.emit(f"  Aggregation started — project: '{self.project_name}'")
        self.log.emit(f"  Task   : {self.task_type.upper()}")
        self.log.emit(f"  Images : {self.raw_images_path}")
        self.log.emit(f"  Labels : {self.raw_labels_path}")
        if self.test_images_path:
            self.log.emit(f"  Test   : {self.test_images_path}")
        else:
            self.log.emit("  Test   : (none — no test folder provided)")
        self.log.emit("=" * 52)
        self.progress.emit(5)

        task = self.task_type.lower()

        # ── Step 1: Detect labels format ─────────────────────────────────
        detected = detect_format(self.raw_labels_path)
        result.format_detected = detected
        self.log.emit(f"\n[1] Format detected in pool: {detected.upper()}")
        
        # Override UI task if we detected exactly yolo or unet
        if detected in ["yolo", "unet"] and task != detected:
            self.log.emit(f"    [!] Auto-correcting task type from {task.upper()} to {detected.upper()}")
            task = detected
        
        self.progress.emit(10)

        # ── Step 2: Collect labeled pairs ─────────────────────────────
        self.log.emit("\n[2] Matching image+label pairs across separate folders...")
        if task == "yolo":
            raw_pairs = _collect_image_label_pairs_yolo(self.raw_images_path, self.raw_labels_path, self.log.emit)
        else:
            raw_pairs = _collect_image_mask_pairs_unet(self.raw_images_path, self.raw_labels_path, self.log.emit)

        if not raw_pairs:
            raise ValueError(
                "No matched pairs found. Ensure that filenames in your "
                "Images Folder and Labels/Masks Folder have exactly matching names."
            )

        self.log.emit(f"   Found {len(raw_pairs)} labeled pairs.")
        result.total_labeled = len(raw_pairs)
        self.progress.emit(25)

        # ── Step 3: Resolve duplicates in labeled pool ────────────────
        self.log.emit("\n[3] Checking for duplicate filenames in labeled pool...")
        resolved_pairs, rename_count = _resolve_duplicates(raw_pairs, self.log.emit)
        result.duplicates_renamed = rename_count
        self.log.emit(f"   {rename_count} duplicate(s) renamed." if rename_count else "   No duplicates found.")
        self.progress.emit(38)

        # ── Step 4: Shuffle + split labeled pool into train/val only ──
        self.log.emit("\n[4] Shuffling and splitting labeled pool -> train / val...")
        random.shuffle(resolved_pairs)
        total   = len(resolved_pairs)
        n_train = max(1, round(total * self.train_pct / 100))
        n_val   = total - n_train

        labeled_splits = {
            "train": resolved_pairs[:n_train],
            "val":   resolved_pairs[n_train:],
        }
        result.train_count = len(labeled_splits["train"])
        result.val_count   = len(labeled_splits["val"])
        self.log.emit(f"   Train: {result.train_count}  |  Val: {result.val_count}")
        self.progress.emit(48)

        # ── Step 5: Collect test images (blind — no labels needed) ────
        test_images = []
        if self.test_images_path and os.path.isdir(self.test_images_path):
            self.log.emit("\n[5] Collecting blind test images...")
            test_images = _collect_test_images(self.test_images_path, self.log.emit)
            result.test_count = len(test_images)
            self.log.emit(f"   Test images: {result.test_count}")
        else:
            self.log.emit("\n[5] No test folder provided — test/images/ will be empty.")
            result.test_count = 0
        self.progress.emit(55)

        # ── Step 6: Create version folder ─────────────────────────────
        version_folder = next_version_folder(task, self.project_name, self.registry_root)
        data_folder    = os.path.join(version_folder, "data")
        result.version_folder = version_folder
        result.data_folder    = data_folder
        self.log.emit(f"\n[6] Creating version folder: {version_folder}")

        # ── Step 7: Build directory structure ──────────────────────────
        self.log.emit("\n[7] Building dataset directory structure...")
        if task == "yolo":
            self._build_yolo_structure(data_folder, labeled_splits, test_images, result)
        else:
            self._build_unet_structure(data_folder, labeled_splits, test_images, result)
        self.progress.emit(82)

        # ── Step 8: Compute dataset fingerprint ───────────────────────
        self.log.emit("\n[8] Computing dataset fingerprint (Quick Hash)...")
        result.dataset_hash = compute_dataset_hash(data_folder)
        self.log.emit(f"   Hash: {result.dataset_hash[:20]}...")
        self.progress.emit(92)

        # ── Step 9: Write hyperparams.json manifest ────────────────────
        self.log.emit("\n[9] Writing hyperparams.json manifest...")
        self._write_manifest(version_folder, result)
        self.progress.emit(100)

        result.success = True
        self.log.emit("\n" + "=" * 52)
        self.log.emit(f"  Aggregation complete!")
        self.log.emit(f"  Dataset at: {data_folder}")
        self.log.emit("=" * 52)

    # -----------------------------------------------------------------------
    def _build_yolo_structure(self, data_folder, labeled_splits, test_images, result):
        """
        YOLO layout:
          images/train/ + labels/train/
          images/val/   + labels/val/
          images/test/                   <- images only, NO labels/
        """
        self.log.emit("   Layout: YOLO (images/ + labels/ for train+val; images/ only for test)")

        # Train and Val — both images AND labels
        for split_name, pairs in labeled_splits.items():
            img_dir = os.path.join(data_folder, "images", split_name)
            lbl_dir = os.path.join(data_folder, "labels", split_name)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            for img_path, lbl_path, new_stem in pairs:
                if not self._running:
                    return
                img_ext = os.path.splitext(img_path)[1]
                lbl_ext = os.path.splitext(lbl_path)[1]
                shutil.copy2(img_path, os.path.join(img_dir, new_stem + img_ext))
                shutil.copy2(lbl_path, os.path.join(lbl_dir, new_stem + lbl_ext))

        # Test — IMAGES ONLY (test labels are never put here)
        if test_images:
            test_img_dir = os.path.join(data_folder, "images", "test")
            os.makedirs(test_img_dir, exist_ok=True)
            for img_path, new_stem in test_images:
                if not self._running:
                    return
                img_ext = os.path.splitext(img_path)[1]
                shutil.copy2(img_path, os.path.join(test_img_dir, new_stem + img_ext))
            self.log.emit(f"   Copied {len(test_images)} blind test images -> images/test/")

        # data.yaml
        yaml_path = os.path.join(data_folder, "data.yaml")
        result.yaml_path = yaml_path
        self._write_data_yaml(yaml_path, data_folder, has_test=bool(test_images))
        self.log.emit(f"   data.yaml -> {yaml_path}")

    def _build_unet_structure(self, data_folder, labeled_splits, test_images, result):
        """
        UNet layout (matching legacy dataloader.py):
          train/images/ + train/masks/
          valid/images/ + valid/masks/
          test/images/                   <- images only, NO masks/
        """
        self.log.emit("   Layout: UNet (train/images/, train/masks/, valid/images/ etc.)")

        # Map internal split name 'val' to 'valid' for dataloader compatibility
        split_map = {"train": "train", "val": "valid"}

        for split_name, pairs in labeled_splits.items():
            out_split = split_map[split_name]
            img_dir  = os.path.join(data_folder, out_split, "images")
            mask_dir = os.path.join(data_folder, out_split, "masks")
            os.makedirs(img_dir,  exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            for img_path, mask_path, new_stem in pairs:
                if not self._running:
                    return
                img_ext  = os.path.splitext(img_path)[1]
                mask_ext = os.path.splitext(mask_path)[1]
                shutil.copy2(img_path,  os.path.join(img_dir,  new_stem + img_ext))
                shutil.copy2(mask_path, os.path.join(mask_dir, new_stem + mask_ext))

        if test_images:
            test_img_dir = os.path.join(data_folder, "test", "images")
            os.makedirs(test_img_dir, exist_ok=True)
            for img_path, new_stem in test_images:
                if not self._running:
                    return
                img_ext = os.path.splitext(img_path)[1]
                shutil.copy2(img_path, os.path.join(test_img_dir, new_stem + img_ext))
            self.log.emit(f"   Copied {len(test_images)} blind test images -> test/images/")

        # Preserve class_map.json if it exists in the original mask export folder
        possible_class_map = os.path.join(self.raw_labels_path, "class_map.json")
        if not os.path.isfile(possible_class_map):
            # Sometimes users point to exports/masks/ instead of the parent
            parent_dir = os.path.dirname(self.raw_labels_path)
            possible_class_map = os.path.join(parent_dir, "class_map.json")
            
        if os.path.isfile(possible_class_map):
            dest_map = os.path.join(data_folder, "class_map.json")
            shutil.copy2(possible_class_map, dest_map)
            self.log.emit("   Copied class_map.json manifest to dataset root.")

        result.yaml_path = ""  # UNet doesn't use data.yaml

    def _write_data_yaml(self, yaml_path: str, data_folder: str, has_test: bool):
        """Generate a YOLO-compatible data.yaml."""
        lines = [
            f"path: {data_folder}",
            f"train: images/train",
            f"val:   images/val",
        ]
        if has_test:
            lines.append("test:  images/test")
        lines += [
            "",
            f"nc: {len(self.class_names)}",
            f"names: {self.class_names}",
        ]
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _write_manifest(self, version_folder: str, result: AggregationResult):
        """Write hyperparams.json — the single source of truth for this model version."""
        manifest = {
            "created_at":         datetime.now().isoformat(),
            "project":            self.project_name,
            "task":               self.task_type,
            "format_detected":    result.format_detected,
            "total_labeled_pairs": result.total_labeled,
            "duplicates_renamed": result.duplicates_renamed,
            "split": {
                "train": result.train_count,
                "val":   result.val_count,
                "test":  result.test_count,
            },
            "dataset_hash":       result.dataset_hash,
            "class_names":        self.class_names,
            "yaml_path":          result.yaml_path,
            # Training fields — filled in by Phase 2 trainer
            "status":             "aggregated",
            "annotator":          "",
            "commit_message":     "",
            "base_model":         "",
            "epochs":             None,
            "batch":              None,
            "map50_95":           None,
            "map50":              None,
            "champion":           False,
        }
        manifest_path = os.path.join(version_folder, "hyperparams.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        self.log.emit(f"   hyperparams.json -> {manifest_path}")
