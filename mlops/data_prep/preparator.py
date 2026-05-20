"""
mlops/data_prep/preparator.py

Pure Python backend for dataset preparation. No PyQt5 imports.
Handles scanning image/label pairs, train/val splitting, YOLO and UNet export layouts,
and writing the dataset_info.json manifest.
"""

import os
import re
import json
import random
import shutil
from datetime import datetime


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


class DataPreparator:
    """Stateless utility class for dataset preparation operations."""

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    @staticmethod
    def scan_pairs(images_folder: str, labels_folder: str, task_type: str) -> dict:
        """
        Match image files to their label/mask counterparts by stem name.

        task_type='yolo'  → label extension is .txt
        task_type='unet'  → label extension is .png

        Both folders are scanned non-recursively.
        Raises FileNotFoundError if either folder does not exist.
        """
        if not os.path.isdir(images_folder):
            raise FileNotFoundError(f"Images folder not found: {images_folder}")
        if not os.path.isdir(labels_folder):
            raise FileNotFoundError(f"Labels folder not found: {labels_folder}")

        label_ext = ".txt" if task_type == "yolo" else ".png"

        # Build stem→path maps
        image_map = {}
        for fname in os.listdir(images_folder):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in _IMAGE_EXTS:
                image_map[stem.lower()] = os.path.join(images_folder, fname)

        label_map = {}
        for fname in os.listdir(labels_folder):
            stem, ext = os.path.splitext(fname)
            if ext.lower() == label_ext:
                label_map[stem.lower()] = os.path.join(labels_folder, fname)

        # Match
        matched_stems = set(image_map) & set(label_map)
        pairs = sorted(
            [(image_map[s], label_map[s]) for s in matched_stems],
            key=lambda p: os.path.basename(p[0]).lower(),
        )
        unpaired_images = sorted(
            image_map[s] for s in image_map if s not in label_map
        )
        unpaired_labels = sorted(
            label_map[s] for s in label_map if s not in image_map
        )

        return {
            "pairs": pairs,
            "unpaired_images": unpaired_images,
            "unpaired_labels": unpaired_labels,
            "total": len(pairs),
        }

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    @staticmethod
    def split(pairs: list, train_pct: int) -> dict:
        """
        Randomly shuffle and split pairs into train and val sets.
        Guarantees at least 1 pair in each set.
        """
        pairs = list(pairs)
        random.shuffle(pairs)
        n = len(pairs)
        train_n = max(1, int(round(n * train_pct / 100)))
        # Ensure at least 1 val pair
        if train_n >= n:
            train_n = n - 1
        return {
            "train": pairs[:train_n],
            "val": pairs[train_n:],
        }

    # ------------------------------------------------------------------
    # Output folder
    # ------------------------------------------------------------------

    @staticmethod
    def make_output_folder(base_dir: str, project_name: str) -> str:
        """
        Create and return {base_dir}/{project_name}_{YYYY-MM-DD}/.
        If that name is taken, appends _2, _3, ... until a free name is found.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        base_name = f"{project_name}_{date_str}"
        candidate = os.path.join(base_dir, base_name)

        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return os.path.abspath(candidate)

        suffix = 2
        while True:
            candidate = os.path.join(base_dir, f"{base_name}_{suffix}")
            if not os.path.exists(candidate):
                os.makedirs(candidate)
                return os.path.abspath(candidate)
            suffix += 1

    # ------------------------------------------------------------------
    # YOLO export
    # ------------------------------------------------------------------

    @staticmethod
    def export_yolo(
        train_pairs: list,
        val_pairs: list,
        test_folder,
        output_folder: str,
        class_names: list,
    ) -> dict:
        """
        Build YOLO dataset layout inside output_folder.
        Returns a dataset_info dict (sha256 left as '').
        """
        # Create directories
        dirs = {
            "img_train":  os.path.join(output_folder, "images", "train"),
            "img_val":    os.path.join(output_folder, "images", "val"),
            "img_test":   os.path.join(output_folder, "images", "test"),
            "lbl_train":  os.path.join(output_folder, "labels", "train"),
            "lbl_val":    os.path.join(output_folder, "labels", "val"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        # Copy train pairs
        for img, lbl in train_pairs:
            shutil.copy2(img, os.path.join(dirs["img_train"], os.path.basename(img)))
            shutil.copy2(lbl, os.path.join(dirs["lbl_train"], os.path.basename(lbl)))

        # Copy val pairs
        for img, lbl in val_pairs:
            shutil.copy2(img, os.path.join(dirs["img_val"], os.path.basename(img)))
            shutil.copy2(lbl, os.path.join(dirs["lbl_val"], os.path.basename(lbl)))

        # Copy test images
        test_count = 0
        has_test = test_folder and os.path.isdir(test_folder)
        if has_test:
            for fname in os.listdir(test_folder):
                if os.path.splitext(fname)[1].lower() in _IMAGE_EXTS:
                    shutil.copy2(
                        os.path.join(test_folder, fname),
                        os.path.join(dirs["img_test"], fname),
                    )
                    test_count += 1

        # Write data.yaml (plain text, no yaml package)
        yaml_path = os.path.join(output_folder, "data.yaml")
        abs_out = output_folder.replace("\\", "/")
        lines = [
            f"path: {abs_out}",
            "train: images/train",
            "val: images/val",
        ]
        if has_test and test_count > 0:
            lines.append("test: images/test")
        lines.append(f"nc: {len(class_names)}")
        lines.append("names:")
        for name in class_names:
            lines.append(f"  - {name}")
        with open(yaml_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        return {
            "project":       "",
            "task_type":     "yolo",
            "output_folder": output_folder.replace("\\", "/"),
            "created_at":    datetime.now().isoformat(timespec="seconds"),
            "train_count":   len(train_pairs),
            "val_count":     len(val_pairs),
            "test_count":    test_count,
            "class_names":   list(class_names),
            "num_classes":   len(class_names),
            "sha256":        "",
            "data_yaml":     yaml_path.replace("\\", "/"),
        }

    # ------------------------------------------------------------------
    # UNet export
    # ------------------------------------------------------------------

    @staticmethod
    def export_unet(
        train_pairs: list,
        val_pairs: list,
        test_folder,
        output_folder: str,
    ) -> dict:
        """
        Build UNet dataset layout inside output_folder.
        Returns a dataset_info dict (sha256 left as '').
        """
        dirs = {
            "train_img":  os.path.join(output_folder, "train", "images"),
            "train_mask": os.path.join(output_folder, "train", "masks"),
            "val_img":    os.path.join(output_folder, "val", "images"),
            "val_mask":   os.path.join(output_folder, "val", "masks"),
            "test_img":   os.path.join(output_folder, "test", "images"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        for img, mask in train_pairs:
            shutil.copy2(img,  os.path.join(dirs["train_img"],  os.path.basename(img)))
            shutil.copy2(mask, os.path.join(dirs["train_mask"], os.path.basename(mask)))

        for img, mask in val_pairs:
            shutil.copy2(img,  os.path.join(dirs["val_img"],  os.path.basename(img)))
            shutil.copy2(mask, os.path.join(dirs["val_mask"], os.path.basename(mask)))

        test_count = 0
        if test_folder and os.path.isdir(test_folder):
            for fname in os.listdir(test_folder):
                if os.path.splitext(fname)[1].lower() in _IMAGE_EXTS:
                    shutil.copy2(
                        os.path.join(test_folder, fname),
                        os.path.join(dirs["test_img"], fname),
                    )
                    test_count += 1

        return {
            "project":       "",
            "task_type":     "unet",
            "output_folder": output_folder.replace("\\", "/"),
            "created_at":    datetime.now().isoformat(timespec="seconds"),
            "train_count":   len(train_pairs),
            "val_count":     len(val_pairs),
            "test_count":    test_count,
            "class_names":   [],
            "num_classes":   0,
            "sha256":        "",
            "data_yaml":     "",
        }

    # ------------------------------------------------------------------
    # Manifest write
    # ------------------------------------------------------------------

    @staticmethod
    def write_dataset_info(output_folder: str, info: dict) -> None:
        """Atomically write info as JSON to {output_folder}/dataset_info.json."""
        path = os.path.join(output_folder, "dataset_info.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Import existing pre-split dataset
    # ------------------------------------------------------------------

    @staticmethod
    def scan_existing_presplit(root: str) -> dict:
        """
        Scan a dataset that already has a train/val folder structure.

        Detects two layouts:
          Layout A — root/train/images/  + root/val/images/   (common export format)
          Layout B — root/images/train/  + root/images/val/   (app-generated format)

        Also auto-reads class names from classes.txt or data.yaml if present.
        Returns a result dict — always safe to call, never raises.
        """
        result = {
            "root":             root,
            "layout":           None,   # "trainval_images" | "images_trainval" | "flat" | "unknown"
            "task_type":        None,   # "yolo" | "unet" | None
            "train_count":      0,
            "val_count":        0,
            "test_count":       0,
            "class_names":      [],
            "has_dataset_info": False,
            "has_data_yaml":    False,
            "errors":           [],
        }

        if not os.path.isdir(root):
            result["errors"].append(f"Folder not found: {root}")
            return result

        def _count_imgs(folder):
            if not os.path.isdir(folder):
                return 0
            return sum(1 for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in _IMAGE_EXTS)

        # ── Detect layout ──────────────────────────────────────────────
        train_img_a = os.path.join(root, "train", "images")
        val_img_a   = os.path.join(root, "val",   "images")
        train_img_b = os.path.join(root, "images", "train")
        val_img_b   = os.path.join(root, "images", "val")
        flat_img    = os.path.join(root, "images")

        if os.path.isdir(train_img_a) and os.path.isdir(val_img_a):
            result["layout"]      = "trainval_images"
            result["train_count"] = _count_imgs(train_img_a)
            result["val_count"]   = _count_imgs(val_img_a)
            result["test_count"]  = _count_imgs(os.path.join(root, "test", "images"))
            # Detect task type from sibling label/mask folders
            if os.path.isdir(os.path.join(root, "train", "labels")):
                result["task_type"] = "yolo"
            elif os.path.isdir(os.path.join(root, "train", "masks")):
                result["task_type"] = "unet"

        elif os.path.isdir(train_img_b) and os.path.isdir(val_img_b):
            result["layout"]      = "images_trainval"
            result["train_count"] = _count_imgs(train_img_b)
            result["val_count"]   = _count_imgs(val_img_b)
            result["test_count"]  = _count_imgs(os.path.join(root, "images", "test"))
            if os.path.isdir(os.path.join(root, "labels", "train")):
                result["task_type"] = "yolo"
            elif os.path.isdir(os.path.join(root, "train", "masks")):
                result["task_type"] = "unet"

        elif os.path.isdir(flat_img):
            # Flat export: images/ + labels/ (no split yet)
            result["layout"]      = "flat"
            result["train_count"] = _count_imgs(flat_img)
            result["val_count"]   = 0
            if os.path.isdir(os.path.join(root, "labels")):
                result["task_type"] = "yolo"
            elif os.path.isdir(os.path.join(root, "masks")):
                result["task_type"] = "unet"
        else:
            result["layout"] = "unknown"
            result["errors"].append(
                "No recognised folder layout found. "
                "Expected train/images/ + val/images/, or images/train/ + images/val/, "
                "or a flat images/ + labels/ folder."
            )

        # ── Auto-read class names ──────────────────────────────────────
        classes_txt = os.path.join(root, "classes.txt")
        data_yaml   = os.path.join(root, "data.yaml")

        if os.path.isfile(classes_txt):
            names = []
            with open(classes_txt, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    # Handle both "0: name" and plain "name" formats
                    names.append(line.split(":", 1)[1].strip() if ":" in line else line)
            result["class_names"] = names
        elif os.path.isfile(data_yaml):
            try:
                with open(data_yaml, "r", encoding="utf-8") as fh:
                    content = fh.read()
                names = re.findall(r"^\s*-\s*(.+)$", content, re.MULTILINE)
                if names:
                    result["class_names"] = [n.strip() for n in names]
            except Exception:
                pass

        result["has_dataset_info"] = os.path.isfile(os.path.join(root, "dataset_info.json"))
        result["has_data_yaml"]    = os.path.isfile(data_yaml)
        return result

    @staticmethod
    def generate_presplit_metadata(root: str, task_type: str,
                                   class_names: list, layout: str) -> None:
        """
        Write dataset_info.json and (for YOLO) data.yaml into an existing
        pre-split dataset root. No files are copied or moved.

        layout: "trainval_images" → train/images/ paths
                "images_trainval" → images/train/ paths
        """
        def _count_imgs(folder):
            if not os.path.isdir(folder):
                return 0
            return sum(1 for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in _IMAGE_EXTS)

        if layout == "trainval_images":
            train_dir  = os.path.join(root, "train", "images")
            val_dir    = os.path.join(root, "val",   "images")
            test_dir   = os.path.join(root, "test",  "images")
            yaml_train = "train/images"
            yaml_val   = "val/images"
            yaml_test  = "test/images"
        else:  # images_trainval
            train_dir  = os.path.join(root, "images", "train")
            val_dir    = os.path.join(root, "images", "val")
            test_dir   = os.path.join(root, "images", "test")
            yaml_train = "images/train"
            yaml_val   = "images/val"
            yaml_test  = "images/test"

        train_count = _count_imgs(train_dir)
        val_count   = _count_imgs(val_dir)
        test_count  = _count_imgs(test_dir)

        info = {
            "task_type":     task_type,
            "output_folder": root.replace("\\", "/"),
            "created_at":    datetime.now().isoformat(timespec="seconds"),
            "train_count":   train_count,
            "val_count":     val_count,
            "test_count":    test_count,
            "class_names":   list(class_names),
            "num_classes":   len(class_names),
            "sha256":        "",
            "data_yaml":     os.path.join(root, "data.yaml").replace("\\", "/")
                             if task_type == "yolo" else "",
        }
        DataPreparator.write_dataset_info(root, info)

        if task_type != "yolo":
            return

        abs_root = root.replace("\\", "/")
        lines = [
            f"path: {abs_root}",
            f"train: {yaml_train}",
            f"val:   {yaml_val}",
        ]
        if test_count > 0:
            lines.append(f"test:  {yaml_test}")
        lines += [f"nc: {len(class_names)}", "names:"]
        for name in class_names:
            lines.append(f"  - {name}")

        yaml_path = os.path.join(root, "data.yaml")
        tmp = yaml_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, yaml_path)
