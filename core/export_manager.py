import os
import json
import random
from datetime import datetime

class ExportManager:
    def __init__(self, project_manager, class_manager):
        self.pm = project_manager       # ProjectManager instance
        self.cm = class_manager         # ClassManager instance
        self._warnings = []

    def run_export(self,
                   output_dir: str,
                   fmt: str,                   # "yolo" | "coco" | "voc"
                   split: bool,
                   train_pct: int,
                   val_pct: int,
                   test_pct: int,
                   seed: int,
                   delta_only: bool,
                   draw_annotated: bool,
                   progress_callback=None      # callable(current, total, message)
                   ) -> dict:                  # returns summary dict
        """
        Orchestrates a full export run.
        """
        self._warnings = []
        summary = {
            "exported": 0,
            "skipped_unchanged": 0,
            "skipped_no_class": 0,
            "total_annotations": 0,
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "output_dir": output_dir,
            "format": fmt,
            "warnings": self._warnings
        }

        # Step 1: Build class index map
        ordered_classes = list(self.cm.classes.values())
        class_index = {cls.id: i for i, cls in enumerate(ordered_classes)}

        # Step 2: Collect exportable images
        exportable_images = []
        if os.path.exists(self.pm.annotations_dir):
            for filename in sorted(os.listdir(self.pm.annotations_dir)):
                if not filename.endswith(".json") or filename.endswith(".tmp"):
                    continue
                    
                path = os.path.join(self.pm.annotations_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    continue

                status = data.get("status", "")
                if status not in ("annotated", "in_progress"):
                    continue

                active_mode = data.get("active_mode", "yolo")
                annotations = data.get("layers", {}).get(active_mode, {}).get("annotations", [])
                
                if not annotations:
                    continue

                if delta_only:
                    last_mod_str = data.get("last_modified")
                    last_exp_str = data.get("last_exported_at")
                    if last_exp_str and last_mod_str:
                        if last_exp_str >= last_mod_str:
                            summary["skipped_unchanged"] += 1
                            continue

                img_file = data.get("image_file")
                image_path = os.path.join(self.pm.image_folder, img_file)
                if not os.path.exists(image_path):
                    self._warnings.append(f"{img_file} not found at expected path — skipped.")
                    continue

                exportable_images.append((img_file, data))

        # Step 3: Train/Val/Test split
        splits = {"train": [], "val": [], "test": []}
        if split and len(exportable_images) >= 3:
            rng = random.Random(seed)
            shuffled = list(exportable_images)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = round(n * train_pct / 100)
            n_val   = round(n * val_pct / 100)
            n_test  = n - n_train - n_val
            splits["train"] = shuffled[:n_train]
            splits["val"] = shuffled[n_train:n_train+n_val]
            splits["test"] = shuffled[n_train+n_val:]
        elif split and len(exportable_images) < 3:
            self._warnings.append(f"Only {len(exportable_images)} images — all placed in train (split requires 3+).")
            splits["train"] = exportable_images
        else:
            splits["train"] = exportable_images

        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)

        # Step 4: Setup Exporter
        exporter = None
        if fmt == "yolo":
            from core.exporters.yolo_exporter import YoloExporter
            exporter = YoloExporter(output_dir, split, ordered_classes)
        elif fmt == "coco":
            from core.exporters.coco_exporter import CocoExporter
            exporter = CocoExporter(output_dir, split, ordered_classes)
        elif fmt == "voc":
            from core.exporters.voc_exporter import VocExporter
            exporter = VocExporter(output_dir, split, ordered_classes)

        total_images = len(exportable_images)
        current_idx = 0

        # Step 5: Export each image
        for split_name in ("train", "val", "test"):
            for filename, data in splits[split_name]:
                current_idx += 1
                if progress_callback:
                    progress_callback(current_idx, total_images, f"Exporting {filename}")
                
                image_path = os.path.join(self.pm.image_folder, filename)
                
                if exporter:
                    written = exporter.export_image(filename, data, class_index, output_dir, split_name, image_path)
                    summary["total_annotations"] += written
                    
                    active_mode = data.get("active_mode", "yolo")
                    annotations = data.get("layers", {}).get(active_mode, {}).get("annotations", [])
                    for ann in annotations:
                        if ann.get("hollow_role") == "inner":
                            continue
                        if not ann.get("class_id") or ann.get("class_id") not in class_index:
                            summary["skipped_no_class"] += 1
                
                if draw_annotated:
                    self._draw_annotated_image(filename, data, class_index, output_dir, image_path)
                
                self.pm.mark_as_exported(filename)
                summary["exported"] += 1
                summary[f"{split_name}_count"] += 1

        # Step 6: Write format-level files
        if exporter:
            if fmt == "yolo":
                exporter.write_data_yaml(output_dir, split, self.pm.image_folder)
                exporter.write_classes_txt(output_dir)
            elif fmt == "coco":
                exporter.finalize(output_dir)
                
        return summary

    def _draw_annotated_image(self, filename, json_data, class_index, output_dir, image_path):
        """Draw bounding boxes on image copy using Pillow."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            self._warnings.append("Pillow not installed — annotated images skipped.")
            return

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        iw, ih = img.size

        active_mode = json_data.get("active_mode", "yolo")
        annotations = json_data.get("layers", {}).get(active_mode, {}).get("annotations", [])

        for ann in annotations:
            if ann.get("hollow_role") == "inner":
                continue
            cid = ann.get("class_id")
            if not cid or cid not in class_index:
                continue

            cls_obj = self.cm.get_class(cid)
            color_hex = cls_obj.color if cls_obj else "#00FF00"
            color = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0,2,4))

            bbox = self._annotation_to_pixel_bbox(ann, iw, ih)
            if bbox:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                label = cls_obj.name if cls_obj else str(class_index[cid])
                draw.text((x1 + 2, y1 + 2), label, fill=color)

        out_path = os.path.join(output_dir, "annotated", filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)

    def _annotation_to_pixel_bbox(self, ann, iw, ih):
        """Convert annotation dict to (x1,y1,x2,y2) pixel bbox."""
        t = ann.get("shape_type")
        np = ann.get("native_params", {})
        pts = ann.get("points", [])

        if t == "box":
            cx = np.get("cx_norm", 0) * iw
            cy = np.get("cy_norm", 0) * ih
            w  = np.get("w_norm", 0) * iw
            h  = np.get("h_norm", 0) * ih
            return (int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2))
        elif t == "circle":
            cx,cy,r = np.get("cx", 0), np.get("cy", 0), np.get("r", 0)
            return (cx-r, cy-r, cx+r, cy+r)
        elif t == "ellipse":
            cx,cy = np.get("cx", 0), np.get("cy", 0)
            rx,ry = np.get("rx", 0), np.get("ry", 0)
            return (cx-rx, cy-ry, cx+rx, cy+ry)
        elif t == "frame":
            return (np.get("x1", 0), np.get("y1", 0), np.get("x2", 0), np.get("y2", 0))
        elif t == "donut":
            cx,cy = np.get("cx", 0), np.get("cy", 0)
            r = np.get("outer_r", 0)
            return (cx-r, cy-r, cx+r, cy+r)
        elif t == "hollow_ellipse":
            cx,cy = np.get("cx", 0), np.get("cy", 0)
            rx = np.get("outer_rx", 0)
            ry = np.get("outer_ry", 0)
            return (cx-rx, cy-ry, cx+rx, cy+ry)
        elif t in ("polygon", "bezier_polygon"):
            if not pts: return None
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        return None
