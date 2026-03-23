import os
import shutil

class YoloExporter:
    def __init__(self, output_dir, split_enabled, ordered_classes):
        self.output_dir = output_dir
        self.split_enabled = split_enabled
        self.ordered_classes = ordered_classes  # List[ClassCategory]
        self._clamped_count = 0   # track warnings

    def export_image(self, filename, json_data, class_index,
                     output_dir, split_name, image_path):
        """Write one .txt label file and copy image to correct folder."""

        active_mode = json_data.get("active_mode", "yolo")
        annotations = (json_data.get("layers", {})
                                 .get(active_mode, {})
                                 .get("annotations", []))
        iw = json_data.get("image_width", 1)
        ih = json_data.get("image_height", 1)

        lines = []
        for ann in annotations:
            if ann.get("hollow_role") == "inner":
                continue
            cid = ann.get("class_id")
            if not cid or cid not in class_index:
                continue
            class_idx = class_index[cid]
            yolo_values = self._ann_to_yolo(ann, iw, ih)
            if yolo_values is None:
                continue
            cx, cy, w, h = yolo_values
            lines.append(f"{class_idx} {cx} {cy} {w} {h}\n")

        # Determine output paths
        stem = os.path.splitext(filename)[0]
        if self.split_enabled:
            label_dir = os.path.join(output_dir, "labels", split_name)
            image_dir = os.path.join(output_dir, "images", split_name)
        else:
            label_dir = os.path.join(output_dir, "labels")
            image_dir = os.path.join(output_dir, "images")

        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # Write label file (always, even if empty - YOLO expects empty .txt
        # for images with no annotations in that split)
        label_path = os.path.join(label_dir, stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Copy image file
        dest_image = os.path.join(image_dir, filename)
        if not os.path.exists(dest_image):
            shutil.copy2(image_path, dest_image)

        return len(lines)   # number of annotations written

    def _ann_to_yolo(self, ann, iw, ih):
        """Convert annotation dict to (cx_norm, cy_norm, w_norm, h_norm).
           Returns None if invalid. All values are clamped strings."""
        t  = ann.get("shape_type")
        np = ann.get("native_params", {})
        pts = ann.get("points", [])

        def clamp(v):
            return round(max(0.0, min(1.0, float(v))), 6)

        if t == "box":
            cx = clamp(np.get("cx_norm", 0))
            cy = clamp(np.get("cy_norm", 0))
            w  = clamp(np.get("w_norm",  0))
            h  = clamp(np.get("h_norm",  0))
        elif t == "circle":
            r  = np.get("r", 0)
            cx = clamp(np.get("cx", 0) / iw)
            cy = clamp(np.get("cy", 0) / ih)
            w  = clamp((2 * r) / iw)
            h  = clamp((2 * r) / ih)
        elif t == "ellipse":
            cx = clamp(np.get("cx", 0) / iw)
            cy = clamp(np.get("cy", 0) / ih)
            w  = clamp((2 * np.get("rx", 0)) / iw)
            h  = clamp((2 * np.get("ry", 0)) / ih)
        elif t == "frame":
            x1, y1 = np.get("x1", 0), np.get("y1", 0)
            x2, y2 = np.get("x2", 0), np.get("y2", 0)
            cx = clamp(((x1 + x2) / 2) / iw)
            cy = clamp(((y1 + y2) / 2) / ih)
            w  = clamp((x2 - x1) / iw)
            h  = clamp((y2 - y1) / ih)
        elif t == "donut":
            r  = np.get("outer_r", 0)
            cx = clamp(np.get("cx", 0) / iw)
            cy = clamp(np.get("cy", 0) / ih)
            w  = clamp((2 * r) / iw)
            h  = clamp((2 * r) / ih)
        elif t == "hollow_ellipse":
            cx = clamp(np.get("cx", 0) / iw)
            cy = clamp(np.get("cy", 0) / ih)
            w  = clamp((2 * np.get("outer_rx", 0)) / iw)
            h  = clamp((2 * np.get("outer_ry", 0)) / ih)
        elif t in ("polygon", "bezier_polygon"):
            if not pts:
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            cx = clamp(((x1 + x2) / 2) / iw)
            cy = clamp(((y1 + y2) / 2) / ih)
            w  = clamp((x2 - x1) / iw)
            h  = clamp((y2 - y1) / ih)
        else:
            return None

        if w <= 0 or h <= 0:
            return None

        return cx, cy, w, h

    def write_data_yaml(self, output_dir, split_enabled, image_folder_abs=None):
        """Write data.yaml for YOLO training."""
        names = [cls.name for cls in self.ordered_classes]
        nc = len(names)

        if split_enabled:
            content = {
                "path": output_dir,
                "train": "images/train",
                "val":   "images/val",
                "test":  "images/test",
                "nc":    nc,
                "names": names
            }
        else:
            content = {
                "path": output_dir,
                "train": "images",
                "val":   "images",
                "nc":    nc,
                "names": names
            }

        yaml_path = os.path.join(output_dir, "data.yaml")
        try:
            import yaml as _yaml
            with open(yaml_path, "w") as f:
                _yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            # Manual YAML write if PyYAML not installed
            with open(yaml_path, "w") as f:
                f.write(f"path: {output_dir}\n")
                if split_enabled:
                    f.write("train: images/train\nval: images/val\ntest: images/test\n")
                else:
                    f.write("train: images\nval: images\n")
                f.write(f"nc: {nc}\n")
                f.write("names:\n")
                for name in names:
                    # Sanitize: replace spaces with underscores
                    safe = name.replace(" ", "_")
                    f.write(f"  - {safe}\n")

    def write_classes_txt(self, output_dir):
        """Write classes.txt mapping index to name."""
        path = os.path.join(output_dir, "classes.txt")
        with open(path, "w", encoding="utf-8") as f:
            for i, cls in enumerate(self.ordered_classes):
                f.write(f"{i}: {cls.name}\n")
