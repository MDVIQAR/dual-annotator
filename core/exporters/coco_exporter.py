import os
import shutil
import json
from datetime import datetime

class CocoExporter:
    def __init__(self, output_dir, split_enabled, ordered_classes):
        self.output_dir = output_dir
        self.split_enabled = split_enabled
        self.ordered_classes = ordered_classes

        info = {
            "description": "DualAnnotator Export",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        }
        
        categories = []
        for i, cls_obj in enumerate(self.ordered_classes):
            categories.append({
                "id": i + 1,  # COCO categories are 1-indexed
                "name": cls_obj.name,
                "supercategory": "object"
            })
            
        def init_coco_dict():
            return {
                "info": info,
                "licenses": [],
                "categories": categories,
                "images": [],
                "annotations": []
            }

        self._data = {
            "train": init_coco_dict(),
            "val":   init_coco_dict(),
            "test":  init_coco_dict()
        }
        if not split_enabled:
            self._data["all"] = init_coco_dict()

        self._image_id_counter = 1
        self._ann_id_counter = 1

    def export_image(self, filename, json_data, class_index,
                     output_dir, split_name, image_path):
                     
        split_key = split_name if self.split_enabled else "all"
        target_dict = self._data.get(split_key)
        if target_dict is None:
            return 0
            
        iw = json_data.get("image_width", 1)
        ih = json_data.get("image_height", 1)
        
        image_id = self._image_id_counter
        self._image_id_counter += 1
        
        target_dict["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": iw,
            "height": ih
        })

        active_mode = json_data.get("active_mode", "yolo")
        annotations = (json_data.get("layers", {})
                                 .get(active_mode, {})
                                 .get("annotations", []))

        written_anns = 0
        for ann in annotations:
            if ann.get("hollow_role") == "inner":
                continue
            cid = ann.get("class_id")
            if not cid or cid not in class_index:
                continue
                
            class_idx = class_index[cid]
            category_id = class_idx + 1 # 1-indexed
            
            coco_bbox = self._ann_to_coco_bbox(ann, iw, ih)
            if coco_bbox is None:
                continue
                
            x1, y1, w, h = coco_bbox
            ann_id = self._ann_id_counter
            self._ann_id_counter += 1
            
            target_dict["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            written_anns += 1

        # Copy image file
        if self.split_enabled:
            image_dir = os.path.join(output_dir, "images", split_name)
        else:
            image_dir = os.path.join(output_dir, "images")
            
        os.makedirs(image_dir, exist_ok=True)
        dest_image = os.path.join(image_dir, filename)
        if not os.path.exists(dest_image):
            shutil.copy2(image_path, dest_image)

        return written_anns

    def _ann_to_coco_bbox(self, ann, iw, ih):
        t  = ann.get("shape_type")
        np = ann.get("native_params", {})
        pts = ann.get("points", [])

        # Get normalized YOLO values first
        def clamp(v):
            return max(0.0, min(1.0, float(v)))
            
        cx, cy, w_norm, h_norm = None, None, None, None

        if t == "box":
            cx, cy = np.get("cx_norm", 0), np.get("cy_norm", 0)
            w_norm, h_norm = np.get("w_norm", 0), np.get("h_norm", 0)
        elif t == "circle":
            r  = np.get("r", 0)
            cx, cy = np.get("cx", 0) / iw, np.get("cy", 0) / ih
            w_norm, h_norm = (2 * r) / iw, (2 * r) / ih
        elif t == "ellipse":
            cx, cy = np.get("cx", 0) / iw, np.get("cy", 0) / ih
            w_norm, h_norm = (2 * np.get("rx", 0)) / iw, (2 * np.get("ry", 0)) / ih
        elif t == "frame":
            x1, y1 = np.get("x1", 0), np.get("y1", 0)
            x2, y2 = np.get("x2", 0), np.get("y2", 0)
            cx, cy = ((x1 + x2) / 2) / iw, ((y1 + y2) / 2) / ih
            w_norm, h_norm = (x2 - x1) / iw, (y2 - y1) / ih
        elif t == "donut":
            r  = np.get("outer_r", 0)
            cx, cy = np.get("cx", 0) / iw, np.get("cy", 0) / ih
            w_norm, h_norm = (2 * r) / iw, (2 * r) / ih
        elif t == "hollow_ellipse":
            cx, cy = np.get("cx", 0) / iw, np.get("cy", 0) / ih
            w_norm, h_norm = (2 * np.get("outer_rx", 0)) / iw, (2 * np.get("outer_ry", 0)) / ih
        elif t in ("polygon", "bezier_polygon"):
            if not pts:
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            cx, cy = ((x1 + x2) / 2) / iw, ((y1 + y2) / 2) / ih
            w_norm, h_norm = (x2 - x1) / iw, (y2 - y1) / ih
        else:
            return None

        cx = clamp(cx)
        cy = clamp(cy)
        w_norm = clamp(w_norm)
        h_norm = clamp(h_norm)

        if w_norm <= 0 or h_norm <= 0:
            return None

        # Convert to COCO pixel coordinates
        x1_px = round((cx - w_norm / 2) * iw, 2)
        y1_px = round((cy - h_norm / 2) * ih, 2)
        w_px = round(w_norm * iw, 2)
        h_px = round(h_norm * ih, 2)

        return x1_px, y1_px, w_px, h_px

    def finalize(self, output_dir):
        if self.split_enabled:
            for split_name in ("train", "val", "test"):
                path = os.path.join(output_dir, f"instances_{split_name}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self._data[split_name], f)
        else:
            path = os.path.join(output_dir, "instances_all.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._data["all"], f)
