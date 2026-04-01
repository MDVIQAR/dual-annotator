import os
import json
import hashlib
import uuid
from datetime import datetime
from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox

class ResumeDialog(QDialog):
    def __init__(self, last_opened, annotated_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resume Project?")
        self.choice = None # "resume" or "fresh"
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Found existing project."))
        layout.addWidget(QLabel(f"{annotated_count} annotated images"))
        layout.addWidget(QLabel(f"Last opened: {last_opened}"))
        btn_row = QHBoxLayout()
        fresh_btn = QPushButton("Start Fresh")
        resume_btn = QPushButton("Resume Project")
        fresh_btn.clicked.connect(lambda: self._pick("fresh"))
        resume_btn.clicked.connect(lambda: self._pick("resume"))
        btn_row.addWidget(fresh_btn)
        btn_row.addWidget(resume_btn)
        layout.addLayout(btn_row)

    def _pick(self, choice):
        self.choice = choice
        self.accept()

class ProjectManager(QObject):
    def __init__(self):
        super().__init__()
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(800) # 800ms debounce
        self._autosave_timer.timeout.connect(self._do_autosave)
        
        self._pending_image = None
        self._pending_shapes = None
        self._pending_mode = None
        self._pending_width = None
        self._pending_height = None
        
        self.image_folder = None
        self.project_dir = None
        self.annotations_dir = None
        self.project_json_path = None
        self.project_data = None
        
        self._hash_cache = {}

    def _write_json_atomic(self, path, data):
        """Write JSON atomically via temp file rename."""
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path) # atomic on all platforms
        except Exception as e:
            print(f"Error saving to {path}: {e}")

    def compute_image_hash(self, image_path):
        """Return md5:hexstring. Cache result."""
        if image_path in self._hash_cache:
            return self._hash_cache[image_path]
        try:
            with open(image_path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
                result = f"md5:{digest}"
                self._hash_cache[image_path] = result
                return result
        except Exception:
            return None

    def open_folder(self, path, parent_widget=None):
        self.image_folder = path
        self.project_dir = os.path.join(path, ".dualannotator")
        self.annotations_dir = os.path.join(self.project_dir, "annotations")
        self.project_json_path = os.path.join(self.project_dir, "project.json")

        if os.path.exists(self.project_json_path):
            with open(self.project_json_path, "r", encoding="utf-8") as f:
                try:
                    self.project_data = json.load(f)
                except json.JSONDecodeError:
                    self.project_data = None
            
            if self.project_data:
                stats = self.project_data.get("stats", {})
                annotated_count = stats.get("annotated_images", 0)
                last_opened = self.project_data.get("last_opened", "Unknown")
                
                dlg = ResumeDialog(last_opened, annotated_count, parent=parent_widget)
                if dlg.exec_() != QDialog.Accepted:
                    return None # user cancelled
                
                if dlg.choice == "fresh":
                    # Warning before deletion
                    reply = QMessageBox.question(parent_widget, "Confirm Start Fresh", 
                                                 "This will delete ALL existing annotations in this folder. Are you sure?",
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.Yes:
                        import shutil
                        shutil.rmtree(self.project_dir, ignore_errors=True)
                        return self._create_new_project(path)
                    else:
                        return None
                elif dlg.choice == "resume":
                    self.project_data["last_opened"] = datetime.now().isoformat()
                    self._write_json_atomic(self.project_json_path, self.project_data)
                    return self.project_data
        
        # Start fresh / no existing project
        return self._create_new_project(path)

    def _create_new_project(self, path):
        os.makedirs(self.annotations_dir, exist_ok=True)
        # Attempt to hide on Windows
        if os.name == 'nt':
            os.system(f'attrib +h "{self.project_dir}"')
            
        self.project_data = {
            "schema_version": "1.0",
            "app": "DualAnnotator",
            "created_at": datetime.now().isoformat(),
            "last_opened": datetime.now().isoformat(),
            "image_dir": path,
            "active_mode": "yolo",
            "classes": [],
            "image_order": [],
            "image_statuses": {},
            "stats": {
                "total_images": 0,
                "annotated_images": 0,
                "in_progress_images": 0,
                "skipped_images": 0
            }
        }
        self._write_json_atomic(self.project_json_path, self.project_data)
        return self.project_data

    def _serialize_shape(self, shape):
        ann_id = "ann_" + getattr(shape, "id", str(uuid.uuid4())[:8])
        
        # Look up the human-readable class name from project_data
        resolved_class_name = ""
        if self.project_data and "classes" in self.project_data:
            for c in self.project_data["classes"]:
                if c.get("id") == shape.class_id:
                    resolved_class_name = c.get("name", "")
                    break

        ann = {
            "id": ann_id,
            "shape_type": shape.type,
            "class_id": shape.class_id,
            "class_name": resolved_class_name,
            "is_hollow": getattr(shape, "is_hollow", False),
            "group_id": getattr(shape, "group_id", None),
            "hollow_role": getattr(shape, "hollow_role", None),
            "source": "manual",
            "confidence": 1.0,
            "from_template": None,
            "native_params": {},
            "points": [],
            "inner": None
        }

        t = shape.type
        if t == "box":
            if hasattr(shape, "to_pixels"):
                x1, y1, x2, y2 = shape.to_pixels()
                ann["native_params"] = {
                    "cx_norm": shape.x, "cy_norm": shape.y,
                    "w_norm": shape.width, "h_norm": shape.height
                }
        elif t == "circle":
            if hasattr(shape, "to_pixels"):
                cx, cy, r = shape.to_pixels()
                ann["native_params"] = {"cx": cx, "cy": cy, "r": r}
        elif t == "ellipse":
            if hasattr(shape, "to_pixels"):
                cx, cy, rx, ry = shape.to_pixels()
                ann["native_params"] = {"cx": cx, "cy": cy, "rx": rx, "ry": ry}
        elif t == "polygon":
            if hasattr(shape, "to_pixel_points"):
                ann["points"] = shape.to_pixel_points()
            if getattr(shape, "inner_points", None):
                cw = shape.image_width if getattr(shape, 'image_width', None) else 1
                ch = shape.image_height if getattr(shape, 'image_height', None) else 1
                if hasattr(shape, "get_inner_pixel_points"):
                    ann["inner_points"] = shape.get_inner_pixel_points()
                else:
                    ann["inner_points"] = [(int(nx * cw), int(ny * ch)) for nx, ny in shape.inner_points]
        elif t == "bezier_polygon":
            if hasattr(shape, "to_pixel_points"):
                ann["points"] = shape.to_pixel_points()
            if getattr(shape, "ctrl", None):
                cw = shape.image_width if getattr(shape, 'image_width', None) else 1
                ch = shape.image_height if getattr(shape, 'image_height', None) else 1
                ann["ctrl_points"] = [(c[0]*cw, c[1]*ch) if c is not None else None for c in shape.ctrl]
            if getattr(shape, "inner_points", None):
                cw = shape.image_width if getattr(shape, 'image_width', None) else 1
                ch = shape.image_height if getattr(shape, 'image_height', None) else 1
                ann["inner_points"] = [(int(nx * cw), int(ny * ch)) for nx, ny in shape.inner_points]
            if getattr(shape, "inner_control_points", None):
                cw = shape.image_width if getattr(shape, 'image_width', None) else 1
                ch = shape.image_height if getattr(shape, 'image_height', None) else 1
                ann["inner_ctrl_points"] = [(c[0]*cw, c[1]*ch) if c is not None else None for c in shape.inner_control_points]
        elif t == "frame":
            if hasattr(shape, "to_pixels"):
                x1, y1, x2, y2 = shape.to_pixels()
                ann["native_params"] = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "t_top": getattr(shape, "t_top", 0), "t_bottom": getattr(shape, "t_bottom", 0),
                    "t_left": getattr(shape, "t_left", 0), "t_right": getattr(shape, "t_right", 0)
                }
        elif t == "donut":
            if hasattr(shape, "to_pixels"):
                cx, cy, outer_r, inner_r = shape.to_pixels()
                ann["native_params"] = {"cx": cx, "cy": cy, "outer_r": outer_r, "inner_r": inner_r}
        elif t == "hollow_ellipse":
            if hasattr(shape, "to_pixels"):
                cx, cy, outer_rx, outer_ry, inner_rx, inner_ry = shape.to_pixels()
                ann["native_params"] = {
                    "cx": cx, "cy": cy, "outer_rx": outer_rx, "outer_ry": outer_ry, 
                    "inner_rx": inner_rx, "inner_ry": inner_ry
                }

        inner_obj = getattr(shape, "inner_shape", None)
        if inner_obj:
            ann["inner"] = self._serialize_shape(inner_obj)

        return ann

    def deserialize_shape(self, data, w, h):
        t = data.get("shape_type", "")
        shape = None
        native = data.get("native_params", {})
        pts = data.get("points", [])
        
        cid = data.get("class_id")

        if t == "box":
            from core.annotation import BoundingBox
            shape = BoundingBox(image_size=(w, h))
            shape.x = native.get("cx_norm", 0)
            shape.y = native.get("cy_norm", 0)
            shape.width = native.get("w_norm", 0)
            shape.height = native.get("h_norm", 0)
            shape.class_id = cid
        elif t == "circle":
            from core.circle_shape import CircleShape
            shape = CircleShape(image_size=(w, h))
            if native:
                shape.from_pixels(native.get("cx", 0), native.get("cy", 0), native.get("r", 0))
            shape.class_id = cid
        elif t == "ellipse":
            from core.ellipse_shape import EllipseShape
            shape = EllipseShape(image_size=(w, h))
            if native:
                shape.from_pixels(native.get("cx", 0), native.get("cy", 0), native.get("rx", 0), native.get("ry", 0))
            shape.class_id = cid
        elif t == "polygon":
            from core.polygon_shape import PolygonShape
            shape = PolygonShape(image_size=(w, h))
            shape.from_pixel_points(pts)
            shape.close_polygon()
            if "inner_points" in data:
                normalized_inner = []
                for px, py in data["inner_points"]:
                    normalized_inner.append((px / w, py / h))
                shape.inner_points = normalized_inner
            shape.class_id = cid
        elif t == "bezier_polygon":
            from core.bezier_shape import BezierPolygonShape
            shape = BezierPolygonShape(image_size=(w, h))
            shape.from_pixel_points(pts)
            if "ctrl_points" in data:
                shape.ctrl = [(c[0]/w, c[1]/h) if c is not None else None for c in data["ctrl_points"]]
            if "inner_points" in data:
                normalized_inner = []
                for px, py in data["inner_points"]:
                    normalized_inner.append((px / w, py / h))
                shape.inner_points = normalized_inner
            if "inner_ctrl_points" in data:
                shape.inner_control_points = [(c[0]/w, c[1]/h) if c is not None else None for c in data["inner_ctrl_points"]]
            shape.close_polygon()
            shape.class_id = cid
        elif t == "frame":
            from core.ring_shape import FrameShape
            shape = FrameShape(image_size=(w, h))
            if native:
                shape.from_pixels(native.get("x1", 0), native.get("y1", 0), native.get("x2", 0), native.get("y2", 0))
                shape.t_top = native.get("t_top", 20)
                shape.t_bottom = native.get("t_bottom", 20)
                shape.t_left = native.get("t_left", 20)
                shape.t_right = native.get("t_right", 20)
            shape.class_id = cid
        elif t == "donut":
            from core.ring_shape import DonutShape
            shape = DonutShape(image_size=(w, h))
            if native:
                shape.from_pixels(native.get("cx", 0), native.get("cy", 0), native.get("outer_r", 0))
                shape.inner_r = native.get("inner_r", 0) / min(w, h) if w and h else 0
            shape.class_id = cid
        elif t == "hollow_ellipse":
            from core.ring_shape import HollowEllipseShape
            shape = HollowEllipseShape(image_size=(w, h))
            if native:
                shape.from_pixels(native.get("cx", 0), native.get("cy", 0), native.get("outer_rx", 0), native.get("outer_ry", 0))
                shape.inner_rx = native.get("inner_rx", 0) / w if w else 0
                shape.inner_ry = native.get("inner_ry", 0) / h if h else 0
            shape.class_id = cid

        if shape:
            ann_id = data.get("id", "")
            if ann_id.startswith("ann_"):
                shape.id = ann_id[4:]
            else:
                shape.id = ann_id

            shape.is_hollow = data.get("is_hollow", False)
            shape.group_id = data.get("group_id", None)
            shape.hollow_role = data.get("hollow_role", None)

            inner_data = data.get("inner")
            if inner_data:
                inner_shape = self.deserialize_shape(inner_data, w, h)
                if inner_shape:
                    # do not recursively call attach_inner if it overwrites group_id unexpectedly,
                    # but spec says call outer_shape.attach_inner(inner_shape)
                    shape.attach_inner(inner_shape)

        return shape

    def load_image_annotations(self, image_filename):
        if not self.annotations_dir: return None
        json_path = os.path.join(self.annotations_dir, f"{image_filename}.json")
        if not os.path.exists(json_path):
            return {"hash_mismatch": False, "layers": {"yolo": {"visible": False, "annotations": []}, "unet": {"visible": True, "annotations": []}, "concentric": {"visible": False, "annotations": []}}}  # CONCENTRIC INTEGRATION
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            img_path = os.path.join(self.image_folder, image_filename)
            stored_hash = data.get("image_hash")
            current_hash = self.compute_image_hash(img_path)
            if stored_hash and current_hash and stored_hash != current_hash:
                data["hash_mismatch"] = True
            else:
                data["hash_mismatch"] = False
                
            return data
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            return {"hash_mismatch": False, "layers": {"yolo": {"visible": False, "annotations": []}, "unet": {"visible": True, "annotations": []}, "concentric": {"visible": False, "annotations": []}}}  # CONCENTRIC INTEGRATION

    def save_image_annotations(self, image_filename, canvas_shapes, mode, width, height):
        if not self.annotations_dir: return
        json_path = os.path.join(self.annotations_dir, f"{image_filename}.json")
        img_path = os.path.join(self.image_folder, image_filename)
        
        # Load existing so we only update current layer
        data = self.load_image_annotations(image_filename)
        if "schema_version" not in data:
            data = {
                "schema_version": "1.0",
                "image_file": image_filename,
                "image_hash": self.compute_image_hash(img_path),
                "image_width": width,
                "image_height": height,
                "status": "in_progress",
                "active_mode": mode,
                "last_modified": datetime.now().isoformat(),
                "last_exported_at": None,  # EXPORT INTEGRATION
                "layers": {
                    "yolo":        {"visible": mode == "yolo",        "annotations": []},
                    "unet":        {"visible": mode == "unet",        "annotations": []},
                    "concentric":  {"visible": mode == "concentric",  "annotations": []}  # CONCENTRIC INTEGRATION
                }
            }
        
        data["last_modified"] = datetime.now().isoformat()
        data["active_mode"] = mode
        
        layer_data = []
        for sh in canvas_shapes:
            if getattr(sh, "hollow_role", None) == "inner":
                continue # Handled by outer
            layer_data.append(self._serialize_shape(sh))
            
        # CONCENTRIC INTEGRATION — backfill missing layer for older JSON files
        if "concentric" not in data.get("layers", {}):
            data.setdefault("layers", {})["concentric"] = {
                "visible": False, "annotations": []}
        if "unet" not in data.get("layers", {}):
            data["layers"]["unet"] = {"visible": False, "annotations": []}

        data["layers"][mode]["annotations"] = layer_data
        
        prev_status = data.get("status", "unannotated")
        if prev_status == "unannotated" and layer_data:
            data["status"] = "in_progress"
            self.set_image_status(image_filename, "in_progress", write_json=False)
            
        self._write_json_atomic(json_path, data)
        self.update_project_stats()

    def schedule_autosave(self, image_filename, canvas_shapes, mode, width, height):
        self._pending_image = image_filename
        self._pending_shapes = list(canvas_shapes)
        self._pending_mode = mode
        self._pending_width = width
        self._pending_height = height
        self._autosave_timer.start()

    def flush_autosave(self):
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
            self._do_autosave()

    def _do_autosave(self):
        if self._pending_image and self._pending_shapes is not None:
            self.save_image_annotations(
                self._pending_image, self._pending_shapes,
                self._pending_mode, self._pending_width, self._pending_height
            )

    def set_layer_visibility(self, image_filename, mode, visible):
        if not self.annotations_dir: return
        json_path = os.path.join(self.annotations_dir, f"{image_filename}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if mode in data.get("layers", {}):
                    data["layers"][mode]["visible"] = visible
                    self._write_json_atomic(json_path, data)
            except Exception:
                pass

    def get_image_status(self, image_filename):
        if not self.project_data: return "unannotated"
        return self.project_data.get("image_statuses", {}).get(image_filename, "unannotated")

    def set_image_status(self, image_filename, status, write_json=True):
        if not self.project_data: return
        valid = ["unannotated", "in_progress", "annotated", "skipped"]
        if status not in valid: return
        self.project_data["image_statuses"][image_filename] = status
        if write_json and self.annotations_dir:
            json_path = os.path.join(self.annotations_dir, f"{image_filename}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["status"] = status
                    self._write_json_atomic(json_path, data)
                except Exception:
                    pass
        self.update_project_stats()

    # EXPORT INTEGRATION
    def mark_as_exported(self, image_filename: str) -> None:
        if not self.annotations_dir: return
        json_path = os.path.join(self.annotations_dir, f"{image_filename}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["last_exported_at"] = datetime.now().isoformat()
                self._write_json_atomic(json_path, data)
            except Exception:
                pass

    def update_project_stats(self):
        if not self.project_data: return
        stats = {"total_images": 0, "annotated_images": 0, "in_progress_images": 0, "skipped_images": 0}
        statuses = self.project_data.get("image_statuses", {})
        stats["total_images"] = len(statuses) if statuses else getattr(self, '_total_images_count', 0)
        for s in statuses.values():
            if s == "annotated": stats["annotated_images"] += 1
            elif s == "in_progress": stats["in_progress_images"] += 1
            elif s == "skipped": stats["skipped_images"] += 1
        self.project_data["stats"] = stats
        self._write_json_atomic(self.project_json_path, self.project_data)

    def update_project_classes(self, classes):
        """Update project class list. classes should be list of dicts with id, name, color."""
        if not self.project_data: return
        self.project_data["classes"] = classes
        self._write_json_atomic(self.project_json_path, self.project_data)

    def get_project_stats(self):
        if self.project_data:
            return self.project_data.get("stats", {})
        return {}
