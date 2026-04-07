"""
gui/template_engine.py
Pure detection logic for template matching auto-annotation.
No Qt dependency — safe to import anywhere.
"""

import cv2
import numpy as np


class TemplateEngine:
    DEFAULT_THRESHOLD = 0.70
    NMS_FRACTION      = 0.85
    PADDING           = 6

    def __init__(self):
        self.threshold  = self.DEFAULT_THRESHOLD
        self.nms_fraction = self.NMS_FRACTION
        self.img  = None
        self.gray = None
        self.iw   = 0
        self.ih   = 0

    def load_image(self, image_path: str) -> bool:
        """Load image. Returns True on success."""
        img = cv2.imread(image_path)
        if img is None:
            return False
        self.img  = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.ih, self.iw = img.shape[:2]
        return True

    def build_template(self, boxes):
        """
        Build best template from list of (x1,y1,x2,y2) pixel boxes.
        Picks highest-contrast crop. Returns (template_array, (px1,py1,w,h)) or (None, None).
        """
        if not boxes:
            return None, None
        best_template = None
        best_score    = -1
        best_box      = None
        for (x1, y1, x2, y2) in boxes:
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            px1 = max(0, x1 - self.PADDING)
            py1 = max(0, y1 - self.PADDING)
            px2 = min(self.iw - 1, x2 + self.PADDING)
            py2 = min(self.ih - 1, y2 + self.PADDING)
            crop = self.gray[py1:py2, px1:px2]
            if crop.size == 0:
                continue
            score = float(crop.std())
            if score > best_score:
                best_score    = score
                best_template = crop
                best_box      = (px1, py1, px2 - px1, py2 - py1)
        return best_template, best_box

    def _nms(self, points, scores, min_dist_px):
        """Non-Maximum Suppression — removes overlapping detections."""
        if not points:
            return []
        order = sorted(range(len(points)), key=lambda i: scores[i], reverse=True)
        kept  = []
        for idx in order:
            pt = points[idx]
            too_close = False
            for k in kept:
                dx = abs(pt[0] - points[k][0])
                dy = abs(pt[1] - points[k][1])
                if dx < min_dist_px and dy < min_dist_px:
                    too_close = True
                    break
            if not too_close:
                kept.append(idx)
        return [points[i] for i in kept]

    def run_detection(self, template, template_box, threshold=None):
        """
        Run cv2.matchTemplate and return list of (x, y, w, h) pixel boxes.
        """
        if template is None or template.size == 0 or self.gray is None:
            return []
        th, tw = template.shape[:2]
        t = threshold if threshold is not None else self.threshold
        result = cv2.matchTemplate(self.gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= t)
        if len(xs) == 0:
            return []
        points = list(zip(xs.tolist(), ys.tolist()))
        scores  = [float(result[y, x]) for x, y in points]
        min_dist = max(tw, th) * self.nms_fraction
        kept_pts = self._nms(points, scores, min_dist)
        return [(x, y, tw, th) for x, y in kept_pts]

    def detect(self, template_boxes_px, existing_boxes_px=None):
        """
        Main entry point.
        template_boxes_px : list of (x1,y1,x2,y2) — shapes used as templates
        existing_boxes_px : list of (x1,y1,x2,y2) — shapes already on canvas
                            (new detections overlapping these are skipped)
        Returns list of (x, y, w, h) NEW detections only.
        """
        template, tmpl_box = self.build_template(template_boxes_px)
        if template is None:
            return []
        raw = self.run_detection(template, tmpl_box)
        if not existing_boxes_px:
            return raw
        # Smart merge: skip detections whose center falls inside an existing bbox
        filtered = []
        for (x, y, w, h) in raw:
            cx, cy   = x + w / 2.0, y + h / 2.0
            overlaps = False
            for (ex1, ey1, ex2, ey2) in existing_boxes_px:
                if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                    overlaps = True
                    break
            if not overlaps:
                filtered.append((x, y, w, h))
        return filtered

    @staticmethod
    def shape_to_pixel_bbox(shape, iw, ih):
        """
        Convert any canvas shape to (x1, y1, x2, y2) pixel bbox.
        Returns None if the shape type is unrecognised.
        """
        t = getattr(shape, 'type', None)
        try:
            if t == 'box':
                x1, y1, x2, y2 = shape.to_pixels()
                return (x1, y1, x2, y2)
            elif t == 'circle':
                cx, cy, r = shape.to_pixels()
                return (cx - r, cy - r, cx + r, cy + r)
            elif t == 'ellipse':
                cx, cy, rx, ry = shape.to_pixels()
                return (cx - rx, cy - ry, cx + rx, cy + ry)
            elif t == 'frame':
                x, y, w, h = shape.to_pixels()   # NOTE: returns x,y,w,h not x1,y1,x2,y2
                return (x, y, x + w, y + h)
            elif t in ('donut', 'hollow_ellipse'):
                pts = shape.to_pixels()
                cx, cy, rx = pts[0], pts[1], pts[2]
                ry = pts[3] if len(pts) > 3 else pts[2]
                return (cx - rx, cy - ry, cx + rx, cy + ry)
            elif t in ('polygon', 'bezier_polygon'):
                px_pts = shape.to_pixel_points()
                if not px_pts:
                    return None
                xs = [p[0] for p in px_pts]
                ys = [p[1] for p in px_pts]
                return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        except Exception:
            pass
        return None
