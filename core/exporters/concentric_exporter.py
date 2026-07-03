# core/exporters/concentric_exporter.py
"""
Concentric zone mask exporter.

Exports annotations from 'concentric' mode as PNG segmentation masks.

Inner-wins rule: the first-drawn shape (index 0 in the annotations list)
has the highest priority. Every later shape has all earlier shapes'
regions subtracted before its pixels are written to the mask.

Output structure (with split):
    output_dir/train/images/stem.png
    output_dir/train/masks/stem.png
    output_dir/overlays/stem.png
    output_dir/class_map.json

Output structure (without split):
    output_dir/images/stem.png
    output_dir/masks/stem.png
    output_dir/overlays/stem.png
    output_dir/class_map.json
"""

import os
import json
from PIL import Image, ImageDraw, ImageEnhance


class ConcentricExporter:

    def __init__(self, output_dir, split_enabled, ordered_classes,
                 mask_mode, class_manager):
        self.output_dir    = output_dir
        self.split_enabled = split_enabled
        self.ordered_classes = ordered_classes
        self.mask_mode     = mask_mode      # "binary_01" | "binary_255" | "multiclass_id" | "multiclass_255" | "auto_01" | "auto_255"
        self.class_manager = class_manager

    # ------------------------------------------------------------------

    def export_image(self, filename, json_data, class_index,
                     output_dir, split_name, image_path):
        """
        Render mask + overlay for one image using the inner-wins rule.
        Annotations are processed in draw order (index 0 = innermost = highest priority).
        Returns count of valid annotations written.
        """
        import PIL.ImageChops as IC

        # ── Output directories ──────────────────────────────────────────
        if self.split_enabled:
            img_dir  = os.path.join(output_dir, split_name, "images")
            mask_dir = os.path.join(output_dir, split_name, "masks")
        else:
            img_dir  = os.path.join(output_dir, "images")
            mask_dir = os.path.join(output_dir, "masks")
        overlay_dir = os.path.join(output_dir, "overlays")

        os.makedirs(img_dir,     exist_ok=True)
        os.makedirs(mask_dir,    exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        iw, ih = img.size

        # ── Blank mask ──────────────────────────────────────────────────
        mask = Image.new("L", (iw, ih), 0)

        # ── Collect valid annotations in draw order ─────────────────────
        # index 0 = first drawn = innermost = highest priority
        active_mode = json_data.get("active_mode", "concentric")
        annotations = (json_data.get("layers", {})
                                 .get(active_mode, {})
                                 .get("annotations", []))

        valid = []
        for ann in annotations:
            if ann.get("hollow_role") == "inner":
                continue
            cid = ann.get("class_id")
            if not cid or cid not in class_index:
                continue
            valid.append(ann)

        if not valid:
            return 0

        # ── Apply inner-wins rule to build mask ─────────────────────────
        # cumulative_mask tracks all pixels claimed by earlier annotations.
        # Each new annotation has those pixels zeroed out before being pasted.
        cumulative_mask = Image.new("L", (iw, ih), 0)
        zero_layer      = Image.new("L", (iw, ih), 0)

        for ann in valid:
            cid       = ann["class_id"]
            class_idx = class_index[cid]
            pv        = self._pixel_value(class_idx, len(valid))

            # Draw this annotation at full size onto a temporary canvas
            temp      = Image.new("L", (iw, ih), 0)
            temp_draw = ImageDraw.Draw(temp)
            self._draw_ann(temp_draw, ann, iw, ih, pv)

            # Cut out all previously claimed pixels
            # composite(img1, img2, mask): img1 where mask=255, img2 where mask=0
            cum_binary = cumulative_mask.point(lambda p: 255 if p > 0 else 0)
            temp_cut   = Image.composite(zero_layer, temp, cum_binary)

            # Paste surviving pixels into the final mask
            nonzero = temp_cut.point(lambda p: 255 if p > 0 else 0)
            mask.paste(temp_cut, mask=nonzero)

            # Mark this annotation's FULL area as claimed (prevents future
            # annotations from overwriting it — even with different pixel values)
            temp_binary     = temp.point(lambda p: 255 if p > 0 else 0)
            cumulative_mask = IC.lighter(cumulative_mask, temp_binary)

        # ── Save outputs ────────────────────────────────────────────────
        stem = os.path.splitext(filename)[0]
        mask.save(os.path.join(mask_dir, stem + ".png"))
        img.save(os.path.join(img_dir,   stem + ".png"))
        self._write_overlay(img, valid, class_index, overlay_dir, stem)

        return len(valid)

    # ------------------------------------------------------------------

    def _draw_ann(self, draw, ann, iw, ih, pixel_value):
        """Draw one annotation onto a grayscale ImageDraw context."""
        t   = ann.get("shape_type")
        np_ = ann.get("native_params", {})
        pts = ann.get("points", [])

        if t == "box":
            cx = np_.get("cx_norm", 0) * iw
            cy = np_.get("cy_norm", 0) * ih
            w  = np_.get("w_norm",  0) * iw
            h  = np_.get("h_norm",  0) * ih
            draw.rectangle([cx-w/2, cy-h/2, cx+w/2, cy+h/2], fill=pixel_value)

        elif t == "circle":
            cx, cy, r = np_.get("cx",0), np_.get("cy",0), np_.get("r",0)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=pixel_value)

        elif t == "ellipse":
            cx, cy = np_.get("cx",0), np_.get("cy",0)
            rx, ry = np_.get("rx",0), np_.get("ry",0)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=pixel_value)

        elif t == "polygon":
            if len(pts) >= 3:
                draw.polygon([tuple(p) for p in pts], fill=pixel_value)

        elif t == "bezier_polygon":
            ctrl = ann.get("ctrl_points", [])
            tess = self._tessellate(pts, ctrl)
            if len(tess) >= 3:
                draw.polygon(tess, fill=pixel_value)

        elif t == "frame":
            x1,y1 = np_.get("x1",0), np_.get("y1",0)
            x2,y2 = np_.get("x2",0), np_.get("y2",0)
            x1,x2 = min(x1,x2), max(x1,x2)
            y1,y2 = min(y1,y2), max(y1,y2)
            draw.rectangle([x1,y1,x2,y2], fill=pixel_value)
            t_top    = np_.get("t_top",    np_.get("thickness", 20))
            t_bottom = np_.get("t_bottom", np_.get("thickness", 20))
            t_left   = np_.get("t_left",   np_.get("thickness", 20))
            t_right  = np_.get("t_right",  np_.get("thickness", 20))
            ix1 = x1 + t_left;  iy1 = y1 + t_top
            ix2 = x2 - t_right; iy2 = y2 - t_bottom
            if ix2 > ix1 and iy2 > iy1:
                draw.rectangle([ix1,iy1,ix2,iy2], fill=0)

        elif t == "donut":
            cx,cy   = np_.get("cx",0), np_.get("cy",0)
            outer_r = np_.get("outer_r",0)
            inner_r = np_.get("inner_r",0)
            draw.ellipse([cx-outer_r, cy-outer_r, cx+outer_r, cy+outer_r],
                         fill=pixel_value)
            if inner_r > 0:
                draw.ellipse([cx-inner_r, cy-inner_r, cx+inner_r, cy+inner_r],
                             fill=0)

        elif t == "hollow_ellipse":
            cx,cy = np_.get("cx",0), np_.get("cy",0)
            orx, ory = np_.get("outer_rx",0), np_.get("outer_ry",0)
            irx, iry = np_.get("inner_rx",0), np_.get("inner_ry",0)
            draw.ellipse([cx-orx, cy-ory, cx+orx, cy+ory], fill=pixel_value)
            if irx > 0 and iry > 0:
                draw.ellipse([cx-irx, cy-iry, cx+irx, cy+iry], fill=0)

    # ------------------------------------------------------------------

    @staticmethod
    def _tessellate(points, ctrl_points, steps=20):
        """Quadratic bezier tessellation (same as unet_exporter)."""
        if not points or len(points) < 3:
            return [tuple(p) for p in points] if points else []
        n = len(points)
        ctrls = ctrl_points or []
        result = []
        for i in range(n):
            p0 = points[i]; p1 = points[(i+1) % n]
            ctrl = ctrls[i] if i < len(ctrls) else None
            if ctrl is None:
                result.append((float(p0[0]), float(p0[1])))
            else:
                for s in range(steps):
                    t = s / steps
                    x = (1-t)**2*p0[0] + 2*(1-t)*t*ctrl[0] + t**2*p1[0]
                    y = (1-t)**2*p0[1] + 2*(1-t)*t*ctrl[1] + t**2*p1[1]
                    result.append((float(x), float(y)))
        return result

    # ------------------------------------------------------------------

    def _pixel_value(self, class_idx, n_classes):
        """
        Always return a distinct pixel value per class index.
        Concentric mode requires separate values for each zone.

        mask_mode controls the scale:
          'multiclass_id'  / 'binary_01'  -> 1-based integer  (1, 2, 3, ...)
          'multiclass_255' / 'binary_255' -> spread across 255 (85, 170, 255, ...)
        """
        n = max(len(self.ordered_classes), 1)
        use_255 = self.mask_mode in ("binary_255", "multiclass_255", "auto_255")
        if use_255:
            return max(1, round((class_idx + 1) / n * 255))
        else:
            return class_idx + 1

    # ------------------------------------------------------------------

    def _write_overlay(self, img, valid_anns, class_index, overlay_dir, stem):
        """
        Write a coloured overlay PNG for visual QA.

        Shapes are drawn in REVERSE order (outer first, inner last) so that
        inner zones always appear on top of outer zones in the final image.
        This matches the inner-wins visual rule without needing cutout logic.
        """
        brightened = ImageEnhance.Brightness(img).enhance(1.2)
        overlay    = brightened.convert("RGBA")
        draw_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw       = ImageDraw.Draw(draw_layer)

        # Reversed: outer drawn first (bottom), inner drawn last (top)
        for ann in reversed(valid_anns):
            cid     = ann.get("class_id")
            cls_obj = self.class_manager.get_class(cid)
            hex_c   = cls_obj.color if cls_obj else "#00FF00"
            r = int(hex_c[1:3], 16)
            g = int(hex_c[3:5], 16)
            b = int(hex_c[5:7], 16)
            self._draw_ann_rgba(draw, ann, img.width, img.height,
                                (r, g, b, 110), (255, 255, 255, 255))

        result = Image.alpha_composite(overlay, draw_layer).convert("RGB")
        result.save(os.path.join(overlay_dir, stem + ".png"))

    def _draw_ann_rgba(self, draw, ann, iw, ih, fill, outline):
        """Draw one annotation in RGBA for overlay (outline only, no cutout)."""
        t   = ann.get("shape_type")
        np_ = ann.get("native_params", {})
        pts = ann.get("points", [])
        ow  = 1

        if t == "box":
            cx = np_.get("cx_norm",0)*iw; cy = np_.get("cy_norm",0)*ih
            w  = np_.get("w_norm", 0)*iw; h  = np_.get("h_norm", 0)*ih
            bbox = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
            draw.rectangle(bbox, fill=fill, outline=outline, width=ow)
        elif t == "circle":
            cx,cy,r = np_.get("cx",0),np_.get("cy",0),np_.get("r",0)
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], fill=fill, outline=outline, width=ow)
        elif t == "ellipse":
            cx,cy=np_.get("cx",0),np_.get("cy",0)
            rx,ry=np_.get("rx",0),np_.get("ry",0)
            draw.ellipse([cx-rx,cy-ry,cx+rx,cy+ry],fill=fill,outline=outline,width=ow)
        elif t == "polygon":
            if len(pts) >= 3:
                flat = [tuple(p) for p in pts]
                draw.polygon(flat, fill=fill)
                draw.line(flat + [flat[0]], fill=outline, width=ow)
        elif t == "bezier_polygon":
            ctrl = ann.get("ctrl_points", [])
            tess = self._tessellate(pts, ctrl)
            if len(tess) >= 3:
                draw.polygon(tess, fill=fill)
                draw.line(tess + [tess[0]], fill=outline, width=ow)
        elif t == "frame":
            x1,y1 = np_.get("x1",0), np_.get("y1",0)
            x2,y2 = np_.get("x2",0), np_.get("y2",0)
            x1,x2 = min(x1,x2), max(x1,x2)
            y1,y2 = min(y1,y2), max(y1,y2)
            draw.rectangle([x1,y1,x2,y2], fill=fill, outline=outline, width=ow)
        elif t == "donut":
            cx,cy=np_.get("cx",0),np_.get("cy",0)
            or_=np_.get("outer_r",0); ir=np_.get("inner_r",0)
            draw.ellipse([cx-or_,cy-or_,cx+or_,cy+or_],fill=fill,outline=outline,width=ow)
            if ir > 0:
                draw.ellipse([cx-ir,cy-ir,cx+ir,cy+ir],fill=(0,0,0,0),outline=outline,width=ow)
        elif t == "hollow_ellipse":
            cx,cy=np_.get("cx",0),np_.get("cy",0)
            orx,ory=np_.get("outer_rx",0),np_.get("outer_ry",0)
            irx,iry=np_.get("inner_rx",0),np_.get("inner_ry",0)
            draw.ellipse([cx-orx,cy-ory,cx+orx,cy+ory],fill=fill,outline=outline,width=ow)
            if irx > 0 and iry > 0:
                draw.ellipse([cx-irx,cy-iry,cx+irx,cy+iry],fill=(0,0,0,0),outline=outline,width=ow)

    # ------------------------------------------------------------------

    def write_class_map(self, output_dir):
        data = {"mask_mode": self.mask_mode, "background": 0, "classes": []}
        for i, cls in enumerate(self.ordered_classes):
            data["classes"].append({
                "id": i + 1, "name": cls.name, "color": cls.color})
        path = os.path.join(output_dir, "class_map.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
