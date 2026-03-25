import os
import json
import shutil
from PIL import Image, ImageDraw


class UNetExporter:
    """
    Exports annotations as UNet-compatible segmentation masks.
    Produces: PNG masks, source image copies, and colored overlay visualizations.
    """

    def __init__(self, output_dir, split_enabled, ordered_classes, mask_mode, class_manager):
        self.output_dir = output_dir
        self.split_enabled = split_enabled
        self.ordered_classes = ordered_classes
        self.mask_mode = mask_mode          # "binary_01" | "binary_255" | "multiclass_id" | "multiclass_255"
        self.class_manager = class_manager

    # ------------------------------------------------------------------
    #  Main export entry point
    # ------------------------------------------------------------------

    def export_image(self, filename, json_data, class_index, output_dir, split_name, image_path):
        """
        Renders mask + overlay for a single image.
        Returns count of valid (non-inner) annotations written.
        """
        # Step 1 — Resolve output directories
        if self.split_enabled:
            img_dir  = os.path.join(output_dir, split_name, "images")
            mask_dir = os.path.join(output_dir, split_name, "masks")
        else:
            img_dir  = os.path.join(output_dir, "images")
            mask_dir = os.path.join(output_dir, "masks")

        overlay_dir = os.path.join(output_dir, "overlays")  # always at root, never split

        os.makedirs(img_dir,     exist_ok=True)
        os.makedirs(mask_dir,    exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        # Step 2 — Open source image
        img = Image.open(image_path).convert("RGB")
        iw, ih = img.size

        # Step 3 — Create blank mask (L = grayscale)
        mask = Image.new("L", (iw, ih), 0)
        mask_draw = ImageDraw.Draw(mask)

        # Step 4 — Iterate annotations and draw on mask
        active_mode = json_data.get("active_mode", "yolo")
        annotations = json_data.get("layers", {}).get(active_mode, {}).get("annotations", [])

        written = 0
        # Count unique classes actually present in this image (non-inner, valid)
        unique_class_ids = set()
        for ann in annotations:
            if ann.get("hollow_role") == "inner":
                continue
            cid = ann.get("class_id")
            if cid and cid in class_index:
                unique_class_ids.add(cid)
        n_unique = len(unique_class_ids)

        for ann in annotations:
            if ann.get("hollow_role") == "inner":
                continue
            cid = ann.get("class_id")
            if not cid or cid not in class_index:
                continue
            pixel_value = self._get_pixel_value(class_index[cid], n_unique)
            self._draw_ann_on_mask(mask_draw, ann, iw, ih, pixel_value)
            # If this is a hollow outer shape, cut out the inner hole
            inner_ann = ann.get("inner")
            if inner_ann and ann.get("hollow_role") == "outer":
                self._cut_inner_from_mask(mask_draw, inner_ann, iw, ih)
            written += 1

        # Step 5 — Save mask and image as PNG
        stem = os.path.splitext(filename)[0]
        mask.save(os.path.join(mask_dir, stem + ".png"))
        img.save(os.path.join(img_dir, stem + ".png"))

        # Step 6 — Generate overlay
        overlay_base = img.convert("RGBA")
        draw_layer = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(draw_layer)

        for ann in annotations:
            if ann.get("hollow_role") == "inner":
                continue
            cid = ann.get("class_id")
            if not cid or cid not in class_index:
                continue
            cls_obj = self.class_manager.get_class(cid)
            color_hex = cls_obj.color if cls_obj else "#00FF00"
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            # Semi-transparent fill at alpha 170 for visibility on thermal images
            fill_rgba = (r, g, b, 170)
            # Bright opaque outline for sharp boundary
            outline_rgba = (255, 255, 255, 230)
            self._draw_ann_on_overlay(overlay_draw, ann, iw, ih, fill_rgba, outline_rgba)

        overlay = Image.alpha_composite(overlay_base, draw_layer).convert("RGB")
        overlay.save(os.path.join(overlay_dir, stem + ".png"))

        return written

    # ------------------------------------------------------------------
    #  Pixel value computation
    # ------------------------------------------------------------------

    def _get_pixel_value(self, class_idx, n_unique_classes=1):
        """
        Auto-detects binary vs multiclass based on how many distinct classes
        appear in this image, then scales by the user-chosen range (0/1 or 0/255).

        auto_01  / auto_255:
          - n_unique <= 1  → binary: all foreground = max scale value
          - n_unique  > 1  → multiclass: each class gets a distinct value

        Legacy modes are kept for backwards compatibility.
        """
        use_255 = self.mask_mode in ("auto_255", "binary_255", "multiclass_255")
        scale = 255 if use_255 else 1
        n = max(len(self.ordered_classes), 1)

        if self.mask_mode in ("auto_01", "auto_255"):
            if n_unique_classes <= 1:
                # Binary: single foreground value
                return scale
            else:
                # Multiclass: spread evenly across [1 .. scale]
                return max(1, round((class_idx + 1) / n * scale))

        # --- Legacy fixed modes ---
        if self.mask_mode == "binary_01":
            return 1
        elif self.mask_mode == "binary_255":
            return 255
        elif self.mask_mode == "multiclass_id":
            return class_idx + 1
        elif self.mask_mode == "multiclass_255":
            return round((class_idx + 1) / n * 255)
        return 1

    # ------------------------------------------------------------------
    #  Draw annotation on MASK (grayscale)
    # ------------------------------------------------------------------

    def _draw_ann_on_mask(self, draw, ann, iw, ih, pixel_value):
        t   = ann.get("shape_type")
        np_ = ann.get("native_params", {})
        pts = ann.get("points", [])

        if t == "box":
            cx = np_.get("cx_norm", 0) * iw
            cy = np_.get("cy_norm", 0) * ih
            w  = np_.get("w_norm",  0) * iw
            h  = np_.get("h_norm",  0) * ih
            draw.rectangle([cx - w/2, cy - h/2, cx + w/2, cy + h/2], fill=pixel_value)

        elif t == "circle":
            cx, cy, r = np_.get("cx", 0), np_.get("cy", 0), np_.get("r", 0)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=pixel_value)

        elif t == "ellipse":
            cx, cy = np_.get("cx", 0), np_.get("cy", 0)
            rx, ry = np_.get("rx", 0), np_.get("ry", 0)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=pixel_value)

        elif t in ("polygon", "bezier_polygon"):
            if len(pts) >= 3:
                draw.polygon([tuple(p) for p in pts], fill=pixel_value)
            # Handle hollow cutout stored in nested inner_shape
            inner = ann.get("inner_shape")
            if inner:
                inner_pts = inner.get("points", [])
                if len(inner_pts) >= 3:
                    draw.polygon([tuple(p) for p in inner_pts], fill=0)

        elif t == "frame":
            x1 = np_.get("x1", 0); y1 = np_.get("y1", 0)
            x2 = np_.get("x2", 0); y2 = np_.get("y2", 0)
            draw.rectangle([x1, y1, x2, y2], fill=pixel_value)
            # Compute inner rect from wall thickness
            t_top    = np_.get("t_top",    np_.get("thickness", 20))
            t_bottom = np_.get("t_bottom", np_.get("thickness", 20))
            t_left   = np_.get("t_left",   np_.get("thickness", 20))
            t_right  = np_.get("t_right",  np_.get("thickness", 20))
            ix1 = x1 + t_left;  iy1 = y1 + t_top
            ix2 = x2 - t_right; iy2 = y2 - t_bottom
            if ix2 > ix1 and iy2 > iy1:
                draw.rectangle([ix1, iy1, ix2, iy2], fill=0)

        elif t == "donut":
            cx, cy  = np_.get("cx", 0), np_.get("cy", 0)
            outer_r = np_.get("outer_r", 0)
            inner_r = np_.get("inner_r", 0)
            draw.ellipse([cx-outer_r, cy-outer_r, cx+outer_r, cy+outer_r], fill=pixel_value)
            if inner_r > 0:
                draw.ellipse([cx-inner_r, cy-inner_r, cx+inner_r, cy+inner_r], fill=0)

        elif t == "hollow_ellipse":
            cx, cy  = np_.get("cx", 0), np_.get("cy", 0)
            orx     = np_.get("outer_rx", 0); ory = np_.get("outer_ry", 0)
            irx     = np_.get("inner_rx", 0); iry = np_.get("inner_ry", 0)
            draw.ellipse([cx-orx, cy-ory, cx+orx, cy+ory], fill=pixel_value)
            if irx > 0 and iry > 0:
                draw.ellipse([cx-irx, cy-iry, cx+irx, cy+iry], fill=0)

    # ------------------------------------------------------------------
    #  Draw annotation on OVERLAY (RGBA, semi-transparent) + outline
    # ------------------------------------------------------------------

    def _draw_ann_on_overlay(self, draw, ann, iw, ih, rgba_color, outline_color=None):
        t   = ann.get("shape_type")
        np_ = ann.get("native_params", {})
        pts = ann.get("points", [])
        ow  = 2  # outline width in pixels

        if t == "box":
            cx = np_.get("cx_norm", 0) * iw
            cy = np_.get("cy_norm", 0) * ih
            w  = np_.get("w_norm",  0) * iw
            h  = np_.get("h_norm",  0) * ih
            bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
            draw.rectangle(bbox, fill=rgba_color)
            if outline_color:
                draw.rectangle(bbox, outline=outline_color, width=ow)

        elif t == "circle":
            cx, cy, r = np_.get("cx", 0), np_.get("cy", 0), np_.get("r", 0)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=rgba_color)
            if outline_color:
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=outline_color, width=ow)
            # Hollow outer/inner pair
            inner_ann = ann.get("inner")
            if inner_ann and ann.get("hollow_role") == "outer":
                self._cut_inner_from_overlay(draw, inner_ann, iw, ih)

        elif t == "ellipse":
            cx, cy = np_.get("cx", 0), np_.get("cy", 0)
            rx, ry = np_.get("rx", 0), np_.get("ry", 0)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=rgba_color)
            if outline_color:
                draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], outline=outline_color, width=ow)
            inner_ann = ann.get("inner")
            if inner_ann and ann.get("hollow_role") == "outer":
                self._cut_inner_from_overlay(draw, inner_ann, iw, ih)

        elif t in ("polygon", "bezier_polygon"):
            if len(pts) >= 3:
                flat = [tuple(p) for p in pts]
                draw.polygon(flat, fill=rgba_color)
                if outline_color:
                    draw.line(flat + [flat[0]], fill=outline_color, width=ow)
            inner = ann.get("inner_shape") or ann.get("inner")
            if inner:
                inner_pts = inner.get("points", [])
                if len(inner_pts) >= 3:
                    draw.polygon([tuple(p) for p in inner_pts], fill=(0, 0, 0, 0))

        elif t == "frame":
            x1 = np_.get("x1", 0); y1 = np_.get("y1", 0)
            x2 = np_.get("x2", 0); y2 = np_.get("y2", 0)
            draw.rectangle([x1, y1, x2, y2], fill=rgba_color)
            if outline_color:
                draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=ow)
            t_top    = np_.get("t_top",    np_.get("thickness", 20))
            t_bottom = np_.get("t_bottom", np_.get("thickness", 20))
            t_left   = np_.get("t_left",   np_.get("thickness", 20))
            t_right  = np_.get("t_right",  np_.get("thickness", 20))
            ix1 = x1 + t_left;  iy1 = y1 + t_top
            ix2 = x2 - t_right; iy2 = y2 - t_bottom
            if ix2 > ix1 and iy2 > iy1:
                draw.rectangle([ix1, iy1, ix2, iy2], fill=(0, 0, 0, 0))

        elif t == "donut":
            cx, cy  = np_.get("cx", 0), np_.get("cy", 0)
            outer_r = np_.get("outer_r", 0)
            inner_r = np_.get("inner_r", 0)
            draw.ellipse([cx-outer_r, cy-outer_r, cx+outer_r, cy+outer_r], fill=rgba_color)
            if outline_color:
                draw.ellipse([cx-outer_r, cy-outer_r, cx+outer_r, cy+outer_r], outline=outline_color, width=ow)
            if inner_r > 0:
                draw.ellipse([cx-inner_r, cy-inner_r, cx+inner_r, cy+inner_r], fill=(0, 0, 0, 0))
                if outline_color:
                    draw.ellipse([cx-inner_r, cy-inner_r, cx+inner_r, cy+inner_r], outline=outline_color, width=ow)

        elif t == "hollow_ellipse":
            cx, cy  = np_.get("cx", 0), np_.get("cy", 0)
            orx     = np_.get("outer_rx", 0); ory = np_.get("outer_ry", 0)
            irx     = np_.get("inner_rx", 0); iry = np_.get("inner_ry", 0)
            draw.ellipse([cx-orx, cy-ory, cx+orx, cy+ory], fill=rgba_color)
            if outline_color:
                draw.ellipse([cx-orx, cy-ory, cx+orx, cy+ory], outline=outline_color, width=ow)
            if irx > 0 and iry > 0:
                draw.ellipse([cx-irx, cy-iry, cx+irx, cy+iry], fill=(0, 0, 0, 0))
                if outline_color:
                    draw.ellipse([cx-irx, cy-iry, cx+irx, cy+iry], outline=outline_color, width=ow)

    # ------------------------------------------------------------------
    #  Helper: cut inner hole from mask / overlay
    # ------------------------------------------------------------------

    def _cut_inner_from_mask(self, draw, inner_ann, iw, ih):
        """Erase the inner shape from the mask (fill=0) for outer/inner hollow pairs."""
        t   = inner_ann.get("shape_type")
        np_ = inner_ann.get("native_params", {})
        pts = inner_ann.get("points", [])

        if t == "box":
            cx = np_.get("cx_norm", 0) * iw
            cy = np_.get("cy_norm", 0) * ih
            w  = np_.get("w_norm",  0) * iw
            h  = np_.get("h_norm",  0) * ih
            draw.rectangle([cx - w/2, cy - h/2, cx + w/2, cy + h/2], fill=0)
        elif t == "circle":
            cx, cy, r = np_.get("cx", 0), np_.get("cy", 0), np_.get("r", 0)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=0)
        elif t == "ellipse":
            cx, cy = np_.get("cx", 0), np_.get("cy", 0)
            rx, ry = np_.get("rx", 0), np_.get("ry", 0)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=0)
        elif t in ("polygon", "bezier_polygon"):
            if len(pts) >= 3:
                draw.polygon([tuple(p) for p in pts], fill=0)

    def _cut_inner_from_overlay(self, draw, inner_ann, iw, ih):
        """Erase the inner shape from the RGBA overlay (fill transparent) for outer/inner hollow pairs."""
        t   = inner_ann.get("shape_type")
        np_ = inner_ann.get("native_params", {})
        pts = inner_ann.get("points", [])

        if t == "box":
            cx = np_.get("cx_norm", 0) * iw
            cy = np_.get("cy_norm", 0) * ih
            w  = np_.get("w_norm",  0) * iw
            h  = np_.get("h_norm",  0) * ih
            draw.rectangle([cx - w/2, cy - h/2, cx + w/2, cy + h/2], fill=(0, 0, 0, 0))
        elif t == "circle":
            cx, cy, r = np_.get("cx", 0), np_.get("cy", 0), np_.get("r", 0)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0, 0, 0, 0))
        elif t == "ellipse":
            cx, cy = np_.get("cx", 0), np_.get("cy", 0)
            rx, ry = np_.get("rx", 0), np_.get("ry", 0)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(0, 0, 0, 0))
        elif t in ("polygon", "bezier_polygon"):
            if len(pts) >= 3:
                draw.polygon([tuple(p) for p in pts], fill=(0, 0, 0, 0))

    # ------------------------------------------------------------------
    #  Write class map JSON
    # ------------------------------------------------------------------

    def write_class_map(self, output_dir):
        data = {
            "mask_mode": self.mask_mode,
            "background": 0,
            "classes": []
        }
        for i, cls in enumerate(self.ordered_classes):
            data["classes"].append({
                "id": i + 1,
                "name": cls.name,
                "color": cls.color
            })
        path = os.path.join(output_dir, "class_map.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
