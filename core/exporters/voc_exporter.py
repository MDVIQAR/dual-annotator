import os
import shutil
import xml.etree.ElementTree as ET
import xml.dom.minidom

class VocExporter:
    def __init__(self, output_dir, split_enabled, ordered_classes):
        self.output_dir = output_dir
        self.split_enabled = split_enabled
        self.ordered_classes = ordered_classes

    def export_image(self, filename, json_data, class_index,
                     output_dir, split_name, image_path):

        if self.split_enabled:
            anno_dir = os.path.join(output_dir, "Annotations", split_name)
            img_dir = os.path.join(output_dir, "JPEGImages", split_name)
        else:
            anno_dir = os.path.join(output_dir, "Annotations")
            img_dir = os.path.join(output_dir, "JPEGImages")

        os.makedirs(anno_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        iw = json_data.get("image_width", 1)
        ih = json_data.get("image_height", 1)
        
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "JPEGImages"
        ET.SubElement(annotation, "filename").text = filename
        dest_image = os.path.join(img_dir, filename)
        ET.SubElement(annotation, "path").text = os.path.abspath(dest_image)
        
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "DualAnnotator"
        
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(iw)
        ET.SubElement(size, "height").text = str(ih)
        ET.SubElement(size, "depth").text = "3"
        
        ET.SubElement(annotation, "segmented").text = "0"

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
            
            bbox = self._ann_to_voc_bbox(ann, iw, ih)
            if bbox is None:
                continue
                
            xmin, ymin, xmax, ymax = bbox
            
            cls_idx = class_index[cid]
            cls_name = "unknown"
            if cls_idx < len(self.ordered_classes):
                cls_name = self.ordered_classes[cls_idx].name
                
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = cls_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)
            
            written_anns += 1

        # Write xml formatted nicely
        xml_str = ET.tostring(annotation, encoding='unicode')
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # Remove extra blank lines from toprettyxml
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
        
        stem = os.path.splitext(filename)[0]
        xml_path = os.path.join(anno_dir, stem + ".xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        if not os.path.exists(dest_image):
            shutil.copy2(image_path, dest_image)

        return written_anns

    def _ann_to_voc_bbox(self, ann, iw, ih):
        t  = ann.get("shape_type")
        np = ann.get("native_params", {})
        pts = ann.get("points", [])

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
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
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

        xmin = int((cx - w_norm / 2) * iw)
        ymin = int((cy - h_norm / 2) * ih)
        xmax = int((cx + w_norm / 2) * iw)
        ymax = int((cy + h_norm / 2) * ih)

        # VOC bndbox requires xmin >= 1
        xmin = max(1, xmin)
        ymin = max(1, ymin)
        xmax = min(iw, xmax)
        ymax = min(ih, ymax)

        return xmin, ymin, xmax, ymax
