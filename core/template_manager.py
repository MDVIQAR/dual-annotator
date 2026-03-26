# core/template_manager.py
"""Session-only polygon template storage for stamp tool"""


class TemplateManager:
    """Manages polygon templates in memory (session-only)"""
    
    def __init__(self):
        self.templates = {}  # name -> list of (dx, dy) offsets from center
    
    def add_template(self, name, pixel_points, image_width, image_height, inner_pixel_points=None, shape_type='polygon', ctrl_points=None, inner_ctrl_points=None, native_params=None):
        """
        Store a polygon template normalized to its own bounding box center.
        Points are stored as offsets from center, normalized by bounding box size,
        so the template can be scaled proportionally when stamped.
        """
        if not name or len(pixel_points) < 3:
            return False
        
        # Find bounding box (using all points including inner ones to perfectly center it)
        all_points = list(pixel_points)
        if inner_pixel_points:
            all_points.extend(inner_pixel_points)
            
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        
        if bbox_w < 5 or bbox_h < 5:
            return False
        
        # Store offsets from center, normalized by bbox half-size
        # So each point is in range roughly [-1, 1] relative to center
        half_w = bbox_w / 2
        half_h = bbox_h / 2
        
        normalized_points = []
        for px, py in pixel_points:
            normalized_points.append((
                (px - cx) / half_w,
                (py - cy) / half_h
            ))
            
        normalized_inner = []
        if inner_pixel_points:
            for px, py in inner_pixel_points:
                normalized_inner.append((
                    (px - cx) / half_w,
                    (py - cy) / half_h
                ))
                
        normalized_ctrl = []
        if ctrl_points:
            for c in ctrl_points:
                if c is not None:
                    normalized_ctrl.append(((c[0] - cx) / half_w, (c[1] - cy) / half_h))
                else:
                    normalized_ctrl.append(None)
                    
        normalized_inner_ctrl = []
        if inner_ctrl_points:
            for c in inner_ctrl_points:
                if c is not None:
                    normalized_inner_ctrl.append(((c[0] - cx) / half_w, (c[1] - cy) / half_h))
                else:
                    normalized_inner_ctrl.append(None)
        
        self.templates[name] = {
            'type': shape_type,
            'orig_w': bbox_w,
            'orig_h': bbox_h,
            'points': normalized_points,
            'inner_points': normalized_inner,
            'ctrl': normalized_ctrl,
            'inner_ctrl': normalized_inner_ctrl,
            'native': native_params or {}
        }
        return True
    
    def get_template(self, name):
        """Get template points (normalized offsets from center)"""
        return self.templates.get(name, None)
        
    def get_template_info(self, name):
        """Return dict with orig_w, orig_h, type"""
        template_obj = self.templates.get(name)
        if isinstance(template_obj, dict):
            return {
                'orig_w': template_obj.get('orig_w', 100),
                'orig_h': template_obj.get('orig_h', 100),
                'type': template_obj.get('type', 'polygon')
            }
        return {'orig_w': 100, 'orig_h': 100, 'type': 'polygon'}
    
    def list_templates(self):
        """Get list of template names"""
        return list(self.templates.keys())
    
    def delete_template(self, name):
        """Delete a template"""
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def get_pixel_points(self, name, center_x, center_y, scale_x, scale_y):
        """
        Get template points as pixel coordinates, placed at (center_x, center_y)
        and scaled by (scale_x, scale_y) which represent the half-width and half-height.
        Returns: (type, outer_points, inner_points, ctrl_points, inner_ctrl_points, native_params)
        """
        template_obj = self.templates.get(name)
        if not template_obj:
            return 'polygon', [], [], [], [], {}
            
        # Handle old template format if needed
        if not isinstance(template_obj, dict):
            points = template_obj
            shape_type = 'polygon'
            inner_points = []
            ctrl_points = []
            inner_ctrl_points = []
            native_params = {}
        else:
            points = template_obj.get('points', [])
            inner_points = template_obj.get('inner_points', [])
            shape_type = template_obj.get('type', 'polygon')
            ctrl_points = template_obj.get('ctrl', [])
            inner_ctrl_points = template_obj.get('inner_ctrl', [])
            native_params = template_obj.get('native', {})
        
        outer_result = []
        for dx, dy in points:
            px = center_x + dx * scale_x
            py = center_y + dy * scale_y
            outer_result.append((int(px), int(py)))
            
        inner_result = []
        for dx, dy in inner_points:
            px = center_x + dx * scale_x
            py = center_y + dy * scale_y
            inner_result.append((int(px), int(py)))
            
        ctrl_result = []
        for c in ctrl_points:
            if c is not None:
                dx, dy = c
                px = center_x + dx * scale_x
                py = center_y + dy * scale_y
                ctrl_result.append((int(px), int(py)))
            else:
                ctrl_result.append(None)
                
        inner_ctrl_result = []
        for c in inner_ctrl_points:
            if c is not None:
                dx, dy = c
                px = center_x + dx * scale_x
                py = center_y + dy * scale_y
                inner_ctrl_result.append((int(px), int(py)))
            else:
                inner_ctrl_result.append(None)
            
        return shape_type, outer_result, inner_result, ctrl_result, inner_ctrl_result, native_params

    def save_to_file(self, filepath):
        """Persist all templates to JSON."""
        import json
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Failed to save templates: {e}")
            return False

    def load_from_file(self, filepath):
        """Load templates from JSON if it exists."""
        import json, os
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Basic validation
            if isinstance(data, dict):
                for name, tmpl in data.items():
                    if isinstance(tmpl, dict) and 'points' in tmpl:
                        self.templates[name] = tmpl
            print(f"📋 Loaded {len(self.templates)} templates from {os.path.basename(filepath)}")
        except Exception as e:
            print(f"❌ Failed to load templates: {e}")
