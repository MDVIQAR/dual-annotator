# core/template_manager.py
"""Session-only polygon template storage for stamp tool"""


class TemplateManager:
    """Manages polygon templates in memory (session-only)"""
    
    def __init__(self):
        self.templates = {}  # name -> list of (dx, dy) offsets from center
    
    def add_template(self, name, pixel_points, image_width, image_height):
        """
        Store a polygon template normalized to its own bounding box center.
        Points are stored as offsets from center, normalized by bounding box size,
        so the template can be scaled proportionally when stamped.
        """
        if not name or len(pixel_points) < 3:
            return False
        
        # Find bounding box
        xs = [p[0] for p in pixel_points]
        ys = [p[1] for p in pixel_points]
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
        
        self.templates[name] = normalized_points
        return True
    
    def get_template(self, name):
        """Get template points (normalized offsets from center)"""
        return self.templates.get(name, None)
    
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
        """
        points = self.templates.get(name)
        if not points:
            return []
        
        result = []
        for dx, dy in points:
            px = center_x + dx * scale_x
            py = center_y + dy * scale_y
            result.append((int(px), int(py)))
        return result
