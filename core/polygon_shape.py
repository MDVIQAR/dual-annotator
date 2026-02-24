# core/polygon_shape.py
import uuid
import math
from .shape_base import Shape

class PolygonShape(Shape):
    """Polygon shape for segmentation"""
    
    def __init__(self, points=None, class_id=None, image_size=(1, 1)):
        super().__init__(class_id, image_size)
        self.type = 'polygon'
        self.points = points or []  # List of (x, y) tuples (normalized)
        self.closed = False
        
    def add_point(self, x, y):
        """Add a point to the polygon (normalized coordinates)"""
        self.points.append((x, y))
        
    def from_pixel_points(self, pixel_points):
        """Set points from pixel coordinates"""
        self.points = []
        for px, py in pixel_points:
            self.points.append((px / self.image_width, py / self.image_height))
            
    def to_pixel_points(self):
        """Convert to pixel coordinates"""
        pixel_points = []
        for nx, ny in self.points:
            px = int(nx * self.image_width)
            py = int(ny * self.image_height)
            pixel_points.append((px, py))
        return pixel_points
    
    def to_pixels(self):
        """Return pixel coordinates for drawing (compatible with other shapes)"""
        return self.to_pixel_points()
    
    def contains_point(self, x, y):
        """Check if point is inside polygon using ray casting algorithm"""
        if not self.closed or len(self.points) < 3:
            return False
            
        pixel_points = self.to_pixel_points()
        n = len(pixel_points)
        inside = False
        
        p1x, p1y = pixel_points[0]
        for i in range(1, n + 1):
            p2x, p2y = pixel_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def move(self, dx, dy):
        """Move the polygon by delta (normalized)"""
        new_points = []
        for nx, ny in self.points:
            new_points.append((nx + dx, ny + dy))
        self.points = new_points
        
    def get_resize_handles(self):
        """Get all vertices as resize handles"""
        pixel_points = self.to_pixel_points()
        handles = {}
        for i, (px, py) in enumerate(pixel_points):
            handles[f'vertex_{i}'] = (px, py)
        return handles
    
    def resize_from_handle(self, handle_name, dx, dy):
        """Move a vertex"""
        if handle_name.startswith('vertex_'):
            idx = int(handle_name.split('_')[1])
            if 0 <= idx < len(self.points):
                nx, ny = self.points[idx]
                self.points[idx] = (
                    nx + dx / self.image_width,
                    ny + dy / self.image_height
                )
                return True
        return False
    
    def close_polygon(self):
        """Close the polygon"""
        self.closed = True
        
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'id': self.id,
            'type': 'polygon',
            'class_id': self.class_id,
            'points': self.points,
            'closed': self.closed
        }
    
    @classmethod
    def from_dict(cls, data, image_size):
        """Create from dictionary"""
        polygon = cls(
            points=data['points'],
            class_id=data['class_id'],
            image_size=image_size
        )
        polygon.id = data['id']
        polygon.closed = data['closed']
        return polygon