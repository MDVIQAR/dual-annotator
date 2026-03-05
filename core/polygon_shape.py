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
        self.inner_points = []  # Inner cutout polygon (normalized), empty = solid
        self.closed = False
        self._resize_origin = None  # Store original points for resizing
        
    def add_point(self, x, y):
        """Add a point to the polygon (normalized coordinates)"""
        self.points.append((x, y))
        
    def from_pixel_points(self, pixel_points):
        """Set points from pixel coordinates"""
        self.points = []
        for px, py in pixel_points:
            self.points.append((px / self.image_width, py / self.image_height))
    
    def set_inner_from_pixels(self, pixel_points):
        """Set inner cutout from pixel coordinates"""
        self.inner_points = []
        for px, py in pixel_points:
            self.inner_points.append((px / self.image_width, py / self.image_height))
    
    def get_inner_pixel_points(self):
        """Get inner cutout as pixel coordinates"""
        return [(int(nx * self.image_width), int(ny * self.image_height)) for nx, ny in self.inner_points]
            
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
        # Also move inner cutout
        if self.inner_points:
            new_inner = []
            for nx, ny in self.inner_points:
                new_inner.append((nx + dx, ny + dy))
            self.inner_points = new_inner
        
    def get_resize_handles(self):
        """Get all vertices as resize handles, plus midpoint handles between edges"""
        pixel_points = self.to_pixel_points()
        handles = {}
        # Vertex handles
        for i, (px, py) in enumerate(pixel_points):
            handles[f'vertex_{i}'] = (px, py)
        # Midpoint handles (between consecutive vertices, including last→first if closed)
        if self.closed and len(pixel_points) >= 3:
            for i in range(len(pixel_points)):
                j = (i + 1) % len(pixel_points)
                mx = (pixel_points[i][0] + pixel_points[j][0]) // 2
                my = (pixel_points[i][1] + pixel_points[j][1]) // 2
                handles[f'mid_{i}'] = (mx, my)
            
        if self.inner_points:
            for i, p in enumerate(self.inner_points):
                px = int(p[0] * self.image_width)
                py = int(p[1] * self.image_height)
                handles[f'inner_{i}'] = (px, py)
                
            if len(self.inner_points) >= 3:
                for i in range(len(self.inner_points)):
                    j = (i + 1) % len(self.inner_points)
                    mx = (self.inner_points[i][0] + self.inner_points[j][0]) / 2
                    my = (self.inner_points[i][1] + self.inner_points[j][1]) / 2
                    handles[f'mid_inner_{i}'] = (int(mx * self.image_width), int(my * self.image_height))
            
        return handles
    
    def resize_from_handle(self, handle_name, dx, dy):
        """Move a vertex or midpoint - dx, dy are cumulative delta from resize start"""
        if self._resize_origin is None:
            return False
        
        if handle_name.startswith('vertex_'):
            idx = int(handle_name.split('_')[1])
            if 0 <= idx < len(self._resize_origin):
                orig_nx, orig_ny = self._resize_origin[idx]
                self.points[idx] = (
                    orig_nx + dx / self.image_width,
                    orig_ny + dy / self.image_height
                )
                return True
        
        elif handle_name.startswith('mid_') and not handle_name.startswith('mid_inner_'):
            # Midpoint handle — this was converted to a vertex in begin_resize
            # Find the actual vertex index for this midpoint
            mid_idx = int(handle_name.split('_')[1])
            # The inserted vertex is at position mid_idx + 1 + number of prior insertions
            # But since we only insert once at begin_resize, the handle maps to
            # the inserted index stored in _mid_vertex_idx
            actual_idx = self._mid_vertex_idx
            if actual_idx is not None and 0 <= actual_idx < len(self._resize_origin):
                orig_nx, orig_ny = self._resize_origin[actual_idx]
                self.points[actual_idx] = (
                    orig_nx + dx / self.image_width,
                    orig_ny + dy / self.image_height
                )
                return True
            
        elif handle_name.startswith('mid_inner_'):
            actual_idx = self._mid_inner_vertex_idx
            if actual_idx is not None and 0 <= actual_idx < len(self._inner_origin):
                orig_nx, orig_ny = self._inner_origin[actual_idx]
                self.inner_points[actual_idx] = (
                    orig_nx + dx / self.image_width,
                    orig_ny + dy / self.image_height
                )
                return True
            
        elif handle_name.startswith('inner_'):
            idx = int(handle_name.split('_')[1])
            if hasattr(self, '_inner_origin') and 0 <= idx < len(self._inner_origin):
                orig_nx, orig_ny = self._inner_origin[idx]
                self.inner_points[idx] = (
                    orig_nx + dx / self.image_width,
                    orig_ny + dy / self.image_height
                )
                return True
    
        return False
    
    def begin_resize(self, handle_name=None):
        """Store original points before resizing starts.
        If handle_name is a midpoint, insert a new vertex first."""
        self._mid_vertex_idx = None
        self._mid_inner_vertex_idx = None
        
        if handle_name and handle_name.startswith('mid_') and not handle_name.startswith('mid_inner_'):
            mid_idx = int(handle_name.split('_')[1])
            insert_at = mid_idx + 1
            p1 = self.points[mid_idx]
            p2 = self.points[(mid_idx + 1) % len(self.points)]
            mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            self.points.insert(insert_at, mid_point)
            self._mid_vertex_idx = insert_at
            
        elif handle_name and handle_name.startswith('mid_inner_'):
            mid_idx = int(handle_name.split('_')[2])
            insert_at = mid_idx + 1
            p1 = self.inner_points[mid_idx]
            p2 = self.inner_points[(mid_idx + 1) % len(self.inner_points)]
            mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            self.inner_points.insert(insert_at, mid_point)
            self._mid_inner_vertex_idx = insert_at
    
        self._resize_origin = list(self.points)
        self._inner_origin = list(self.inner_points) if self.inner_points else []
        return True
    
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
            'inner_points': self.inner_points,
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
        polygon.inner_points = data.get('inner_points', [])
        return polygon