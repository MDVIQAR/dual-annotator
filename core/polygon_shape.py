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
            
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'move'):
            self.inner_shape.move(dx, dy)
        
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
                    
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'get_resize_handles'):
            inner_handles = self.inner_shape.get_resize_handles()
            for k, (hx, hy) in inner_handles.items():
                handles[f'inner_{k}'] = (hx, hy)
            
        return handles
    
    def resize_from_handle(self, handle_name, dx, dy):
        """Move a vertex or midpoint - dx, dy are cumulative delta from resize start"""
        if handle_name.startswith('inner_') and getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'resize_from_handle'):
            # It's an inner shape handle passed up
            inner_handle = handle_name.replace('inner_', '', 1)
            # Avoid intercepting old hardcoded inner ring logic if inner_handle is just a digit '0', '1', etc 
            if not inner_handle.isdigit():
                return self.inner_shape.resize_from_handle(inner_handle, dx, dy)
                
        if self._resize_origin is None:
            return False
        
        result = False
        
        if handle_name.startswith('vertex_'):
            idx = int(handle_name.split('_')[1])
            if 0 <= idx < len(self._resize_origin):
                orig_nx, orig_ny = self._resize_origin[idx]
                self.points[idx] = (
                    orig_nx + dx / self.image_width,
                    orig_ny + dy / self.image_height
                )
                result = True
        
        elif handle_name.startswith('mid_') and not handle_name.startswith('mid_inner_'):
            actual_idx = self._mid_vertex_idx
            if actual_idx is not None and 0 <= actual_idx < len(self._resize_origin):
                orig_nx, orig_ny = self._resize_origin[actual_idx]
                self.points[actual_idx] = (
                    orig_nx + dx / self.image_width,
                    orig_ny + dy / self.image_height
                )
                result = True
            
        elif handle_name.startswith('mid_inner_'):
            actual_idx = self._mid_inner_vertex_idx
            if actual_idx is not None and 0 <= actual_idx < len(self._inner_origin):
                orig_nx, orig_ny = self._inner_origin[actual_idx]
                self.inner_points[actual_idx] = (
                    orig_nx + dx / self.image_width,
                    orig_ny + dy / self.image_height
                )
                result = True
            
        elif handle_name.startswith('inner_'):
            idx = int(handle_name.split('_')[1])
            if hasattr(self, '_inner_origin') and 0 <= idx < len(self._inner_origin):
                orig_nx, orig_ny = self._inner_origin[idx]
                self.inner_points[idx] = (
                    orig_nx + dx / self.image_width,
                    orig_ny + dy / self.image_height
                )
                result = True
        
        # After any resize, constrain inner points/shape to stay within outer polygon
        if result and len(self.points) >= 3:
            if self.inner_points:
                self._constrain_inner_points()
            if getattr(self, 'inner_shape', None):
                self._constrain_inner_shape_object()

        return result
    
    def _constrain_inner_points(self):
        """Clamp each inner point so it stays inside the outer polygon
        and at least a small distance away from the outer boundary."""
        if not self.inner_points or len(self.points) < 3:
            return

        # Minimum gap from outer boundary in pixel space
        MIN_GAP_PX = 4.0

        # Compute outer centroid (normalized coords)
        cx = sum(p[0] for p in self.points) / len(self.points)
        cy = sum(p[1] for p in self.points) / len(self.points)

        # Pre-compute outer polygon vertices in pixel coordinates for distance checks
        outer_px = self.to_pixel_points()

        def _point_segment_distance(px, py, x1, y1, x2, y2):
            """Return distance from point (px, py) to segment (x1, y1)-(x2, y2)."""
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                # Degenerate segment
                return math.hypot(px - x1, py - y1)
            t = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            return math.hypot(px - proj_x, py - proj_y)

        for i, (ix, iy) in enumerate(self.inner_points):
            best_x, best_y = ix, iy

            # First, ensure the point is inside the outer polygon.
            if not self._point_in_polygon(ix, iy, self.points):
                # Project the point back inside: move it toward the outer centroid
                # until it is inside
                for t_int in range(1, 21):
                    t = t_int / 20.0
                    tx = ix + (cx - ix) * t
                    ty = iy + (cy - iy) * t
                    if self._point_in_polygon(tx, ty, self.points):
                        best_x, best_y = tx, ty
                        break
                else:
                    # Fallback: place at 90% toward centroid
                    best_x = ix + (cx - ix) * 0.9
                    best_y = iy + (cy - iy) * 0.9
            # Now also enforce a minimum distance from the outer boundary.
            # Work in pixel space for a consistent visual gap.
            for _ in range(10):
                px = int(best_x * self.image_width)
                py = int(best_y * self.image_height)
                min_dist = float("inf")
                for j in range(len(outer_px)):
                    x1, y1 = outer_px[j]
                    x2, y2 = outer_px[(j + 1) % len(outer_px)]
                    dist = _point_segment_distance(px, py, x1, y1, x2, y2)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist >= MIN_GAP_PX:
                    break
                # Move a bit further toward centroid and try again.
                best_x = best_x + (cx - best_x) * 0.3
                best_y = best_y + (cy - best_y) * 0.3

            self.inner_points[i] = (best_x, best_y)
    
    def _constrain_inner_shape_object(self):
        """Constrain an attached inner_shape object so it stays inside the
        outer polygon's bounding box, with a small safety gap."""
        inner = getattr(self, 'inner_shape', None)
        if not inner or len(self.points) < 3:
            return

        MIN_GAP = 4  # pixels

        # Compute outer polygon bounding box in pixel space
        outer_px = self.to_pixel_points()
        ox_min = min(p[0] for p in outer_px)
        ox_max = max(p[0] for p in outer_px)
        oy_min = min(p[1] for p in outer_px)
        oy_max = max(p[1] for p in outer_px)

        # Try to get inner bounding box via to_pixels() or similar
        if hasattr(inner, 'to_pixels'):
            result = inner.to_pixels()
            # Could be (cx, cy, rx, ry) for circles/ellipses or pixel point list for polygons
            if isinstance(result, (list, tuple)) and len(result) == 4 and not isinstance(result[0], (list, tuple)):
                cx, cy, rx, ry = result
                # Clamp radii
                max_rx = max(1, (ox_max - ox_min) // 2 - MIN_GAP)
                max_ry = max(1, (oy_max - oy_min) // 2 - MIN_GAP)
                rx = min(rx, max_rx)
                ry = min(ry, max_ry)
                # Clamp center so bounding box stays inside outer bbox
                cx = max(ox_min + rx + MIN_GAP, min(cx, ox_max - rx - MIN_GAP))
                cy = max(oy_min + ry + MIN_GAP, min(cy, oy_max - ry - MIN_GAP))
                if hasattr(inner, 'from_pixels'):
                    inner.from_pixels(int(cx), int(cy), int(rx), int(ry))
        # For polygon-type inner shapes, constrain via move if centre is out
        elif hasattr(inner, 'to_pixel_points') and hasattr(inner, 'move'):
            pts = inner.to_pixel_points()
            if pts:
                ix_min = min(p[0] for p in pts)
                ix_max = max(p[0] for p in pts)
                iy_min = min(p[1] for p in pts)
                iy_max = max(p[1] for p in pts)
                shift_x = shift_y = 0
                if ix_min < ox_min + MIN_GAP:
                    shift_x = (ox_min + MIN_GAP - ix_min) / self.image_width
                elif ix_max > ox_max - MIN_GAP:
                    shift_x = (ox_max - MIN_GAP - ix_max) / self.image_width
                if iy_min < oy_min + MIN_GAP:
                    shift_y = (oy_min + MIN_GAP - iy_min) / self.image_height
                elif iy_max > oy_max - MIN_GAP:
                    shift_y = (oy_max - MIN_GAP - iy_max) / self.image_height
                if shift_x or shift_y:
                    inner.move(shift_x, shift_y)

    @staticmethod
    def _point_in_polygon(x, y, polygon):
        """Ray-casting point-in-polygon test on normalized coords"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
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
        
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'begin_resize'):
            inner_handle = handle_name.replace('inner_', '', 1) if handle_name and handle_name.startswith('inner_') else None
            self.inner_shape.begin_resize(inner_handle)
            
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