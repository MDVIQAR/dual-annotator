# core/ellipse_shape.py
import uuid
import math
from .shape_base import Shape

class EllipseShape(Shape):
    """Ellipse shape for segmentation"""
    
    def __init__(self, center=(0, 0), radius_x=0, radius_y=0, class_id=None, image_size=(1, 1)):
        super().__init__(class_id, image_size)
        self.type = 'ellipse'
        self.center_x, self.center_y = center  # Normalized coordinates
        self.radius_x = radius_x  # Normalized horizontal radius
        self.radius_y = radius_y  # Normalized vertical radius
        self._resize_origin = None # Store original state for resizing
        
    def from_pixels(self, center_x, center_y, radius_x, radius_y):
        """Set from pixel coordinates"""
        self.center_x = center_x / self.image_width
        self.center_y = center_y / self.image_height
        self.radius_x = radius_x / self.image_width
        self.radius_y = radius_y / self.image_height
        
    def to_pixels(self):
        """Convert to pixel coordinates"""
        cx = int(self.center_x * self.image_width)
        cy = int(self.center_y * self.image_height)
        rx = int(self.radius_x * self.image_width)
        ry = int(self.radius_y * self.image_height)
        return cx, cy, rx, ry
    
    def contains_point(self, x, y):
        """Check if point is inside the ellipse using ellipse equation"""
        cx, cy, rx, ry = self.to_pixels()
        if rx == 0 or ry == 0:
            return False
        normalized_x = ((x - cx) / rx) ** 2
        normalized_y = ((y - cy) / ry) ** 2
        return (normalized_x + normalized_y) <= 1
    
    def move(self, dx, dy):
        """Move the ellipse by delta (normalized)"""
        self.center_x += dx
        self.center_y += dy
        
    def get_resize_handles(self):
        """Get resize handles - simple corner handles"""
        cx, cy, rx, ry = self.to_pixels()
        
        # Use 4 corner handles like a box
        handles = {
            'top_left': (cx - rx, cy - ry),
            'top_right': (cx + rx, cy - ry),
            'bottom_left': (cx - rx, cy + ry),
            'bottom_right': (cx + rx, cy + ry)
        }
        return handles
    
    def resize_from_handle(self, handle_name, dx, dy):
        if self._resize_origin is None:
            return False

        orig_cx, orig_cy, orig_rx, orig_ry = self._resize_origin

        left = orig_cx - orig_rx
        right = orig_cx + orig_rx
        top = orig_cy - orig_ry
        bottom = orig_cy + orig_ry

        if handle_name == 'top_left':
            left += dx
            top += dy
        elif handle_name == 'top_right':
            right += dx
            top += dy
        elif handle_name == 'bottom_left':
            left += dx
            bottom += dy
        elif handle_name == 'bottom_right':
            right += dx
            bottom += dy
        else:
            return False

        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1

        new_cx = (left + right) / 2
        new_cy = (top + bottom) / 2
        new_rx = (right - left) / 2
        new_ry = (bottom - top) / 2

        min_size = 5
        max_rx = self.image_width // 2
        max_ry = self.image_height // 2

        new_rx = max(min_size, min(new_rx, max_rx))
        new_ry = max(min_size, min(new_ry, max_ry))

        self.center_x = new_cx / self.image_width
        self.center_y = new_cy / self.image_height
        self.radius_x = new_rx / self.image_width
        self.radius_y = new_ry / self.image_height

        return True
    
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'id': self.id,
            'type': 'ellipse',
            'class_id': self.class_id,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'radius_x': self.radius_x,
            'radius_y': self.radius_y
        }
    def begin_resize(self):
        """Store original geometry before resizing starts"""
        self._resize_origin = self.to_pixels()  # Stores (cx, cy, rx, ry)
        return True
        
    @classmethod
    def from_dict(cls, data, image_size):
        """Create from dictionary"""
        ellipse = cls(
            center=(data['center_x'], data['center_y']),
            radius_x=data['radius_x'],
            radius_y=data['radius_y'],
            class_id=data['class_id'],
            image_size=image_size
        )
        ellipse.id = data['id']
        return ellipse