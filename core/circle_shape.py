# core/circle_shape.py
import uuid
import math
from .shape_base import Shape

class CircleShape(Shape):
    """Circle shape for segmentation"""
    
    def __init__(self, center=(0, 0), radius=0, class_id=None, image_size=(1, 1)):
        super().__init__(class_id, image_size)
        self.type = 'circle'
        self.center_x, self.center_y = center  # Normalized coordinates
        self.radius = radius  # Normalized radius
        self._resize_origin = None  # Store original state for resizing
        
    def from_pixels(self, center_x, center_y, radius):
        """Set from pixel coordinates"""
        self.center_x = center_x / self.image_width
        self.center_y = center_y / self.image_height
        self.radius = radius / max(self.image_width, self.image_height)
        
    def to_pixels(self):
        """Convert to pixel coordinates"""
        cx = int(self.center_x * self.image_width)
        cy = int(self.center_y * self.image_height)
        r = int(self.radius * max(self.image_width, self.image_height))
        return cx, cy, r
    
    def contains_point(self, x, y):
        """Check if point is inside the circle"""
        cx, cy, r = self.to_pixels()
        distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return distance <= r
    
    def move(self, dx, dy):
        """Move the circle by delta (normalized)"""
        self.center_x += dx
        self.center_y += dy
        
    def get_resize_handles(self):
        """Get resize handles - only corner handles for simplicity"""
        cx, cy, r = self.to_pixels()
        
        # Just use 4 corner handles like a box
        handles = {
            'top_left': (cx - r, cy - r),
            'top_right': (cx + r, cy - r),
            'bottom_left': (cx - r, cy + r),
            'bottom_right': (cx + r, cy + r)
        }
        return handles
    
    def resize_from_handle(self, handle_name, dx, dy):
        if self._resize_origin is None:
            return False

        orig_cx, orig_cy, orig_r = self._resize_origin

        # Original bounding box
        left = orig_cx - orig_r
        right = orig_cx + orig_r
        top = orig_cy - orig_r
        bottom = orig_cy + orig_r

        # Apply delta relative to ORIGINAL box
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
        new_r = min((right - left), (bottom - top)) / 2

        min_radius = 5
        max_radius = min(self.image_width, self.image_height) // 2
        new_r = max(min_radius, min(new_r, max_radius))

        self.center_x = new_cx / self.image_width
        self.center_y = new_cy / self.image_height
        self.radius = new_r / max(self.image_width, self.image_height)

        return True
    
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'id': self.id,
            'type': 'circle',
            'class_id': self.class_id,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'radius': self.radius
        }
    
    def begin_resize(self):
        """Store original geometry before resizing starts"""
        self._resize_origin = self.to_pixels()  # Stores (cx, cy, r)
        return True
        
    @classmethod
    def from_dict(cls, data, image_size):
        """Create from dictionary"""
        circle = cls(
            center=(data['center_x'], data['center_y']),
            radius=data['radius'],
            class_id=data['class_id'],
            image_size=image_size
        )
        circle.id = data['id']
        return circle