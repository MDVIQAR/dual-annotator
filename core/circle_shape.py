# core/circle_shape.py
import uuid
import math
import numpy as np
from .shape_base import Shape

class CircleShape(Shape):
    """Circle shape for segmentation"""
    
    def __init__(self, center=(0, 0), radius=0, class_id=None, image_size=(1, 1)):
        super().__init__(class_id, image_size)
        self.type = 'circle'
        self.center_x, self.center_y = center  # Normalized coordinates
        self.radius = radius  # Normalized radius
        
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
        """Get resize handles (center and 4 cardinal points)"""
        cx, cy, r = self.to_pixels()
        
        handles = {
            'center': (cx, cy),
            'top': (cx, cy - r),
            'right': (cx + r, cy),
            'bottom': (cx, cy + r),
            'left': (cx - r, cy)
        }
        return handles
    
    def resize_from_handle(self, handle_name, dx, dy):
        """Resize from a specific handle"""
        cx, cy, r = self.to_pixels()
        
        if handle_name == 'center':
            # Move the entire circle
            self.center_x += dx / self.image_width
            self.center_y += dy / self.image_height
        elif handle_name in ['top', 'bottom', 'left', 'right']:
            # Resize radius based on handle movement
            if handle_name == 'top' or handle_name == 'bottom':
                new_r = abs(cy + dy - cy) if handle_name == 'bottom' else abs(cy - (cy + dy))
            else:  # left or right
                new_r = abs(cx + dx - cx) if handle_name == 'right' else abs(cx - (cx + dx))
            
            if new_r > 5:  # Minimum radius
                self.radius = new_r / max(self.image_width, self.image_height)
        else:
            return False
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