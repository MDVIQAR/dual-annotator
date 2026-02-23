# core/annotation.py
import uuid
import numpy as np

class BoundingBox:
    """Represents a YOLO-style bounding box annotation"""
    
    def __init__(self, x=0, y=0, width=0, height=0, class_id=None, image_size=(1, 1)):
        self.id = str(uuid.uuid4())[:8]
        self.class_id = class_id
        self.x = x  # Center x (normalized)
        self.y = y  # Center y (normalized)
        self.width = width  # Width (normalized)
        self.height = height  # Height (normalized)
        self.image_width, self.image_height = image_size
        self.selected = False
        self.created_at = None
        
    def from_pixels(self, x1, y1, x2, y2, image_width, image_height):
        """Convert pixel coordinates to normalized YOLO format"""
        self.image_width = image_width
        self.image_height = image_height
        
        # Convert to center x, center y, width, height (normalized)
        box_width = abs(x2 - x1)
        box_height = abs(y2 - y1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        self.x = center_x / image_width
        self.y = center_y / image_height
        self.width = box_width / image_width
        self.height = box_height / image_height
        
    def to_pixels(self):
        """Convert normalized coordinates to pixel coordinates"""
        center_x = self.x * self.image_width
        center_y = self.y * self.image_height
        box_width = self.width * self.image_width
        box_height = self.height * self.image_height
        
        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)
        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)
        
        return x1, y1, x2, y2
    
    def contains_point(self, px, py):
        """Check if point is inside the box (pixel coordinates)"""
        x1, y1, x2, y2 = self.to_pixels()
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def to_yolo_string(self):
        """Convert to YOLO format string"""
        return f"{self.class_id} {self.x:.6f} {self.y:.6f} {self.width:.6f} {self.height:.6f}"
    
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'id': self.id,
            'class_id': self.class_id,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }
    
    @classmethod
    def from_dict(cls, data, image_size):
        """Create from dictionary"""
        box = cls(
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            class_id=data['class_id'],
            image_size=image_size
        )
        box.id = data['id']
        return box