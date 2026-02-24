# core/annotation.py
import uuid

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
        
    def copy(self):
        """Create a copy of this bounding box"""
        print(f"📋 Copying box {self.id}")  # Debug print
        new_box = BoundingBox(
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            class_id=self.class_id,
            image_size=(self.image_width, self.image_height)
        )
        new_box.id = str(uuid.uuid4())[:8]  # New unique ID
        print(f"✅ Created new box {new_box.id}")
        return new_box
        
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
    
    def get_resize_handles(self):
        """Get the positions of resize handles (corners)"""
        x1, y1, x2, y2 = self.to_pixels()
        
        handles = {
            'top_left': (x1, y1),
            'top_right': (x2, y1),
            'bottom_left': (x1, y2),
            'bottom_right': (x2, y2)
        }
        return handles
    
    def resize_from_handle(self, handle_name, dx, dy):
        """Resize the box from a specific handle"""
        x1, y1, x2, y2 = self.to_pixels()
        
        if handle_name == 'top_left':
            x1 += dx
            y1 += dy
        elif handle_name == 'top_right':
            x2 += dx
            y1 += dy
        elif handle_name == 'bottom_left':
            x1 += dx
            y2 += dy
        elif handle_name == 'bottom_right':
            x2 += dx
            y2 += dy
        
        # Ensure minimum size
        if x2 - x1 < 10 or y2 - y1 < 10:
            return False
        
        # Update normalized coordinates
        self.x = ((x1 + x2) / 2) / self.image_width
        self.y = ((y1 + y2) / 2) / self.image_height
        self.width = (x2 - x1) / self.image_width
        self.height = (y2 - y1) / self.image_height
        
        return True
    
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