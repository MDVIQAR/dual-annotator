# core/annotation.py
import uuid  # Added this import
import math

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
        self.type = 'box'
        self._resize_origin = None  # Add this line
        
    def copy(self):
        """Create a copy of this bounding box"""
        new_box = BoundingBox(
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            class_id=self.class_id,
            image_size=(self.image_width, self.image_height)
        )
        new_box.id = str(uuid.uuid4())[:8]
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
    
    def begin_resize(self):
        """Store original geometry before resizing starts"""
        self._resize_origin = self.to_pixels()  # Stores (x1, y1, x2, y2)
        print(f"📦 Box begin_resize() called, origin: {self._resize_origin}")
        return True
    
    def resize_from_handle(self, handle_name, dx, dy):
        """Resize the box from a specific handle"""
        if self._resize_origin is None:
            print("❌ Box _resize_origin is None")
            return False
        
        orig_x1, orig_y1, orig_x2, orig_y2 = self._resize_origin
        print(f"📐 Box original: ({orig_x1}, {orig_y1}) - ({orig_x2}, {orig_y2})")
        print(f"📐 Delta: ({dx}, {dy})")
        
        # Start with original coordinates
        left = orig_x1
        right = orig_x2
        top = orig_y1
        bottom = orig_y2
        
        # Apply delta relative to ORIGINAL box
        if handle_name == 'top_left':
            left += dx
            top += dy
            print(f"↖️ Moving top_left: left={left}, top={top}")
        elif handle_name == 'top_right':
            right += dx
            top += dy
            print(f"↗️ Moving top_right: right={right}, top={top}")
        elif handle_name == 'bottom_left':
            left += dx
            bottom += dy
            print(f"↙️ Moving bottom_left: left={left}, bottom={bottom}")
        elif handle_name == 'bottom_right':
            right += dx
            bottom += dy
            print(f"↘️ Moving bottom_right: right={right}, bottom={bottom}")
        else:
            print(f"❌ Unknown handle: {handle_name}")
            return False
        
        # Prevent inversion
        if right <= left:
            right = left + 1
            print(f"⚠️ Fixed right inversion: {right}")
        if bottom <= top:
            bottom = top + 1
            print(f"⚠️ Fixed bottom inversion: {bottom}")
        
        # Ensure coordinates stay within image bounds
        left = max(0, min(left, self.image_width - 1))
        right = max(0, min(right, self.image_width - 1))
        top = max(0, min(top, self.image_height - 1))
        bottom = max(0, min(bottom, self.image_height - 1))
        
        print(f"✨ New pixel bounds: left={left}, right={right}, top={top}, bottom={bottom}")
        
        # Calculate new center and dimensions
        new_center_x = (left + right) / 2
        new_center_y = (top + bottom) / 2
        new_width = right - left
        new_height = bottom - top
        
        print(f"✨ New values: center=({new_center_x}, {new_center_y}), size=({new_width}, {new_height})")
        
        # Update normalized coordinates
        self.x = new_center_x / self.image_width
        self.y = new_center_y / self.image_height
        self.width = new_width / self.image_width
        self.height = new_height / self.image_height
        
        print(f"✅ Final normalized: x={self.x:.3f}, y={self.y:.3f}, w={self.width:.3f}, h={self.height:.3f}")
        
        return True
    
    def to_yolo_string(self):
        """Convert to YOLO format string"""
        return f"{self.class_id} {self.x:.6f} {self.y:.6f} {self.width:.6f} {self.height:.6f}"
    
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'id': self.id,
            'type': 'box',
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