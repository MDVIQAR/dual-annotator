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
        self._resize_origin = None
        
        # Hollow pair support
        self.inner_shape = None
        self.hollow_role = None  # 'outer' | 'inner' | None
        self.group_id = None
        self.is_hollow = False

    def copy(self):
        """Create a deep copy of this bounding box, including any hollow inner shape."""
        new_box = BoundingBox(
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            class_id=self.class_id,
            image_size=(self.image_width, self.image_height)
        )
        new_box.id = str(uuid.uuid4())[:8]
        new_box.is_hollow = self.is_hollow
        new_box.hollow_role = self.hollow_role

        # Deep-copy the inner shape if this is a hollow pair
        if self.inner_shape is not None and hasattr(self.inner_shape, 'copy'):
            new_inner = self.inner_shape.copy()
            new_gid = str(uuid.uuid4())[:8]
            new_box.inner_shape = new_inner
            new_box.group_id = new_gid
            new_inner.hollow_role = 'inner'
            new_inner.group_id = new_gid
        else:
            new_box.inner_shape = None
            new_box.group_id = self.group_id

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
        
    def move(self, dx, dy):
        """Move the bounding box by delta (normalized)"""
        self.x += dx
        self.y += dy
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'move'):
            self.inner_shape.move(dx, dy)
    
    def get_resize_handles(self):
        """Get the positions of resize handles (corners)"""
        x1, y1, x2, y2 = self.to_pixels()
        
        handles = {
            'top_left': (x1, y1),
            'top_right': (x2, y1),
            'bottom_left': (x1, y2),
            'bottom_right': (x2, y2)
        }
        
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'get_resize_handles'):
            inner_handles = self.inner_shape.get_resize_handles()
            for k, (hx, hy) in inner_handles.items():
                handles[f'inner_{k}'] = (hx, hy)
                
        return handles
    
    def begin_resize(self):
        """Store original geometry before resizing starts"""
        self._resize_origin = self.to_pixels()  # Stores (x1, y1, x2, y2)
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'begin_resize'):
            self.inner_shape.begin_resize()
        return True
    
    def resize_from_handle(self, handle_name, dx, dy):
        """Resize the box from a specific handle"""
        if handle_name.startswith('inner_'):
            if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'resize_from_handle'):
                inner_handle = handle_name.replace('inner_', '', 1)
                result = self.inner_shape.resize_from_handle(inner_handle, dx, dy)
                if result:
                    self._constrain_inner_shape()
                return result
            return False
            
        if self._resize_origin is None:
            return False
        
        orig_x1, orig_y1, orig_x2, orig_y2 = self._resize_origin
        
        # Start with original coordinates
        left = orig_x1
        right = orig_x2
        top = orig_y1
        bottom = orig_y2
        
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
        
        # Prevent inversion
        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1

        # Ensure outer always contains inner + gap (hollow shapes)
        min_gap = 4
        if getattr(self, 'inner_shape', None) and isinstance(self.inner_shape, BoundingBox):
            ix1, iy1, ix2, iy2 = self.inner_shape.to_pixels()
            left = min(left, ix1 - min_gap)
            right = max(right, ix2 + min_gap)
            top = min(top, iy1 - min_gap)
            bottom = max(bottom, iy2 + min_gap)
            if right <= left:
                right = left + 1
            if bottom <= top:
                bottom = top + 1
        
        # Ensure coordinates stay within image bounds
        left = max(0, min(left, self.image_width - 1))
        right = max(0, min(right, self.image_width - 1))
        top = max(0, min(top, self.image_height - 1))
        bottom = max(0, min(bottom, self.image_height - 1))
        
        # Calculate new center and dimensions
        new_center_x = (left + right) / 2
        new_center_y = (top + bottom) / 2
        new_width = right - left
        new_height = bottom - top
        
        # Update normalized coordinates
        self.x = new_center_x / self.image_width
        self.y = new_center_y / self.image_height
        self.width = new_width / self.image_width
        self.height = new_height / self.image_height
        
        # If this shape has a hollow inner, prevent outer from shrinking
        # so small that it would pass through or swallow the inner shape.
        if getattr(self, 'inner_shape', None) and isinstance(self.inner_shape, BoundingBox):
            MIN_GAP = 4
            ix1, iy1, ix2, iy2 = self.inner_shape.to_pixels()
            inner_w = ix2 - ix1
            inner_h = iy2 - iy1
            # Outer must be at least inner + gap on every side
            min_outer_w = inner_w + 2 * MIN_GAP
            min_outer_h = inner_h + 2 * MIN_GAP
            if new_width < min_outer_w or new_height < min_outer_h:
                # Reject this resize — outer would swallow inner
                return False
            self._constrain_inner_shape()

        return True
    
    def _constrain_inner_shape(self):
        """Ensure inner shape stays fully within the outer box.
        Works for BoundingBox inner shapes only — circle/ellipse inner
        shapes are constrained directly in their own resize_from_handle."""
        inner = self.inner_shape
        if not inner or not isinstance(inner, BoundingBox):
            return

        MIN_GAP = 4
        ox1, oy1, ox2, oy2 = self.to_pixels()
        ix1, iy1, ix2, iy2 = inner.to_pixels()

        # Clamp inner to stay inside outer with gap on all sides
        ix1 = max(ix1, ox1 + MIN_GAP)
        iy1 = max(iy1, oy1 + MIN_GAP)
        ix2 = min(ix2, ox2 - MIN_GAP)
        iy2 = min(iy2, oy2 - MIN_GAP)

        # Enforce minimum inner size — but never push it outside the outer.
        # Compute how much space is actually available inside the outer.
        available_w = (ox2 - ox1) - 2 * MIN_GAP
        available_h = (oy2 - oy1) - 2 * MIN_GAP
        if available_w < 2 or available_h < 2:
            # Outer is too small to contain any inner shape — do nothing
            return

        min_w = min(10, available_w)
        min_h = min(10, available_h)

        if ix2 - ix1 < min_w:
            mid_x = (ox1 + ox2) / 2
            ix1 = mid_x - min_w / 2
            ix2 = mid_x + min_w / 2
        if iy2 - iy1 < min_h:
            mid_y = (oy1 + oy2) / 2
            iy1 = mid_y - min_h / 2
            iy2 = mid_y + min_h / 2

        inner.from_pixels(ix1, iy1, ix2, iy2, self.image_width, self.image_height)
    
    def to_yolo_string(self):
        """Convert to YOLO format string"""
        return f"{self.class_id} {self.x:.6f} {self.y:.6f} {self.width:.6f} {self.height:.6f}"
    
    def to_dict(self):
        """Convert to dictionary for saving"""
        d = {
            'id': self.id,
            'type': 'box',
            'class_id': self.class_id,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'is_hollow': self.is_hollow,
            'group_id': self.group_id,
            'hollow_role': self.hollow_role
        }
        if self.inner_shape and hasattr(self.inner_shape, 'to_dict'):
            d['inner_shape'] = self.inner_shape.to_dict()
        return d
    
    def attach_inner(self, inner_shape):
        """Link inner_shape to this shape as the hollow cutout."""
        gid = str(uuid.uuid4())[:8]
        self.inner_shape = inner_shape
        self.is_hollow = True
        self.hollow_role = 'outer'
        self.group_id = gid
        if hasattr(inner_shape, 'hollow_role'):
            inner_shape.hollow_role = 'inner'
            inner_shape.group_id = gid

    def detach_inner(self):
        """Detach inner shape and reset hollow state."""
        if self.inner_shape:
            if hasattr(self.inner_shape, 'hollow_role'):
                self.inner_shape.hollow_role = None
                self.inner_shape.group_id = None
        self.inner_shape = None
        self.is_hollow = False
        self.hollow_role = None
    
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