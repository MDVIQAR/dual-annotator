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
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'move'):
            self.inner_shape.move(dx, dy)
        
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
        
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'get_resize_handles'):
            inner_handles = self.inner_shape.get_resize_handles()
            for k, (hx, hy) in inner_handles.items():
                handles[f'inner_{k}'] = (hx, hy)
                
        return handles
    
    def begin_resize(self):
        """Store original geometry before resizing starts"""
        self._resize_origin = self.to_pixels()  # Stores (cx, cy, r)
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'begin_resize'):
            self.inner_shape.begin_resize()
        return True
        
    def _constrain_inner_shape(self):
        """Ensure inner circle stays fully inside the outer circle."""
        inner = getattr(self, 'inner_shape', None)
        if not inner or not hasattr(inner, 'to_pixels') or not hasattr(inner, 'from_pixels'):
            return
        min_gap = 4
        cx_o, cy_o, r_o = self.to_pixels()
        cx_i, cy_i, r_i = inner.to_pixels()
        # Clamp inner radius so inner stays inside outer with gap
        r_i = max(5, min(r_i, r_o - min_gap))
        # Clamp inner center so inner circle stays inside outer
        dist = math.sqrt((cx_i - cx_o) ** 2 + (cy_i - cy_o) ** 2)
        if dist + r_i > r_o - min_gap and dist > 1e-6:
            new_dist = max(0, r_o - min_gap - r_i)
            scale = new_dist / dist
            cx_i = cx_o + (cx_i - cx_o) * scale
            cy_i = cy_o + (cy_i - cy_o) * scale
        inner.from_pixels(int(cx_i), int(cy_i), int(r_i))

    def resize_from_handle(self, handle_name, dx, dy):
        MIN_GAP = 4
        MIN_R   = 4

        # ── Inner handle drag ──────────────────────────────────────────
        if handle_name.startswith('inner_'):
            if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'resize_from_handle'):
                inner_handle = handle_name.replace('inner_', '', 1)
                
                if hasattr(self.inner_shape, 'to_dict'):
                    old_state = self.inner_shape.to_dict()
                    result = self.inner_shape.resize_from_handle(inner_handle, dx, dy)
                    if not result: return False
                    
                    cx_o, cy_o, r_o = self.to_pixels()
                    # It might be a circle, ellipse, or polygon inside
                    if hasattr(self.inner_shape, 'to_pixels'):
                        res = self.inner_shape.to_pixels()
                        if len(res) == 3: # Circle
                            cx_i, cy_i, r_i = res
                            dist = math.hypot(cx_i - cx_o, cy_i - cy_o)
                            if dist + r_i > r_o - MIN_GAP:
                                restored = self.inner_shape.__class__.from_dict(old_state, (self.image_width, self.image_height))
                                for k, v in restored.__dict__.items():
                                    if not k.startswith('_') and k != 'id':
                                        setattr(self.inner_shape, k, v)
                                return False
                        elif len(res) == 4: # Box or Ellipse
                            ix1, iy1, ix2, iy2 = res
                            bcx = (ix1 + ix2) / 2
                            bcy = (iy1 + iy2) / 2
                            br = math.hypot(ix1 - bcx, iy1 - bcy)
                            if math.hypot(bcx - cx_o, bcy - cy_o) + br > r_o - MIN_GAP:
                                restored = self.inner_shape.__class__.from_dict(old_state, (self.image_width, self.image_height))
                                for k, v in restored.__dict__.items():
                                    if not k.startswith('_') and k != 'id':
                                        setattr(self.inner_shape, k, v)
                                return False
                    return True
                else:
                    return self.inner_shape.resize_from_handle(inner_handle, dx, dy)
            return False

        # ── Outer handle drag ──────────────────────────────────────────
        if self._resize_origin is None:
            return False

        orig_cx, orig_cy, orig_r = self._resize_origin

        left   = orig_cx - orig_r
        right  = orig_cx + orig_r
        top    = orig_cy - orig_r
        bottom = orig_cy + orig_r

        if handle_name == 'top_left':
            left += dx;  top    += dy
        elif handle_name == 'top_right':
            right += dx; top    += dy
        elif handle_name == 'bottom_left':
            left += dx;  bottom += dy
        elif handle_name == 'bottom_right':
            right += dx; bottom += dy
        else:
            return False

        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1

        new_cx = (left + right) / 2
        new_cy = (top + bottom) / 2
        new_r  = min((right - left), (bottom - top)) / 2

        min_radius = 5
        max_radius = min(self.image_width, self.image_height) // 2
        new_r = max(min_radius, min(new_r, max_radius))

        # Clamp: outer radius must always be larger than inner radius + gap
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'to_pixels'):
            res = self.inner_shape.to_pixels()
            if len(res) == 3: # Circle inside
                cx_i, cy_i, r_i = res
                dist = math.hypot(cx_i - new_cx, cy_i - new_cy)
                if dist + r_i > new_r - MIN_GAP:
                    return False
            elif len(res) == 4: # Ellipse or Box inside
                ix1, iy1, ix2, iy2 = res
                bcx = (ix1 + ix2) / 2
                bcy = (iy1 + iy2) / 2
                br = math.hypot(ix1 - bcx, iy1 - bcy)
                if math.hypot(bcx - new_cx, bcy - new_cy) + br > new_r - MIN_GAP:
                    return False

        self.center_x = new_cx / self.image_width
        self.center_y = new_cy / self.image_height
        self.radius   = new_r / max(self.image_width, self.image_height)

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