# core/ring_shape.py
import uuid
import math
import copy
from .shape_base import Shape

class FrameShape(Shape):
    """Hollow rectangle (frame) with wall thickness"""
    
    def __init__(self, x=0.0, y=0.0, w=0.2, h=0.15, thickness=20.0, class_id=None, image_size=(1, 1)):
        super().__init__(class_id, image_size)
        self.type = 'frame'
        self.id = str(uuid.uuid4())[:8]
        self.x = x  # left edge (normalized)
        self.y = y  # top edge (normalized)
        self.w = w  # width (normalized)
        self.h = h  # height (normalized)
        self.thickness = thickness  # wall thickness in pixels
        self.selected = False
        self._resize_origin = None
        
    def to_pixels(self):
        """Convert to pixel coordinates"""
        x = int(self.x * self.image_width)
        y = int(self.y * self.image_height)
        w = int(self.w * self.image_width)
        h = int(self.h * self.image_height)
        return x, y, w, h
    
    def from_pixels(self, x1, y1, x2, y2):
        """Set from pixel coordinates (for drawing)"""
        self.x = min(x1, x2) / self.image_width
        self.y = min(y1, y2) / self.image_height
        self.w = abs(x2 - x1) / self.image_width
        self.h = abs(y2 - y1) / self.image_height
        
    def get_inner_rect(self):
        """Get inner rectangle (hole) pixel coordinates"""
        x, y, w, h = self.to_pixels()
        t = min(self.thickness, min(w, h) // 2 - 1)
        ix = x + t
        iy = y + t
        iw = w - 2 * t
        ih = h - 2 * t
        return ix, iy, iw, ih
    
    def contains_point(self, px, py):
        """Check if point is in ring (in outer but not inner)"""
        x, y, w, h = self.to_pixels()
        ix, iy, iw, ih = self.get_inner_rect()
        
        in_outer = x <= px <= x + w and y <= py <= y + h
        in_inner = ix <= px <= ix + iw and iy <= py <= iy + ih
        
        return in_outer and not in_inner
    
    def move(self, dx, dy):
        """Move the frame by delta (normalized)"""
        self.x += dx
        self.y += dy
        # Keep within bounds
        self.x = max(0, min(1 - self.w, self.x))
        self.y = max(0, min(1 - self.h, self.y))
    
    def get_resize_handles(self):
        """8 outer handles for resizing - returns list of tuples for drawing"""
        x, y, w, h = self.to_pixels()
        return [
            (x, y),           # 0 TL
            (x + w//2, y),    # 1 TC
            (x + w, y),       # 2 TR
            (x, y + h//2),    # 3 ML
            (x + w, y + h//2),# 4 MR
            (x, y + h),       # 5 BL
            (x + w//2, y + h),# 6 BC
            (x + w, y + h),   # 7 BR
        ]
    
    def get_inner_handles(self):
        """4 inner handles for thickness adjustment - returns list of tuples"""
        ix, iy, iw, ih = self.get_inner_rect()
        return [
            (ix + iw//2, iy),      # top
            (ix + iw, iy + ih//2), # right
            (ix + iw//2, iy + ih), # bottom
            (ix, iy + ih//2),      # left
        ]
    
    def get_handle_at_pos(self, px, py, handle_size=14):
        """Check if position is over any handle and return handle info"""
        # Check outer handles
        for i, (hx, hy) in enumerate(self.get_resize_handles()):
            if abs(px - hx) <= handle_size and abs(py - hy) <= handle_size:
                return ('outer', i)
        
        # Check inner handles
        for i, (hx, hy) in enumerate(self.get_inner_handles()):
            if abs(px - hx) <= handle_size and abs(py - hy) <= handle_size:
                return ('inner', i)
        
        return None
    
    def resize_from_handle(self, handle_name, dx, dy):
        """Resize from handle - dx, dy are cumulative delta from resize start"""
        # Must have _resize_origin set by begin_resize()
        if self._resize_origin is None:
            return False
        
        orig_x, orig_y, orig_w, orig_h = self._resize_origin
        min_size = 20
        
        # Parse handle info
        if isinstance(handle_name, tuple):
            handle_type, idx = handle_name
        else:
            return False
        
        if handle_type == 'outer':
            # Start from ORIGINAL geometry (not current) to prevent compounding
            new_x, new_y = orig_x, orig_y
            new_w, new_h = orig_w, orig_h
            
            # Handle index mapping from get_resize_handles():
            # 0:TL  1:TC  2:TR  3:ML  4:MR  5:BL  6:BC  7:BR
            if idx in [0, 3, 5]:  # left side handles (TL, ML, BL)
                new_x = orig_x + dx
                new_w = orig_w - dx
            if idx in [2, 4, 7]:  # right side handles (TR, MR, BR)
                new_w = orig_w + dx
            if idx in [0, 1, 2]:  # top side handles (TL, TC, TR)
                new_y = orig_y + dy
                new_h = orig_h - dy
            if idx in [5, 6, 7]:  # bottom side handles (BL, BC, BR)
                new_h = orig_h + dy
            
            if new_w < min_size or new_h < min_size:
                return False
            
            self.x = new_x / self.image_width
            self.y = new_y / self.image_height
            self.w = new_w / self.image_width
            self.h = new_h / self.image_height
            return True
            
        elif handle_type == 'inner':
            # Use original thickness from _resize_origin to prevent compounding
            orig_x, orig_y, orig_w, orig_h = self._resize_origin
            orig_thickness = self._resize_orig_thickness
            
            new_t = orig_thickness
            if idx == 0:  # top handle - dragging down increases thickness
                new_t = orig_thickness + dy
            elif idx == 1:  # right handle - dragging left increases thickness
                new_t = orig_thickness - dx
            elif idx == 2:  # bottom handle - dragging up increases thickness
                new_t = orig_thickness - dy
            elif idx == 3:  # left handle - dragging right increases thickness
                new_t = orig_thickness + dx
            
            # Use current outer dimensions for max thickness
            x, y, w, h = self.to_pixels()
            max_t = min(w, h) // 2 - 2
            self.thickness = max(4, min(new_t, max_t))
            return True
        
        return False
    
    def begin_resize(self):
        """Store original geometry before resizing starts"""
        self._resize_origin = self.to_pixels()
        self._resize_orig_thickness = self.thickness
        return True
    
    def copy(self):
        """Create a copy of this frame"""
        new = FrameShape(
            x=self.x, y=self.y,
            w=self.w, h=self.h,
            thickness=self.thickness,
            class_id=self.class_id,
            image_size=(self.image_width, self.image_height)
        )
        new.id = str(uuid.uuid4())[:8]
        new.selected = self.selected
        return new


class DonutShape(Shape):
    """Hollow circle (donut) with inner radius"""
    
    def __init__(self, cx=0.5, cy=0.5, outer_r=0.12, inner_r=0.055, class_id=None, image_size=(1, 1)):
        super().__init__(class_id, image_size)
        self.type = 'donut'
        self.id = str(uuid.uuid4())[:8]
        self.cx = cx  # center x (normalized)
        self.cy = cy  # center y (normalized)
        self.outer_r = outer_r  # outer radius normalized to min(W,H)
        self.inner_r = inner_r  # inner radius normalized to min(W,H)
        self.selected = False
        self._resize_origin = None
        
    def to_pixels(self):
        """Convert to pixel coordinates"""
        cx = int(self.cx * self.image_width)
        cy = int(self.cy * self.image_height)
        d = min(self.image_width, self.image_height)
        outer_r = int(self.outer_r * d)
        inner_r = int(self.inner_r * d)
        return cx, cy, outer_r, inner_r
    
    def from_pixels(self, cx, cy, r):
        """Set from pixel coordinates (for drawing)"""
        self.cx = cx / self.image_width
        self.cy = cy / self.image_height
        d = min(self.image_width, self.image_height)
        self.outer_r = r / d
        self.inner_r = r * 0.45 / d  # Inner radius 45% of outer
    
    def contains_point(self, px, py):
        """Check if point is in ring (in outer but not inner)"""
        cx, cy, outer_r, inner_r = self.to_pixels()
        dist = math.sqrt((px - cx)**2 + (py - cy)**2)
        return dist <= outer_r and dist >= inner_r
    
    def move(self, dx, dy):
        """Move the donut by delta (normalized)"""
        self.cx += dx
        self.cy += dy
        # Keep within bounds
        d = min(self.image_width, self.image_height)
        max_r = self.outer_r * d / min(self.image_width, self.image_height)
        self.cx = max(max_r, min(1 - max_r, self.cx))
        self.cy = max(max_r, min(1 - max_r, self.cy))
    
    def get_outer_handles(self):
        """4 cardinal handles on outer circle - returns list of tuples"""
        cx, cy, outer_r, _ = self.to_pixels()
        return [
            (cx + outer_r, cy),      # east
            (cx, cy + outer_r),      # south
            (cx - outer_r, cy),      # west
            (cx, cy - outer_r),      # north
        ]
    
    def get_inner_handles(self):
        """4 cardinal handles on inner circle - returns list of tuples"""
        cx, cy, _, inner_r = self.to_pixels()
        return [
            (cx + inner_r, cy),      # east
            (cx, cy + inner_r),      # south
            (cx - inner_r, cy),      # west
            (cx, cy - inner_r),      # north
        ]
    
    def get_resize_handles(self):
        """Get all handles - returns list of tuples for drawing"""
        return self.get_outer_handles()
    
    def get_handle_at_pos(self, px, py, handle_size=14):
        """Check if position is over any handle and return handle info"""
        # Check outer handles
        for i, (hx, hy) in enumerate(self.get_outer_handles()):
            if abs(px - hx) <= handle_size and abs(py - hy) <= handle_size:
                return ('outer', i)
        
        # Check inner handles
        for i, (hx, hy) in enumerate(self.get_inner_handles()):
            if abs(px - hx) <= handle_size and abs(py - hy) <= handle_size:
                return ('inner', i)
        
        return None
    
    def resize_from_handle(self, handle_name, dx, dy):
        """Resize from handle - dx, dy are cumulative delta from resize start"""
        if self._resize_origin is None:
            return False
        
        orig_cx, orig_cy, orig_outer_r, orig_inner_r = self._resize_origin
        d = min(self.image_width, self.image_height)
        
        # Parse handle info
        if isinstance(handle_name, tuple):
            handle_type, idx = handle_name
        else:
            return False
        
        if handle_type == 'outer':
            # Outer handles are at cardinal points on the outer circle
            # 0:east  1:south  2:west  3:north
            # Calculate original handle position and add delta to get new handle position
            if idx == 0:    # east handle
                orig_hx, orig_hy = orig_cx + orig_outer_r, orig_cy
            elif idx == 1:  # south handle
                orig_hx, orig_hy = orig_cx, orig_cy + orig_outer_r
            elif idx == 2:  # west handle
                orig_hx, orig_hy = orig_cx - orig_outer_r, orig_cy
            elif idx == 3:  # north handle
                orig_hx, orig_hy = orig_cx, orig_cy - orig_outer_r
            else:
                return False
            
            # New handle position after drag
            new_hx = orig_hx + dx
            new_hy = orig_hy + dy
            
            # New radius = distance from center to new handle position
            new_r = math.sqrt((new_hx - orig_cx)**2 + (new_hy - orig_cy)**2)
            
            # Get current inner radius for min constraint
            _, _, _, current_inner_r = self.to_pixels()
            min_r = current_inner_r + 10
            if new_r >= min_r:
                self.outer_r = new_r / d
                return True
                
        elif handle_type == 'inner':
            # Inner handles are at cardinal points on the inner circle
            if idx == 0:    # east handle
                orig_hx, orig_hy = orig_cx + orig_inner_r, orig_cy
            elif idx == 1:  # south handle
                orig_hx, orig_hy = orig_cx, orig_cy + orig_inner_r
            elif idx == 2:  # west handle
                orig_hx, orig_hy = orig_cx - orig_inner_r, orig_cy
            elif idx == 3:  # north handle
                orig_hx, orig_hy = orig_cx, orig_cy - orig_inner_r
            else:
                return False
            
            # New handle position after drag
            new_hx = orig_hx + dx
            new_hy = orig_hy + dy
            
            # New radius = distance from center to new handle position
            new_r = math.sqrt((new_hx - orig_cx)**2 + (new_hy - orig_cy)**2)
            
            # Get current outer radius for max constraint
            _, _, current_outer_r, _ = self.to_pixels()
            max_r = current_outer_r - 10
            if new_r <= max_r and new_r >= 5:
                self.inner_r = new_r / d
                return True
        
        return False
    
    def begin_resize(self):
        """Store original geometry before resizing starts"""
        self._resize_origin = self.to_pixels()
        return True
    
    def copy(self):
        """Create a copy of this donut"""
        new = DonutShape(
            cx=self.cx, cy=self.cy,
            outer_r=self.outer_r,
            inner_r=self.inner_r,
            class_id=self.class_id,
            image_size=(self.image_width, self.image_height)
        )
        new.id = str(uuid.uuid4())[:8]
        new.selected = self.selected
        return new


class HollowEllipseShape(Shape):
    """Hollow ellipse — elliptical ring with inner cutout"""
    
    def __init__(self, cx=0.5, cy=0.5, outer_rx=0.15, outer_ry=0.10,
                 inner_rx=0.07, inner_ry=0.045, class_id=None, image_size=(1, 1)):
        super().__init__(class_id, image_size)
        self.type = 'hollow_ellipse'
        self.id = str(uuid.uuid4())[:8]
        self.cx = cx          # center x (normalized)
        self.cy = cy          # center y (normalized)
        self.outer_rx = outer_rx  # outer horizontal radius (normalized to image_width)
        self.outer_ry = outer_ry  # outer vertical radius (normalized to image_height)
        self.inner_rx = inner_rx  # inner horizontal radius (normalized to image_width)
        self.inner_ry = inner_ry  # inner vertical radius (normalized to image_height)
        self.selected = False
        self._resize_origin = None
    
    def to_pixels(self):
        """Convert to pixel coordinates"""
        cx = int(self.cx * self.image_width)
        cy = int(self.cy * self.image_height)
        orx = int(self.outer_rx * self.image_width)
        ory = int(self.outer_ry * self.image_height)
        irx = int(self.inner_rx * self.image_width)
        iry = int(self.inner_ry * self.image_height)
        return cx, cy, orx, ory, irx, iry
    
    def from_pixels(self, cx, cy, rx, ry):
        """Set from pixel coordinates (for initial drawing)"""
        self.cx = cx / self.image_width
        self.cy = cy / self.image_height
        self.outer_rx = rx / self.image_width
        self.outer_ry = ry / self.image_height
        # Inner radii default to 45% of outer
        self.inner_rx = (rx * 0.45) / self.image_width
        self.inner_ry = (ry * 0.45) / self.image_height
    
    def contains_point(self, px, py):
        """Check if point is in ring (in outer ellipse but not inner)"""
        cx, cy, orx, ory, irx, iry = self.to_pixels()
        if orx == 0 or ory == 0:
            return False
        # Check outer ellipse
        outer_val = ((px - cx) / orx) ** 2 + ((py - cy) / ory) ** 2
        in_outer = outer_val <= 1
        # Check inner ellipse
        if irx > 0 and iry > 0:
            inner_val = ((px - cx) / irx) ** 2 + ((py - cy) / iry) ** 2
            in_inner = inner_val <= 1
        else:
            in_inner = False
        return in_outer and not in_inner
    
    def move(self, dx, dy):
        """Move the hollow ellipse by delta (normalized)"""
        self.cx += dx
        self.cy += dy
    
    def get_outer_handles(self):
        """4 cardinal handles on outer ellipse"""
        cx, cy, orx, ory, _, _ = self.to_pixels()
        return [
            (cx + orx, cy),      # 0: east
            (cx, cy + ory),      # 1: south
            (cx - orx, cy),      # 2: west
            (cx, cy - ory),      # 3: north
        ]
    
    def get_inner_handles(self):
        """4 cardinal handles on inner ellipse"""
        cx, cy, _, _, irx, iry = self.to_pixels()
        return [
            (cx + irx, cy),      # 0: east
            (cx, cy + iry),      # 1: south
            (cx - irx, cy),      # 2: west
            (cx, cy - iry),      # 3: north
        ]
    
    def get_resize_handles(self):
        """Get outer handles for drawing"""
        return self.get_outer_handles()
    
    def get_handle_at_pos(self, px, py, handle_size=14):
        """Check if position is over any handle and return handle info"""
        # Check outer handles
        for i, (hx, hy) in enumerate(self.get_outer_handles()):
            if abs(px - hx) <= handle_size and abs(py - hy) <= handle_size:
                return ('outer', i)
        # Check inner handles
        for i, (hx, hy) in enumerate(self.get_inner_handles()):
            if abs(px - hx) <= handle_size and abs(py - hy) <= handle_size:
                return ('inner', i)
        return None
    
    def resize_from_handle(self, handle_name, dx, dy):
        """Resize from handle — dx, dy are cumulative delta from resize start"""
        if self._resize_origin is None:
            return False
        
        orig_cx, orig_cy, orig_orx, orig_ory, orig_irx, orig_iry = self._resize_origin
        
        if isinstance(handle_name, tuple):
            handle_type, idx = handle_name
        else:
            return False
        
        if handle_type == 'outer':
            # Outer cardinal handles: 0=east, 1=south, 2=west, 3=north
            # East/West affect outer_rx, North/South affect outer_ry
            new_orx = orig_orx
            new_ory = orig_ory
            
            if idx == 0:    # east — dragging right increases rx
                new_orx = orig_orx + dx
            elif idx == 2:  # west — dragging left increases rx
                new_orx = orig_orx - dx
            elif idx == 1:  # south — dragging down increases ry
                new_ory = orig_ory + dy
            elif idx == 3:  # north — dragging up increases ry
                new_ory = orig_ory - dy
            else:
                return False
            
            # Get current inner for min constraint
            _, _, _, _, cur_irx, cur_iry = self.to_pixels()
            min_rx = cur_irx + 10
            min_ry = cur_iry + 10
            
            if new_orx >= min_rx:
                self.outer_rx = new_orx / self.image_width
            if new_ory >= min_ry:
                self.outer_ry = new_ory / self.image_height
            return True
        
        elif handle_type == 'inner':
            new_irx = orig_irx
            new_iry = orig_iry
            
            if idx == 0:    # east
                new_irx = orig_irx + dx
            elif idx == 2:  # west
                new_irx = orig_irx - dx
            elif idx == 1:  # south
                new_iry = orig_iry + dy
            elif idx == 3:  # north
                new_iry = orig_iry - dy
            else:
                return False
            
            # Get current outer for max constraint
            _, _, cur_orx, cur_ory, _, _ = self.to_pixels()
            max_rx = cur_orx - 10
            max_ry = cur_ory - 10
            
            if new_irx <= max_rx and new_irx >= 5:
                self.inner_rx = new_irx / self.image_width
            if new_iry <= max_ry and new_iry >= 5:
                self.inner_ry = new_iry / self.image_height
            return True
        
        return False
    
    def begin_resize(self):
        """Store original geometry before resizing starts"""
        self._resize_origin = self.to_pixels()
        return True
    
    def copy(self):
        """Create a copy of this hollow ellipse"""
        new = HollowEllipseShape(
            cx=self.cx, cy=self.cy,
            outer_rx=self.outer_rx, outer_ry=self.outer_ry,
            inner_rx=self.inner_rx, inner_ry=self.inner_ry,
            class_id=self.class_id,
            image_size=(self.image_width, self.image_height)
        )
        new.id = str(uuid.uuid4())[:8]
        new.selected = self.selected
        return new
    
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'id': self.id,
            'type': 'hollow_ellipse',
            'class_id': self.class_id,
            'cx': self.cx,
            'cy': self.cy,
            'outer_rx': self.outer_rx,
            'outer_ry': self.outer_ry,
            'inner_rx': self.inner_rx,
            'inner_ry': self.inner_ry
        }
    
    @classmethod
    def from_dict(cls, data, image_size):
        """Create from dictionary"""
        shape = cls(
            cx=data['cx'], cy=data['cy'],
            outer_rx=data['outer_rx'], outer_ry=data['outer_ry'],
            inner_rx=data['inner_rx'], inner_ry=data['inner_ry'],
            class_id=data['class_id'],
            image_size=image_size
        )
        shape.id = data['id']
        return shape