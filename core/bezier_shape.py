# core/bezier_shape.py
"""
BezierPolygonShape: A closed polygon where each edge can be curved
using a single midpoint control handle per edge (like MS Paint curve tool).

- Click to place vertices
- Close by clicking near first point or pressing Enter
- After closing: drag the midpoint handle on any edge to curve it
- Each edge stores one 'pull point' (bezier control midpoint)
"""

import uuid
import math
from PyQt5.QtGui import QPainterPath, QPolygonF
from PyQt5.QtCore import QPointF
from core.shape_base import Shape


class BezierPolygonShape(Shape):
    """Polygon with per-edge curve control handles (quadratic bezier curves)"""

    def __init__(self, points=None, class_id=None, image_size=(1, 1)):
        super().__init__(class_id, image_size)
        self.type = 'bezier_polygon'
        # Outer vertices (normalized)
        self.points = points or []
        # Per-edge control points: ctrl[i] controls the curve from points[i] to points[i+1]
        # Each ctrl is (nx, ny) normalized. If None, edge is straight.
        self.ctrl = []
        self.closed = False
        self.inner_points = []  # For hollow support (same as PolygonShape)
        self._resize_origin = None
        self._ctrl_origin = None
        self._mid_vertex_idx = None

    # ------------------------------------------------------------------
    #  Coordinate helpers
    # ------------------------------------------------------------------

    def _norm_to_px(self, nx, ny):
        return int(nx * self.image_width), int(ny * self.image_height)

    def _px_to_norm(self, px, py):
        return px / self.image_width, py / self.image_height

    def to_pixel_points(self):
        return [self._norm_to_px(nx, ny) for nx, ny in self.points]

    def to_pixels(self):
        return self.to_pixel_points()

    # ------------------------------------------------------------------
    #  Building QPainterPath
    # ------------------------------------------------------------------

    def _make_path(self, widget_scale=1.0, widget_ox=0, widget_oy=0):
        """Build QPainterPath for this bezier polygon in widget coords."""
        if len(self.points) < 2:
            return QPainterPath()

        def to_w(nx, ny):
            px, py = self._norm_to_px(nx, ny)
            return QPointF(px * widget_scale + widget_ox, py * widget_scale + widget_oy)

        path = QPainterPath()
        p0 = to_w(*self.points[0])
        path.moveTo(p0)

        n = len(self.points)
        for i in range(n):
            j = (i + 1) % n
            if not self.closed and j == 0:
                break  # Don't close if not closed
            p_end = to_w(*self.points[j])
            ctrl_pt = self.ctrl[i] if i < len(self.ctrl) and self.ctrl[i] is not None else None
            if ctrl_pt is not None:
                cp = to_w(*ctrl_pt)
                # Quadratic bezier via cubic (duplicate control point)
                path.quadTo(cp, p_end)
            else:
                path.lineTo(p_end)

        if self.closed:
            path.closeSubpath()

        return path

    # ------------------------------------------------------------------
    #  contains_point
    # ------------------------------------------------------------------

    def contains_point(self, x, y):
        if not self.closed or len(self.points) < 3:
            return False
        path = self._make_path(widget_scale=1.0, widget_ox=0, widget_oy=0)
        return path.contains(QPointF(x, y))

    # ------------------------------------------------------------------
    #  move
    # ------------------------------------------------------------------

    def move(self, dx, dy):
        if not self.points:
            return

        all_x = [nx + dx for nx, ny in self.points]
        all_y = [ny + dy for nx, ny in self.points]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        if min_x < 0: dx -= min_x
        if max_x > 1: dx -= (max_x - 1)
        if min_y < 0: dy -= min_y
        if max_y > 1: dy -= (max_y - 1)

        self.points = [(nx + dx, ny + dy) for nx, ny in self.points]
        new_ctrl = []
        for c in self.ctrl:
            if c is not None:
                new_ctrl.append((c[0] + dx, c[1] + dy))
            else:
                new_ctrl.append(None)
        self.ctrl = new_ctrl
        if self.inner_points:
            self.inner_points = [(nx + dx, ny + dy) for nx, ny in self.inner_points]

        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'move'):
            self.inner_shape.move(dx, dy)

    # ------------------------------------------------------------------
    #  Resize handles
    # ------------------------------------------------------------------

    def get_resize_handles(self):
        """Vertex handles + per-edge curve handles (if edge has ctrl pt)"""
        handles = {}
        pxpts = self.to_pixel_points()
        for i, (px, py) in enumerate(pxpts):
            handles[f'vertex_{i}'] = (px, py)

        if self.closed and len(pxpts) >= 3:
            n = len(pxpts)
            for i in range(n):
                j = (i + 1) % n
                if i < len(self.ctrl) and self.ctrl[i] is not None:
                    # Show actual control point
                    cpx, cpy = self._norm_to_px(*self.ctrl[i])
                    handles[f'ctrl_{i}'] = (cpx, cpy)
                else:
                    # Show midpoint as a "drag to curve" handle
                    mx = (pxpts[i][0] + pxpts[j][0]) // 2
                    my = (pxpts[i][1] + pxpts[j][1]) // 2
                    handles[f'ctrl_{i}'] = (mx, my)

        # Inner shape handles
        if self.inner_points:
            inner_ctrl = getattr(self, 'inner_control_points', [])
            n_inner = len(self.inner_points)
            for i, p in enumerate(self.inner_points):
                px, py = self._norm_to_px(*p)
                handles[f'inner_{i}'] = (px, py)
                
                if n_inner >= 3:
                    j = (i + 1) % n_inner
                    if i < len(inner_ctrl) and inner_ctrl[i] is not None:
                        cpx, cpy = self._norm_to_px(*inner_ctrl[i])
                        handles[f'ctrl_inner_{i}'] = (cpx, cpy)
                    else:
                        mx = (self.inner_points[i][0] + self.inner_points[j][0]) / 2
                        my = (self.inner_points[i][1] + self.inner_points[j][1]) / 2
                        wmx, wmy = self._norm_to_px(mx, my)
                        handles[f'ctrl_inner_{i}'] = (wmx, wmy)

        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'get_resize_handles'):
            inner_handles = self.inner_shape.get_resize_handles()
            for k, (hx, hy) in inner_handles.items():
                handles[f'inner_{k}'] = (hx, hy)
                
        return handles

    def begin_resize(self, handle_name=None):
        self._resize_origin = list(self.points)
        self._ctrl_origin = list(self.ctrl)
        self._inner_origin = list(self.inner_points) if self.inner_points else []
        self._inner_ctrl_origin = list(getattr(self, 'inner_control_points', []))
        self._mid_vertex_idx = None
        
        # Ensure inner_control_points array matches inner_points length
        if self.inner_points:
            inner_ctrl = getattr(self, 'inner_control_points', [])
            if len(inner_ctrl) != len(self.inner_points):
                self.inner_control_points = inner_ctrl + [None] * (len(self.inner_points) - len(inner_ctrl))
                self._inner_ctrl_origin = list(self.inner_control_points)
                
        if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'begin_resize'):
            inner_handle = handle_name.replace('inner_', '', 1) if handle_name and handle_name.startswith('inner_') else None
            self.inner_shape.begin_resize(inner_handle)
            
        return True

    def _constrain_inner_points(self):
        """Clamp each inner anchor so it stays inside the outer bezier polygon
        and at least a small distance away from the outer boundary."""
        from core.polygon_shape import PolygonShape
        
        if not getattr(self, 'inner_points', None) or len(self.points) < 3:
            return

        MIN_GAP_PX = 4.0

        # Centroid of outer anchors (normalized)
        cx = sum(p[0] for p in self.points) / len(self.points)
        cy = sum(p[1] for p in self.points) / len(self.points)

        # Approximate outer curve with straight segments between anchors.
        outer_px = [self._norm_to_px(nx, ny) for (nx, ny) in self.points]

        def _point_segment_distance(px, py, x1, y1, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                return math.hypot(px - x1, py - y1)
            t = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            return math.hypot(px - proj_x, py - proj_y)

        for i, (ix, iy) in enumerate(self.inner_points):
            best_x, best_y = ix, iy

            # Rough inside check using polygon made from outer anchors.
            if not PolygonShape._point_in_polygon(best_x, best_y, self.points):
                # Project toward centroid until inside or give up.
                for t_int in range(1, 21):
                    t = t_int / 20.0
                    tx = ix + (cx - ix) * t
                    ty = iy + (cy - iy) * t
                    if PolygonShape._point_in_polygon(tx, ty, self.points):
                        best_x, best_y = tx, ty
                        break
                else:
                    best_x = ix + (cx - ix) * 0.9
                    best_y = iy + (cy - iy) * 0.9

            # Enforce minimum distance from outer boundary in pixel space.
            for _ in range(10):
                px, py = self._norm_to_px(best_x, best_y)
                min_dist = float("inf")
                for j in range(len(outer_px)):
                    x1, y1 = outer_px[j]
                    x2, y2 = outer_px[(j + 1) % len(outer_px)]
                    dist = _point_segment_distance(px, py, x1, y1, x2, y2)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist >= MIN_GAP_PX:
                    break
                best_x = best_x + (cx - best_x) * 0.3
                best_y = best_y + (cy - best_y) * 0.3

            self.inner_points[i] = (best_x, best_y)

    def _constrain_inner_shape_object(self):
        """Constrain an attached inner_shape object so it stays inside the
        outer bezier polygon's bounding box, with a small safety gap."""
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

        # Handle circle/ellipse-type inner shapes with (cx, cy, rx, ry)
        if hasattr(inner, 'to_pixels'):
            result = inner.to_pixels()
            if isinstance(result, (list, tuple)) and len(result) == 4 and not isinstance(result[0], (list, tuple)):
                cx, cy, rx, ry = result
                max_rx = max(1, (ox_max - ox_min) // 2 - MIN_GAP)
                max_ry = max(1, (oy_max - oy_min) // 2 - MIN_GAP)
                rx = min(rx, max_rx)
                ry = min(ry, max_ry)
                cx = max(ox_min + rx + MIN_GAP, min(cx, ox_max - rx - MIN_GAP))
                cy = max(oy_min + ry + MIN_GAP, min(cy, oy_max - ry - MIN_GAP))
                if hasattr(inner, 'from_pixels'):
                    inner.from_pixels(int(cx), int(cy), int(rx), int(ry))
        # Handle polygon-type inner shapes
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

    def resize_from_handle(self, handle_name, dx, dy):
        if handle_name.startswith('inner_') and getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'resize_from_handle'):
            inner_handle = handle_name.replace('inner_', '', 1)
            if not inner_handle.isdigit():
                if hasattr(self.inner_shape, 'to_dict'):
                    old_state = self.inner_shape.to_dict()
                    res = self.inner_shape.resize_from_handle(inner_handle, dx, dy)
                    if not res: return False
                    
                    from shapely.geometry import Polygon as ShapelyPolygon
                    try:
                        if len(self.points) >= 3 and hasattr(self.inner_shape, 'points') and len(self.inner_shape.points) >= 3:
                            outer_poly = ShapelyPolygon(self.points)
                            inner_poly = ShapelyPolygon(self.inner_shape.points)
                            if not outer_poly.is_valid: outer_poly = outer_poly.buffer(0)
                            if not inner_poly.is_valid: inner_poly = inner_poly.buffer(0)
                            
                            if not inner_poly.within(outer_poly.buffer(1e-5)):
                                restored = self.inner_shape.__class__.from_dict(old_state, (self.image_width, self.image_height))
                                for k, v in restored.__dict__.items():
                                    if not k.startswith('_') and k != 'id':
                                        setattr(self.inner_shape, k, v)
                                return False
                    except Exception:
                        pass
                    return True
                else:
                    return self.inner_shape.resize_from_handle(inner_handle, dx, dy)
        if self._resize_origin is None:
            return False

        result = False

        if handle_name.startswith('vertex_'):
            idx = int(handle_name.split('_')[1])
            if 0 <= idx < len(self._resize_origin):
                ox, oy = self._resize_origin[idx]
                self.points[idx] = (ox + dx / self.image_width, oy + dy / self.image_height)
                result = True

        elif handle_name.startswith('ctrl_inner_'):
            idx = int(handle_name.split('_')[2])
            if hasattr(self, '_inner_ctrl_origin') and idx < len(self._inner_ctrl_origin):
                orig = self._inner_ctrl_origin[idx]
                if orig is not None:
                    ox, oy = orig
                else:
                    n = len(self.inner_points)
                    j = (idx + 1) % n
                    ox = (self.inner_points[idx][0] + self.inner_points[j][0]) / 2
                    oy = (self.inner_points[idx][1] + self.inner_points[j][1]) / 2
                if not hasattr(self, 'inner_control_points'):
                    self.inner_control_points = [None] * len(self.inner_points)
                self.inner_control_points[idx] = (ox + dx / self.image_width, oy + dy / self.image_height)
                result = True

        elif handle_name.startswith('ctrl_'):
            idx = int(handle_name.split('_')[1])
            if idx < len(self._ctrl_origin):
                orig = self._ctrl_origin[idx]
                if orig is not None:
                    ox, oy = orig
                else:
                    # Midpoint of edge
                    n = len(self.points)
                    j = (idx + 1) % n
                    ox = (self.points[idx][0] + self.points[j][0]) / 2
                    oy = (self.points[idx][1] + self.points[j][1]) / 2
                self.ctrl[idx] = (ox + dx / self.image_width, oy + dy / self.image_height)
                result = True

        elif handle_name.startswith('inner_'):
            idx = int(handle_name.split('_')[1])
            if hasattr(self, '_inner_origin') and 0 <= idx < len(self._inner_origin):
                ox, oy = self._inner_origin[idx]
                self.inner_points[idx] = (ox + dx / self.image_width, oy + dy / self.image_height)
                if hasattr(self, 'inner_control_points') and self.inner_control_points:
                    inner_ctrl_orig = getattr(self, '_inner_ctrl_origin', [])
                    prev_idx = (idx - 1) % len(self.inner_points)
                    if prev_idx < len(inner_ctrl_orig) and inner_ctrl_orig[prev_idx] is not None:
                        cox, coy = inner_ctrl_orig[prev_idx]
                        self.inner_control_points[prev_idx] = (cox + dx / self.image_width, coy + dy / self.image_height)
                    if idx < len(inner_ctrl_orig) and inner_ctrl_orig[idx] is not None:
                        cox, coy = inner_ctrl_orig[idx]
                        self.inner_control_points[idx] = (cox + dx / self.image_width, coy + dy / self.image_height)
                result = True

        # If any inner or outer geometry changed and we have inner anchors/shapes,
        # keep them constrained inside the outer curve.
        if result and len(self.points) >= 3:
            from shapely.geometry import Polygon as ShapelyPolygon
            try:
                outer_poly = ShapelyPolygon(self.points)
                if not outer_poly.is_valid: outer_poly = outer_poly.buffer(0)
                
                # Check inner object
                if getattr(self, 'inner_shape', None) and hasattr(self.inner_shape, 'points') and len(self.inner_shape.points) >= 3:
                    inner_poly = ShapelyPolygon(self.inner_shape.points)
                    if not inner_poly.is_valid: inner_poly = inner_poly.buffer(0)
                    if not inner_poly.within(outer_poly.buffer(1e-5)):
                        self.points = list(self._resize_origin)
                        self.ctrl = list(self._ctrl_origin)
                        return False
                        
                # Check legacy inner_points
                if getattr(self, 'inner_points', None) and len(self.inner_points) >= 3:
                    inner_poly = ShapelyPolygon(self.inner_points)
                    if not inner_poly.is_valid: inner_poly = inner_poly.buffer(0)
                    if not inner_poly.within(outer_poly.buffer(1e-5)):
                        self.points = list(self._resize_origin)
                        self.ctrl = list(self._ctrl_origin)
                        if hasattr(self, '_inner_origin'):
                            self.inner_points = list(self._inner_origin)
                        if hasattr(self, '_inner_ctrl_origin'):
                            self.inner_control_points = list(self._inner_ctrl_origin)
                        return False
            except Exception:
                pass
                
            if getattr(self, 'inner_shape', None):
                self._constrain_inner_shape_object()

        return result

    def insert_vertex_at_ctrl(self, handle_name):
        """Insert a new vertex at the position of a control handle."""
        if handle_name.startswith('ctrl_'):
            idx = int(handle_name.split('_')[1])
            if idx < len(self.ctrl):
                if self.ctrl[idx] is not None:
                    mx, my = self.ctrl[idx]
                else:
                    n = len(self.points)
                    j = (idx + 1) % n
                    mx = (self.points[idx][0] + self.points[j][0]) / 2
                    my = (self.points[idx][1] + self.points[j][1]) / 2
                
                # Insert the new point
                self.points.insert(idx + 1, (mx, my))
                
                # The single curve is split into two, we reset both segments to straight lines (None)
                self.ctrl.pop(idx)
                self.ctrl.insert(idx, None)
                self.ctrl.insert(idx + 1, None)
                return True
                
        elif handle_name.startswith('ctrl_inner_'):
            idx = int(handle_name.split('_')[2])
            if self.inner_points:
                n = len(self.inner_points)
                # Ensure we have inner control points array
                if not hasattr(self, 'inner_control_points'):
                    self.inner_control_points = [None] * n
                elif len(self.inner_control_points) < n:
                    self.inner_control_points.extend([None] * (n - len(self.inner_control_points)))

                if idx < len(self.inner_control_points):
                    if self.inner_control_points[idx] is not None:
                        mx, my = self.inner_control_points[idx]
                    else:
                        j = (idx + 1) % n
                        mx = (self.inner_points[idx][0] + self.inner_points[j][0]) / 2
                        my = (self.inner_points[idx][1] + self.inner_points[j][1]) / 2
                    
                    self.inner_points.insert(idx + 1, (mx, my))
                    self.inner_control_points.pop(idx)
                    self.inner_control_points.insert(idx, None)
                    self.inner_control_points.insert(idx + 1, None)
                    return True
        return False

    def get_closest_edge(self, image_x, image_y, threshold_px=10.0):
        """Find the closest edge segment to a pixel coordinate within threshold."""
        import math
        norm_threshold_x = threshold_px / self.image_width
        norm_threshold_y = threshold_px / self.image_height
        norm_threshold = max(norm_threshold_x, norm_threshold_y)
        nx, ny = self._px_to_norm(image_x, image_y)
        
        min_dist = float('inf')
        closest_info = None
        
        def check_segments(points, ctrls, is_inner):
            nonlocal min_dist, closest_info
            n = len(points)
            if n < 2: return
            for i in range(n):
                p1 = points[i]
                p2 = points[(i+1)%n]
                ctrl = ctrls[i] if i < len(ctrls) else None
                
                # Sample 20 points along this segment
                for step in range(21):
                    t = step / 20.0
                    if ctrl is None:
                        px = p1[0] * (1-t) + p2[0] * t
                        py = p1[1] * (1-t) + p2[1] * t
                    else:
                        px = (1-t)**2 * p1[0] + 2*(1-t)*t * ctrl[0] + t**2 * p2[0]
                        py = (1-t)**2 * p1[1] + 2*(1-t)*t * ctrl[1] + t**2 * p2[1]
                        
                    dist = math.hypot(px - nx, py - ny)
                    if dist < min_dist:
                        min_dist = dist
                        closest_info = (i, (px, py), is_inner)

        # Check outer
        check_segments(self.points, self.ctrl, False)
        # Check inner
        if self.inner_points:
            inner_ctrls = getattr(self, 'inner_control_points', [None]*len(self.inner_points))
            check_segments(self.inner_points, inner_ctrls, True)
            
        if min_dist <= norm_threshold:
            return closest_info
        return None

    def insert_vertex(self, idx, pt, is_inner=False):
        """Insert a point pt at segment idx."""
        if not is_inner:
            self.points.insert(idx + 1, pt)
            if idx < len(self.ctrl):
                self.ctrl.pop(idx)
                self.ctrl.insert(idx, None)
            else:
                while len(self.ctrl) < len(self.points):
                    self.ctrl.append(None)
            self.ctrl.insert(idx + 1, None)
        else:
            self.inner_points.insert(idx + 1, pt)
            if not hasattr(self, 'inner_control_points'):
                self.inner_control_points = [None] * len(self.inner_points)
            if idx < len(self.inner_control_points):
                self.inner_control_points.pop(idx)
                self.inner_control_points.insert(idx, None)
            else:
                while len(self.inner_control_points) < len(self.inner_points):
                    self.inner_control_points.append(None)
            self.inner_control_points.insert(idx + 1, None)

    # ------------------------------------------------------------------
    #  Close
    # ------------------------------------------------------------------

    def from_pixel_points(self, pixel_points):
        self.points = [self._px_to_norm(px, py) for px, py in pixel_points]

    def close_polygon(self):
        self.closed = True
        # Ensure ctrl list matches points length
        n = len(self.points)
        while len(self.ctrl) < n:
            self.ctrl.append(None)
        self.ctrl = self.ctrl[:n]

    # ------------------------------------------------------------------
    #  Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            'id': self.id,
            'type': 'bezier_polygon',
            'class_id': self.class_id,
            'points': self.points,
            'ctrl': self.ctrl,
            'inner_points': self.inner_points,
            'closed': self.closed,
        }

    @classmethod
    def from_dict(cls, data, image_size):
        shape = cls(
            points=data['points'],
            class_id=data['class_id'],
            image_size=image_size,
        )
        shape.id = data['id']
        shape.closed = data['closed']
        shape.ctrl = data.get('ctrl', [None] * len(data['points']))
        shape.inner_points = data.get('inner_points', [])
        return shape
