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
        return True

    def resize_from_handle(self, handle_name, dx, dy):
        if self._resize_origin is None:
            return False

        if handle_name.startswith('vertex_'):
            idx = int(handle_name.split('_')[1])
            if 0 <= idx < len(self._resize_origin):
                ox, oy = self._resize_origin[idx]
                self.points[idx] = (ox + dx / self.image_width, oy + dy / self.image_height)
                return True

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
            return True

        elif handle_name.startswith('inner_'):
            idx = int(handle_name.split('_')[1])
            if hasattr(self, '_inner_origin') and 0 <= idx < len(self._inner_origin):
                ox, oy = self._inner_origin[idx]
                self.inner_points[idx] = (ox + dx / self.image_width, oy + dy / self.image_height)
                
                # move control points attached to this inner node along with it
                if hasattr(self, 'inner_control_points') and self.inner_control_points:
                    inner_ctrl_orig = getattr(self, '_inner_ctrl_origin', [])
                    # The control point preceding this vertex
                    prev_idx = (idx - 1) % len(self.inner_points)
                    if prev_idx < len(inner_ctrl_orig) and inner_ctrl_orig[prev_idx] is not None:
                        cox, coy = inner_ctrl_orig[prev_idx]
                        self.inner_control_points[prev_idx] = (cox + dx / self.image_width, coy + dy / self.image_height)
                    # The control point succeeding this vertex
                    if idx < len(inner_ctrl_orig) and inner_ctrl_orig[idx] is not None:
                        cox, coy = inner_ctrl_orig[idx]
                        self.inner_control_points[idx] = (cox + dx / self.image_width, coy + dy / self.image_height)
                
                return True

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
                return True

        return False

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
