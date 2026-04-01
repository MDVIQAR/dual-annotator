# gui/concentric_renderer.py
"""
Concentric Zone Mode renderer.

Renders shapes with the inner-wins cutout rule:
each shape has all previously-drawn shapes subtracted from its fill area.
The outline of the original shape is always drawn at full size.
"""

from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath, QPolygonF, QFont
from PyQt5.QtCore import Qt, QRectF, QPointF


class ConcentricRenderer:
    """
    Renders concentric annotation shapes with inner-wins cutout logic.

    Each shape has all previously-drawn shapes subtracted from its fill.
    The outline of the original shape is always drawn at full size.
    Selected shape shows resize handles exactly as in normal UNet mode.
    """

    FILL_ALPHA   = 160   # Fill transparency (0=invisible, 255=opaque)
    OUTLINE_WIDTH = 2    # Outline stroke width in pixels
    SELECTED_WIDTH = 3
    HANDLE_SIZE = 10

    @staticmethod
    def draw(painter, shapes, class_manager, scale, offset_x, offset_y,
             handle_size=10):
        """
        Main entry point. Called from canvas.paintEvent() when mode=='concentric'.

        Parameters
        ----------
        painter      : QPainter (active)
        shapes       : list of shape objects (canvas.shapes)
        class_manager: ClassManager instance
        scale        : float (canvas.scale)
        offset_x     : float (canvas.offset_x)
        offset_y     : float (canvas.offset_y)
        handle_size  : int (canvas.handle_size)
        """
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Build per-shape QPainterPaths once (in widget coordinate space)
        shape_paths = []
        for shape in shapes:
            if getattr(shape, 'hollow_role', None) == 'inner':
                # Inner shapes of hollow pairs are handled by their outer shape
                shape_paths.append(None)
                continue
            path = ConcentricRenderer._shape_to_path(
                shape, scale, offset_x, offset_y)
            shape_paths.append(path)

        # Draw each shape with earlier shapes cut out
        for i, shape in enumerate(shapes):
            if getattr(shape, 'hollow_role', None) == 'inner':
                continue
            if shape_paths[i] is None or shape_paths[i].isEmpty():
                continue

            # Get class color
            color = QColor(0, 255, 0)  # default
            if class_manager and getattr(shape, 'class_id', None):
                cls = class_manager.get_class(shape.class_id)
                if cls:
                    color = QColor(cls.color)

            # Build the filled region = shape[i] minus all shapes[0..i-1]
            fill_path = QPainterPath(shape_paths[i])
            for j in range(i):
                if shape_paths[j] is not None and not shape_paths[j].isEmpty():
                    fill_path = fill_path.subtracted(shape_paths[j])

            # Fill the cut region
            fill_color = QColor(color.red(), color.green(), color.blue(),
                                ConcentricRenderer.FILL_ALPHA)
            painter.setBrush(QBrush(fill_color))
            painter.setPen(Qt.NoPen)
            painter.drawPath(fill_path)

            # Draw the full outline (uncut) so the shape boundary is always visible
            if getattr(shape, 'selected', False):
                painter.setPen(QPen(QColor(255, 255, 0),
                                    ConcentricRenderer.SELECTED_WIDTH))
            else:
                painter.setPen(QPen(color, ConcentricRenderer.OUTLINE_WIDTH))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(shape_paths[i])

            # Draw resize handles if selected
            if getattr(shape, 'selected', False):
                ConcentricRenderer._draw_handles(
                    painter, shape, scale, offset_x, offset_y, handle_size)

            # Draw class label
            ConcentricRenderer._draw_label(
                painter, shape, class_manager, scale, offset_x, offset_y)

    @staticmethod
    def _shape_to_path(shape, scale, offset_x, offset_y):
        """
        Convert a shape to a QPainterPath in widget coordinates.
        For hollow shapes (donut, frame, hollow_ellipse) the path already
        has the inner hole subtracted.
        """
        from core.annotation import BoundingBox
        path = QPainterPath()
        t = getattr(shape, 'type', 'box')

        def wx(px): return float(px) * scale + offset_x
        def wy(py): return float(py) * scale + offset_y

        try:
            if t == 'box' or isinstance(shape, BoundingBox):
                x1, y1, x2, y2 = shape.to_pixels()
                path.addRect(QRectF(wx(x1), wy(y1),
                                    wx(x2) - wx(x1), wy(y2) - wy(y1)))
                # Handle inner shape (hollow box)
                if getattr(shape, 'inner_shape', None):
                    in_x1, in_y1, in_x2, in_y2 = shape.inner_shape.to_pixels()
                    inner_path = QPainterPath()
                    inner_path.addRect(QRectF(wx(in_x1), wy(in_y1),
                                              wx(in_x2) - wx(in_x1), wy(in_y2) - wy(in_y1)))
                    path = path.subtracted(inner_path)

            elif t == 'circle':
                cx, cy, r = shape.to_pixels()
                r_w = r * scale
                path.addEllipse(QPointF(wx(cx), wy(cy)), r_w, r_w)
                # Handle inner shape (hollow circle via attach_inner)
                if getattr(shape, 'inner_shape', None):
                    inner = shape.inner_shape
                    if inner.type == 'circle':
                        icx, icy, ir = inner.to_pixels()
                        inner_path = QPainterPath()
                        inner_path.addEllipse(QPointF(wx(icx), wy(icy)), ir * scale, ir * scale)
                        path = path.subtracted(inner_path)

            elif t == 'ellipse':
                cx, cy, rx, ry = shape.to_pixels()
                path.addEllipse(QPointF(wx(cx), wy(cy)),
                                rx * scale, ry * scale)
                # Handle inner shape
                if getattr(shape, 'inner_shape', None):
                    inner = shape.inner_shape
                    if inner.type == 'ellipse':
                        icx, icy, irx, iry = inner.to_pixels()
                        inner_path = QPainterPath()
                        inner_path.addEllipse(QPointF(wx(icx), wy(icy)), irx * scale, iry * scale)
                        path = path.subtracted(inner_path)

            elif t in ('polygon', 'bezier_polygon'):
                pts = shape.to_pixel_points()
                if len(pts) >= 3:
                    poly = QPolygonF([QPointF(wx(p[0]), wy(p[1])) for p in pts])
                    path.addPolygon(poly)
                    path.closeSubpath()
                # Handle inner shape or inner_points
                inner_path = None
                if getattr(shape, 'inner_shape', None):
                    inner = shape.inner_shape
                    if hasattr(inner, '_make_path'):
                        inner_path = inner._make_path(scale, offset_x, offset_y)
                    elif hasattr(inner, 'to_pixel_points'):
                        ipts = inner.to_pixel_points()
                        if len(ipts) >= 3:
                            inner_path = QPainterPath()
                            ipoly = QPolygonF([QPointF(wx(p[0]), wy(p[1])) for p in ipts])
                            inner_path.addPolygon(ipoly)
                            inner_path.closeSubpath()
                elif getattr(shape, 'inner_points', None):
                    iw = getattr(shape, 'image_width', 1) or 1
                    ih = getattr(shape, 'image_height', 1) or 1
                    ipts_pixel = [(nx * iw, ny * ih) for nx, ny in shape.inner_points]
                    if len(ipts_pixel) >= 3:
                        inner_path = QPainterPath()
                        ipoly = QPolygonF([QPointF(wx(p[0]), wy(p[1])) for p in ipts_pixel])
                        inner_path.addPolygon(ipoly)
                        inner_path.closeSubpath()
                if inner_path is not None:
                    path = path.subtracted(inner_path)

            elif t == 'frame':
                x1, y1, x2, y2 = shape.to_pixels()
                outer_path = QPainterPath()
                outer_path.addRect(QRectF(wx(x1), wy(y1),
                                    wx(x2) - wx(x1), wy(y2) - wy(y1)))
                # Frame has thickness — compute inner rect
                t_top = getattr(shape, 't_top', 20)
                t_bottom = getattr(shape, 't_bottom', 20)
                t_left = getattr(shape, 't_left', 20)
                t_right = getattr(shape, 't_right', 20)
                ix1 = x1 + t_left
                iy1 = y1 + t_top
                ix2 = x2 - t_right
                iy2 = y2 - t_bottom
                if ix2 > ix1 and iy2 > iy1:
                    inner_path = QPainterPath()
                    inner_path.addRect(QRectF(wx(ix1), wy(iy1),
                                              wx(ix2) - wx(ix1), wy(iy2) - wy(iy1)))
                    path = outer_path.subtracted(inner_path)
                else:
                    path = outer_path

            elif t == 'donut':
                cx, cy, outer_r, inner_r = shape.to_pixels()
                outer = QPainterPath()
                outer.addEllipse(QPointF(wx(cx), wy(cy)),
                                 outer_r * scale, outer_r * scale)
                inner = QPainterPath()
                inner.addEllipse(QPointF(wx(cx), wy(cy)),
                                 inner_r * scale, inner_r * scale)
                path = outer.subtracted(inner)

            elif t == 'hollow_ellipse':
                cx, cy, orx, ory, irx, iry = shape.to_pixels()
                outer = QPainterPath()
                outer.addEllipse(QPointF(wx(cx), wy(cy)),
                                 orx * scale, ory * scale)
                inner = QPainterPath()
                inner.addEllipse(QPointF(wx(cx), wy(cy)),
                                 irx * scale, iry * scale)
                path = outer.subtracted(inner)

        except Exception:
            pass  # return empty path on any coordinate error

        return path

    @staticmethod
    def _draw_handles(painter, shape, scale, offset_x, offset_y, handle_size):
        """Draw resize handles for selected shape (same as normal UNet mode)."""
        half = handle_size // 2
        mid_r = max(handle_size // 3, 4)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))

        if not hasattr(shape, 'get_resize_handles'):
            return
        try:
            handles = shape.get_resize_handles()
        except Exception:
            return

        for handle_name, (hx, hy) in handles.items():
            whx = int(float(hx) * scale + offset_x)
            why = int(float(hy) * scale + offset_y)
            if 'mid' in handle_name:
                painter.drawEllipse(QPointF(whx, why), mid_r, mid_r)
            elif 'ctrl' in handle_name:
                # Control point handle (yellow diamond) for bezier shapes
                painter.setBrush(QBrush(QColor(255, 255, 0)))
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                diamond = QPolygonF([
                    QPointF(whx, why - half),
                    QPointF(whx + half, why),
                    QPointF(whx, why + half),
                    QPointF(whx - half, why)
                ])
                painter.drawPolygon(diamond)
                # Reset brush
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(QColor(0, 0, 0), 1))
            else:
                painter.drawRect(whx - half, why - half, handle_size, handle_size)

    @staticmethod
    def _draw_label(painter, shape, class_manager, scale, offset_x, offset_y):
        """Draw class name label near the top-left of the shape."""
        if not class_manager or not getattr(shape, 'class_id', None):
            return
        cls = class_manager.get_class(shape.class_id)
        if not cls:
            return

        # Find label position: top-left of bounding box
        try:
            t = getattr(shape, 'type', 'box')
            if t in ('polygon', 'bezier_polygon'):
                pts = shape.to_pixel_points()
                if not pts:
                    return
                px = min(p[0] for p in pts) * scale + offset_x
                py = min(p[1] for p in pts) * scale + offset_y
            elif hasattr(shape, 'to_pixels'):
                pixels = shape.to_pixels()
                if t == 'circle':
                    px = (pixels[0] - pixels[2]) * scale + offset_x
                    py = (pixels[1] - pixels[2]) * scale + offset_y
                elif t == 'ellipse':
                    px = (pixels[0] - pixels[2]) * scale + offset_x
                    py = (pixels[1] - pixels[3]) * scale + offset_y
                elif t == 'donut':
                    px = (pixels[0] - pixels[2]) * scale + offset_x
                    py = (pixels[1] - pixels[2]) * scale + offset_y
                elif t == 'hollow_ellipse':
                    px = (pixels[0] - pixels[2]) * scale + offset_x
                    py = (pixels[1] - pixels[3]) * scale + offset_y
                else:
                    px = pixels[0] * scale + offset_x
                    py = pixels[1] * scale + offset_y
            else:
                return
        except Exception:
            return

        painter.setFont(QFont("Arial", 8))
        text = cls.name
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()

        bg = QColor(cls.color) if cls.color else QColor(0, 0, 0)
        bg = bg.darker(150)
        bg.setAlpha(200)
        painter.fillRect(int(px), int(py) - th - 5, tw + 10, th + 5, bg)
        painter.setPen(QPen(Qt.white, 1))
        painter.drawText(int(px) + 5, int(py) - 8, text)
