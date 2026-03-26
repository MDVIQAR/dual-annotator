# gui/canvas.py
from PyQt5.QtWidgets import QWidget, QApplication, QMenu, QAction
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QRect, QRectF, QPointF, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen, QBrush, QFont, QPolygonF, QCursor, QPainterPath
import os
import math

from core.annotation import BoundingBox
from core.polygon_shape import PolygonShape
from core.bezier_shape import BezierPolygonShape
from core.circle_shape import CircleShape
from core.ellipse_shape import EllipseShape
from core.ring_shape import FrameShape, DonutShape, HollowEllipseShape
from core.template_manager import TemplateManager

class AnnotationCanvas(QWidget):
    """Canvas widget for displaying images and annotations"""
    
    # Signals
    position_changed = pyqtSignal(int, int)  # Emitted when mouse moves
    shape_selected = pyqtSignal(str)  # Emitted when shape is selected
    annotation_changed = pyqtSignal()  # AUTOSAVE INTEGRATION
    
    def __init__(self):
        """Initialize the canvas"""
        super().__init__()
        
        # Set canvas properties
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #1e1e1e;")
        
        # Image related variables
        self.image = None
        self.image_path = None
        self.pixmap = None
        self.scaled_pixmap = None
        self.image_width = 0
        self.image_height = 0
        
        # Shape drawing variables
        self.current_shape_type = 'box'  # 'box', 'polygon', 'circle', 'ellipse', 'frame', 'donut'
        self.polygon_points = []  # Temporary points for polygon drawing
        self.circle_center = None  # Center point for circle drawing
        self.circle_radius = 0  # Radius for circle drawing
        self.ellipse_center = None  # Center point for ellipse drawing
        self.ellipse_radius_x = 0  # Horizontal radius for ellipse
        self.ellipse_radius_y = 0  # Vertical radius for ellipse
        self.donut_center = None  # Center point for donut drawing
        self.donut_radius = 0  # Radius for donut drawing
        self.hollow_ellipse_center = None  # Center point for hollow ellipse drawing
        self.hollow_ellipse_rx = 0  # Horizontal radius for hollow ellipse
        self.hollow_ellipse_ry = 0  # Vertical radius for hollow ellipse
        
        # Template/Stamp variables
        self.template_manager = TemplateManager()
        self.stamping = False  # Whether we're placing a stamp
        self.stamp_template_name = None  # Currently selected template
        self.stamp_center = None  # Click point for stamp placement (image coords)
        self.stamp_current_pos = None  # Current mouse pos during stamp drag
        
        # View parameters
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_mouse_pos = None
        self.pan_mode = False  # Whether we're in pan mode
        self.original_cursor = None  # Store original cursor
        
        # Mode
        self.mode = 'yolo'  # 'yolo' or 'unet'
        
        # Class manager reference
        self.class_manager = None
        
        # Parent window reference (for callbacks)
        self.parent_window = None  # Reference to parent main window
        
        # Shape storage
        self.shapes = []  # List of all shapes (boxes, polygons, circles, ellipses, frames, donuts)
        self.drawing = False
        self.start_point = None
        self.current_shape = None
        self.selected_shape = None
        
        # Resize variables
        self.resizing = False
        self.resizing_handle = None
        self.resize_start_pos = None
        
        # Move variables
        self.moving = False
        self.move_start_pos = None
        self.move_original_positions = []
        
        # Drag-copy variables
        self.drag_copy = False
        self.drag_copy_shape = None
        self.drag_start_pos = None
        self.original_shape = None
        
        # Paste variables
        self.clipboard_shape = None
        self.pasting = False
        self.paste_shape = None
        self.paste_start_pos = None
        self.paste_confirmed = False
        
        # Nudge step (pixels) — configurable via Edit > Nudge Step
        self.nudge_step = 1

        # Multi-selection state
        self.selected_shapes = []           # All currently selected shapes
        self.rubber_band_active = False     # Rubber band drag in progress
        self.rubber_band_start = None       # QPoint where rubber band started (widget coords)
        self.rubber_band_end = None         # QPoint current rubber band end (widget coords)
        self.multi_moving = False           # Group move in progress
        self.multi_move_start = None        # Image-coord (x,y) where group move started
        self.multi_drag_copy = False        # Group Ctrl+drag-copy in progress
        self.multi_drag_copies = []         # Copied shapes during drag-copy
        self.multi_drag_originals = []      # Original shapes during drag-copy
        self.clipboard_shapes = []          # Multi-shape clipboard (separate from clipboard_shape)
        
        # Polygon drawing state
        self.drawing_polygon = False
        self.drawing_inner_cutout = False
        self.cutout_target_shape = None

        # Bezier polygon drawing state
        self.drawing_bezier = False
        self.bezier_points = []  # pixel points being drawn
        
        # Pan mode
        self.pan_mode = False
        self.original_cursor = None
        
        # Ring drawing state
        self.drawing_ring = False
        self.ring_stage = 'outer'
        self.ring_outer_points = []
        self.ring_inner_points = []
        self.ring_outer_center = None
        self.ring_outer_radius = 0
        
        # Undo/Redo stacks
        self.undo_stack = []  # Stack of actions for undo
        self.redo_stack = []  # Stack of actions for redo
        self.max_stack_size = 50  # Maximum undo steps
        
        # Resize handle size (pixels)
        self.handle_size = 10
        
        # Enable mouse tracking for position updates
        self.setMouseTracking(True)
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        
        # Mouse tracking for previews
        self.current_mouse_pos = None

        # Hollow Shape Preview states
        self._hollow_preview_shape = None
        self._hollow_preview_offset = 0.0
        self._hollow_preview_mode = None
        
        print("✅ Canvas initialized")

        
    def set_parent_window(self, parent):
        """Set reference to parent main window"""
        self.parent_window = parent
        print(f"✅ Parent window set")
        
    def set_class_manager(self, class_manager):
        """Set the class manager reference"""
        self.class_manager = class_manager
        
    def load_image(self, image_path):
        """Load an image from file"""
        try:
            self.image_path = image_path
            self.pixmap = QPixmap(image_path)
            
            if not self.pixmap.isNull():
                self.image_width = self.pixmap.width()
                self.image_height = self.pixmap.height()
                
                # Clear previous shapes when loading new image
                self.shapes = []
                self.selected_shape = None
                self.selected_shapes = []
                
                # Reset all drawing states
                self.reset_all_states()
                
                self.fit_to_window()
                self.update()
                
                print(f"Loaded image: {os.path.basename(image_path)} ({self.image_width}x{self.image_height})")
            else:
                print(f"Failed to load image: {image_path}")
                
        except Exception as e:
            print(f"Error loading image: {e}")

    def set_mode(self, mode):
        """Set the annotation mode"""
        self.mode = mode
        self.reset_all_states()
        self.update()
                
    def fit_to_window(self):
        """Scale image to fit the window"""
        if self.pixmap and not self.pixmap.isNull():
            widget_width = self.width()
            widget_height = self.height()
            
            if widget_width > 0 and widget_height > 0:
                scale_x = widget_width / self.image_width
                scale_y = widget_height / self.image_height
                self.scale = min(scale_x, scale_y) * 0.9  # 90% to leave margin
                
                # Center the image
                scaled_width = self.image_width * self.scale
                scaled_height = self.image_height * self.scale
                self.offset_x = (widget_width - scaled_width) / 2
                self.offset_y = (widget_height - scaled_height) / 2
                
                self.update()
                
    def _zoom_at_cursor(self, zoom_factor):
        if not self.pixmap or self.pixmap.isNull():
            return
            
        # Get global mouse position and map it to the widget
        global_pos = QCursor.pos()
        cursor_pos = self.mapFromGlobal(global_pos)
        
        # If cursor is not within the canvas, default to center of canvas
        if not self.rect().contains(cursor_pos):
            cursor_pos = QPoint(self.width() // 2, self.height() // 2)

        # Convert to image coordinates before zoom
        image_pos = self.widget_to_image(cursor_pos)
        
        new_scale = self.scale * zoom_factor
        new_scale = max(0.1, min(10.0, new_scale))
        
        if new_scale != self.scale:
            # Current screen position of the image point
            screen_x = image_pos[0] * self.scale + self.offset_x
            screen_y = image_pos[1] * self.scale + self.offset_y
            
            self.scale = new_scale
            
            # Adjust offset so the image point stays at the same screen position
            self.offset_x = screen_x - image_pos[0] * self.scale
            self.offset_y = screen_y - image_pos[1] * self.scale
            
            self.update()

    def zoom_in(self):
        """Zoom in by 20% at cursor position"""
        self._zoom_at_cursor(1.2)
        
    def zoom_out(self):
        """Zoom out by 20% at cursor position"""
        self._zoom_at_cursor(0.8)
        
    def start_drawing(self, pos):
        """Start drawing a new shape"""
        # Force reset all states before starting to draw
        self.force_reset_for_drawing()
        
        if self.class_manager and self.class_manager.get_current_class():
            if self.current_shape_type == 'box':
                self.drawing = True
                self.start_point = self.widget_to_image(pos)
                self.current_shape = BoundingBox(
                    image_size=(self.image_width, self.image_height)
                )
                print(f"✏️ Started drawing new box")
            else:
                print(f"⚠️ Cannot draw - invalid shape type")
        else:
            print("⚠️ Cannot draw - no class selected")
        
    def update_drawing(self, pos):
        """Update the current shape while drawing"""
        if self.drawing and self.start_point and self.current_shape:
            current_pos = self.widget_to_image(pos)
            
            x1 = min(self.start_point[0], current_pos[0])
            y1 = min(self.start_point[1], current_pos[1])
            x2 = max(self.start_point[0], current_pos[0])
            y2 = max(self.start_point[1], current_pos[1])
            
            # Handle different shape types
            if hasattr(self.current_shape, 'type'):
                if self.current_shape.type == 'box':
                    # Box needs image dimensions
                    self.current_shape.from_pixels(
                        x1, y1, x2, y2,
                        self.image_width, self.image_height
                    )
                elif self.current_shape.type == 'frame':
                    # Frame has simple from_pixels
                    self.current_shape.from_pixels(x1, y1, x2, y2)
            
            self.update()
        
    def finish_drawing(self):
        """Finish drawing and add the shape to the list"""
        if self.drawing and self.current_shape:
            current_class = self.class_manager.get_current_class()
            if current_class:
                self.current_shape.class_id = current_class.id
                self.save_state()
                self.shapes.append(self.current_shape)
                shape_type = getattr(self.current_shape, 'type', 'box')
                print(f"✅ Added new {shape_type}")
                self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
        
        self.drawing = False
        self.start_point = None
        self.current_shape = None
        self.update()

    def find_shape_at(self, pos):
        """Hit-test all shapes at pos, return topmost."""
        image_x, image_y = self.widget_to_image(pos)
        for shape in reversed(self.shapes):
            if hasattr(shape, 'hollow_role') and shape.hollow_role == 'inner':
                continue # hit outer instead
            if hasattr(shape, 'contains_point') and shape.contains_point(image_x, image_y):
                return shape
        return None

    # ==========================================
    #  Multi-select helper methods
    # ==========================================

    def _get_shape_bbox_image(self, shape):
        """Return (x1,y1,x2,y2) bounding box in image pixel coords for any shape type."""
        t = getattr(shape, 'type', None)
        if t == 'box':
            return shape.to_pixels()  # returns (x1,y1,x2,y2)
        elif t == 'circle':
            cx, cy, r = shape.to_pixels()
            return (cx - r, cy - r, cx + r, cy + r)
        elif t == 'ellipse':
            cx, cy, rx, ry = shape.to_pixels()
            return (cx - rx, cy - ry, cx + rx, cy + ry)
        elif t == 'frame':
            x, y, w, h = shape.to_pixels()
            return (x, y, x + w, y + h)
        elif t in ('donut', 'hollow_ellipse'):
            pts = shape.to_pixels()
            cx, cy = pts[0], pts[1]
            rx = pts[2]
            ry = pts[3] if len(pts) > 3 else pts[2]
            return (cx - rx, cy - ry, cx + rx, cy + ry)
        elif t in ('polygon', 'bezier_polygon'):
            px_pts = shape.to_pixel_points()
            if not px_pts:
                return None
            xs = [p[0] for p in px_pts]
            ys = [p[1] for p in px_pts]
            return (min(xs), min(ys), max(xs), max(ys))
        return None

    def _rects_intersect(self, r1, r2):
        """Return True if two (x1,y1,x2,y2) rects intersect."""
        return not (r1[2] < r2[0] or r2[2] < r1[0] or r1[3] < r2[1] or r2[3] < r1[1])

    def _clear_multi_selection(self):
        """Deselect all shapes and clear multi-select state."""
        for shape in self.selected_shapes:
            shape.selected = False
        self.selected_shapes = []
        self.selected_shape = None

    def _get_group_bounding_box(self):
        """Return (wx1,wy1,wx2,wy2) group bbox in widget coords, or None."""
        if not self.selected_shapes:
            return None
        all_x1, all_y1, all_x2, all_y2 = [], [], [], []
        for shape in self.selected_shapes:
            bbox = self._get_shape_bbox_image(shape)
            if bbox:
                all_x1.append(bbox[0]); all_y1.append(bbox[1])
                all_x2.append(bbox[2]); all_y2.append(bbox[3])
        if not all_x1:
            return None
        ix1, iy1 = min(all_x1), min(all_y1)
        ix2, iy2 = max(all_x2), max(all_y2)
        wx1 = int(ix1 * self.scale + self.offset_x)
        wy1 = int(iy1 * self.scale + self.offset_y)
        wx2 = int(ix2 * self.scale + self.offset_x)
        wy2 = int(iy2 * self.scale + self.offset_y)
        return (wx1, wy1, wx2, wy2)

    def _finish_rubber_band_selection(self):
        """Finalize rubber band: select all shapes intersecting the drag rect."""
        if not self.rubber_band_start or not self.rubber_band_end:
            self.rubber_band_active = False
            return
        w = abs(self.rubber_band_end.x() - self.rubber_band_start.x())
        h = abs(self.rubber_band_end.y() - self.rubber_band_start.y())
        if w < 5 and h < 5:
            # Too small — treat as a deselect click
            self._clear_multi_selection()
            self.rubber_band_active = False
            self.rubber_band_start = None
            self.rubber_band_end = None
            self.update()
            return
        # Build image-coord rect
        sx = min(self.rubber_band_start.x(), self.rubber_band_end.x())
        sy = min(self.rubber_band_start.y(), self.rubber_band_end.y())
        ex = max(self.rubber_band_start.x(), self.rubber_band_end.x())
        ey = max(self.rubber_band_start.y(), self.rubber_band_end.y())
        x1, y1 = self.widget_to_image(QPoint(sx, sy))
        x2, y2 = self.widget_to_image(QPoint(ex, ey))
        band_rect = (x1, y1, x2, y2)
        # Find intersecting shapes
        newly_selected = []
        for shape in self.shapes:
            if getattr(shape, 'hollow_role', None) == 'inner':
                continue
            bbox = self._get_shape_bbox_image(shape)
            if bbox and self._rects_intersect(band_rect, bbox):
                newly_selected.append(shape)
        # Apply selection
        for shape in self.shapes:
            shape.selected = False
        self.selected_shape = None
        self.selected_shapes = newly_selected
        for shape in newly_selected:
            shape.selected = True
        if len(newly_selected) == 1:
            self.selected_shape = newly_selected[0]
        self.rubber_band_active = False
        self.rubber_band_start = None
        self.rubber_band_end = None
        print(f"🔲 Rubber band selected {len(newly_selected)} shapes")
        self.update()

    def select_all(self):
        """Select all non-inner shapes on the canvas."""
        for shape in self.shapes:
            shape.selected = False
        self.selected_shapes = []
        self.selected_shape = None
        for shape in self.shapes:
            if getattr(shape, 'hollow_role', None) == 'inner':
                continue
            shape.selected = True
            self.selected_shapes.append(shape)
        if len(self.selected_shapes) == 1:
            self.selected_shape = self.selected_shapes[0]
        self.update()
        print(f"✅ Selected all: {len(self.selected_shapes)} shapes")

    def delete_selected_group(self):
        """Delete all shapes in selected_shapes."""
        if not self.selected_shapes:
            return
        self.save_state()
        ids_to_remove = set(id(s) for s in self.selected_shapes)
        # Also remove any inner_shapes attached to selected outer shapes
        extras = []
        for shape in self.selected_shapes:
            inner = getattr(shape, 'inner_shape', None)
            if inner and id(inner) not in ids_to_remove:
                extras.append(inner)
        self.shapes = [s for s in self.shapes if id(s) not in ids_to_remove and s not in extras]
        n = len(self.selected_shapes)
        self._clear_multi_selection()
        self.shape_selected.emit("none")
        self.annotation_changed.emit()
        self.update()
        print(f"🗑️ Deleted group of {n} shapes")

    def copy_selected_group(self):
        """Copy all selected shapes into clipboard_shapes."""
        if not self.selected_shapes:
            return
        self.clipboard_shapes = [s.copy() for s in self.selected_shapes]
        print(f"📋 Copied {len(self.clipboard_shapes)} shapes to clipboard")

    def paste_group(self):
        """Paste clipboard_shapes with a small offset."""
        if not getattr(self, 'clipboard_shapes', None):
            if self.clipboard_shape and self.pixmap and not self.pixmap.isNull():
                cursor_pos = self.mapFromGlobal(self.cursor().pos())
                self.start_paste(cursor_pos)
            return
        self.save_state()
        offset_x = 20 / max(self.image_width, 1)
        offset_y = 20 / max(self.image_height, 1)
        new_shapes = []
        for s in self.clipboard_shapes:
            c = s.copy()
            new_shapes.append(c)
        # Skip inner shapes whose outer is also being pasted (move() recurses)
        new_ids = set(id(c) for c in new_shapes)
        inner_ids_to_skip = set()
        for c in new_shapes:
            inner = getattr(c, 'inner_shape', None)
            if inner and id(inner) in new_ids:
                inner_ids_to_skip.add(id(inner))
        for c in new_shapes:
            if id(c) not in inner_ids_to_skip and getattr(c, 'hollow_role', None) != 'inner':
                if hasattr(c, 'move'):
                    c.move(offset_x, offset_y)
        self._clear_multi_selection()
        for s in new_shapes:
            s.selected = True
            self.shapes.append(s)
        self.selected_shapes = new_shapes
        if len(new_shapes) == 1:
            self.selected_shape = new_shapes[0]
        self.annotation_changed.emit()
        self.update()
        print(f"📋 Pasted {len(new_shapes)} shapes")

    def start_multi_move(self, pos):
        """Start moving all selected shapes as a group."""
        self.multi_moving = True
        self.multi_move_start = self.widget_to_image(pos)
        self.setCursor(Qt.ClosedHandCursor)

    def update_multi_move(self, pos):
        """Update position of all selected shapes during group move."""
        if not self.multi_moving or not self.multi_move_start:
            return
        current_pos = self.widget_to_image(pos)
        dx = (current_pos[0] - self.multi_move_start[0]) / self.image_width
        dy = (current_pos[1] - self.multi_move_start[1]) / self.image_height
        # Build set of ids of all inner_shapes owned by selected outers.
        # shape.move() already recurses into inner_shape, so skip inner shapes
        # that are inner children of another selected shape to prevent double-move.
        selected_ids = set(id(s) for s in self.selected_shapes)
        inner_ids_to_skip = set()
        for shape in self.selected_shapes:
            inner = getattr(shape, 'inner_shape', None)
            if inner and id(inner) in selected_ids:
                inner_ids_to_skip.add(id(inner))
        for shape in self.selected_shapes:
            if id(shape) in inner_ids_to_skip:
                continue
            # Also skip bare 'inner' hollow-role shapes (inner of a non-selected outer)
            if getattr(shape, 'hollow_role', None) == 'inner':
                continue
            if hasattr(shape, 'move'):
                shape.move(dx, dy)
        self.multi_move_start = current_pos
        self.update()

    def finish_multi_move(self):
        """Finish group move and save state."""
        if self.multi_moving:
            self.save_state()
            self.annotation_changed.emit()
        self.multi_moving = False
        self.multi_move_start = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def start_multi_drag_copy(self, pos):
        """Start Ctrl+drag copy of all selected shapes."""
        self.multi_drag_originals = list(self.selected_shapes)
        self.multi_drag_copies = [s.copy() for s in self.selected_shapes]
        self.multi_drag_copy = True
        self.multi_move_start = self.widget_to_image(pos)
        for s in self.multi_drag_originals:
            s.selected = False
        # Apply the same inner-skip logic as update_multi_move so copies
        # don't double-move on first drag tick
        selected_ids = set(id(s) for s in self.multi_drag_originals)
        inner_ids_to_skip = set()
        for shape in self.multi_drag_originals:
            inner = getattr(shape, 'inner_shape', None)
            if inner and id(inner) in selected_ids:
                inner_ids_to_skip.add(id(inner))
        # Give copies a small initial offset (10px) so they're visually
        # distinct from originals immediately on Ctrl+click
        init_dx = 10 / max(self.image_width, 1)
        init_dy = 10 / max(self.image_height, 1)
        for i, c in enumerate(self.multi_drag_copies):
            orig = self.multi_drag_originals[i]
            if id(orig) not in inner_ids_to_skip and getattr(orig, 'hollow_role', None) != 'inner':
                if hasattr(c, 'move'):
                    c.move(init_dx, init_dy)
            c.selected = True
            self.shapes.append(c)
        self.selected_shapes = list(self.multi_drag_copies)
        self.setCursor(Qt.ClosedHandCursor)
        print(f"📋 Multi drag-copy started ({len(self.multi_drag_copies)} shapes)")

    def finish_multi_drag_copy(self):
        """Finish group drag-copy and save state."""
        self.save_state()
        self.multi_drag_copy = False
        self.multi_drag_copies = []
        self.multi_drag_originals = []
        self.multi_moving = False
        self.multi_move_start = None
        self.annotation_changed.emit()
        self.setCursor(Qt.ArrowCursor)
        print("✅ Multi drag-copy completed")
        self.update()

    def cancel_multi_drag_copy(self):
        """Cancel group drag-copy and restore originals."""
        for c in self.multi_drag_copies:
            if c in self.shapes:
                self.shapes.remove(c)
        for s in self.multi_drag_originals:
            s.selected = True
        self.selected_shapes = list(self.multi_drag_originals)
        self.multi_drag_copy = False
        self.multi_drag_copies = []
        self.multi_drag_originals = []
        self.multi_moving = False
        self.multi_move_start = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def nudge_selected(self, dx_px, dy_px):
        """Move selected shapes by pixel amount. Works for single or group selection."""
        targets = self.selected_shapes if self.selected_shapes else (
            [self.selected_shape] if self.selected_shape else []
        )
        if not targets:
            return
        self.save_state()
        dx = dx_px / max(self.image_width, 1)
        dy = dy_px / max(self.image_height, 1)
        # Skip inner shapes owned by a selected outer (move() recurses into them)
        target_ids = set(id(s) for s in targets)
        inner_ids_to_skip = set()
        for shape in targets:
            inner = getattr(shape, 'inner_shape', None)
            if inner and id(inner) in target_ids:
                inner_ids_to_skip.add(id(inner))
        for shape in targets:
            if id(shape) in inner_ids_to_skip:
                continue
            if getattr(shape, 'hollow_role', None) == 'inner':
                continue
            if hasattr(shape, 'move'):
                shape.move(dx, dy)
        self.annotation_changed.emit()
        self.update()

    def _show_multi_select_context_menu(self, global_pos):
        """Show right-click context menu when 2+ shapes are selected."""
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #1e1e2e; color: #d8e0f8; border: 1px solid #2e3a58; }"
            "QMenu::item { padding: 6px 20px; }"
            "QMenu::item:selected { background-color: #2e3a58; }"
            "QMenu::separator { height: 1px; background: #2e3a58; margin: 4px 0; }"
        )
        n = len(self.selected_shapes)
        title_act = menu.addAction(f"  {n} shapes selected")
        title_act.setEnabled(False)
        menu.addSeparator()
        act_copy     = menu.addAction("📋  Copy Group        Ctrl+C")
        act_paste    = menu.addAction("📌  Paste Group       Ctrl+V")
        act_delete   = menu.addAction("🗑️   Delete Group      Del")
        menu.addSeparator()
        act_desel    = menu.addAction("✖   Deselect All      Esc")
        act_sel_all  = menu.addAction("☑   Select All        Ctrl+A")
        chosen = menu.exec_(global_pos)
        if chosen == act_copy:
            self.copy_selected_group()
        elif chosen == act_paste:
            self.paste_group()
        elif chosen == act_delete:
            self.delete_selected_group()
        elif chosen == act_desel:
            self._clear_multi_selection()
            self.update()
        elif chosen == act_sel_all:
            self.select_all()

    def show_thickness_dialog(self, mode, shape):
        if not hasattr(self, 'thickness_dialog'):
            from gui.thickness_dialog import ThicknessDialog
            self.thickness_dialog = ThicknessDialog(self)
            self.thickness_dialog.offset_changed.connect(self.preview_hollow)
            self.thickness_dialog.applied.connect(self.commit_hollow)
            self.thickness_dialog.cancelled.connect(self.cancel_hollow_preview)
        self.thickness_dialog.show_near_shape(shape, self, mode)
        
    def preview_hollow(self, mode, offset):
        self._hollow_preview_shape = self.selected_shape
        self._hollow_preview_mode = mode
        self._hollow_preview_offset = offset
        self.update()

    def commit_hollow(self, mode, offset):
        shape = self._hollow_preview_shape
        if not shape: return
        self.save_state()
        
        from core.hollow_ops import offset_rect, offset_circle, offset_ellipse, offset_polygon, offset_bezier
        
        inner = None
        if shape.type == 'box':
            x1, y1, x2, y2 = shape.to_pixels()
            p = offset_rect(x1, y1, x2 - x1, y2 - y1, offset, mode)
            from core.annotation import BoundingBox
            inner = BoundingBox(image_size=(self.image_width, self.image_height))
            inner.from_pixels(p['x'], p['y'], p['x']+p['w'], p['y']+p['h'], self.image_width, self.image_height)
        elif shape.type == 'circle':
            cx, cy, r = shape.to_pixels()
            p = offset_circle(cx, cy, r, offset, mode)
            from core.circle_shape import CircleShape
            inner = CircleShape(image_size=(self.image_width, self.image_height))
            inner.from_pixels(p['cx'], p['cy'], p['r'])
        elif shape.type == 'ellipse':
            cx, cy, rx, ry = shape.to_pixels()
            p = offset_ellipse(cx, cy, rx, ry, offset, mode)
            from core.ellipse_shape import EllipseShape
            inner = EllipseShape(image_size=(self.image_width, self.image_height))
            inner.from_pixels(p['cx'], p['cy'], p['rx'], p['ry'])
        elif shape.type == 'polygon':
            pts = offset_polygon(shape.to_pixel_points(), offset, mode)
            from core.polygon_shape import PolygonShape
            inner = PolygonShape(image_size=(self.image_width, self.image_height))
            inner.from_pixel_points(pts)
            inner.closed = getattr(shape, 'closed', False)
        elif shape.type == 'bezier_polygon':
            px_ctrls = []
            for c in shape.ctrl:
                if c is not None:
                    px_ctrls.append((c[0] * self.image_width, c[1] * self.image_height))
                else:
                    px_ctrls.append(None)
            ap, cp = offset_bezier(shape.to_pixel_points(), px_ctrls, offset, mode)
            from core.bezier_shape import BezierPolygonShape
            inner = BezierPolygonShape(image_size=(self.image_width, self.image_height))
            inner.from_pixel_points(ap)
            # Convert cp back to normalized
            norm_cp = []
            for c in cp:
                if c is not None:
                    norm_cp.append((c[0] / self.image_width, c[1] / self.image_height))
                else:
                    norm_cp.append(None)
            inner.ctrl = norm_cp
            inner.closed = getattr(shape, 'closed', False)

            
        if inner:
            inner.class_id = shape.class_id
        if mode == 'outer':
            if shape in self.shapes:
                self.shapes.remove(shape)
            inner.attach_inner(shape)
        else:
            shape.attach_inner(inner)
        self.shapes.append(inner)
            
        self._hollow_preview_mode = None
        self._hollow_preview_shape = None
        self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
        self.update()

    def save_template_from_shape(self, shape):
        """Unified method to save a shape as a template, preserving its identity."""
        from PyQt5.QtWidgets import QInputDialog, QMessageBox
        import math
        
        name, ok = QInputDialog.getText(self, "Save Template", "Enter a name for the template:")
        if not (ok and name and name.strip()):
            return
            
        name = name.strip()
        t = shape.type
        
        # 1. Get pixel points representation (for preview/fallback)
        pixel_points = []
        if t == 'polygon' or t == 'bezier_polygon':
            if hasattr(shape, 'closed') and not shape.closed:
                QMessageBox.warning(self, "Error", f"Cannot save unclosed {t} as template")
                return
            pixel_points = shape.to_pixel_points()
        elif t == 'box':
            x1, y1, x2, y2 = shape.to_pixels()
            pixel_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        elif t == 'circle':
            cx, cy, r = shape.to_pixels()
            for i in range(36):
                angle = 2 * math.pi * i / 36
                pixel_points.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        elif t == 'ellipse':
            cx, cy, rx, ry = shape.to_pixels()
            for i in range(36):
                angle = 2 * math.pi * i / 36
                pixel_points.append((int(cx + rx * math.cos(angle)), int(cy + ry * math.sin(angle))))
        elif t in ('frame', 'donut', 'hollow_ellipse'):
            # Convert to outer points for standard template representation
            if t == 'frame':
                x1, y1, x2, y2 = shape.to_pixels()
                pixel_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                pts = shape.to_pixels()
                cx, cy = pts[0], pts[1]
                rx = pts[2]
                ry = pts[3] if len(pts) > 3 else pts[2]
                for i in range(36):
                    angle = 2 * math.pi * i / 36
                    pixel_points.append((int(cx + rx * math.cos(angle)), int(cy + ry * math.sin(angle))))
        
        if not pixel_points or len(pixel_points) < 3:
            QMessageBox.warning(self, "Error", f"Invalid shape '{t}' for template")
            return

        # 2. Get inner points if any
        inner_pixel_points = None
        inner_type = None
        
        # Priority 1: Object-based inner shape (for Boxes, Circles, Ellipses, and some Polygons)
        if getattr(shape, 'inner_shape', None):
            inner_type = shape.inner_shape.type
            if hasattr(shape.inner_shape, 'to_pixel_points'):
                inner_pixel_points = shape.inner_shape.to_pixel_points()
            else:
                ipts = shape.inner_shape.to_pixels()
                if shape.inner_shape.type == 'box':
                    ix1, iy1, ix2, iy2 = ipts
                    inner_pixel_points = [(ix1, iy1), (ix2, iy1), (ix2, iy2), (ix1, iy2)]
                elif shape.inner_shape.type == 'circle':
                    icx, icy, ir = ipts
                    inner_pixel_points = [(int(icx + ir * math.cos(2*math.pi*i/36)), int(icy + ir * math.sin(2*math.pi*i/36))) for i in range(36)]
                elif shape.inner_shape.type == 'ellipse':
                    icx, icy, irx, iry = ipts
                    inner_pixel_points = [(int(icx + irx * math.cos(2*math.pi*i/36)), int(icy + iry * math.sin(2*math.pi*i/36))) for i in range(36)]
        
        # Priority 2: List-based inner points (for standard PolygonShape)
        if not inner_pixel_points and getattr(shape, 'inner_points', None):
            inner_type = 'polygon'
            if hasattr(shape, 'get_inner_pixel_points'):
                inner_pixel_points = shape.get_inner_pixel_points()
            else:
                cw, ch = self.image_width, self.image_height
                inner_pixel_points = [(int(nx * cw), int(ny * ch)) for nx, ny in shape.inner_points]
        
        # 3. Get control points for bezier
        
        # 3. Get control points for bezier
        ctrl_points = None
        inner_ctrl_points = None
        if t == 'bezier_polygon':
            ctrl_points = []
            for c in getattr(shape, 'ctrl', []):
                ctrl_points.append((c[0] * self.image_width, c[1] * self.image_height) if c is not None else None)
            
            # Use inner_shape for bezier inner control points if it exists as a bezier
            target_inner = shape.inner_shape if getattr(shape, 'inner_shape', None) else shape
            if getattr(target_inner, 'inner_control_points', None):
                inner_ctrl_points = []
                for c in target_inner.inner_control_points:
                    inner_ctrl_points.append((c[0] * self.image_width, c[1] * self.image_height) if c is not None else None)
            elif getattr(target_inner, 'ctrl', None) and target_inner != shape:
                # If it's an attached inner shape that is a bezier
                inner_ctrl_points = []
                for c in target_inner.ctrl:
                    inner_ctrl_points.append((c[0] * self.image_width, c[1] * self.image_height) if c is not None else None)

        # 4. Get native parameters
        native = {'inner_type': inner_type}
        if t == 'box':
            x1, y1, x2, y2 = shape.to_pixels()
            native.update({'w': x2 - x1, 'h': y2 - y1})
        elif t == 'circle':
            _, _, r = shape.to_pixels()
            native.update({'r': r})
        elif t == 'ellipse':
            _, _, rx, ry = shape.to_pixels()
            native.update({'rx': rx, 'ry': ry})
        elif t == 'frame':
            _, _, w, h = shape.to_pixels()
            native.update({
                'w': w, 'h': h,
                't_top': getattr(shape, 't_top', 20.0), 't_bottom': getattr(shape, 't_bottom', 20.0),
                't_left': getattr(shape, 't_left', 20.0), 't_right': getattr(shape, 't_right', 20.0)
            })
        elif t == 'donut':
            _, _, outer_r, inner_r = shape.to_pixels()
            native.update({'outer_r': outer_r, 'inner_r': inner_r})
        elif t == 'hollow_ellipse':
            _, _, outer_rx, outer_ry, inner_rx, inner_ry = shape.to_pixels()
            native.update({'outer_rx': outer_rx, 'outer_ry': outer_ry, 'inner_rx': inner_rx, 'inner_ry': inner_ry})

        # 5. Add to manager
        success = self.template_manager.add_template(
            name, pixel_points, self.image_width, self.image_height,
            inner_pixel_points=inner_pixel_points,
            shape_type=t,
            ctrl_points=ctrl_points,
            inner_ctrl_points=inner_ctrl_points,
            native_params=native
        )
        
        if success:
            print(f"✅ Saved template: {name} (Type: {t})")
            if hasattr(self.parent_window, 'update_template_dropdown'):
                self.parent_window.update_template_dropdown(name)
        else:
            QMessageBox.warning(self, "Error", "Failed to save template")


    def cancel_hollow_preview(self):
        self._hollow_preview_mode = None
        self._hollow_preview_shape = None
        self.update()
        
    def select_shape(self, pos):

        """Select a shape at the given position - only one shape at a time"""
        if not self.shapes:
            self.selected_shape = None
            self.selected_shapes = []
            self.shape_selected.emit("none")
            return
            
        # First, deselect ALL shapes and clear multi-select
        for shape in self.shapes:
            shape.selected = False
        self.selected_shapes = []
            
        # Check each shape (from top to bottom)
        image_x, image_y = self.widget_to_image(pos)
        selected = False
        for shape in reversed(self.shapes):
            if hasattr(shape, 'contains_point') and shape.contains_point(image_x, image_y):
                shape.selected = True
                self.selected_shape = shape
                self.selected_shapes = [shape]
                selected = True
                # Print selected shape info
                if self.class_manager and shape.class_id:
                    cls = self.class_manager.get_class(shape.class_id)
                    if cls:
                        shape_type = getattr(shape, 'type', 'box')
                        print(f"🔍 Selected {shape_type}: {cls.name}")
                        self.shape_selected.emit(shape_type)
                break
        
        if not selected:
            self.selected_shape = None
            self.selected_shapes = []
            self.shape_selected.emit("none")
            print("👆 Clicked on empty area")
                    
        self.update()
        
    def delete_selected(self):
        """Delete the selected shape"""
        if self.selected_shape:
            self.save_state()  # Save state before deleting
            self.shapes.remove(self.selected_shape)
            self.selected_shape = None
            self.shape_selected.emit("none")
            self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
            self.update()
            print("🗑️ Deleted selected shape")
            
    def delete_all(self):
        """Delete all shapes from the canvas"""
        if self.shapes:
            self.save_state()  # Save state before deleting all
            self.shapes.clear()
            self.selected_shape = None
            self.shape_selected.emit("none")
            self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
            self.update()
            print("🗑️ Deleted all shapes")
        
    def paintEvent(self, event):
        """Handle painting events"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Draw image if loaded
        if self.pixmap and not self.pixmap.isNull():
            # Calculate scaled dimensions
            scaled_width = int(self.image_width * self.scale)
            scaled_height = int(self.image_height * self.scale)
            
            # Scale the pixmap
            self.scaled_pixmap = self.pixmap.scaled(
                scaled_width, scaled_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Draw the image
            painter.drawPixmap(
                int(self.offset_x), int(self.offset_y),
                self.scaled_pixmap
            )
            
            # Draw all shapes
            self.draw_shapes(painter)
            
            # Draw current shape if drawing
            if self.drawing and self.current_shape:
                if isinstance(self.current_shape, BoundingBox):
                    self.draw_single_box(painter, self.current_shape, QColor(255, 255, 0))
                elif isinstance(self.current_shape, FrameShape):
                    self.draw_frame(painter, self.current_shape, QColor(255, 255, 0))
            
            # Draw polygon preview if drawing polygon
            if self.polygon_points and len(self.polygon_points) > 0:
                self.draw_polygon_preview(painter)

            # Draw bezier preview if drawing bezier
            if self.bezier_points and len(self.bezier_points) > 0:
                self.draw_bezier_preview(painter)
                
            # Draw circle preview if drawing circle
            if self.circle_center and self.circle_radius > 0:
                self.draw_circle_preview(painter)
                
            # Draw ellipse preview if drawing ellipse
            if hasattr(self, 'ellipse_center') and self.ellipse_center and self.ellipse_radius_x > 0:
                self.draw_ellipse_preview(painter)
                
            # Draw donut preview if drawing donut
            if hasattr(self, 'donut_center') and self.donut_center and self.donut_radius > 0:
                self.draw_donut_preview(painter)
                
            # Draw hollow ellipse preview if drawing
            if hasattr(self, 'hollow_ellipse_center') and self.hollow_ellipse_center and self.hollow_ellipse_rx > 0:
                self.draw_hollow_ellipse_preview(painter)
            
            # Draw hollow offset preview
            if getattr(self, '_hollow_preview_mode', None) and getattr(self, '_hollow_preview_shape', None):
                self._draw_hollow_preview(painter)
            
            # Draw stamp preview if stamping
            if hasattr(self, 'stamping') and self.stamping and self.stamp_center and self.stamp_current_pos:
                self.draw_stamp_preview(painter)
            
            # Draw group bounding box when 2+ shapes are selected
            if len(self.selected_shapes) > 1:
                bbox = self._get_group_bounding_box()
                if bbox:
                    x1, y1, x2, y2 = bbox
                    painter.setPen(QPen(QColor(100, 180, 255), 1, Qt.DashLine))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                    h = 6
                    painter.setBrush(QBrush(QColor(100, 180, 255)))
                    painter.setPen(Qt.NoPen)
                    for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                        painter.drawRect(hx - h//2, hy - h//2, h, h)
                    painter.setPen(QPen(QColor(100, 180, 255)))
                    f = QFont()
                    f.setPointSize(9)
                    painter.setFont(f)
                    painter.drawText(x1 + 4, y1 - 6, f"{len(self.selected_shapes)} selected")

            # Draw rubber band selection rectangle
            if self.rubber_band_active and self.rubber_band_start and self.rubber_band_end:
                rx = min(self.rubber_band_start.x(), self.rubber_band_end.x())
                ry = min(self.rubber_band_start.y(), self.rubber_band_end.y())
                rw = abs(self.rubber_band_end.x() - self.rubber_band_start.x())
                rh = abs(self.rubber_band_end.y() - self.rubber_band_start.y())
                painter.setPen(QPen(QColor(100, 180, 255), 1, Qt.DashLine))
                painter.setBrush(QBrush(QColor(100, 180, 255, 30)))
                painter.drawRect(rx, ry, rw, rh)
                
        # Draw mode indicator
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        mode_text = f"Mode: {self.mode.upper()}"
        painter.drawText(10, 20, mode_text)
        
        # Draw shape type indicator
        if self.current_shape_type:
            shape_text = f"Shape: {self.current_shape_type.upper()}"
        else:
            shape_text = "Shape: NONE"
            painter.setPen(QPen(QColor(255, 100, 100), 1))
        painter.drawText(10, 40, shape_text)
        
        # Draw pan mode indicator
        if self.pan_mode:
            painter.setPen(QPen(QColor(100, 200, 255), 1))
            painter.drawText(10, 60, "Pan Mode: ON (Space to toggle)")
        
        # Draw current class indicator
        if self.class_manager:
            current_class = self.class_manager.get_current_class()
            if current_class:
                painter.setPen(QPen(QColor(current_class.color), 2))
                painter.setFont(QFont("Arial", 10))
                painter.drawText(10, 90, f"Class: {current_class.name}")

    def _draw_hollow_preview(self, painter):
        shape = self._hollow_preview_shape
        offset = self._hollow_preview_offset
        mode = self._hollow_preview_mode
        if not shape or not offset: return
        
        from core.hollow_ops import offset_rect, offset_circle, offset_ellipse, offset_polygon, offset_bezier
        
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        
        if shape.type == 'box':
            x1_p, y1_p, x2_p, y2_p = shape.to_pixels()
            p = offset_rect(x1_p, y1_p, x2_p - x1_p, y2_p - y1_p, offset, mode)
            x1, y1 = int(p['x']*self.scale + self.offset_x), int(p['y']*self.scale + self.offset_y)
            w, h = int(p['w']*self.scale), int(p['h']*self.scale)
            painter.drawRect(x1, y1, w, h)
        elif shape.type == 'circle':
            cx, cy, r = shape.to_pixels()
            p = offset_circle(cx, cy, r, offset, mode)
            wx, wy = int(p['cx']*self.scale + self.offset_x), int(p['cy']*self.scale + self.offset_y)
            wr = int(p['r']*self.scale)
            painter.drawEllipse(wx - wr, wy - wr, wr * 2, wr * 2)
        elif shape.type == 'ellipse':
            cx, cy, rx, ry = shape.to_pixels()
            p = offset_ellipse(cx, cy, rx, ry, offset, mode)
            wx, wy = int(p['cx']*self.scale + self.offset_x), int(p['cy']*self.scale + self.offset_y)
            wrx, wry = int(p['rx']*self.scale), int(p['ry']*self.scale)
            painter.drawEllipse(wx - wrx, wy - wry, wrx * 2, wry * 2)
        elif shape.type in ('polygon', 'bezier_polygon'):
            pts = offset_polygon(shape.to_pixel_points(), offset, mode)
            w_pts = [QPointF(int(pt[0]*self.scale + self.offset_x), int(pt[1]*self.scale + self.offset_y)) for pt in pts]
            if w_pts:
                painter.drawPolygon(QPolygonF(w_pts))
        
    def draw_shapes(self, painter):
        """Draw all shapes"""
        for shape in self.shapes:
            if getattr(shape, 'hollow_role', None) == 'inner':
                continue
                
            # Get class color
            color = QColor(0, 255, 0)  # Default green
            if self.class_manager and hasattr(shape, 'class_id') and shape.class_id:
                cls = self.class_manager.get_class(shape.class_id)
                if cls:
                    color = QColor(cls.color)
            
            # Draw based on shape type
            if hasattr(shape, 'type'):
                if shape.type == 'box' or isinstance(shape, BoundingBox):
                    self.draw_single_box(painter, shape, color)
                elif shape.type == 'polygon':
                    self.draw_polygon(painter, shape, color)
                elif shape.type == 'bezier_polygon':
                    self.draw_bezier_shape(painter, shape, color)
                elif shape.type == 'circle':
                    self.draw_circle(painter, shape, color)
                elif shape.type == 'ellipse':
                    self.draw_ellipse(painter, shape, color)
                elif shape.type == 'frame':
                    self.draw_frame(painter, shape, color)
                elif shape.type == 'donut':
                    self.draw_donut(painter, shape, color)
                elif shape.type == 'hollow_ellipse':
                    self.draw_hollow_ellipse(painter, shape, color)
            else:
                # Default to box for backward compatibility
                self.draw_single_box(painter, shape, color)
                
            # Draw class label for the shape
            if not (self.drag_copy or self.drawing or self.moving) and getattr(shape, 'class_id', None):
                if self.class_manager:
                    cls = self.class_manager.get_class(shape.class_id)
                    if cls:
                        px, py = 0, 0
                        stype = getattr(shape, 'type', 'box')
                        if hasattr(shape, 'to_pixels') or hasattr(shape, 'to_pixel_points'):
                            try:
                                if stype in ('polygon', 'bezier_polygon') and hasattr(shape, 'to_pixel_points'):
                                    pts = shape.to_pixel_points()
                                    if pts and len(pts) > 0:
                                        if isinstance(pts[0], (list, tuple)):
                                            px = min(p[0] for p in pts)
                                            py = min(p[1] for p in pts)
                                        else:
                                            px, py = pts[0], pts[1]
                                elif hasattr(shape, 'to_pixels'):
                                    pixels = shape.to_pixels()
                                    if stype == 'circle' and len(pixels) >= 3:
                                        px, py = pixels[0] - pixels[2], pixels[1] - pixels[2]
                                    elif stype == 'ellipse' and len(pixels) >= 4:
                                        px, py = pixels[0] - pixels[2], pixels[1] - pixels[3]
                                    elif stype == 'donut' and len(pixels) >= 4:
                                        px, py = pixels[0] - pixels[2], pixels[1] - pixels[2]
                                    elif stype == 'hollow_ellipse' and len(pixels) >= 6:
                                        px, py = pixels[0] - pixels[2], pixels[1] - pixels[3]
                                    elif len(pixels) >= 2:
                                        if isinstance(pixels[0], (int, float)):
                                            px, py = pixels[0], pixels[1]
                                        elif isinstance(pixels[0], (list, tuple)) and len(pixels[0]) >= 2:
                                            px, py = pixels[0][0], pixels[0][1]
                            except Exception:
                                pass 
                        wx = int(px * self.scale + self.offset_x)
                        wy = int(py * self.scale + self.offset_y)
                        
                        painter.setPen(QPen(Qt.white, 1))
                        painter.setFont(QFont("Arial", 8))
                        text = cls.name
                        text_width = painter.fontMetrics().horizontalAdvance(text)
                        text_height = painter.fontMetrics().height()
                        
                        label_color = QColor(cls.color) if cls.color else QColor(0, 0, 0, 200)
                        label_bg = label_color.darker(150)
                        label_bg.setAlpha(200)
                        
                        painter.fillRect(wx, wy - text_height - 5, text_width + 10, text_height + 5, label_bg)
                        painter.drawText(wx + 5, wy - 8, text)
        
    def draw_single_box(self, painter, box, color):
        """Draw a single bounding box"""
        x1, y1, x2, y2 = box.to_pixels()
        
        # Convert to widget coordinates
        x1 = int(x1 * self.scale + self.offset_x)
        y1 = int(y1 * self.scale + self.offset_y)
        x2 = int(x2 * self.scale + self.offset_x)
        y2 = int(y2 * self.scale + self.offset_y)
        
        # Set pen based on selection
        if box.selected:
            pen = QPen(QColor(255, 255, 0), 3)  # Yellow, thicker
        else:
            pen = QPen(color, 2)
            
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        
        path = QPainterPath()
        path.addRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        
        inner_rect = None
        if getattr(box, 'inner_shape', None):
            in_x1, in_y1, in_x2, in_y2 = box.inner_shape.to_pixels()
            wx1 = int(in_x1 * self.scale + self.offset_x)
            wy1 = int(in_y1 * self.scale + self.offset_y)
            wx2 = int(in_x2 * self.scale + self.offset_x)
            wy2 = int(in_y2 * self.scale + self.offset_y)
            
            inner_path = QPainterPath()
            inner_path.addRect(QRectF(wx1, wy1, wx2 - wx1, wy2 - wy1))
            path = path.subtracted(inner_path)
            inner_rect = QRect(wx1, wy1, wx2 - wx1, wy2 - wy1)

        painter.drawPath(path)
        if inner_rect:
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(inner_rect)
        
        # Draw resize handles for selected box
        if box.selected:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            
            # Draw corner handles
            half = self.handle_size // 2
            handles = box.get_resize_handles()
            for handle_name, (hx, hy) in handles.items():
                whx = int(hx * self.scale + self.offset_x)
                why = int(hy * self.scale + self.offset_y)
                painter.drawRect(whx - half, why - half, self.handle_size, self.handle_size)
        # Resize handles have been drawn; class label is now handled globally in draw_shapes.

    def draw_bezier_shape(self, painter, bezier_shape, color):
        """Draw a bezier polygon shape"""
        if not bezier_shape.points:
            return
            
        # Draw the main path
        path = bezier_shape._make_path(self.scale, self.offset_x, self.offset_y)
        
        # Hollow inner cutout support (like regular polygon)
        inner_path = None
        if getattr(bezier_shape, 'inner_shape', None) and getattr(bezier_shape.inner_shape, 'type', '') == 'bezier_polygon':
            inner_path = bezier_shape.inner_shape._make_path(self.scale, self.offset_x, self.offset_y)
        elif bezier_shape.inner_points:
            inner_bezier = BezierPolygonShape(bezier_shape.inner_points)
            inner_bezier.closed = True
            inner_bezier.image_width = bezier_shape.image_width
            inner_bezier.image_height = bezier_shape.image_height
            if hasattr(bezier_shape, 'inner_control_points'):
                inner_bezier.ctrl = list(bezier_shape.inner_control_points)
            inner_path = inner_bezier._make_path(self.scale, self.offset_x, self.offset_y)
        
        if inner_path is not None:
            path = path.subtracted(inner_path)
            
        if bezier_shape.selected:
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
            painter.setPen(QPen(color, 2, Qt.DashLine))
        else:
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
            painter.setPen(QPen(color, 2))
            
        painter.drawPath(path)
        
        # Draw handles if selected
        if bezier_shape.selected:
            half = self.handle_size // 2
            handles = bezier_shape.get_resize_handles()
            
            for handle_name, (hx, hy) in handles.items():
                whx = int(hx * self.scale + self.offset_x)
                why = int(hy * self.scale + self.offset_y)
                
                if handle_name.startswith('vertex_') or handle_name.startswith('inner_vertex_'):
                    # Vertex handle (white square)
                    painter.setBrush(QBrush(QColor(255, 255, 255)))
                    painter.setPen(QPen(QColor(0, 0, 0), 1))
                    painter.drawRect(whx - half, why - half, self.handle_size, self.handle_size)
                elif handle_name.startswith('ctrl_') or handle_name.startswith('inner_ctrl_') or handle_name.startswith('ctrl_inner_'):
                    # Control handle (yellow diamond)
                    painter.setBrush(QBrush(QColor(255, 255, 0)))
                    painter.setPen(QPen(QColor(0, 0, 0), 1))
                    diamond = QPolygonF([
                        QPointF(whx, why - half),
                        QPointF(whx + half, why),
                        QPointF(whx, why + half),
                        QPointF(whx - half, why)
                    ])
                    painter.drawPolygon(diamond)
                    
        # Draw class label if not dragging or copying
        if not (self.drag_copy or self.drawing or self.moving) and bezier_shape.class_id:
            cls = self.class_manager.get_class(bezier_shape.class_id)
            if cls and bezier_shape.points:
                wx, wy = bezier_shape._norm_to_px(*bezier_shape.points[0])
                wx = wx * self.scale + self.offset_x
                wy = wy * self.scale + self.offset_y
                painter.setPen(QPen(Qt.white, 1))
                painter.setFont(QFont("Arial", 8))
                text = cls.name
                text_width = painter.fontMetrics().horizontalAdvance(text)
                text_height = painter.fontMetrics().height()
                
                # Use class color for background to make it visible!
                label_color = QColor(cls.color) if cls.color else QColor(0, 0, 0, 200)
                # Darken the background color slightly to ensure white text is visible
                label_bg = label_color.darker(150)
                label_bg.setAlpha(200)
                
                painter.fillRect(int(wx), int(wy) - text_height - 5, text_width + 10, text_height + 5, label_bg)
                painter.drawText(int(wx) + 5, int(wy) - 5, text)

    def draw_bezier_preview(self, painter):
        """Draw bezier polygon while it is being created (straight lines preview)"""
        if not self.bezier_points:
            return
            
        # Draw lines between points
        painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
        for i in range(len(self.bezier_points) - 1):
            p1 = self.bezier_points[i]
            p2 = self.bezier_points[i + 1]
            wp1 = QPoint(int(p1[0] * self.scale + self.offset_x), int(p1[1] * self.scale + self.offset_y))
            wp2 = QPoint(int(p2[0] * self.scale + self.offset_x), int(p2[1] * self.scale + self.offset_y))
            painter.drawLine(wp1, wp2)
            
        # Draw line to cursor
        if self.current_mouse_pos:
            last_p = self.bezier_points[-1]
            w_last = QPoint(int(last_p[0] * self.scale + self.offset_x), int(last_p[1] * self.scale + self.offset_y))
            painter.drawLine(w_last, self.current_mouse_pos)
            
        # Draw points
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        half = self.handle_size // 2
        for p in self.bezier_points:
            wpx = int(p[0] * self.scale + self.offset_x)
            wpy = int(p[1] * self.scale + self.offset_y)
            painter.drawRect(wpx - half, wpy - half, self.handle_size, self.handle_size)

    def draw_polygon(self, painter, polygon, color):
        """Draw a polygon shape with highlighting when selected"""
        from PyQt5.QtGui import QPainterPath
        
        points = polygon.to_pixel_points()
        
        # Convert to widget coordinates
        widget_points = []
        for px, py in points:
            wx = int(px * self.scale + self.offset_x)
            wy = int(py * self.scale + self.offset_y)
            widget_points.append(QPointF(wx, wy))
        
        # Set pen based on selection - YELLOW when selected
        if polygon.selected:
            pen = QPen(QColor(255, 255, 0), 3)
        else:
            pen = QPen(color, 2)
        
        painter.setPen(pen)
        
        # Build outer polygon
        if len(widget_points) >= 3:
            outer_poly = QPolygonF(widget_points)
            
            # Check if this is a hollow polygon (has inner cutout)
            inner_pixel = None
            if getattr(polygon, 'inner_shape', None) and hasattr(polygon.inner_shape, 'to_pixel_points'):
                inner_pixel = polygon.inner_shape.to_pixel_points()
            elif polygon.inner_points:
                inner_pixel = polygon.get_inner_pixel_points()
                
            if inner_pixel:
                inner_widget = []
                for px, py in inner_pixel:
                    wx = int(px * self.scale + self.offset_x)
                    wy = int(py * self.scale + self.offset_y)
                    inner_widget.append(QPointF(wx, wy))
                
                if len(inner_widget) >= 3:
                    # Use QPainterPath subtraction for hollow effect
                    outer_path = QPainterPath()
                    outer_path.addPolygon(outer_poly)
                    outer_path.closeSubpath()
                    
                    inner_poly = QPolygonF(inner_widget)
                    inner_path = QPainterPath()
                    inner_path.addPolygon(inner_poly)
                    inner_path.closeSubpath()
                    
                    hollow_path = outer_path.subtracted(inner_path)
                    
                    if polygon.selected:
                        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
                    else:
                        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
                    
                    painter.drawPath(hollow_path)
                    
                    # Draw inner outline
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPolygon(inner_poly)
                else:
                    # Fallback: just draw outer
                    if polygon.selected:
                        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
                    else:
                        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
                    painter.drawPolygon(outer_poly)
            else:
                # Solid polygon (no inner cutout)
                if polygon.selected:
                    painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
                else:
                    painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
                painter.drawPolygon(outer_poly)
            
            # Draw vertex handles for selected polygon
        if polygon.selected:
            half = self.handle_size // 2
            mid_r = max(self.handle_size // 3, 4)
            handles = polygon.get_resize_handles()
            
            for handle_name, (hx, hy) in handles.items():
                wx = int(hx * self.scale + self.offset_x)
                wy = int(hy * self.scale + self.offset_y)
                
                if handle_name.startswith('vertex_') or handle_name.startswith('inner_'):
                    painter.setBrush(QBrush(QColor(255, 255, 255)))
                    painter.setPen(QPen(QColor(0, 0, 0), 1))
                    painter.drawRect(wx - half, wy - half, self.handle_size, self.handle_size)
                elif handle_name.startswith('mid_') or handle_name.startswith('mid_inner_'):
                    painter.setBrush(QBrush(QColor(180, 180, 255)))
                    painter.setPen(QPen(QColor(0, 0, 0), 1))
                    painter.drawEllipse(QPointF(wx, wy), mid_r, mid_r)
        
        # Draw class name if available
        if polygon.selected and self.class_manager and hasattr(polygon, 'class_id') and polygon.class_id:
            cls = self.class_manager.get_class(polygon.class_id)
            if cls and widget_points:
                # Use first point for text placement
                wx, wy = widget_points[0].x(), widget_points[0].y()
                painter.setPen(QPen(Qt.white, 1))
                painter.setFont(QFont("Arial", 8))
                
                # Draw background for text
                text = cls.name
                text_width = painter.fontMetrics().horizontalAdvance(text)
                text_height = painter.fontMetrics().height()
                painter.fillRect(int(wx), int(wy) - text_height - 5, text_width + 10, text_height + 5, QColor(0, 0, 0, 150))
                
                painter.drawText(int(wx) + 5, int(wy) - 8, text)
    
    def draw_circle(self, painter, circle, color):
        """Draw a circle shape with highlighting when selected"""
        cx, cy, r = circle.to_pixels()
        
        # Convert to widget coordinates
        wx = int(cx * self.scale + self.offset_x)
        wy = int(cy * self.scale + self.offset_y)
        wr = int(r * self.scale)
        
        # Set pen based on selection - YELLOW when selected
        if circle.selected:
            pen = QPen(QColor(255, 255, 0), 3)  # Yellow, thicker
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
        else:
            pen = QPen(color, 2)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        
        painter.setPen(pen)
        
        path = QPainterPath()
        path.addEllipse(QRectF(wx - wr, wy - wr, wr * 2, wr * 2))
        
        inner_ellipse = None
        if getattr(circle, 'inner_shape', None):
            in_cx, in_cy, in_r = circle.inner_shape.to_pixels()
            iwx = int(in_cx * self.scale + self.offset_x)
            iwy = int(in_cy * self.scale + self.offset_y)
            iwr = int(in_r * self.scale)
            
            inner_path = QPainterPath()
            inner_path.addEllipse(QRectF(iwx - iwr, iwy - iwr, iwr * 2, iwr * 2))
            path = path.subtracted(inner_path)
            inner_ellipse = QRect(iwx - iwr, iwy - iwr, iwr * 2, iwr * 2)

        painter.drawPath(path)
        if inner_ellipse:
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(inner_ellipse)
        
        # Draw resize handles for selected circle
        if circle.selected:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            
            half = self.handle_size // 2
            handles = circle.get_resize_handles()
            for handle_name, (hx, hy) in handles.items():
                whx = int(hx * self.scale + self.offset_x)
                why = int(hy * self.scale + self.offset_y)
                painter.drawRect(whx - half, why - half, self.handle_size, self.handle_size)
        
        # Draw class name if available
        if circle.selected and self.class_manager and hasattr(circle, 'class_id') and circle.class_id:
            cls = self.class_manager.get_class(circle.class_id)
            if cls:
                painter.setPen(QPen(Qt.white, 1))
                painter.setFont(QFont("Arial", 8))
                
                # Draw background for text
                text = cls.name
                text_width = painter.fontMetrics().horizontalAdvance(text)
                text_height = painter.fontMetrics().height()
                painter.fillRect(wx - wr, wy - wr - text_height - 5, text_width + 10, text_height + 5, QColor(0, 0, 0, 150))
                
                painter.drawText(wx - wr + 5, wy - wr - 8, text)
    
    def draw_ellipse(self, painter, ellipse, color):
        """Draw an ellipse shape with highlighting when selected"""
        cx, cy, rx, ry = ellipse.to_pixels()
        
        # Convert to widget coordinates
        wx = int(cx * self.scale + self.offset_x)
        wy = int(cy * self.scale + self.offset_y)
        wrx = int(rx * self.scale)
        wry = int(ry * self.scale)
        
        # Set pen based on selection - YELLOW when selected
        if ellipse.selected:
            pen = QPen(QColor(255, 255, 0), 3)  # Yellow, thicker
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
        else:
            pen = QPen(color, 2)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        
        painter.setPen(pen)
        
        path = QPainterPath()
        path.addEllipse(QRectF(wx - wrx, wy - wry, wrx * 2, wry * 2))
        
        inner_rect = None
        if getattr(ellipse, 'inner_shape', None):
            in_cx, in_cy, in_rx, in_ry = ellipse.inner_shape.to_pixels()
            iwx = int(in_cx * self.scale + self.offset_x)
            iwy = int(in_cy * self.scale + self.offset_y)
            iwrx = int(in_rx * self.scale)
            iwry = int(in_ry * self.scale)
            
            inner_path = QPainterPath()
            inner_path.addEllipse(QRectF(iwx - iwrx, iwy - iwry, iwrx * 2, iwry * 2))
            path = path.subtracted(inner_path)
            inner_rect = QRect(iwx - iwrx, iwy - iwry, iwrx * 2, iwry * 2)

        painter.drawPath(path)
        if inner_rect:
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(inner_rect)
        
        # Draw resize handles for selected ellipse
        if ellipse.selected:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            
            half = self.handle_size // 2
            handles = ellipse.get_resize_handles()
            for handle_name, (hx, hy) in handles.items():
                whx = int(hx * self.scale + self.offset_x)
                why = int(hy * self.scale + self.offset_y)
                painter.drawRect(whx - half, why - half, self.handle_size, self.handle_size)
        
        # Draw class name if available
        if ellipse.selected and self.class_manager and hasattr(ellipse, 'class_id') and ellipse.class_id:
            cls = self.class_manager.get_class(ellipse.class_id)
            if cls:
                painter.setPen(QPen(Qt.white, 1))
                painter.setFont(QFont("Arial", 8))
                
                # Draw background for text
                text = cls.name
                text_width = painter.fontMetrics().horizontalAdvance(text)
                text_height = painter.fontMetrics().height()
                painter.fillRect(wx - wrx, wy - wry - text_height - 5, text_width + 10, text_height + 5, QColor(0, 0, 0, 150))
                
                painter.drawText(wx - wrx + 5, wy - wry - 8, text)
    
    def draw_frame(self, painter, frame, color):
        """Draw a frame (hollow rectangle)"""
        x, y, w, h = frame.to_pixels()
        ix, iy, iw, ih = frame.get_inner_rect()
        
        # Convert to widget coordinates
        x1 = int(x * self.scale + self.offset_x)
        y1 = int(y * self.scale + self.offset_y)
        x2 = int((x + w) * self.scale + self.offset_x)
        y2 = int((y + h) * self.scale + self.offset_y)
        ix1 = int(ix * self.scale + self.offset_x)
        iy1 = int(iy * self.scale + self.offset_y)
        ix2 = int((ix + iw) * self.scale + self.offset_x)
        iy2 = int((iy + ih) * self.scale + self.offset_y)
        
        # Set pen based on selection
        if frame.selected:
            pen = QPen(QColor(255, 255, 0), 3)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
        else:
            pen = QPen(color, 2)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        
        painter.setPen(pen)
        
        # Draw outer rectangle
        painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
        
        # Draw inner rectangle (hole) with background color
        if iw > 0 and ih > 0:
            painter.setBrush(QBrush(QColor(30, 30, 30)))  # Background color
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRect(ix1, iy1, ix2 - ix1, iy2 - iy1))
            
            # Redraw outer border
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
        
        # Draw handles if selected
        if frame.selected:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            half = self.handle_size // 2
            
            # Outer handles
            for hx, hy in frame.get_resize_handles():
                whx = int(hx * self.scale + self.offset_x)
                why = int(hy * self.scale + self.offset_y)
                painter.drawRect(whx - half, why - half, self.handle_size, self.handle_size)
            
            # Inner handles
            for hx, hy in frame.get_inner_handles():
                whx = int(hx * self.scale + self.offset_x)
                why = int(hy * self.scale + self.offset_y)
                painter.drawEllipse(whx - half//2, why - half//2, half, half)
    
    def draw_donut(self, painter, donut, color):
        """Draw a donut (hollow circle)"""
        cx, cy, outer_r, inner_r = donut.to_pixels()
        
        # Convert to widget coordinates
        wcx = int(cx * self.scale + self.offset_x)
        wcy = int(cy * self.scale + self.offset_y)
        wor = int(outer_r * self.scale)
        wir = int(inner_r * self.scale)
        
        # Set pen based on selection
        if donut.selected:
            pen = QPen(QColor(255, 255, 0), 3)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
        else:
            pen = QPen(color, 2)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        
        painter.setPen(pen)
        
        # Draw outer circle
        painter.drawEllipse(wcx - wor, wcy - wor, wor * 2, wor * 2)
        
        # Draw inner circle (hole) with background color
        if wir > 0:
            painter.setBrush(QBrush(QColor(30, 30, 30)))  # Background color
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(wcx - wir, wcy - wir, wir * 2, wir * 2)
            
            # Redraw outer border
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(wcx - wor, wcy - wor, wor * 2, wor * 2)
        
        # Draw handles if selected
        if donut.selected:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            half = self.handle_size // 2
            
            # Outer handles
            for hx, hy in donut.get_outer_handles():
                whx = int(hx * self.scale + self.offset_x)
                why = int(hy * self.scale + self.offset_y)
                painter.drawEllipse(whx - half//2, why - half//2, half, half)
            
            # Inner handles
            for hx, hy in donut.get_inner_handles():
                whx = int(hx * self.scale + self.offset_x)
                why = int(hy * self.scale + self.offset_y)
                painter.drawRect(whx - half//2, why - half//2, half, half)
    
    def draw_hollow_ellipse(self, painter, shape, color):
        """Draw a hollow ellipse"""
        cx, cy, orx, ory, irx, iry = shape.to_pixels()
        
        # Convert to widget coordinates
        wcx = int(cx * self.scale + self.offset_x)
        wcy = int(cy * self.scale + self.offset_y)
        worx = int(orx * self.scale)
        wory = int(ory * self.scale)
        wirx = int(irx * self.scale)
        wiry = int(iry * self.scale)
        
        # Set pen based on selection
        if shape.selected:
            pen = QPen(QColor(255, 255, 0), 3)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
        else:
            pen = QPen(color, 2)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        
        painter.setPen(pen)
        
        # Draw outer ellipse
        painter.drawEllipse(wcx - worx, wcy - wory, worx * 2, wory * 2)
        
        # Draw inner ellipse (hole) with background color
        if wirx > 0 and wiry > 0:
            painter.setBrush(QBrush(QColor(30, 30, 30)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(wcx - wirx, wcy - wiry, wirx * 2, wiry * 2)
            
            # Redraw outer border
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(wcx - worx, wcy - wory, worx * 2, wory * 2)
        
        # Draw handles if selected
        if shape.selected:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            half = self.handle_size // 2
            
            # Outer handles (circles)
            for hx, hy in shape.get_outer_handles():
                whx = int(hx * self.scale + self.offset_x)
                why = int(hy * self.scale + self.offset_y)
                painter.drawEllipse(whx - half//2, why - half//2, half, half)
            
            # Inner handles (squares)
            for hx, hy in shape.get_inner_handles():
                whx = int(hx * self.scale + self.offset_x)
                why = int(hy * self.scale + self.offset_y)
                painter.drawRect(whx - half//2, why - half//2, half, half)
    
    def draw_hollow_ellipse_preview(self, painter):
        """Draw hollow ellipse preview"""
        if not hasattr(self, 'hollow_ellipse_center') or not self.hollow_ellipse_center or self.hollow_ellipse_rx <= 0:
            return
            
        cx, cy = self.hollow_ellipse_center
        wcx = int(cx * self.scale + self.offset_x)
        wcy = int(cy * self.scale + self.offset_y)
        wrx = int(self.hollow_ellipse_rx * self.scale)
        wry = int(self.hollow_ellipse_ry * self.scale)
        wirx = int(getattr(self, 'hollow_ellipse_inner_rx', 0) * self.scale)
        wiry = int(getattr(self, 'hollow_ellipse_inner_ry', 0) * self.scale)
        
        # Draw outer ellipse preview
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(wcx, wcy), wrx, wry)
        
        # Draw inner ellipse preview
        if wirx > 0 and wiry > 0:
            painter.setPen(QPen(QColor(255, 200, 0), 1, Qt.DashLine))
            painter.drawEllipse(QPointF(wcx, wcy), wirx, wiry)
            
        # Draw center point
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        half = self.handle_size // 2
        painter.drawRect(wcx - half, wcy - half, self.handle_size, self.handle_size)

    def draw_stamp_preview(self, painter):
        """Draw template stamp preview based on drag distance"""
        if not hasattr(self, 'stamping') or not self.stamping or not self.stamp_center or not self.stamp_current_pos:
            return
            
        dx = abs(self.stamp_current_pos[0] - self.stamp_center[0])
        dy = abs(self.stamp_current_pos[1] - self.stamp_center[1])
        
        # If user hasn't dragged much, show default preview size
        info = self.template_manager.get_template_info(self.stamp_template_name)
        orig_half_w = info['orig_w'] / 2.0
        orig_half_h = info['orig_h'] / 2.0
        if orig_half_w == 0: orig_half_w = 50.0
        if orig_half_h == 0: orig_half_h = 50.0

        if dx < 5 and dy < 5:
            scale_x = orig_half_w
            scale_y = orig_half_h
        else:
            scale_x = max(dx, 10)
            scale_y = max(dy, 10)
            
        ratio_x = scale_x / orig_half_w
        ratio_y = scale_y / orig_half_h
        
        shape_type, outer_points, inner_points, ctrl_points, inner_ctrl_points, native_params = self.template_manager.get_pixel_points(
            self.stamp_template_name,
            self.stamp_center[0], self.stamp_center[1],
            scale_x, scale_y
        )
        
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.setBrush(QBrush(QColor(255, 255, 0, 50)))
        
        from PyQt5.QtGui import QPainterPath, QPolygonF
        from PyQt5.QtCore import QPointF
        
        wcx = int(self.stamp_center[0] * self.scale + self.offset_x)
        wcy = int(self.stamp_center[1] * self.scale + self.offset_y)
        
        # Custom draws for native shapes so the preview matches the rehydrated shape
        if shape_type in ('box', 'frame', 'circle', 'ellipse', 'donut', 'hollow_ellipse'):
            if shape_type == 'box' or shape_type == 'frame':
                orig_w = native_params.get('w', 100)
                orig_h = native_params.get('h', 100)
                w = int(orig_w * ratio_x * self.scale)
                h = int(orig_h * ratio_y * self.scale)
                
                if shape_type == 'frame':
                    # Draw outer frame
                    path = QPainterPath()
                    path.addRect(wcx - w//2, wcy - h//2, w, h)
                    
                    # Draw inner cutout
                    w_ratio = (orig_w * ratio_x) / max(orig_w, 1)
                    h_ratio = (orig_h * ratio_y) / max(orig_h, 1)
                    
                    t_top = int(native_params.get('t_top', 20.0) * h_ratio * self.scale)
                    t_bottom = int(native_params.get('t_bottom', 20.0) * h_ratio * self.scale)
                    t_left = int(native_params.get('t_left', 20.0) * w_ratio * self.scale)
                    t_right = int(native_params.get('t_right', 20.0) * w_ratio * self.scale)
                    
                    ix = (wcx - w//2) + t_left
                    iy = (wcy - h//2) + t_top
                    iw = max(w - t_left - t_right, 4)
                    ih = max(h - t_top - t_bottom, 4)
                    
                    inner_path = QPainterPath()
                    inner_path.addRect(ix, iy, iw, ih)
                    path = path.subtracted(inner_path)
                    painter.drawPath(path)
                else:
                    painter.drawRect(wcx - w//2, wcy - h//2, w, h)
                    
            elif shape_type == 'circle':
                r = int(native_params.get('r', 50) * ratio_x * self.scale)
                painter.drawEllipse(QPointF(wcx, wcy), r, r)
                
            elif shape_type == 'ellipse':
                rx = int(native_params.get('rx', 50) * ratio_x * self.scale)
                ry = int(native_params.get('ry', 50) * ratio_y * self.scale)
                painter.drawEllipse(QPointF(wcx, wcy), rx, ry)
                
            elif shape_type == 'donut':
                outer_r = int(native_params.get('outer_r', 50) * ratio_x * self.scale)
                inner_r = int(native_params.get('inner_r', 25) * ratio_x * self.scale)
                
                path = QPainterPath()
                path.addEllipse(QPointF(wcx, wcy), outer_r, outer_r)
                inner_path = QPainterPath()
                inner_path.addEllipse(QPointF(wcx, wcy), inner_r, inner_r)
                path = path.subtracted(inner_path)
                painter.drawPath(path)
                
            elif shape_type == 'hollow_ellipse':
                outer_rx = int(native_params.get('outer_rx', 50) * ratio_x * self.scale)
                outer_ry = int(native_params.get('outer_ry', 50) * ratio_y * self.scale)
                inner_rx = int(native_params.get('inner_rx', 25) * ratio_x * self.scale)
                inner_ry = int(native_params.get('inner_ry', 25) * ratio_y * self.scale)
                
                path = QPainterPath()
                path.addEllipse(QPointF(wcx, wcy), outer_rx, outer_ry)
                inner_path = QPainterPath()
                inner_path.addEllipse(QPointF(wcx, wcy), inner_rx, inner_ry)
                path = path.subtracted(inner_path)
                painter.drawPath(path)

        else:
            # Generic Polygon/Bezier path
            if outer_points and len(outer_points) >= 3:
                outer_poly = QPolygonF()
                for px, py in outer_points:
                    wx = int(px * self.scale + self.offset_x)
                    wy = int(py * self.scale + self.offset_y)
                    outer_poly.append(QPointF(wx, wy))
                
                outer_path = QPainterPath()
                outer_path.addPolygon(outer_poly)
                outer_path.closeSubpath()
                
                if inner_points and len(inner_points) >= 3:
                    inner_poly = QPolygonF()
                    for px, py in inner_points:
                        wx = int(px * self.scale + self.offset_x)
                        wy = int(py * self.scale + self.offset_y)
                        inner_poly.append(QPointF(wx, wy))
                        
                    inner_path = QPainterPath()
                    inner_path.addPolygon(inner_poly)
                    inner_path.closeSubpath()
                    outer_path = outer_path.subtracted(inner_path)
                
                painter.drawPath(outer_path)

    def draw_polygon_preview(self, painter):
        """Draw polygon preview while drawing"""
        if len(self.polygon_points) < 1:
            return
            
        # Convert points to widget coordinates
        widget_points = []
        for px, py in self.polygon_points:
            wx = int(px * self.scale + self.offset_x)
            wy = int(py * self.scale + self.offset_y)
            widget_points.append(QPointF(wx, wy))
        
        # Draw lines between points
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        for i in range(len(widget_points) - 1):
            painter.drawLine(widget_points[i], widget_points[i + 1])
        
        # Draw vertices
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        half = self.handle_size // 2
        for wx, wy in [(p.x(), p.y()) for p in widget_points]:
            painter.drawRect(int(wx - half), int(wy - half), self.handle_size, self.handle_size)
    
    def draw_circle_preview(self, painter):
        """Draw circle preview while drawing"""
        if not self.circle_center:
            return
            
        cx, cy = self.circle_center
        wx = int(cx * self.scale + self.offset_x)
        wy = int(cy * self.scale + self.offset_y)
        wr = int(self.circle_radius * self.scale)
        
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(wx - wr, wy - wr, wr * 2, wr * 2)
        
        # Draw center point
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        half = self.handle_size // 2
        painter.drawRect(wx - half, wy - half, self.handle_size, self.handle_size)
    
    def draw_ellipse_preview(self, painter):
        """Draw ellipse preview while drawing"""
        if not hasattr(self, 'ellipse_center') or not self.ellipse_center:
            return
            
        cx, cy = self.ellipse_center
        wx = int(cx * self.scale + self.offset_x)
        wy = int(cy * self.scale + self.offset_y)
        wrx = int(self.ellipse_radius_x * self.scale)
        wry = int(self.ellipse_radius_y * self.scale)
        
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(wx - wrx, wy - wry, wrx * 2, wry * 2)
        
        # Draw center point
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        half = self.handle_size // 2
        painter.drawRect(wx - half, wy - half, self.handle_size, self.handle_size)
    
    def draw_donut_preview(self, painter):
        """Draw donut preview while drawing"""
        if not hasattr(self, 'donut_center') or not self.donut_center:
            return
            
        cx, cy = self.donut_center
        wx = int(cx * self.scale + self.offset_x)
        wy = int(cy * self.scale + self.offset_y)
        wr = int(self.donut_radius * self.scale)
        
        # Draw outer circle preview
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(wx - wr, wy - wr, wr * 2, wr * 2)
        
        # Draw inner circle preview (45% of outer)
        wir = int(wr * 0.45)
        painter.setPen(QPen(QColor(255, 200, 0), 1, Qt.DashLine))
        painter.drawEllipse(wx - wir, wy - wir, wir * 2, wir * 2)
        
        # Draw center point
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        half = self.handle_size // 2
        painter.drawRect(wx - half, wy - half, self.handle_size, self.handle_size)
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        # Ensure canvas has keyboard focus for shortcuts
        self.setFocus()
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.pan_mode):
            # Pan mode
            self.dragging = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            
        elif event.button() == Qt.RightButton:
            # Get shape at position
            shape = self.find_shape_at(event.pos())
            
            # Handle right click vertex insertion for bezier polygons
            if shape and getattr(shape, 'type', '') == 'bezier_polygon':
                image_x, image_y = self.widget_to_image(event.pos())
                if hasattr(shape, 'get_closest_edge'):
                    # Check if hovering near boundary (10 pixels threshold scaled)
                    edge_info = shape.get_closest_edge(image_x, image_y, 10.0 / self.scale)
                    if edge_info:
                        idx, pt, is_inner = edge_info
                        menu = QMenu(self)
                        add_action = menu.addAction("Add Curve Point Here")
                        action = menu.exec_(self.mapToGlobal(event.pos()))
                        if action == add_action:
                            self.save_state()  # Save state for undo
                            shape.insert_vertex(idx, pt, is_inner)
                            self.update()
                            print(f"✅ Inserted new curve point via context menu")
                        return

                # Fallback to direct ctrl point click
                handle = self.get_resize_handle_at_pos(event.pos(), shape)
                if handle and (handle.startswith('ctrl_') or handle.startswith('ctrl_inner_')):
                    self.save_state()  # Save before modifying
                    if hasattr(shape, 'insert_vertex_at_ctrl'):
                        inserted = shape.insert_vertex_at_ctrl(handle)
                        if inserted:
                            print(f"✅ Inserted new vertex at {handle} via right-click")
                            self.update()
                            return

            if shape and getattr(shape, 'hollow_role', None) != 'inner':
                # Need to select it if not selected
                if not shape.selected:
                    self.select_shape(event.pos())
                    
                self._invoke_context_menu(event.globalPos(), event.pos())
                return
            else:
                self._invoke_context_menu(event.globalPos(), event.pos())
                return
            
        elif event.button() == Qt.LeftButton and not self.pan_mode:
            if self.pixmap and not self.pixmap.isNull():
                
                # Check if we have a selected class
                current_class = self.class_manager.get_current_class() if self.class_manager else None
                
                # Check modifier keys
                modifiers = QApplication.keyboardModifiers()
                ctrl_pressed = bool(modifiers & Qt.ControlModifier)
                shift_pressed = bool(modifiers & Qt.ShiftModifier)
                
                # ── MULTI-SELECT: Shift+click toggles a single shape ──
                if shift_pressed and (not self.current_shape_type or self.current_shape_type == 'none'):
                    image_x, image_y = self.widget_to_image(event.pos())
                    clicked = None
                    for shape in reversed(self.shapes):
                        if getattr(shape, 'hollow_role', None) == 'inner':
                            continue
                        if hasattr(shape, 'contains_point') and shape.contains_point(image_x, image_y):
                            clicked = shape
                            break
                    if clicked:
                        if clicked in self.selected_shapes:
                            self.selected_shapes.remove(clicked)
                            clicked.selected = False
                        else:
                            self.selected_shapes.append(clicked)
                            clicked.selected = True
                        self.selected_shape = self.selected_shapes[-1] if self.selected_shapes else None
                        self.update()
                    return
                
                # FIRST: Check if we're over a resize handle of selected shape (only single-select)
                if self.selected_shape and len(self.selected_shapes) <= 1:
                    handle = self.get_resize_handle_at_pos(event.pos(), self.selected_shape)
                    if handle:
                        # Start resizing
                        self.resizing = True
                        self.resizing_handle = handle
                        self.resize_start_pos = self.widget_to_image(event.pos())
                        
                        # Call begin_resize on the shape
                        if hasattr(self.selected_shape, 'begin_resize'):
                            import inspect
                            sig = inspect.signature(self.selected_shape.begin_resize)
                            if len(sig.parameters) > 0:
                                self.selected_shape.begin_resize(handle)
                            else:
                                self.selected_shape.begin_resize()
                        
                        return
                
                # SECOND: Check if we're clicking on a shape (for moving or selecting)
                image_x, image_y = self.widget_to_image(event.pos())
                clicked_shape = None
                for shape in reversed(self.shapes):
                    if getattr(shape, 'hollow_role', None) == 'inner':
                        continue
                    if hasattr(shape, 'contains_point') and shape.contains_point(image_x, image_y):
                        clicked_shape = shape
                        break
                
                if clicked_shape:
                    # If Ctrl is pressed, start drag-copy (single or group)
                    if ctrl_pressed:
                        if len(self.selected_shapes) > 1 and clicked_shape in self.selected_shapes:
                            self.start_multi_drag_copy(event.pos())
                        else:
                            self.start_drag_copy(clicked_shape, event.pos())
                        return
                    
                    # If the clicked shape is already selected, start moving it (single or group)
                    if clicked_shape.selected:
                        if len(self.selected_shapes) > 1 and clicked_shape in self.selected_shapes:
                            self.start_multi_move(event.pos())
                        else:
                            self.start_move(clicked_shape, event.pos())
                        return
                    else:
                        # Just select the shape (clears multi-select)
                        self.select_shape(event.pos())
                        return
                
                # THIRD: Handle inner cutout drawing (priority over tool selection)
                if self.drawing_inner_cutout:
                    self.start_polygon_drawing(event.pos())
                    return
                
                # FOURTH: Handle drawing based on current tool
                if self.current_shape_type and self.current_shape_type != 'none':
                    if self.current_shape_type == 'polygon':
                        self.start_polygon_drawing(event.pos())
                    elif self.current_shape_type == 'circle':
                        self.start_circle_drawing(event.pos())
                    elif self.current_shape_type == 'ellipse':
                        self.start_ellipse_drawing(event.pos())
                    elif self.current_shape_type == 'box':
                        self.start_drawing(event.pos())
                    elif self.current_shape_type == 'frame':
                        self.start_frame_drawing(event.pos())
                    elif self.current_shape_type == 'donut':
                        self.start_donut_drawing(event.pos())
                    elif self.current_shape_type == 'hollow_ellipse':
                        self.start_hollow_ellipse_drawing(event.pos())
                    elif self.current_shape_type == 'template':
                        self.start_polygon_drawing(event.pos())  # reuse polygon flow
                    elif self.current_shape_type == 'bezier_polygon':
                        self.start_bezier_drawing(event.pos())
                    elif self.current_shape_type == 'stamp':
                        self.start_stamp(event.pos())
                else:
                    # No tool selected — click on empty area starts rubber band
                    self._clear_multi_selection()
                    self.rubber_band_active = True
                    self.rubber_band_start = event.pos()
                    self.rubber_band_end = event.pos()
                    self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        # Convert widget coordinates to image coordinates
        image_x, image_y = self.widget_to_image(event.pos())
        self.current_mouse_pos = event.pos()
        
        # Emit position signal
        self.position_changed.emit(image_x, image_y)
        
        # Rubber band update
        if self.rubber_band_active and self.rubber_band_start:
            self.rubber_band_end = event.pos()
            self.update()
            return
        
        # Multi-group move
        if self.multi_moving:
            self.update_multi_move(event.pos())
            return
        
        # Multi drag-copy move
        if self.multi_drag_copy:
            self.update_multi_move(event.pos())  # selected_shapes already points to copies
            return
        
        # Handle dragging for panning
        if self.dragging and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_mouse_pos = event.pos()
            self.update()
            
        # Handle moving shapes
        elif self.moving and not self.resizing:
            self.update_move(event.pos())
        
        # Handle drawing
        elif self.drawing:
            self.update_drawing(event.pos())
        
        # Handle drag-copy
        elif self.drag_copy and not self.resizing:
            self.update_drag_copy(event.pos())
        
        # Handle circle drawing
        elif self.current_shape_type == 'circle' and self.circle_center:
            self.update_circle_drawing(event.pos())
        
        # Handle ellipse drawing
        elif self.current_shape_type == 'ellipse' and hasattr(self, 'ellipse_center') and self.ellipse_center:
            self.update_ellipse_drawing(event.pos())
        
        # Handle donut drawing (preview only)
        elif self.current_shape_type == 'donut' and hasattr(self, 'donut_center') and self.donut_center:
            self.update_donut_drawing(event.pos())
        
        # Handle hollow ellipse drawing
        elif self.current_shape_type == 'hollow_ellipse' and hasattr(self, 'hollow_ellipse_center') and self.hollow_ellipse_center:
            self.update_hollow_ellipse_drawing(event.pos())
        
        # Handle stamp placement
        elif self.stamping and self.stamp_center:
            self.stamp_current_pos = self.widget_to_image(event.pos())
            self.update()
        
        # Handle resizing - with reduced sensitivity
        elif self.resizing and self.resizing_handle and self.selected_shape:
            current_pos = self.widget_to_image(event.pos())
            dx = current_pos[0] - self.resize_start_pos[0]
            dy = current_pos[1] - self.resize_start_pos[1]
            
            # Apply deadzone to prevent tiny movements (less sensitivity)
            if abs(dx) < 2 and abs(dy) < 2:
                return
                
            if hasattr(self.selected_shape, 'resize_from_handle'):
                self.selected_shape.resize_from_handle(self.resizing_handle, dx, dy)
                self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.dragging):
            self.dragging = False
            self.setCursor(Qt.OpenHandCursor if self.pan_mode else Qt.ArrowCursor)
            
        elif event.button() == Qt.LeftButton:
            # Multi-select release handlers
            if self.rubber_band_active:
                self._finish_rubber_band_selection()
                return
            if self.multi_moving:
                self.finish_multi_move()
                return
            if self.multi_drag_copy:
                self.finish_multi_drag_copy()
                return
            
            was_moving_or_resizing = self.moving or self.resizing  # AUTOSAVE INTEGRATION
            if self.moving:
                self.finish_move()
            elif self.drawing:
                self.finish_drawing()
            elif self.drag_copy:
                self.finish_drag_copy()
            elif self.current_shape_type == 'circle' and self.circle_center:
                self.finish_circle()
            elif self.current_shape_type == 'ellipse' and hasattr(self, 'ellipse_center') and self.ellipse_center:
                self.finish_ellipse()
            elif self.current_shape_type == 'donut' and hasattr(self, 'donut_center') and self.donut_center:
                self.finish_donut()
            elif self.current_shape_type == 'hollow_ellipse' and hasattr(self, 'hollow_ellipse_center') and self.hollow_ellipse_center:
                self.finish_hollow_ellipse()
            elif self.stamping and self.stamp_center:
                self.finish_stamp()
            elif self.resizing:
                self.resizing = False
                self.resizing_handle = None
                self.resize_start_pos = None
                
                # Clear resize origin
                if self.selected_shape and hasattr(self.selected_shape, '_resize_origin'):
                    self.selected_shape._resize_origin = None
                
                print("✅ Resizing complete")
            
            # Ensure we're not stuck in any special state
            self.drag_copy = False
            self.pasting = False
            
            if was_moving_or_resizing:  # AUTOSAVE INTEGRATION
                self.annotation_changed.emit()
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming - zooms to cursor position"""
        if not self.pixmap or self.pixmap.isNull():
            return
            
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier and self.selected_shape:
            # Scale selected shape
            scale_factor = 1.05 if event.angleDelta().y() > 0 else 0.95
            self.scale_selected_shape(scale_factor)
            return
        
        # Get cursor position in widget coordinates
        cursor_pos = event.pos()
        
        # Convert to image coordinates before zoom
        image_pos = self.widget_to_image(cursor_pos)
        
        # Zoom factor
        zoom_factor = 1.1
        if event.angleDelta().y() > 0:
            new_scale = self.scale * zoom_factor
        else:
            new_scale = self.scale / zoom_factor
        
        # Keep zoom within limits
        new_scale = max(0.1, min(10.0, new_scale))
        
        if new_scale != self.scale:
            # Calculate new offset to keep cursor position fixed
            image_x = image_pos[0] * self.scale + self.offset_x
            image_y = image_pos[1] * self.scale + self.offset_y
            
            self.scale = new_scale
            
            # Adjust offset to keep the point under cursor at same screen position
            self.offset_x = image_x - image_pos[0] * self.scale
            self.offset_y = image_y - image_pos[1] * self.scale
            
            self.update()
            
    def scale_selected_shape(self, factor):
        """Scale the currently selected shape evenly by a given factor"""
        if not self.selected_shape:
            return
            
        self.save_state()
        
        # Determine the group center (from the outermost shape)
        def get_center(sh):
            if sh.type == 'box':
                return sh.x, sh.y
            elif sh.type == 'circle':
                return sh.center_x, sh.center_y
            elif sh.type == 'ellipse':
                return sh.center_x, sh.center_y
            elif sh.type == 'frame':
                return sh.x + sh.w / 2, sh.y + sh.h / 2
            elif sh.type in ('donut', 'hollow_ellipse'):
                return sh.cx, sh.cy
            elif sh.type in ('polygon', 'bezier_polygon'):
                if not sh.points: return 0.5, 0.5
                return (min(p[0] for p in sh.points) + max(p[0] for p in sh.points)) / 2.0, \
                       (min(p[1] for p in sh.points) + max(p[1] for p in sh.points)) / 2.0
            return 0.5, 0.5

        group_cx, group_cy = get_center(self.selected_shape)

        def do_scale(sh, f, cx, cy):
            # Scale dimensions and radii
            if sh.type == 'box':
                sh.width = min(1.0, sh.width * f)
                sh.height = min(1.0, sh.height * f)
                # If it's an inner shape, move it relative to group center
                if sh != self.selected_shape:
                    sh.x = cx + (sh.x - cx) * f
                    sh.y = cy + (sh.y - cy) * f
            
            elif sh.type == 'circle':
                sh.radius = min(1.0, sh.radius * f)
                if sh != self.selected_shape:
                    sh.center_x = cx + (sh.center_x - cx) * f
                    sh.center_y = cy + (sh.center_y - cy) * f
                    
            elif sh.type == 'ellipse':
                sh.radius_x = min(1.0, sh.radius_x * f)
                sh.radius_y = min(1.0, sh.radius_y * f)
                if sh != self.selected_shape:
                    sh.center_x = cx + (sh.center_x - cx) * f
                    sh.center_y = cy + (sh.center_y - cy) * f

            elif sh.type == 'frame':
                cur_cx = sh.x + sh.w / 2
                cur_cy = sh.y + sh.h / 2
                sh.w = min(1.0, sh.w * f)
                sh.h = min(1.0, sh.h * f)
                # Scale thicknesses
                sh.t_top *= f
                sh.t_bottom *= f
                sh.t_left *= f
                sh.t_right *= f
                # Apply movement relative to group center
                new_center_x = cx + (cur_cx - cx) * f
                new_center_y = cy + (cur_cy - cy) * f
                sh.x = max(0.0, new_center_x - sh.w / 2)
                sh.y = max(0.0, new_center_y - sh.h / 2)

            elif sh.type == 'donut':
                d = min(self.image_width, self.image_height)
                max_r = min(sh.cx, 1 - sh.cx, sh.cy, 1 - sh.cy) * min(self.image_width, self.image_height) / d
                sh.outer_r = min(max_r, sh.outer_r * f)
                sh.inner_r *= f
                if sh != self.selected_shape:
                    sh.cx = cx + (sh.cx - cx) * f
                    sh.cy = cy + (sh.cy - cy) * f

            elif sh.type == 'hollow_ellipse':
                sh.outer_rx = min(1.0, sh.outer_rx * f)
                sh.outer_ry = min(1.0, sh.outer_ry * f)
                sh.inner_rx *= f
                sh.inner_ry *= f
                if sh != self.selected_shape:
                    sh.cx = cx + (sh.cx - cx) * f
                    sh.cy = cy + (sh.cy - cy) * f

            elif sh.type in ('polygon', 'bezier_polygon'):
                if not sh.points: return
                sh.points = [(cx + (p[0] - cx) * f, cy + (p[1] - cy) * f) for p in sh.points]
                
                if sh.type == 'bezier_polygon':
                    # Fix: use 'ctrl' instead of 'control_points' which might not exist in actual core
                    if hasattr(sh, 'ctrl') and sh.ctrl:
                        sh.ctrl = [(cx + (c[0] - cx) * f, cy + (c[1] - cy) * f) if c is not None else None for c in sh.ctrl]
                    # Check for older/alternative attribute names just in case
                    if hasattr(sh, 'control_points') and sh.control_points:
                        sh.control_points = [(cx + (c[0] - cx) * f, cy + (c[1] - cy) * f) if c is not None else None for c in sh.control_points]
                
                # Internal list-based inner_points (PolygonShape style)
                if getattr(sh, 'inner_points', None):
                    sh.inner_points = [(cx + (p[0] - cx) * f, cy + (p[1] - cy) * f) for p in sh.inner_points]
                    if hasattr(sh, 'inner_control_points') and sh.inner_control_points:
                        sh.inner_control_points = [(cx + (c[0] - cx) * f, cy + (c[1] - cy) * f) if c is not None else None for c in sh.inner_control_points]

            # Recurse to object-based inner_shape
            if getattr(sh, 'inner_shape', None):
                do_scale(sh.inner_shape, f, cx, cy)

        do_scale(self.selected_shape, factor, group_cx, group_cy)
        self.update()
            
    def event(self, event):
        from PyQt5.QtCore import QEvent
        if event.type() in (QEvent.KeyPress, QEvent.ShortcutOverride):
            if hasattr(event, 'key') and event.key() == Qt.Key_Tab:
                if hasattr(self, 'parent_window') and self.parent_window:
                    self.parent_window.next_unannotated()
                return True
        return super().event(event)

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        # Get the key and convert to uppercase for comparison
        key = event.key()
        key_text = event.text().upper()
        
        # ===== CTRL+A SELECT ALL (must come before bare 'A' handler) =====
        if key == Qt.Key_A and event.modifiers() == Qt.ControlModifier:
            self.select_all()
            return
        
        # ===== SHAPE TOOL SHORTCUTS =====
        elif key_text == 'B':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.set_shape_type('box')
            print("🔷 Box tool selected")
            return
            
        elif key_text == 'P':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.set_shape_type('polygon')
            print("🔷 Polygon tool selected")
            return
            
        elif key_text == 'C':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.set_shape_type('circle')
            print("🔷 Circle tool selected")
            return
            
        elif key_text == 'E':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.set_shape_type('ellipse')
            print("🔷 Ellipse tool selected")
            return
        
        elif key_text == 'F':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.set_shape_type('frame')
            print("🔷 Frame tool selected")
            return
            
        elif key_text == 'O':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.set_shape_type('donut')
            print("🔷 Donut tool selected")
            return
        
        elif key_text == 'N':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.set_shape_type(None)
            print("🔷 Selection mode activated")
            return
        
        elif key_text == 'H':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.set_shape_type('hollow_ellipse')
            print("🔷 Hollow Ellipse tool selected")
            return
        
        elif key_text == 'T':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.set_shape_type('template')
            print("🔷 Template drawing mode")
            return
        
        elif key_text == 'Q':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.set_shape_type('bezier_polygon')
            print("🔷 Bezier Polygon tool selected")
            return
        
        # ===== NAVIGATION =====
        elif key_text == 'A':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.prev_image()
            return
            
        elif key_text == 'D':
            if hasattr(self, 'parent_window') and self.parent_window:
                self.parent_window.next_image()
            return
        
        # ===== PAN MODE =====
        elif key == Qt.Key_Space:
            self.pan_mode = not self.pan_mode
            if self.pan_mode:
                self.original_cursor = self.cursor()
                self.setCursor(Qt.OpenHandCursor)
                print("🖐️ Pan mode activated")
            else:
                self.setCursor(self.original_cursor or Qt.ArrowCursor)
                print("🖐️ Pan mode deactivated")
            return
        
        # ===== ESCAPE =====
        elif key == Qt.Key_Escape:
            if self.pan_mode:
                self.pan_mode = False
                self.setCursor(self.original_cursor or Qt.ArrowCursor)
            elif self.multi_drag_copy:
                self.cancel_multi_drag_copy()
            elif self.multi_moving:
                self.finish_multi_move()
            elif self.rubber_band_active:
                self.rubber_band_active = False
                self.rubber_band_start = None
                self.rubber_band_end = None
                self.update()
            elif self.moving:
                self.cancel_move()
            elif self.drag_copy:
                self.cancel_drag_copy()
            elif self.drawing:
                self.drawing = False
                self.current_shape = None
                self.update()
            elif self.polygon_points:
                self.cancel_polygon()
            elif self.circle_center:
                self.cancel_circle()
            elif hasattr(self, 'ellipse_center') and self.ellipse_center:
                self.cancel_ellipse()
            elif hasattr(self, 'donut_center') and self.donut_center:
                self.cancel_donut()
            elif hasattr(self, 'hollow_ellipse_center') and self.hollow_ellipse_center:
                self.cancel_hollow_ellipse()
            elif self.bezier_points:
                self.cancel_bezier()
            elif self.selected_shapes:
                self._clear_multi_selection()
                self.update()
            print("❌ Operation cancelled")
            return
        
        # ===== ENTER =====
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            if self.polygon_points:
                self.finish_polygon()
            elif self.bezier_points:
                self.finish_bezier()
            return
        
        # ===== DELETE =====
        elif key == Qt.Key_Delete or key == Qt.Key_Backspace:
            if not self.drag_copy and not self.pan_mode and not self.moving:
                if len(self.selected_shapes) > 1:
                    self.delete_selected_group()
                elif event.modifiers() == Qt.ControlModifier and key == Qt.Key_Delete:
                    self.delete_all()
                else:
                    self.delete_selected()
            return
        
        # ===== UNDO/REDO =====
        elif key == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.undo()
            return
        
        elif key == Qt.Key_Y and event.modifiers() == Qt.ControlModifier:
            self.redo()
            return
        
        # ===== COPY/PASTE =====
        elif key == Qt.Key_C and event.modifiers() == Qt.ControlModifier:
            if len(self.selected_shapes) > 1:
                self.copy_selected_group()
            else:
                self.copy_selected()
            return
        
        elif key == Qt.Key_V and event.modifiers() == Qt.ControlModifier:
            if getattr(self, 'clipboard_shapes', None):
                self.paste_group()
            elif self.clipboard_shape and self.pixmap and not self.pixmap.isNull():
                cursor_pos = self.mapFromGlobal(self.cursor().pos())
                self.start_paste(cursor_pos)
            return
        
        # ===== ARROW KEY NUDGE =====
        elif key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            if self.selected_shape or self.selected_shapes:
                step = self.nudge_step * 10 if event.modifiers() == Qt.ShiftModifier else self.nudge_step
                dx = dy = 0
                if key == Qt.Key_Left:  dx = -step
                if key == Qt.Key_Right: dx =  step
                if key == Qt.Key_Up:    dy = -step
                if key == Qt.Key_Down:  dy =  step
                self.nudge_selected(dx, dy)
                return
        
        super().keyPressEvent(event)

    def reset_all_states(self):
        """Reset all drawing and interaction states"""
        self.drawing = False
        self.drag_copy = False
        self.resizing = False
        self.resizing_handle = None
        self.drag_copy_shape = None
        self.current_shape = None
        self.start_point = None
        self.pasting = False
        self.polygon_points = []
        self.circle_center = None
        self.circle_radius = 0
        self.ellipse_center = None
        self.ellipse_radius_x = 0
        self.ellipse_radius_y = 0
        self.donut_center = None
        self.donut_radius = 0
        self.hollow_ellipse_center = None
        self.hollow_ellipse_rx = 0
        self.hollow_ellipse_ry = 0
        self.stamping = False
        self.stamp_center = None
        self.stamp_current_pos = None
        self.drawing_polygon = False
        
        # Ring states
        self.drawing_ring = False
        self.ring_stage = 'outer'
        self.ring_outer_points = []
        self.ring_inner_points = []
        self.ring_outer_center = None
        self.ring_outer_radius = 0
        
        # Multi-select interaction states (do NOT clear selected_shapes — tool switch should not deselect)
        self.rubber_band_active = False
        self.rubber_band_start = None
        self.rubber_band_end = None
        self.multi_moving = False
        self.multi_move_start = None
        self.multi_drag_copy = False
        self.multi_drag_copies = []
        self.multi_drag_originals = []
        
        print("🔄 All states reset")       
        
    def widget_to_image(self, pos):
        """Convert widget coordinates to image coordinates"""
        if not self.pixmap or self.pixmap.isNull():
            return 0, 0
            
        # Calculate image position
        image_x = (pos.x() - self.offset_x) / self.scale
        image_y = (pos.y() - self.offset_y) / self.scale
        
        # Clamp to image boundaries
        image_x = max(0, min(self.image_width - 1, int(image_x)))
        image_y = max(0, min(self.image_height - 1, int(image_y)))
        
        return image_x, image_y
        
    def resizeEvent(self, event):
        """Handle resize events"""
        if self.pixmap and not self.pixmap.isNull():
            self.fit_to_window()
        super().resizeEvent(event)

    def copy_selected(self):
        """Copy the selected shape to clipboard"""
        if self.selected_shape and hasattr(self.selected_shape, 'copy'):
            self.clipboard_shape = self.selected_shape.copy()
            shape_type = getattr(self.selected_shape, 'type', 'box')
            print(f"📋 Copied {shape_type}")
            return True
        else:
            print("⚠️ No shape selected to copy or copy not supported")
            return False    
    
    def start_paste(self, pos):
        """Start pasting a shape at cursor position"""
        if not self.clipboard_shape:
            print("⚠️ No shape in clipboard to paste")
            return False
        
        print(f"📋 Pasting at cursor position: ({pos.x()}, {pos.y()})")
        
        # Create a copy of the clipboard shape
        self.paste_shape = self.clipboard_shape.copy()
        self.pasting = True
        self.paste_confirmed = False
        
        # Position the shape exactly at cursor position
        image_x, image_y = self.widget_to_image(pos)
        
        # Update position based on shape type
        if hasattr(self.paste_shape, 'x') and hasattr(self.paste_shape, 'y'):
            # For boxes and frames
            self.paste_shape.x = image_x / self.image_width
            self.paste_shape.y = image_y / self.image_height
        elif hasattr(self.paste_shape, 'center_x') and hasattr(self.paste_shape, 'center_y'):
            # For circles, ellipses, donuts
            self.paste_shape.center_x = image_x / self.image_width
            self.paste_shape.center_y = image_y / self.image_height
        elif hasattr(self.paste_shape, 'cx') and hasattr(self.paste_shape, 'cy'):
            # For donuts with cx/cy
            self.paste_shape.cx = image_x / self.image_width
            self.paste_shape.cy = image_y / self.image_height
        elif hasattr(self.paste_shape, 'points'):
            # For polygons - move all points
            if self.paste_shape.points:
                # Calculate center offset
                points = self.paste_shape.points
                center_x = sum(p[0] for p in points) / len(points)
                center_y = sum(p[1] for p in points) / len(points)
                dx = (image_x / self.image_width) - center_x
                dy = (image_y / self.image_height) - center_y
                self.paste_shape.move(dx, dy)

        # Deselect any selected shape
        if self.selected_shape:
            self.selected_shape.selected = False
            self.selected_shape = None

        # Select the paste shape
        self.paste_shape.selected = True
        self.selected_shape = self.paste_shape

        # Add to shapes list
        self.shapes.append(self.paste_shape)
        
        shape_type = getattr(self.paste_shape, 'type', 'box')
        print(f"📋 Pasting {shape_type} - drag to position, Enter to confirm, Esc to cancel")
        self.update()
        return True

    def update_paste_position(self, pos):
        """Update the position of the box being pasted (without resizing)"""
        if self.pasting and self.paste_box and not self.resizing_handle:
            image_x, image_y = self.widget_to_image(pos)
            self.paste_box.x = image_x / self.image_width
            self.paste_box.y = image_y / self.image_height
            self.update()

    def start_resize_paste(self, pos, handle):
        """Start resizing the pasted box"""
        if self.pasting and self.paste_box:
            self.resizing_handle = handle
            self.paste_start_pos = self.widget_to_image(pos)
            print(f"📏 Resizing from {handle} handle")  

    def update_paste_resize(self, pos):
        """Update the size while dragging a handle"""
        if self.pasting and self.paste_box and self.resizing_handle and self.paste_start_pos:
            current_pos = self.widget_to_image(pos)
            
            # Calculate delta
            dx = current_pos[0] - self.paste_start_pos[0]
            dy = current_pos[1] - self.paste_start_pos[1]
            
            # Apply resize
            success = self.paste_box.resize_from_handle(self.resizing_handle, dx, dy)
            
            if success:
                self.paste_start_pos = current_pos
                self.update()              

    def confirm_paste(self):
        """Confirm the paste and add the shape to the list"""
        if self.pasting and self.paste_shape:
            print("✅ Paste confirmed")
            self.pasting = False
            self.resizing_handle = None
            self.paste_start_pos = None
            self.paste_confirmed = True
            self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
            self.update()
            return True
        return False   

    def cancel_paste(self):
        """Cancel the paste operation"""
        if self.pasting:
            print("❌ Paste cancelled")
            if self.paste_shape in self.shapes:
                self.shapes.remove(self.paste_shape)
            self.pasting = False
            self.paste_shape = None
            self.resizing_handle = None
            self.paste_start_pos = None
            self.selected_shape = None
            self.update()
            return True
        return False         
    
    def get_resize_handle_at_pos(self, pos, shape):
        """Check if position is over a resize handle of a shape"""
        if not shape or not shape.selected:
            return None
        
        # Convert screen coordinates to image pixel coordinates
        image_x = (pos.x() - self.offset_x) / self.scale
        image_y = (pos.y() - self.offset_y) / self.scale
        
        # For frame and donut shapes, use their specific handle detection
        if hasattr(shape, 'get_handle_at_pos'):
            return shape.get_handle_at_pos(image_x, image_y, self.handle_size)
        
        # For other shapes, use the existing handle detection
        if hasattr(shape, 'get_resize_handles'):
            handles = shape.get_resize_handles()
            
            # Convert handle positions to widget coordinates
            half = self.handle_size // 2
            px, py = pos.x(), pos.y()
            
            for handle_name, (hx, hy) in handles.items():
                wx = int(hx * self.scale + self.offset_x)
                wy = int(hy * self.scale + self.offset_y)
                
                if (wx - half <= px <= wx + half) and (wy - half <= py <= wy + half):
                    return handle_name
        
        return None
    
    def start_drag_copy(self, shape, pos):
        """Start dragging a copy of the selected shape"""
        if not shape or not hasattr(shape, 'copy'):
            return False
        
        print(f"📋 Starting drag copy")
        
        # Create a copy
        self.drag_copy_shape = shape.copy()
        self.drag_copy = True
        self.original_shape = shape
        self.drag_start_pos = self.widget_to_image(pos)
        
        # Store original shape selection state
        shape.selected = False
        
        # Add the copy to shapes list immediately
        self.shapes.append(self.drag_copy_shape)
        self.drag_copy_shape.selected = True
        self.selected_shape = self.drag_copy_shape
        
        self.update()
        return True

    def update_drag_copy(self, pos):
        """Update position of dragged copy using delta-based move() so inner shapes move together."""
        if self.drag_copy and self.drag_copy_shape and self.drag_start_pos:
            image_x, image_y = self.widget_to_image(pos)
            dx = (image_x - self.drag_start_pos[0]) / self.image_width
            dy = (image_y - self.drag_start_pos[1]) / self.image_height
            self.drag_copy_shape.move(dx, dy)
            self.drag_start_pos = (image_x, image_y)
            self.update()

    def finish_drag_copy(self):
        """Finish dragging copy"""
        if self.drag_copy and self.drag_copy_shape:
            self.save_state()  # Save state for the new copy
            print("✅ Drag copy completed")
            self.drag_copy = False
            self.drag_copy_shape = None
            self.drag_start_pos = None
            self.original_shape = None
            self.resizing = False
            self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
            self.update()
            return True
        return False

    def cancel_drag_copy(self):
        """Cancel drag copy"""
        if self.drag_copy and self.drag_copy_shape:
            print("❌ Drag copy cancelled")
            
            # Remove the temporary drag copy shape from shapes list
            if self.drag_copy_shape in self.shapes:
                self.shapes.remove(self.drag_copy_shape)
            
            # Restore original shape selection
            if self.original_shape:
                self.original_shape.selected = True
                self.selected_shape = self.original_shape
            else:
                self.selected_shape = None
            
            # Clear all drag-copy related states
            self.drag_copy = False
            self.drag_copy_shape = None
            self.drag_start_pos = None
            self.original_shape = None
            
            # Reset all interaction states
            self.resizing = False
            self.resizing_handle = None
            self.pasting = False
            
            print("✅ Ready to draw new shapes")
            self.update()
            return True
        return False
    
    def debug_state(self):
        """Print current state for debugging"""
        print(f"🔍 Canvas State - Drawing: {self.drawing}, DragCopy: {self.drag_copy}, "
            f"Resizing: {self.resizing}, Pasting: {self.pasting}, "
            f"Selected Shape: {self.selected_shape is not None}")
        
    def force_reset_for_drawing(self):
        """Force reset all states to allow drawing"""
        print("🔄 Force resetting all states for drawing")
        self.drawing = False
        self.drag_copy = False
        self.drag_copy_shape = None
        self.resizing = False
        self.resizing_handle = None
        self.pasting = False
        self.paste_shape = None
        # Don't clear selected_shape here - let that be handled by click on empty area 

    def set_shape_type(self, shape_type):
        """Set the current shape type"""
        self.current_shape_type = shape_type
        self.reset_all_states()
        if shape_type and shape_type != 'none':
            print(f"🔷 Shape type set to: {shape_type}")
        else:
            print("⬜ No drawing tool selected - click on shapes to select them")
        
        # Cancel any ongoing drawing
        if self.polygon_points:
            self.cancel_polygon()
        if self.circle_center:
            self.cancel_circle()
        if hasattr(self, 'ellipse_center') and self.ellipse_center:
            self.cancel_ellipse()
        if hasattr(self, 'donut_center') and self.donut_center:
            self.cancel_donut()
        if hasattr(self, 'hollow_ellipse_center') and self.hollow_ellipse_center:
            self.cancel_hollow_ellipse()
        if self.drawing:
            self.drawing = False
            self.current_shape = None
    
    def start_polygon_drawing(self, pos):
        """Start or continue polygon drawing"""
        if self.class_manager and self.class_manager.get_current_class():
            image_x, image_y = self.widget_to_image(pos)
            
            # If this is the first point, create a new polygon
            if not self.polygon_points:
                self.polygon_points = [(image_x, image_y)]
                print(f"🔷 Started polygon at ({image_x}, {image_y})")
            else:
                # Check if clicking near the first point to close
                first_x, first_y = self.polygon_points[0]
                distance = math.sqrt((image_x - first_x)**2 + (image_y - first_y)**2)
                
                if distance < 10 and len(self.polygon_points) >= 3:
                    self.finish_polygon()
                else:
                    self.polygon_points.append((image_x, image_y))
                    print(f"🔷 Added polygon point ({image_x}, {image_y})")
            
            self.update()
            
    def finish_polygon(self):
        """Finish drawing polygon"""
        if len(self.polygon_points) >= 3:
            # Check if we're in template drawing mode
            if self.current_shape_type == 'template':
                self.finish_template()
            elif self.drawing_inner_cutout and self.cutout_target_shape:
                # OUTER DRAWING MODE: new polygon = outer, original = inner
                self.save_state()
                target = self.cutout_target_shape
                
                # Save original points as inner (the hole)
                original_points = list(target.points)
                
                # Set new outer shape from the just-drawn polygon
                target.from_pixel_points(self.polygon_points)
                target.closed = True
                
                # Set original as inner cutout
                target.inner_points = original_points
                
                print(f"✅ Hollow polygon created: outer={len(self.polygon_points)} pts, inner={len(original_points)} pts")
                self.drawing_inner_cutout = False
                self.cutout_target_shape = None
                self.polygon_points = []
                self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
                self.update()
            else:
                # Create polygon shape
                polygon = PolygonShape(
                    class_id=self.class_manager.get_current_class().id,
                    image_size=(self.image_width, self.image_height)
                )
                polygon.from_pixel_points(self.polygon_points)
                polygon.close_polygon()
                self.save_state()  # Save state before adding
                self.shapes.append(polygon)
                self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
                print(f"✅ Polygon completed with {len(self.polygon_points)} points")
        
        # Reset polygon drawing state (unless template dialog is open)
        if self.current_shape_type != 'template':
            self.polygon_points = []
            self.update()
    
    def finish_template(self):
        """Save current polygon points as a named template"""
        from PyQt5.QtWidgets import QInputDialog
        
        if len(self.polygon_points) < 3:
            print("❌ Need at least 3 points for a template")
            self.polygon_points = []
            self.update()
            return
        
        name, ok = QInputDialog.getText(
            self, "Save Template",
            "Enter template name:",
        )
        
        if ok and name and name.strip():
            name = name.strip()
            success = self.template_manager.add_template(
                name, self.polygon_points,
                self.image_width, self.image_height
            )
            if success:
                print(f"✅ Template '{name}' saved with {len(self.polygon_points)} points")
                # Notify parent to update dropdown
                if hasattr(self, 'parent_window') and self.parent_window:
                    self.parent_window.update_template_dropdown()
            else:
                print("❌ Failed to save template (too small?)")
        else:
            print("❌ Template save cancelled")
        
        self.polygon_points = []
        self.update()
    
    def start_stamp(self, pos):
        """Start placing a stamp template"""
        if not self.stamp_template_name:
            print("❌ No template selected")
            return
        
        template = self.template_manager.get_template(self.stamp_template_name)
        if not template:
            print(f"❌ Template '{self.stamp_template_name}' not found")
            return
        
        self.stamping = True
        self.stamp_center = self.widget_to_image(pos)
        self.stamp_current_pos = self.stamp_center
        print(f"🔷 Placing template '{self.stamp_template_name}' - drag to scale")
    
    def finish_stamp(self):
        """Finish placing stamp — creates a PolygonShape"""
        if not self.stamping or not getattr(self, 'stamp_center', None) or not getattr(self, 'stamp_current_pos', None):
            self.stamping = False
            return
        
        # Calculate scale from drag distance
        dx = abs(self.stamp_current_pos[0] - self.stamp_center[0])
        dy = abs(self.stamp_current_pos[1] - self.stamp_center[1])
        
        # If user just clicks without dragging, use a decent default scale
        info = self.template_manager.get_template_info(self.stamp_template_name)
        orig_half_w = info['orig_w'] / 2.0
        orig_half_h = info['orig_h'] / 2.0
        if orig_half_w == 0: orig_half_w = 50.0
        if orig_half_h == 0: orig_half_h = 50.0

        if dx < 5 and dy < 5:
            scale_x = orig_half_w
            scale_y = orig_half_h
        else:
            scale_x = max(dx, 10)  # minimum 10px
            scale_y = max(dy, 10)
            
        ratio_x = scale_x / orig_half_w
        ratio_y = scale_y / orig_half_h
        
        # Get pixel points from template
        shape_type, outer_points, inner_points, ctrl_points, inner_ctrl_points, native_params = self.template_manager.get_pixel_points(
            self.stamp_template_name,
            self.stamp_center[0], self.stamp_center[1],
            scale_x, scale_y
        )
        
        current_class = self.class_manager.get_current_class()
        if not current_class:
            self.stamping = False
            self.stamp_center = None
            self.stamp_current_pos = None
            self.update()
            return

        shape = None
        cw, ch = self.image_width, self.image_height

        if shape_type == 'box':
            from core.annotation import BoundingBox
            shape = BoundingBox(class_id=current_class.id, image_size=(cw, ch))
            cx, cy = self.stamp_center
            # Correct scaling relative to the template's bounding box
            w = native_params.get('w', info['orig_w']) * ratio_x
            h = native_params.get('h', info['orig_h']) * ratio_y
            shape.from_pixels(cx - w/2, cy - h/2, cx + w/2, cy + h/2, cw, ch)
            
        elif shape_type == 'circle':
            from core.circle_shape import CircleShape
            shape = CircleShape(class_id=current_class.id, image_size=(cw, ch))
            # Circles scale by x drag by default
            r = native_params.get('r', info['orig_w']/2) * ratio_x
            shape.from_pixels(self.stamp_center[0], self.stamp_center[1], r)
            
        elif shape_type == 'ellipse':
            from core.ellipse_shape import EllipseShape
            shape = EllipseShape(class_id=current_class.id, image_size=(cw, ch))
            rx = native_params.get('rx', info['orig_w']/2) * ratio_x
            ry = native_params.get('ry', info['orig_h']/2) * ratio_y
            shape.from_pixels(self.stamp_center[0], self.stamp_center[1], rx, ry)
            
        elif shape_type == 'frame':
            from core.ring_shape import FrameShape
            shape = FrameShape(class_id=current_class.id, image_size=(cw, ch))
            cx, cy = self.stamp_center
            
            orig_w = native_params.get('w', info['orig_w'])
            orig_h = native_params.get('h', info['orig_h'])
            target_w = orig_w * ratio_x
            target_h = orig_h * ratio_y
            
            shape.from_pixels(cx - target_w/2, cy - target_h/2, cx + target_w/2, cy + target_h/2)
            
            # Apply proportionally scaled thicknesses
            shape.t_top = native_params.get('t_top', 20.0) * ratio_y
            shape.t_bottom = native_params.get('t_bottom', 20.0) * ratio_y
            shape.t_left = native_params.get('t_left', 20.0) * ratio_x
            shape.t_right = native_params.get('t_right', 20.0) * ratio_x
            
        elif shape_type == 'donut':
            from core.ring_shape import DonutShape
            shape = DonutShape(class_id=current_class.id, image_size=(cw, ch))
            outer_r = native_params.get('outer_r', info['orig_w']/2) * ratio_x
            inner_r_px = native_params.get('inner_r', info['orig_w']/4) * ratio_x
            shape.from_pixels(self.stamp_center[0], self.stamp_center[1], outer_r)
            # donut.inner_r is normalized by min(W,H) usually, but from_pixels handles some setups.
            # Let's manually set the normalized inner_r if needed, or assume from_pixels handles it.
            # DonutShape.from_pixels only sets outer_radius.
            d = min(cw, ch)
            shape.inner_r = inner_r_px / d
            
        elif shape_type == 'hollow_ellipse':
            from core.ring_shape import HollowEllipseShape
            shape = HollowEllipseShape(class_id=current_class.id, image_size=(cw, ch))
            outer_rx = native_params.get('outer_rx', info['orig_w']/2) * ratio_x
            outer_ry = native_params.get('outer_ry', info['orig_h']/2) * ratio_y
            inner_rx_px = native_params.get('inner_rx', info['orig_w']/4) * ratio_x
            inner_ry_px = native_params.get('inner_ry', info['orig_h']/4) * ratio_y
            shape.from_pixels(self.stamp_center[0], self.stamp_center[1], outer_rx, outer_ry)
            shape.inner_rx = inner_rx_px / cw
            shape.inner_ry = inner_ry_px / ch
            
        else:
            # Fallback to polygon logic (including bezier_polygon)
            if outer_points and len(outer_points) >= 3:
                if shape_type == 'bezier_polygon':
                    from core.bezier_shape import BezierPolygonShape
                    shape = BezierPolygonShape(class_id=current_class.id, image_size=(cw, ch))
                else:
                    from core.polygon_shape import PolygonShape
                    shape = PolygonShape(class_id=current_class.id, image_size=(cw, ch))
                    
                shape.from_pixel_points(outer_points)
                
                if inner_points and len(inner_points) >= 3:
                    normalized_inner = []
                    for px, py in inner_points:
                        normalized_inner.append((px / cw, py / ch))
                    shape.inner_points = normalized_inner
                    
                if shape_type == 'bezier_polygon':
                    shape.ctrl = []
                    for c in ctrl_points:
                        shape.ctrl.append((c[0] / cw, c[1] / ch) if c is not None else None)
                            
                    if inner_ctrl_points:
                        shape.inner_control_points = []
                        for c in inner_ctrl_points:
                            shape.inner_control_points.append((c[0] / cw, c[1] / ch) if c is not None else None)
                    
                shape.close_polygon()
                
        if shape is not None:
            # Reconstruct inner shape for hallowed pairs (Box, Circle, Ellipse, Bezier)
            inner_type = native_params.get('inner_type')
            if inner_points and inner_type:
                inner_shape = None
                if inner_type == 'box':
                    from core.annotation import BoundingBox
                    ix1 = min(p[0] for p in inner_points); iy1 = min(p[1] for p in inner_points)
                    ix2 = max(p[0] for p in inner_points); iy2 = max(p[1] for p in inner_points)
                    inner_shape = BoundingBox(class_id=current_class.id, image_size=(cw, ch))
                    inner_shape.from_pixels(ix1, iy1, ix2, iy2, cw, ch)
                elif inner_type == 'circle':
                    from core.circle_shape import CircleShape
                    icx = sum(p[0] for p in inner_points) / len(inner_points)
                    icy = sum(p[1] for p in inner_points) / len(inner_points)
                    ir = math.hypot(inner_points[0][0] - icx, inner_points[0][1] - icy)
                    inner_shape = CircleShape(class_id=current_class.id, image_size=(cw, ch))
                    inner_shape.from_pixels(icx, icy, ir)
                elif inner_type == 'ellipse':
                    from core.ellipse_shape import EllipseShape
                    icx = sum(p[0] for p in inner_points) / len(inner_points)
                    icy = sum(p[1] for p in inner_points) / len(inner_points)
                    irx = max(abs(p[0] - icx) for p in inner_points)
                    iry = max(abs(p[1] - icy) for p in inner_points)
                    inner_shape = EllipseShape(class_id=current_class.id, image_size=(cw, ch))
                    inner_shape.from_pixels(icx, icy, irx, iry)
                elif inner_type == 'bezier_polygon':
                    from core.bezier_shape import BezierPolygonShape
                    inner_shape = BezierPolygonShape(class_id=current_class.id, image_size=(cw, ch))
                    inner_shape.from_pixel_points(inner_points)
                    if inner_ctrl_points:
                        inner_shape.ctrl = [(c[0]/cw, c[1]/ch) if c is not None else None for c in inner_ctrl_points]
                    inner_shape.close_polygon()
                elif inner_type == 'polygon':
                    from core.polygon_shape import PolygonShape
                    inner_shape = PolygonShape(class_id=current_class.id, image_size=(cw, ch))
                    inner_shape.from_pixel_points(inner_points)
                    inner_shape.close_polygon()
                
                if inner_shape:
                    shape.attach_inner(inner_shape)

            self.save_state()
            self.shapes.append(shape)
            self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
            print(f"✅ Stamp placed as natively constructed '{shape_type}' (Hollow: {True if inner_type else False})")
            
            self.stamping = False
            self.stamp_center = None
            self.stamp_current_pos = None
            self.update()
    
    
    def cancel_polygon(self):
        """Cancel polygon drawing"""
        self.polygon_points = []
        self.drawing_inner_cutout = False
        self.cutout_target_shape = None
        print("❌ Polygon cancelled")
        self.update()

    # ==========================================
    #  Bezier Polygon Drawing
    # ==========================================

    def start_bezier_drawing(self, pos):
        """Add a point to the bezier polygon being drawn"""
        if not self.class_manager or not self.class_manager.get_current_class():
            print("⚠️ Cannot draw - no class selected")
            return
            
        img_pos = self.widget_to_image(pos)
        
        # If this is the first point
        if not self.drawing_bezier:
            self.drawing_bezier = True
            self.bezier_points = [img_pos]
            self.undone_bezier_points = [] # Clear redo stack
            print(f"🔷 Started bezier polygon at ({img_pos[0]}, {img_pos[1]})")
        else:
            # Check if closing the polygon (near first point)
            if len(self.bezier_points) >= 3:
                first_pos = self.bezier_points[0]
                dist = math.hypot(img_pos[0] - first_pos[0], img_pos[1] - first_pos[1])
                # If closer than 10 pixels (scaled)
                if dist < 10.0 / self.scale:
                    self.finish_bezier()
                    return
            
            self.bezier_points.append(img_pos)
            self.undone_bezier_points = [] # Clear redo stack on new point
            print(f"🔷 Added bezier point ({img_pos[0]}, {img_pos[1]})")
        
        self.update()

    def update_bezier_drawing(self, pos):
        """Update cursor tracking for bezier polygon"""
        pass  # Just need a method signature to match patterns, current_mouse_pos handles the preview drawing

    def finish_bezier(self):
        """Finish drawing bezier polygon"""
        if len(self.bezier_points) >= 3:
            if self.drawing_inner_cutout and self.cutout_target_shape:
                # OUTER DRAWING MODE: new bezier = outer, original = inner
                self.save_state()
                target = self.cutout_target_shape
                
                # Save original points as inner (the hole)
                original_points = list(target.points)
                # For bezier shapes, we also need to preserve control points if we want them editable later,
                # but currently hollow shapes just draw the inner_points as straight lines. Let's just store the points.
                original_ctrl_points = getattr(target, 'control_points', None)
                if original_ctrl_points:
                    target.inner_control_points = list(original_ctrl_points)
                
                # Set new outer shape from the just-drawn bezier
                target.from_pixel_points(self.bezier_points)
                target.close_polygon()
                
                # Set original as inner cutout
                target.inner_points = original_points
                
                print(f"✅ Hollow bezier created: outer={len(self.bezier_points)} pts, inner={len(original_points)} pts")
                self.drawing_inner_cutout = False
                self.cutout_target_shape = None
                self.bezier_points = []
                if hasattr(self, 'undone_bezier_points'):
                    self.undone_bezier_points = []
                self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
                self.update()
            else:
                current_class = self.class_manager.get_current_class()
                if current_class:
                    bezier = BezierPolygonShape(
                        class_id=current_class.id,
                        image_size=(self.image_width, self.image_height)
                    )
                    bezier.from_pixel_points(self.bezier_points)
                    bezier.close_polygon()
                    self.save_state()
                    self.shapes.append(bezier)
                    self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
                    print(f"✅ Bezier polygon completed with {len(self.bezier_points)} points")
                
        self.drawing_bezier = False
        self.bezier_points = []
        if hasattr(self, 'undone_bezier_points'):
            self.undone_bezier_points = []
        self.update()

    def cancel_bezier(self):
        """Cancel bezier drawing"""
        self.drawing_bezier = False
        self.bezier_points = []
        if hasattr(self, 'undone_bezier_points'):
            self.undone_bezier_points = []
        print("❌ Bezier cancelled")
        self.update()
        
    def start_circle_drawing(self, pos):
        """Start drawing a circle"""
        if self.class_manager and self.class_manager.get_current_class():
            self.circle_center = self.widget_to_image(pos)
            self.circle_radius = 0
            print(f"⭕ Started circle at center ({self.circle_center[0]}, {self.circle_center[1]})")
            
    def update_circle_drawing(self, pos):
        """Update circle while drawing"""
        if self.circle_center:
            current_pos = self.widget_to_image(pos)
            dx = current_pos[0] - self.circle_center[0]
            dy = current_pos[1] - self.circle_center[1]
            self.circle_radius = int(math.sqrt(dx*dx + dy*dy))
            self.update()
            
    def finish_circle(self):
        """Finish drawing circle"""
        if self.circle_center and self.circle_radius > 5:
            # Create circle shape
            circle = CircleShape(
                class_id=self.class_manager.get_current_class().id,
                image_size=(self.image_width, self.image_height)
            )
            circle.from_pixels(
                self.circle_center[0],
                self.circle_center[1],
                self.circle_radius
            )
            self.save_state()  # Save state before adding
            self.shapes.append(circle)
            self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
            print(f"✅ Circle completed with radius {self.circle_radius}")
        
        # Reset circle drawing state
        self.circle_center = None
        self.circle_radius = 0
        self.update()
        
    def cancel_circle(self):
        """Cancel circle drawing"""
        self.circle_center = None
        self.circle_radius = 0
        print("❌ Circle cancelled")
        self.update()
        
    def start_ellipse_drawing(self, pos):
        """Start drawing an ellipse"""
        if self.class_manager and self.class_manager.get_current_class():
            self.ellipse_center = self.widget_to_image(pos)
            self.ellipse_radius_x = 0
            self.ellipse_radius_y = 0
            print(f"🟢 Started ellipse at center ({self.ellipse_center[0]}, {self.ellipse_center[1]})")
            
    def update_ellipse_drawing(self, pos):
        """Update ellipse while drawing"""
        if hasattr(self, 'ellipse_center') and self.ellipse_center:
            current_pos = self.widget_to_image(pos)
            dx = current_pos[0] - self.ellipse_center[0]
            dy = current_pos[1] - self.ellipse_center[1]
            self.ellipse_radius_x = abs(dx)
            self.ellipse_radius_y = abs(dy)
            self.update()
            
    def finish_ellipse(self):
        """Finish drawing ellipse"""
        if hasattr(self, 'ellipse_center') and self.ellipse_center and self.ellipse_radius_x > 5 and self.ellipse_radius_y > 5:
            # Create ellipse shape
            ellipse = EllipseShape(
                class_id=self.class_manager.get_current_class().id,
                image_size=(self.image_width, self.image_height)
            )
            ellipse.from_pixels(
                self.ellipse_center[0],
                self.ellipse_center[1],
                self.ellipse_radius_x,
                self.ellipse_radius_y
            )
            self.save_state()  # Save state before adding
            self.shapes.append(ellipse)
            self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
            print(f"✅ Ellipse completed with radii ({self.ellipse_radius_x}, {self.ellipse_radius_y})")
        
        # Reset ellipse drawing state
        self.ellipse_center = None
        self.ellipse_radius_x = 0
        self.ellipse_radius_y = 0
        self.update()
        
    def cancel_ellipse(self):
        """Cancel ellipse drawing"""
        self.ellipse_center = None
        self.ellipse_radius_x = 0
        self.ellipse_radius_y = 0
        print("❌ Ellipse cancelled")
        self.update()
        
    def start_frame_drawing(self, pos):
        """Start drawing a frame (hollow rectangle)"""
        if self.class_manager and self.class_manager.get_current_class():
            self.drawing = True
            self.start_point = self.widget_to_image(pos)
            self.current_shape = FrameShape(
                class_id=self.class_manager.get_current_class().id,
                image_size=(self.image_width, self.image_height)
            )
            print("🔷 Started drawing frame")
    
    def start_donut_drawing(self, pos):
        """Start drawing a donut (hollow circle)"""
        if self.class_manager and self.class_manager.get_current_class():
            # Reset any existing donut state
            self.donut_center = self.widget_to_image(pos)
            self.donut_radius = 0
            # NOTE: Do NOT set self.drawing = True here.
            # Donut uses self.donut_center as its state indicator,
            # just like circle uses self.circle_center and ellipse uses self.ellipse_center.
            # Setting self.drawing would cause mouseMoveEvent/mouseReleaseEvent to
            # route to generic update_drawing()/finish_drawing() instead of donut-specific handlers.
            self.current_shape = None
            print("🔷 Started drawing donut - drag to set size")
    
    def update_donut_drawing(self, pos):
        """Update donut while drawing - only preview, no shape created yet"""
        if hasattr(self, 'donut_center') and self.donut_center:
            current_pos = self.widget_to_image(pos)
            dx = current_pos[0] - self.donut_center[0]
            dy = current_pos[1] - self.donut_center[1]
            self.donut_radius = int(math.sqrt(dx*dx + dy*dy))
            self.update()  # This triggers paintEvent which shows preview
    
    def finish_donut(self):
        """Finish drawing donut - create the actual shape"""
        if hasattr(self, 'donut_center') and self.donut_center and self.donut_radius > 10:
            current_class = self.class_manager.get_current_class()
            if current_class:
                donut = DonutShape(
                    class_id=current_class.id,
                    image_size=(self.image_width, self.image_height)
                )
                donut.from_pixels(
                    self.donut_center[0],
                    self.donut_center[1],
                    self.donut_radius
                )
                self.shapes.append(donut)
                self.save_state()
                self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
                print("✅ Donut created")
        
        # Reset all donut drawing states
        self.donut_center = None
        self.donut_radius = 0
        self.update()
    
    def cancel_donut(self):
        """Cancel donut drawing"""
        self.donut_center = None
        self.donut_radius = 0
        print("❌ Donut cancelled")
        self.update()

    def start_hollow_ellipse_drawing(self, pos):
        """Start drawing a hollow ellipse"""
        if self.class_manager and self.class_manager.get_current_class():
            self.hollow_ellipse_center = self.widget_to_image(pos)
            self.hollow_ellipse_rx = 0
            self.hollow_ellipse_ry = 0
            self.current_shape = None
            print("🔷 Started drawing hollow ellipse - drag to set size")
    
    def update_hollow_ellipse_drawing(self, pos):
        """Update hollow ellipse while drawing"""
        if hasattr(self, 'hollow_ellipse_center') and self.hollow_ellipse_center:
            current_pos = self.widget_to_image(pos)
            dx = current_pos[0] - self.hollow_ellipse_center[0]
            dy = current_pos[1] - self.hollow_ellipse_center[1]
            self.hollow_ellipse_rx = abs(dx)
            self.hollow_ellipse_ry = abs(dy)
            self.update()
    
    def finish_hollow_ellipse(self):
        """Finish drawing hollow ellipse"""
        if (hasattr(self, 'hollow_ellipse_center') and self.hollow_ellipse_center 
                and self.hollow_ellipse_rx > 10 and self.hollow_ellipse_ry > 10):
            current_class = self.class_manager.get_current_class()
            if current_class:
                shape = HollowEllipseShape(
                    class_id=current_class.id,
                    image_size=(self.image_width, self.image_height)
                )
                shape.from_pixels(
                    self.hollow_ellipse_center[0],
                    self.hollow_ellipse_center[1],
                    self.hollow_ellipse_rx,
                    self.hollow_ellipse_ry
                )
                self.shapes.append(shape)
                self.save_state()
                self.annotation_changed.emit()  # AUTOSAVE INTEGRATION
                print("✅ Hollow ellipse created")
        
        self.hollow_ellipse_center = None
        self.hollow_ellipse_rx = 0
        self.hollow_ellipse_ry = 0
        self.update()
    
    def cancel_hollow_ellipse(self):
        """Cancel hollow ellipse drawing"""
        self.hollow_ellipse_center = None
        self.hollow_ellipse_rx = 0
        self.hollow_ellipse_ry = 0
        print("❌ Hollow ellipse cancelled")
        self.update()

    def start_move(self, shape, pos):
        """Start moving a selected shape"""
        if not shape or not shape.selected:
            return False
        
        print(f"↔️ Starting move operation")
        self.moving = True
        self.selected_shape = shape
        self.move_start_pos = self.widget_to_image(pos)
        
        # Store original position for undo (based on shape type)
        if hasattr(shape, 'x') and hasattr(shape, 'y'):  # Box, Frame
            self.move_original_positions = [(shape.x, shape.y)]
        elif hasattr(shape, 'center_x') and hasattr(shape, 'center_y'):  # Circle, Ellipse
            self.move_original_positions = [(shape.center_x, shape.center_y)]
        elif hasattr(shape, 'cx') and hasattr(shape, 'cy'):  # Donut
            self.move_original_positions = [(shape.cx, shape.cy)]
        elif hasattr(shape, 'points'):  # Polygon
            self.move_original_positions = shape.points.copy()
        
        self.setCursor(Qt.ClosedHandCursor)
        return True

    def update_move(self, pos):
        """Update shape position while moving"""
        if not self.moving or not self.selected_shape:
            return

        current_pos = self.widget_to_image(pos)
        dx = (current_pos[0] - self.move_start_pos[0]) / self.image_width
        dy = (current_pos[1] - self.move_start_pos[1]) / self.image_height

        # Use the shape's own move() method so inner+outer always move as one unit
        if hasattr(self.selected_shape, 'move'):
            self.selected_shape.move(dx, dy)
        else:
            # Fallback for any shape without a move() method
            if hasattr(self.selected_shape, 'x') and hasattr(self.selected_shape, 'y'):
                self.selected_shape.x += dx
                self.selected_shape.y += dy
            elif hasattr(self.selected_shape, 'cx') and hasattr(self.selected_shape, 'cy'):
                self.selected_shape.cx += dx
                self.selected_shape.cy += dy

        self.move_start_pos = current_pos
        self.update()

    def finish_move(self):
        """Finish moving shape"""
        if self.moving and self.selected_shape:
            self.save_state()  # Save state for undo
            print("✅ Move completed")
        
        self.moving = False
        self.move_start_pos = None
        self.move_original_positions = []
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def cancel_move(self):
        """Cancel move operation and restore original position"""
        if self.moving and self.selected_shape and self.move_original_positions:
            # Restore original position based on shape type
            if hasattr(self.selected_shape, 'x') and hasattr(self.selected_shape, 'y'):  # Box, Frame
                self.selected_shape.x, self.selected_shape.y = self.move_original_positions[0]
            elif hasattr(self.selected_shape, 'center_x') and hasattr(self.selected_shape, 'center_y'):  # Circle, Ellipse
                self.selected_shape.center_x, self.selected_shape.center_y = self.move_original_positions[0]
            elif hasattr(self.selected_shape, 'cx') and hasattr(self.selected_shape, 'cy'):  # Donut
                self.selected_shape.cx, self.selected_shape.cy = self.move_original_positions[0]
            elif hasattr(self.selected_shape, 'points'):  # Polygon
                self.selected_shape.points = self.move_original_positions.copy()
            
            print("❌ Move cancelled")
        
        self.moving = False
        self.move_start_pos = None
        self.move_original_positions = []
        self.setCursor(Qt.ArrowCursor)
        self.update()

    # ===== UNDO/REDO METHODS =====
    def save_state(self):
        """Save current state for undo"""
        # Create a deep copy of all shapes
        state = [shape.copy() for shape in self.shapes]
        self.undo_stack.append(state)
        
        # Clear redo stack when a new action is performed
        self.redo_stack.clear()
        
        # Limit stack size
        if len(self.undo_stack) > self.max_stack_size:
            self.undo_stack.pop(0)
            
        print(f"💾 State saved (undo stack: {len(self.undo_stack)})")
        
    def undo(self):
        """Undo last action"""
        if not self.undo_stack:
            print("⚠️ Nothing to undo")
            return
            
        # Save current state to redo stack
        current_state = [shape.copy() for shape in self.shapes]
        self.redo_stack.append(current_state)
        
        # Restore previous state
        self.shapes = self.undo_stack.pop()
        
        # Deselect any selected shape to prevent issues
        self.selected_shape = None
        self.selected_shapes = []
        
        # Notify UI and trigger autosave
        self.annotation_changed.emit()
        
        self.update()
        print(f"↩ Undo completed (undo: {len(self.undo_stack)}, redo: {len(self.redo_stack)})")
    
    def redo(self):
        """Redo last undone action"""
        if not self.redo_stack:
            print("⚠️ Nothing to redo")
            return
            
        # Save current state to undo stack
        current_state = [shape.copy() for shape in self.shapes]
        self.undo_stack.append(current_state)
        
        # Restore next state
        self.shapes = self.redo_stack.pop()
        
        # Deselect any selected shape to prevent issues
        self.selected_shape = None
        self.selected_shapes = []
        
        # Notify UI and trigger autosave
        self.annotation_changed.emit()
        
        self.update()
        print(f"↪ Redo completed (undo: {len(self.undo_stack)}, redo: {len(self.redo_stack)})")

    def undo_last_point(self):
        """Undo the last drawn point in any multi-point shape (polygon, bezier)"""
        if self.polygon_points and len(self.polygon_points) > 0:
            pt = self.polygon_points.pop()
            if not hasattr(self, 'undone_polygon_points'):
                self.undone_polygon_points = []
            self.undone_polygon_points.append(pt)
            
            # Warp cursor to the new last point
            if self.polygon_points:
                last_pt = self.polygon_points[-1]
                wx = int(last_pt[0] * self.scale + self.offset_x)
                wy = int(last_pt[1] * self.scale + self.offset_y)
                QCursor.setPos(self.mapToGlobal(QPoint(wx, wy)))
                
            self.update()
            print(f"⟲ Undid last polygon point: {pt}")
        elif self.bezier_points and len(self.bezier_points) > 0:
            pt = self.bezier_points.pop()
            if not hasattr(self, 'undone_bezier_points'):
                self.undone_bezier_points = []
            self.undone_bezier_points.append(pt)
            
            # Warp cursor to the new last point
            if self.bezier_points:
                last_pt = self.bezier_points[-1]
                wx = int(last_pt[0] * self.scale + self.offset_x)
                wy = int(last_pt[1] * self.scale + self.offset_y)
                QCursor.setPos(self.mapToGlobal(QPoint(wx, wy)))
                
            self.update()
            print(f"⟲ Undid last bezier point: {pt}")

    def redo_last_point(self):
        """Redo the last drawn point in any multi-point shape"""
        if hasattr(self, 'undone_polygon_points') and self.undone_polygon_points and self.current_shape_type == 'polygon':
            pt = self.undone_polygon_points.pop()
            self.polygon_points.append(pt)
            
            # Warp cursor to the re-added point
            wx = int(pt[0] * self.scale + self.offset_x)
            wy = int(pt[1] * self.scale + self.offset_y)
            QCursor.setPos(self.mapToGlobal(QPoint(wx, wy)))
            
            self.update()
            print(f"⟳ Redid last polygon point: {pt}")
        elif hasattr(self, 'undone_bezier_points') and self.undone_bezier_points and self.current_shape_type == 'bezier_polygon':
            pt = self.undone_bezier_points.pop()
            self.bezier_points.append(pt)
            
            # Warp cursor to the re-added point
            wx = int(pt[0] * self.scale + self.offset_x)
            wy = int(pt[1] * self.scale + self.offset_y)
            QCursor.setPos(self.mapToGlobal(QPoint(wx, wy)))
            
            self.update()
            print(f"⟳ Redid last bezier point: {pt}")
    
    # ==========================================
    #  Right-click context menu
    # ==========================================
    
    def contextMenuEvent(self, event):
        """Show context menu on right-click"""
        self._invoke_context_menu(event.globalPos(), event.pos())
        
    def _invoke_context_menu(self, global_pos, local_pos):
        # Group context menu when 2+ shapes selected
        if len(self.selected_shapes) > 1:
            self._show_multi_select_context_menu(global_pos)
            return
        
        if not hasattr(self, 'canvas_context_menu'):
            from gui.context_menu import ShapeContextMenu
            self.canvas_context_menu = ShapeContextMenu(self)
            self.canvas_context_menu.request_inner.connect(lambda s: self.show_thickness_dialog('inner', s))
            self.canvas_context_menu.request_outer.connect(lambda s: self.show_thickness_dialog('outer', s))
            self.canvas_context_menu.request_save_template.connect(lambda s: self.save_template_from_shape(s) if s else self.save_selected_as_template())
            self.canvas_context_menu.request_duplicate.connect(lambda s: self.copy_selected())
            self.canvas_context_menu.request_delete.connect(lambda s: self.delete_selected())
            self.canvas_context_menu.request_remove_hollow.connect(lambda s: getattr(self, 'remove_inner_cutout', lambda: None)())
            self.canvas_context_menu.request_undo.connect(self.undo)
            self.canvas_context_menu.request_redo.connect(self.redo)
            self.canvas_context_menu.request_paste.connect(lambda p: getattr(self, 'paste_at_cursor', lambda x: None)(p))
            
            self.canvas_context_menu.request_undo_pt.connect(self.undo_last_point)
            self.canvas_context_menu.request_redo_pt.connect(lambda: self.redo_last_point())
            self.canvas_context_menu.request_finish.connect(lambda: self.finish_bezier() if getattr(self, 'drawing_bezier', False) else self.finish_polygon())
            self.canvas_context_menu.request_cancel.connect(lambda: self.cancel_bezier() if getattr(self, 'drawing_bezier', False) else self.cancel_polygon())
            
        self.canvas_context_menu.build_for_canvas(self, global_pos, local_pos)
    
    def undo_last_polygon_point(self):
        """Remove the last point added during polygon drawing"""
        self.undo_last_point()
    
    def start_outer_drawing(self):
        """Start drawing outer shape for the selected polygon (to make it hollow).
        The current polygon becomes the inner, and the new drawing becomes the outer."""
        if self.selected_shape and self.selected_shape.type == 'polygon':
            self.drawing_inner_cutout = True  # reuse the flag name
            self.cutout_target_shape = self.selected_shape
            self.polygon_points = []
            if hasattr(self, 'undone_polygon_points'):
                self.undone_polygon_points = []
            # Deselect so we can draw around it
            self.selected_shape.selected = False
            self.selected_shape = None
            print("✂ Draw outer shape around the polygon, then press Enter to finish")
            self.update()
    
    def start_inner_cutout(self):
        """Alias for backward compat"""
        self.start_outer_drawing()
    
    def remove_inner_cutout(self):
        """Remove hollow effect from selected polygon (make solid)"""
        if self.selected_shape and self.selected_shape.type == 'polygon':
            self.save_state()
            self.selected_shape.inner_points = []
            print("⊘ Hollow effect removed")
            self.update()
    
    def save_selected_as_template(self):
        """Save the selected shape as a reusable template"""
        if self.selected_shape:
            self.save_template_from_shape(self.selected_shape)
    
    def paste_at_cursor(self, pos):
        """Paste clipboard shape at cursor position"""
        if self.clipboard_shape and self.pixmap and not self.pixmap.isNull():
            image_x, image_y = self.widget_to_image(pos)
            pasted = self.clipboard_shape.copy()
            pasted.move(
                (image_x / self.image_width) - pasted.x if hasattr(pasted, 'x') else 0,
                (image_y / self.image_height) - pasted.y if hasattr(pasted, 'y') else 0
            )
            self.save_state()
            self.shapes.append(pasted)
            self.update()
            print("📋 Shape pasted")