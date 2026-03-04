# gui/canvas.py
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QRect, QPointF, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen, QBrush, QFont, QPolygonF, QCursor
import os
import math

from core.annotation import BoundingBox
from core.polygon_shape import PolygonShape
from core.circle_shape import CircleShape
from core.ellipse_shape import EllipseShape
from core.ring_shape import FrameShape, DonutShape, HollowEllipseShape
from core.template_manager import TemplateManager

class AnnotationCanvas(QWidget):
    """Canvas widget for displaying images and annotations"""
    
    # Signals
    position_changed = pyqtSignal(int, int)  # Emitted when mouse moves
    shape_selected = pyqtSignal(str)  # Emitted when shape is selected
    
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
        
        # Polygon drawing state
        self.drawing_polygon = False
        
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
                
    def zoom_in(self):
        """Zoom in by 20%"""
        self.scale *= 1.2
        self.scale = min(10.0, self.scale)
        self.update()
        
    def zoom_out(self):
        """Zoom out by 20%"""
        self.scale *= 0.8
        self.scale = max(0.1, self.scale)
        self.update()
        
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
        
        self.drawing = False
        self.start_point = None
        self.current_shape = None
        self.update()
        
    def select_shape(self, pos):
        """Select a shape at the given position - only one shape at a time"""
        if not self.shapes:
            self.selected_shape = None
            self.shape_selected.emit("none")
            return
            
        # First, deselect ALL shapes
        for shape in self.shapes:
            shape.selected = False
            
        # Check each shape (from top to bottom)
        image_x, image_y = self.widget_to_image(pos)
        selected = False
        for shape in reversed(self.shapes):
            if hasattr(shape, 'contains_point') and shape.contains_point(image_x, image_y):
                shape.selected = True
                self.selected_shape = shape
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
            self.update()
            print("🗑️ Deleted selected shape")
        
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
            
            # Draw stamp preview if stamping
            if hasattr(self, 'stamping') and self.stamping and self.stamp_center and self.stamp_current_pos:
                self.draw_stamp_preview(painter)
                
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
        
    def draw_shapes(self, painter):
        """Draw all shapes"""
        for shape in self.shapes:
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
        
        # Draw rectangle
        rect = QRect(x1, y1, x2 - x1, y2 - y1)
        painter.drawRect(rect)
        
        # Draw resize handles for selected box
        if box.selected:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            
            # Draw corner handles
            half = self.handle_size // 2
            painter.drawRect(x1 - half, y1 - half, self.handle_size, self.handle_size)  # TL
            painter.drawRect(x2 - half, y1 - half, self.handle_size, self.handle_size)  # TR
            painter.drawRect(x1 - half, y2 - half, self.handle_size, self.handle_size)  # BL
            painter.drawRect(x2 - half, y2 - half, self.handle_size, self.handle_size)  # BR
        
        # Draw class name if available
        if self.class_manager and hasattr(box, 'class_id') and box.class_id:
            cls = self.class_manager.get_class(box.class_id)
            if cls:
                painter.setPen(QPen(Qt.white, 1))
                painter.setFont(QFont("Arial", 8))
                
                # Draw background for text
                text = cls.name
                text_width = painter.fontMetrics().horizontalAdvance(text)
                text_height = painter.fontMetrics().height()
                painter.fillRect(x1, y1 - text_height - 5, text_width + 10, text_height + 5, QColor(0, 0, 0, 150))
                
                painter.drawText(x1 + 5, y1 - 8, text)
    
    def draw_polygon(self, painter, polygon, color):
        """Draw a polygon shape with highlighting when selected"""
        points = polygon.to_pixel_points()
        
        # Convert to widget coordinates
        widget_points = []
        for px, py in points:
            wx = int(px * self.scale + self.offset_x)
            wy = int(py * self.scale + self.offset_y)
            widget_points.append(QPointF(wx, wy))
        
        # Set pen based on selection - YELLOW when selected
        if polygon.selected:
            pen = QPen(QColor(255, 255, 0), 3)  # Yellow, thicker for selected
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 80)))
        else:
            pen = QPen(color, 2)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        
        painter.setPen(pen)
        
        # Draw polygon
        poly = QPolygonF()
        for point in widget_points:
            poly.append(point)
        
        if len(poly) >= 3:
            painter.drawPolygon(poly)
            
            # Draw vertices for selected polygon
            if polygon.selected:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                
                half = self.handle_size // 2
                for wx, wy in [(p.x(), p.y()) for p in widget_points]:
                    painter.drawRect(int(wx - half), int(wy - half), self.handle_size, self.handle_size)
        
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
        
        # Draw circle
        painter.drawEllipse(wx - wr, wy - wr, wr * 2, wr * 2)
        
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
        
        # Draw ellipse
        painter.drawEllipse(wx - wrx, wy - wry, wrx * 2, wry * 2)
        
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
        """Draw hollow ellipse preview while drawing"""
        if not hasattr(self, 'hollow_ellipse_center') or not self.hollow_ellipse_center:
            return
        
        cx, cy = self.hollow_ellipse_center
        wx = int(cx * self.scale + self.offset_x)
        wy = int(cy * self.scale + self.offset_y)
        wrx = int(self.hollow_ellipse_rx * self.scale)
        wry = int(self.hollow_ellipse_ry * self.scale)
        
        # Draw outer ellipse preview
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(wx - wrx, wy - wry, wrx * 2, wry * 2)
        
        # Draw inner ellipse preview (45% of outer)
        wirx = int(wrx * 0.45)
        wiry = int(wry * 0.45)
        painter.setPen(QPen(QColor(255, 200, 0), 1, Qt.DashLine))
        painter.drawEllipse(wx - wirx, wy - wiry, wirx * 2, wiry * 2)
        
        # Draw center point
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        half = self.handle_size // 2
        painter.drawRect(wx - half, wy - half, self.handle_size, self.handle_size)

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
            
        elif event.button() == Qt.LeftButton and not self.pan_mode:
            if self.pixmap and not self.pixmap.isNull():
                
                # Check if we have a selected class
                current_class = self.class_manager.get_current_class() if self.class_manager else None
                
                # Check if Ctrl is pressed for drag-copy
                modifiers = QApplication.keyboardModifiers()
                ctrl_pressed = modifiers == Qt.ControlModifier
                
                # FIRST: Check if we're over a resize handle of selected shape
                if self.selected_shape:
                    handle = self.get_resize_handle_at_pos(event.pos(), self.selected_shape)
                    if handle:
                        # Start resizing
                        self.resizing = True
                        self.resizing_handle = handle
                        self.resize_start_pos = self.widget_to_image(event.pos())
                        
                        # Call begin_resize on the shape
                        if hasattr(self.selected_shape, 'begin_resize'):
                            self.selected_shape.begin_resize()
                        
                        return
                
                # SECOND: Check if we're clicking on a shape (for moving or selecting)
                image_x, image_y = self.widget_to_image(event.pos())
                clicked_shape = None
                for shape in reversed(self.shapes):
                    if hasattr(shape, 'contains_point') and shape.contains_point(image_x, image_y):
                        clicked_shape = shape
                        break
                
                if clicked_shape:
                    # If Ctrl is pressed, start drag-copy
                    if ctrl_pressed:
                        self.start_drag_copy(clicked_shape, event.pos())
                        return
                    
                    # If the clicked shape is already selected, start moving it
                    if clicked_shape.selected:
                        self.start_move(clicked_shape, event.pos())
                        return
                    else:
                        # Just select the shape
                        self.select_shape(event.pos())
                        return
                
                # THIRD: Handle drawing based on current tool
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
                    elif self.current_shape_type == 'stamp':
                        self.start_stamp(event.pos())
                else:
                    # No tool selected, just click on empty area (deselect)
                    self.select_shape(event.pos())

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        # Convert widget coordinates to image coordinates
        image_x, image_y = self.widget_to_image(event.pos())
        
        # Emit position signal
        self.position_changed.emit(image_x, image_y)
        
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
                
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming - zooms to cursor position"""
        if not self.pixmap or self.pixmap.isNull():
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
            
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        # Get the key and convert to uppercase for comparison
        key = event.key()
        key_text = event.text().upper()
        
        # ===== SHAPE TOOL SHORTCUTS =====
        if key_text == 'B':
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
            print("❌ Operation cancelled")
            return
        
        # ===== ENTER =====
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            if self.polygon_points:
                self.finish_polygon()
            return
        
        # ===== DELETE =====
        elif key == Qt.Key_Delete or key == Qt.Key_Backspace:
            if not self.drag_copy and not self.pan_mode and not self.moving:
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
            self.copy_selected()
            return
        
        elif key == Qt.Key_V and event.modifiers() == Qt.ControlModifier:
            if self.clipboard_shape and self.pixmap and not self.pixmap.isNull():
                cursor_pos = self.mapFromGlobal(self.cursor().pos())
                self.start_paste(cursor_pos)
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
        """Update position of dragged copy"""
        if self.drag_copy and self.drag_copy_shape:
            image_x, image_y = self.widget_to_image(pos)
            
            # Update position based on shape type
            if hasattr(self.drag_copy_shape, 'x') and hasattr(self.drag_copy_shape, 'y'):
                # For boxes and frames
                self.drag_copy_shape.x = image_x / self.image_width
                self.drag_copy_shape.y = image_y / self.image_height
            elif hasattr(self.drag_copy_shape, 'center_x') and hasattr(self.drag_copy_shape, 'center_y'):
                # For circles and ellipses
                self.drag_copy_shape.center_x = image_x / self.image_width
                self.drag_copy_shape.center_y = image_y / self.image_height
            elif hasattr(self.drag_copy_shape, 'cx') and hasattr(self.drag_copy_shape, 'cy'):
                # For donuts
                self.drag_copy_shape.cx = image_x / self.image_width
                self.drag_copy_shape.cy = image_y / self.image_height
            elif hasattr(self.drag_copy_shape, 'points'):
                # For polygons - move all points
                if self.drag_start_pos:
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
        if not self.stamping or not self.stamp_center or not self.stamp_current_pos:
            self.stamping = False
            return
        
        # Calculate scale from drag distance
        dx = abs(self.stamp_current_pos[0] - self.stamp_center[0])
        dy = abs(self.stamp_current_pos[1] - self.stamp_center[1])
        scale_x = max(dx, 10)  # minimum 10px
        scale_y = max(dy, 10)
        
        # Get pixel points from template
        pixel_points = self.template_manager.get_pixel_points(
            self.stamp_template_name,
            self.stamp_center[0], self.stamp_center[1],
            scale_x, scale_y
        )
        
        if pixel_points and len(pixel_points) >= 3:
            current_class = self.class_manager.get_current_class()
            if current_class:
                polygon = PolygonShape(
                    class_id=current_class.id,
                    image_size=(self.image_width, self.image_height)
                )
                polygon.from_pixel_points(pixel_points)
                polygon.close_polygon()
                self.save_state()
                self.shapes.append(polygon)
                print(f"✅ Stamp placed with {len(pixel_points)} points")
        
        self.stamping = False
        self.stamp_center = None
        self.stamp_current_pos = None
        self.update()
    
    def draw_stamp_preview(self, painter):
        """Draw preview of stamp being placed"""
        if not self.stamp_center or not self.stamp_current_pos or not self.stamp_template_name:
            return
        
        # Calculate scale
        dx = abs(self.stamp_current_pos[0] - self.stamp_center[0])
        dy = abs(self.stamp_current_pos[1] - self.stamp_center[1])
        scale_x = max(dx, 10)
        scale_y = max(dy, 10)
        
        # Get pixel points
        pixel_points = self.template_manager.get_pixel_points(
            self.stamp_template_name,
            self.stamp_center[0], self.stamp_center[1],
            scale_x, scale_y
        )
        
        if not pixel_points or len(pixel_points) < 3:
            return
        
        # Draw preview polygon
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        painter.setBrush(QBrush(QColor(255, 255, 0, 40)))
        
        points = QPolygonF()
        for px, py in pixel_points:
            wx = px * self.scale + self.offset_x
            wy = py * self.scale + self.offset_y
            points.append(QPointF(wx, wy))
        points.append(points[0])  # close the polygon
        painter.drawPolygon(points)
        
        # Draw center marker
        cx = self.stamp_center[0] * self.scale + self.offset_x
        cy = self.stamp_center[1] * self.scale + self.offset_y
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        half = self.handle_size // 2
        painter.drawRect(int(cx) - half, int(cy) - half, self.handle_size, self.handle_size)
        
    def cancel_polygon(self):
        """Cancel polygon drawing"""
        self.polygon_points = []
        print("❌ Polygon cancelled")
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
        
        # Move shape based on type
        if hasattr(self.selected_shape, 'x') and hasattr(self.selected_shape, 'y'):  # Box, Frame
            self.selected_shape.x += dx
            self.selected_shape.y += dy
        elif hasattr(self.selected_shape, 'center_x') and hasattr(self.selected_shape, 'center_y'):  # Circle, Ellipse
            self.selected_shape.center_x += dx
            self.selected_shape.center_y += dy
        elif hasattr(self.selected_shape, 'cx') and hasattr(self.selected_shape, 'cy'):  # Donut
            self.selected_shape.cx += dx
            self.selected_shape.cy += dy
        elif hasattr(self.selected_shape, 'points'):  # Polygon
            new_points = []
            for nx, ny in self.selected_shape.points:
                new_points.append((nx + dx, ny + dy))
            self.selected_shape.points = new_points
        
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
        # Create a copy of current shapes
        state = []
        for shape in self.shapes:
            if hasattr(shape, 'copy'):
                state.append(shape.copy())
        
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_stack_size:
            self.undo_stack.pop(0)
        self.redo_stack.clear()  # Clear redo stack on new action
        print(f"💾 State saved (undo stack: {len(self.undo_stack)})")

    def undo(self):
        """Undo last action"""
        if not self.undo_stack:
            print("⚠️ Nothing to undo")
            return False
        
        # Save current state to redo stack
        current_state = []
        for shape in self.shapes:
            if hasattr(shape, 'copy'):
                current_state.append(shape.copy())
        self.redo_stack.append(current_state)
        
        # Restore previous state
        self.shapes = self.undo_stack.pop()
        self.selected_shape = None
        self.shape_selected.emit("none")
        self.update()
        print(f"↩ Undo completed (undo: {len(self.undo_stack)}, redo: {len(self.redo_stack)})")
        return True

    def redo(self):
        """Redo last undone action"""
        if not self.redo_stack:
            print("⚠️ Nothing to redo")
            return False
        
        # Save current state to undo stack
        current_state = []
        for shape in self.shapes:
            if hasattr(shape, 'copy'):
                current_state.append(shape.copy())
        self.undo_stack.append(current_state)
        
        # Restore next state
        self.shapes = self.redo_stack.pop()
        self.selected_shape = None
        self.shape_selected.emit("none")
        self.update()
        print(f"↪ Redo completed (undo: {len(self.undo_stack)}, redo: {len(self.redo_stack)})")
        return True