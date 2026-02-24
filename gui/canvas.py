# gui/canvas.py
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QRect, QPointF
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen, QBrush, QFont, QPolygonF
import os
import math

from core.annotation import BoundingBox
from core.polygon_shape import PolygonShape
from core.circle_shape import CircleShape

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
        self.setStyleSheet("background-color: #2b2b2b;")
        
        # Image related variables
        self.image = None
        self.image_path = None
        self.pixmap = None
        self.scaled_pixmap = None
        self.image_width = 0
        self.image_height = 0
        
        # Shape drawing variables
        self.current_shape_type = 'box'  # 'box', 'polygon', 'circle'
        self.polygon_points = []  # Temporary points for polygon drawing
        self.circle_center = None  # Center point for circle drawing
        self.circle_radius = 0  # Radius for circle drawing
        
        # View parameters
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_mouse_pos = None
        
        # Mode
        self.mode = 'yolo'  # 'yolo' or 'unet'
        
        # Class manager reference
        self.class_manager = None
        
        # Shape storage
        self.shapes = []  # List of all shapes (boxes, polygons, circles)
        self.drawing = False
        self.start_point = None
        self.current_shape = None
        self.selected_shape = None
        
        # Resize variables
        self.resizing = False
        self.resizing_handle = None
        self.resize_start_pos = None
        
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
        
        # Resize handle size (pixels)
        self.handle_size = 8
        
        # Enable mouse tracking for position updates
        self.setMouseTracking(True)
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        
        print("✅ Canvas initialized")
        
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
        self.update()
        
    def zoom_out(self):
        """Zoom out by 20%"""
        self.scale *= 0.8
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
            
            if isinstance(self.current_shape, BoundingBox):
                x1 = min(self.start_point[0], current_pos[0])
                y1 = min(self.start_point[1], current_pos[1])
                x2 = max(self.start_point[0], current_pos[0])
                y2 = max(self.start_point[1], current_pos[1])
                
                self.current_shape.from_pixels(
                    x1, y1, x2, y2,
                    self.image_width, self.image_height
                )
            self.update()
        
    def finish_drawing(self):
        """Finish drawing and add the shape to the list"""
        if self.drawing and self.current_shape:
            # Set the class ID from current class
            current_class = self.class_manager.get_current_class()
            if current_class:
                self.current_shape.class_id = current_class.id
                self.shapes.append(self.current_shape)
                shape_type = getattr(self.current_shape, 'type', 'box')
                print(f"✅ Added new {shape_type} with class: {current_class.name}")
                
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
        painter.fillRect(self.rect(), QColor(43, 43, 43))
        
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
            
            # Draw polygon preview if drawing polygon
            if self.polygon_points and len(self.polygon_points) > 0:
                self.draw_polygon_preview(painter)
                
            # Draw circle preview if drawing circle
            if self.circle_center and self.circle_radius > 0:
                self.draw_circle_preview(painter)
                
        # Draw mode indicator
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        mode_text = f"Mode: {self.mode.upper()}"
        painter.drawText(10, 20, mode_text)
        
        # Draw shape type indicator (show "None" if no shape selected)
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        if self.current_shape_type:
            shape_text = f"Shape: {self.current_shape_type.upper()}"
        else:
            shape_text = "Shape: NONE (click 'None' to select shapes)"
            painter.setPen(QPen(QColor(255, 100, 100), 1))  # Red color for warning
        painter.drawText(10, 40, shape_text)
        
        # Draw current class indicator
        if self.class_manager:
            current_class = self.class_manager.get_current_class()
            if current_class:
                painter.setPen(QPen(QColor(current_class.color), 2))
                painter.setFont(QFont("Arial", 10))
                painter.drawText(10, 70, f"Class: {current_class.name}")
        
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
            # Add a glow effect
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
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MiddleButton:  # Pan
            self.dragging = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            
        elif event.button() == Qt.LeftButton:
            if self.pixmap and not self.pixmap.isNull():
                
                # Check if we have a selected class
                current_class = self.class_manager.get_current_class() if self.class_manager else None
                if not current_class:
                    print("⚠️ Please select a class first")
                    return
                
                # Allow shapes in YOLO mode too!
                self.handle_shape_press(event)
    
    def handle_shape_press(self, event):
        """Handle mouse press for shape drawing (works in any mode)"""
        # Check if a shape type is selected
        if not self.current_shape_type:
            print("⚠️ Please select a shape tool from the toolbar first")
            return
        
        # Check if Ctrl is pressed for drag-copy
        modifiers = QApplication.keyboardModifiers()
        ctrl_pressed = modifiers == Qt.ControlModifier
        
        # Check if we're over a resize handle of selected shape
        if self.selected_shape:
            handle = self.get_resize_handle_at_pos(event.pos(), self.selected_shape)
            if handle:
                # Start resizing
                self.resizing = True
                self.resizing_handle = handle
                self.resize_start_pos = self.widget_to_image(event.pos())
                return
        
        # If Ctrl is pressed and we click on a shape, start drag-copy
        if ctrl_pressed:
            image_x, image_y = self.widget_to_image(event.pos())
            for shape in reversed(self.shapes):
                if hasattr(shape, 'contains_point') and shape.contains_point(image_x, image_y):
                    self.start_drag_copy(shape, event.pos())
                    return
        
        # Handle shape-specific drawing based on current_shape_type
        if self.current_shape_type == 'polygon':
            self.start_polygon_drawing(event.pos())
        elif self.current_shape_type == 'circle':
            self.start_circle_drawing(event.pos())
        elif self.current_shape_type == 'box':
            # Default to box
            # First try to select a shape
            self.select_shape(event.pos())
            # If we didn't select anything and not dragging, start drawing
            if not self.selected_shape and not self.drag_copy:
                self.start_drawing(event.pos())
        # If current_shape_type is None, we already returned at the beginning
    
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
            
        # Handle drawing (works in any mode)
        elif self.drawing:
            self.update_drawing(event.pos())
        elif self.drag_copy and not self.resizing:
            self.update_drag_copy(event.pos())
        elif self.current_shape_type == 'circle' and self.circle_center:
            self.update_circle_drawing(event.pos())
        elif self.resizing and self.resizing_handle and self.selected_shape:
            current_pos = self.widget_to_image(event.pos())
            dx = current_pos[0] - self.resize_start_pos[0]
            dy = current_pos[1] - self.resize_start_pos[1]
            
            if hasattr(self.selected_shape, 'resize_from_handle'):
                self.selected_shape.resize_from_handle(self.resizing_handle, dx, dy)
            self.resize_start_pos = current_pos
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.MiddleButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
            
        elif event.button() == Qt.LeftButton:
            if self.drawing:
                self.finish_drawing()
            elif self.drag_copy:
                self.finish_drag_copy()
            elif self.current_shape_type == 'circle' and self.circle_center:
                self.finish_circle()
            elif self.resizing:
                self.resizing = False
                self.resizing_handle = None
                self.resize_start_pos = None
                print("✅ Resizing complete")
            
            # Ensure we're not stuck in any special state
            self.drag_copy = False
            self.pasting = False
                
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.1
        if event.angleDelta().y() > 0:
            self.scale *= zoom_factor
        else:
            self.scale /= zoom_factor
                
        # Keep zoom within reasonable limits
        self.scale = max(0.1, min(10.0, self.scale))
        self.update()
            
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        # Only handle essential keys
        if event.key() == Qt.Key_Escape:
            if self.drag_copy:
                self.cancel_drag_copy()
            elif self.drawing:
                self.drawing = False
                self.current_shape = None
                self.update()
                print("❌ Drawing cancelled")
            elif self.polygon_points:
                self.cancel_polygon()
            elif self.circle_center:
                self.cancel_circle()
                
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.polygon_points:
                self.finish_polygon()
                
        elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            if not self.drag_copy:
                self.delete_selected()
                
        else:
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
        self.drawing_polygon = False
        # Don't clear shapes or selected_shape
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
        """Start pasting a shape at the given position"""
        if not self.clipboard_shape:
            print("⚠️ No shape in clipboard to paste")
            return False
        
        print(f"📋 Starting paste")
        
        # Create a copy of the clipboard shape
        self.paste_shape = self.clipboard_shape.copy()
        self.pasting = True
        self.paste_confirmed = False
        
        # Position the shape at mouse cursor
        image_x, image_y = self.widget_to_image(pos)
        
        # Update position based on shape type
        if hasattr(self.paste_shape, 'x') and hasattr(self.paste_shape, 'y'):
            # For boxes
            self.paste_shape.x = image_x / self.image_width
            self.paste_shape.y = image_y / self.image_height
        elif hasattr(self.paste_shape, 'center_x') and hasattr(self.paste_shape, 'center_y'):
            # For circles
            self.paste_shape.center_x = image_x / self.image_width
            self.paste_shape.center_y = image_y / self.image_height
        elif hasattr(self.paste_shape, 'points'):
            # For polygons - move all points
            if self.paste_shape.points:
                dx = (image_x / self.image_width) - self.paste_shape.points[0][0]
                dy = (image_y / self.image_height) - self.paste_shape.points[0][1]
                self.paste_shape.move(dx, dy)

        # Deselect any selected shape
        if self.selected_shape:
            self.selected_shape.selected = False
            self.selected_shape = None
    
        # Select the paste shape
        self.paste_shape.selected = True
        self.selected_shape = self.paste_shape

        # Add to shapes list temporarily for display
        self.shapes.append(self.paste_shape)
        
        shape_type = getattr(self.paste_shape, 'type', 'box')
        print(f"📋 Started pasting {shape_type} - drag corners to resize, Enter to confirm, Esc to cancel")
        self.update()
        return True

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
                # For boxes
                self.drag_copy_shape.x = image_x / self.image_width
                self.drag_copy_shape.y = image_y / self.image_height
            elif hasattr(self.drag_copy_shape, 'center_x') and hasattr(self.drag_copy_shape, 'center_y'):
                # For circles
                self.drag_copy_shape.center_x = image_x / self.image_width
                self.drag_copy_shape.center_y = image_y / self.image_height
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
            print("✅ Drag copy completed")
            
            # Keep the copied shape but make sure it's not selected
            if self.drag_copy_shape:
                self.drag_copy_shape.selected = False
            
            # Clear all drag-copy related states
            self.drag_copy = False
            self.drag_copy_shape = None
            self.drag_start_pos = None
            self.original_shape = None
            
            # Clear any lingering selection if needed
            if self.selected_shape and not self.selected_shape.selected:
                self.selected_shape = None
                
            # Reset all interaction states
            self.resizing = False
            self.resizing_handle = None
            self.pasting = False
            
            print("✅ Ready to draw new shapes")
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
        """Set the current shape type (None means no shape selected)"""
        self.current_shape_type = shape_type
        self.reset_all_states()
        if shape_type:
            print(f"🔷 Shape type set to: {shape_type}")
        else:
            print("⬜ Shape type cleared - no drawing tool selected")
        
        # If we were drawing a polygon, cancel it
        if self.polygon_points:
            self.cancel_polygon()
        if self.circle_center:
            self.cancel_circle()
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
                # Check if we're clicking near the first point to close the polygon
                first_x, first_y = self.polygon_points[0]
                distance = math.sqrt((image_x - first_x)**2 + (image_y - first_y)**2)
                
                # If close to first point, finish the polygon
                if distance < 20 and len(self.polygon_points) >= 3:
                    self.finish_polygon()
                else:
                    # Add point to existing polygon
                    self.polygon_points.append((image_x, image_y))
                    print(f"🔷 Added polygon point ({image_x}, {image_y})")
            
            self.update()
            
    def finish_polygon(self):
        """Finish drawing polygon"""
        if len(self.polygon_points) >= 3:
            # Create polygon shape
            polygon = PolygonShape(
                class_id=self.class_manager.get_current_class().id,
                image_size=(self.image_width, self.image_height)
            )
            polygon.from_pixel_points(self.polygon_points)
            polygon.close_polygon()
            self.shapes.append(polygon)
            print(f"✅ Polygon completed with {len(self.polygon_points)} points")
        
        # Reset polygon drawing state
        self.polygon_points = []
        self.update()
        
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