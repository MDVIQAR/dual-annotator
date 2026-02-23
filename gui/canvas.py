# gui/canvas.py
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QRect
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen, QBrush, QFont
import os

from core.annotation import BoundingBox

class AnnotationCanvas(QWidget):
    """Canvas widget for displaying images and annotations"""
    
    # Signals
    position_changed = pyqtSignal(int, int)  # Emitted when mouse moves
    
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
        
        # Box drawing variables
        self.boxes = []  # List of bounding boxes
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.selected_box = None
        self.resizing = False
        self.resize_corner = None
        
        # Enable mouse tracking for position updates
        self.setMouseTracking(True)
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
        
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
                
                # Clear previous boxes when loading new image
                self.boxes = []
                self.selected_box = None
                self.drawing = False
                self.current_box = None
                
                self.fit_to_window()
                self.update()
                
                # Print loaded image info
                print(f"Loaded image: {os.path.basename(image_path)} ({self.image_width}x{self.image_height})")
            else:
                print(f"Failed to load image: {image_path}")
                
        except Exception as e:
            print(f"Error loading image: {e}")
            
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
        
    def set_mode(self, mode):
        """Set the annotation mode"""
        self.mode = mode
        self.update()
        
    def start_drawing(self, pos):
        """Start drawing a new bounding box"""
        if self.mode == 'yolo' and self.class_manager and self.class_manager.get_current_class():
            self.drawing = True
            self.start_point = self.widget_to_image(pos)
            self.current_box = BoundingBox(
                image_size=(self.image_width, self.image_height)
            )
        
    def update_drawing(self, pos):
        """Update the current box while drawing"""
        if self.drawing and self.start_point and self.current_box:
            current_pos = self.widget_to_image(pos)
            
            x1 = min(self.start_point[0], current_pos[0])
            y1 = min(self.start_point[1], current_pos[1])
            x2 = max(self.start_point[0], current_pos[0])
            y2 = max(self.start_point[1], current_pos[1])
            
            self.current_box.from_pixels(
                x1, y1, x2, y2,
                self.image_width, self.image_height
            )
            self.update()
        
    def finish_drawing(self):
        """Finish drawing and add the box to the list"""
        if self.drawing and self.current_box:
            # Set the class ID from current class
            current_class = self.class_manager.get_current_class()
            if current_class:
                self.current_box.class_id = current_class.id
                self.boxes.append(self.current_box)
                
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.update()
        
    def select_box(self, pos):
        """Select a box at the given position"""
        if not self.boxes:
            return
            
        # Deselect all boxes first
        for box in self.boxes:
            box.selected = False
            
        # Check each box (from top to bottom)
        image_x, image_y = self.widget_to_image(pos)
        for box in reversed(self.boxes):
            if box.contains_point(image_x, image_y):
                box.selected = True
                self.selected_box = box
                break
                
        self.update()
        
    def delete_selected(self):
        """Delete the selected box"""
        if self.selected_box:
            self.boxes.remove(self.selected_box)
            self.selected_box = None
            self.update()
        
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
            
            # Draw all bounding boxes
            self.draw_boxes(painter)
            
            # Draw current box if drawing
            if self.drawing and self.current_box:
                self.draw_single_box(painter, self.current_box, QColor(255, 255, 0))
                
        # Draw mode indicator (even if no image)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        mode_text = f"Mode: {self.mode.upper()}"
        painter.drawText(10, 20, mode_text)
        
    def draw_boxes(self, painter):
        """Draw all bounding boxes"""
        for box in self.boxes:
            # Get class color
            color = QColor(0, 255, 0)  # Default green
            if self.class_manager and box.class_id:
                cls = self.class_manager.get_class(box.class_id)
                if cls:
                    color = QColor(cls.color)
                    
            self.draw_single_box(painter, box, color)
        
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
        
        # Draw class name if available
        if self.class_manager and box.class_id:
            cls = self.class_manager.get_class(box.class_id)
            if cls:
                painter.setPen(QPen(Qt.white, 1))
                painter.setFont(QFont("Arial", 8))
                painter.drawText(x1, y1 - 5, cls.name)
        
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
            
        # Handle drawing
        if self.drawing:
            self.update_drawing(event.pos())
            
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MiddleButton:  # Pan
            self.dragging = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:  # Left click for drawing/selecting
            if self.mode == 'yolo' and self.pixmap and not self.pixmap.isNull():
                # Check if we're selecting a box
                self.select_box(event.pos())
                if not self.selected_box:
                    # Start drawing new box
                    self.start_drawing(event.pos())
                    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.MiddleButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton and self.drawing:
            self.finish_drawing()
            
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
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_selected()
        elif event.key() == Qt.Key_Escape:
            self.drawing = False
            self.current_box = None
            self.update()
        else:
            super().keyPressEvent(event)
        
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