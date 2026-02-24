# gui/canvas.py
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QRect, QTimer
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
        
        # Resize variables
        self.resizing = False
        self.resizing_handle = None
        self.resize_start_pos = None
        
        # Drag-copy variables
        self.drag_copy = False
        self.drag_copy_box = None
        self.drag_start_pos = None
        self.original_box = None
        
        # Paste variables (keeping for compatibility)
        self.clipboard_box = None
        self.pasting = False
        self.paste_box = None
        self.paste_start_pos = None
        self.paste_confirmed = False
        
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
                
                # Clear previous boxes when loading new image
                self.boxes = []
                self.selected_box = None
                
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
        """Start drawing a new bounding box"""
        # Force reset all states before starting to draw
        self.force_reset_for_drawing()
        
        if self.mode == 'yolo' and self.class_manager and self.class_manager.get_current_class():
            self.drawing = True
            self.start_point = self.widget_to_image(pos)
            self.current_box = BoundingBox(
                image_size=(self.image_width, self.image_height)
            )
            print(f"✏️ Started drawing new box")
        else:
            print("⚠️ Cannot draw - no class selected or not in YOLO mode")
        
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
                print(f"✅ Added new box with class: {current_class.name}")
                
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.update()
        
    def select_box(self, pos):
        """Select a box at the given position"""
        if not self.boxes:
            self.selected_box = None
            return
            
        # Deselect all boxes first
        for box in self.boxes:
            box.selected = False
            
        # Check each box (from top to bottom)
        image_x, image_y = self.widget_to_image(pos)
        selected = False
        for box in reversed(self.boxes):
            if box.contains_point(image_x, image_y):
                box.selected = True
                self.selected_box = box
                selected = True
                # Print selected box info
                if self.class_manager and box.class_id:
                    cls = self.class_manager.get_class(box.class_id)
                    if cls:
                        print(f"🔍 Selected box: {cls.name}")
                break
        
        if not selected:
            self.selected_box = None
            print("👆 Clicked on empty area")
                    
        self.update()
        
    def delete_selected(self):
        """Delete the selected box"""
        if self.selected_box:
            self.boxes.remove(self.selected_box)
            self.selected_box = None
            self.update()
            print("🗑️ Deleted selected box")
        
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
                
        # Draw mode indicator
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        mode_text = f"Mode: {self.mode.upper()}"
        painter.drawText(10, 20, mode_text)
        
        # Draw current class indicator
        if self.class_manager:
            current_class = self.class_manager.get_current_class()
            if current_class:
                painter.setPen(QPen(QColor(current_class.color), 2))
                painter.setFont(QFont("Arial", 10))
                painter.drawText(10, 50, f"Current class: {current_class.name}")
        
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
        if self.class_manager and box.class_id:
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
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MiddleButton:  # Pan
            self.dragging = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            
        elif event.button() == Qt.LeftButton:
            if self.mode == 'yolo' and self.pixmap and not self.pixmap.isNull():
                
                # Check if we have a selected class
                current_class = self.class_manager.get_current_class() if self.class_manager else None
                if not current_class:
                    print("⚠️ Please select a class first")
                    return
                
                # Check if Ctrl is pressed for drag-copy
                modifiers = QApplication.keyboardModifiers()
                ctrl_pressed = modifiers == Qt.ControlModifier
                
                # First, check if we're clicking on an existing box
                image_x, image_y = self.widget_to_image(event.pos())
                clicked_on_box = False
                
                for box in reversed(self.boxes):
                    if box.contains_point(image_x, image_y):
                        clicked_on_box = True
                        break
                
                # If Ctrl is pressed and we click on a box, start drag-copy
                if ctrl_pressed and clicked_on_box:
                    for box in reversed(self.boxes):
                        if box.contains_point(image_x, image_y):
                            self.start_drag_copy(box, event.pos())
                            return
                
                # Check if we're over a resize handle of selected box
                if self.selected_box:
                    handle = self.get_resize_handle_at_pos(event.pos(), self.selected_box)
                    if handle:
                        # Start resizing
                        self.resizing = True
                        self.resizing_handle = handle
                        self.resize_start_pos = self.widget_to_image(event.pos())
                        return
                
                # If we're not doing anything else, handle selection/drawing
                if not ctrl_pressed:
                    # First try to select a box
                    self.select_box(event.pos())
                    
                    # If we didn't select a box (clicked on empty area), start drawing
                    if not self.selected_box:
                        # Make sure we're not in any special state
                        self.drag_copy = False
                        self.drag_copy_box = None
                        self.resizing = False
                        self.resizing_handle = None
                        self.start_drawing(event.pos())

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
        elif self.drawing:
            self.update_drawing(event.pos())
            
        # Handle drag-copy
        elif self.drag_copy and not self.resizing:
            self.update_drag_copy(event.pos())
            
        # Handle resizing
        elif self.resizing and self.resizing_handle and self.selected_box:
            current_pos = self.widget_to_image(event.pos())
            dx = current_pos[0] - self.resize_start_pos[0]
            dy = current_pos[1] - self.resize_start_pos[1]
            
            # Call resize_from_handle on the selected_box
            self.selected_box.resize_from_handle(self.resizing_handle, dx, dy)
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
            elif self.resizing:
                # Finished resizing
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
                self.current_box = None
                self.update()
                print("❌ Drawing cancelled")
                
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
        self.drag_copy_box = None
        self.current_box = None
        self.start_point = None
        self.pasting = False
        # Don't clear boxes or selected_box
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
        """Copy the selected box to clipboard"""
        if self.selected_box:
            self.clipboard_box = self.selected_box.copy()
            print(f"📋 Copied box - Class: {self.clipboard_box.class_id}, Size: {self.clipboard_box.width:.3f}x{self.clipboard_box.height:.3f}")
            print(f"   Position: ({self.clipboard_box.x:.3f}, {self.clipboard_box.y:.3f})")
            return True
        else:
            print("⚠️ No box selected to copy")
            return False    
    
    def start_paste(self, pos):
        """Start pasting a box at the given position"""
        if not self.clipboard_box:
            print("⚠️ No box in clipboard to paste")
            return False
        
        print(f"📋 Starting paste at widget position: ({pos.x()}, {pos.y()})")
        
        # Create a copy of the clipboard box
        self.paste_box = self.clipboard_box.copy()
        self.pasting = True
        self.paste_confirmed = False
        
        # Position the box at mouse cursor
        image_x, image_y = self.widget_to_image(pos)
        print(f"   Image coordinates: ({image_x}, {image_y})")
        
        # Update coordinates to center at mouse
        self.paste_box.x = image_x / self.image_width
        self.paste_box.y = image_y / self.image_height

        print(f"   Normalized position: ({self.paste_box.x:.3f}, {self.paste_box.y:.3f})")

        # Deselect any selected box
        if self.selected_box:
            self.selected_box.selected = False
            self.selected_box = None
    
        # Select the paste box
        self.paste_box.selected = True
        self.selected_box = self.paste_box

        # Add to boxes list temporarily for display
        self.boxes.append(self.paste_box)
        
        print("📋 Started pasting - drag corners to resize, Enter to confirm, Esc to cancel")
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
        """Confirm the paste and add the box to the list"""
        if self.pasting and self.paste_box:
            print("✅ Paste confirmed")
            # The box is already in self.boxes, just update state
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
            # Remove the temporary paste box from boxes list
            if self.paste_box in self.boxes:
                self.boxes.remove(self.paste_box)
            self.pasting = False
            self.paste_box = None
            self.resizing_handle = None
            self.paste_start_pos = None
            self.selected_box = None
            self.update()
            return True
        return False         
    
    def get_resize_handle_at_pos(self, pos, box):
        """Check if position is over a resize handle of a box"""
        if not box or not box.selected:
            return None
        
        # Get box pixel coordinates
        x1, y1, x2, y2 = box.to_pixels()
        
        # Convert to widget coordinates
        x1 = int(x1 * self.scale + self.offset_x)
        y1 = int(y1 * self.scale + self.offset_y)
        x2 = int(x2 * self.scale + self.offset_x)
        y2 = int(y2 * self.scale + self.offset_y)

        # Define handle regions
        half = self.handle_size // 2
        px, py = pos.x(), pos.y()
        
        handles = {
            'top_left': (x1 - half, y1 - half, x1 + half, y1 + half),
            'top_right': (x2 - half, y1 - half, x2 + half, y1 + half),
            'bottom_left': (x1 - half, y2 - half, x1 + half, y2 + half),
            'bottom_right': (x2 - half, y2 - half, x2 + half, y2 + half)
        }
        
        for handle_name, (hx1, hy1, hx2, hy2) in handles.items():
            if hx1 <= px <= hx2 and hy1 <= py <= hy2:
                return handle_name
        
        return None
    
    def start_drag_copy(self, box, pos):
        """Start dragging a copy of the selected box"""
        if not box:
            return False
        
        print(f"📋 Starting drag copy of box")
        
        # Create a copy
        self.drag_copy_box = box.copy()
        self.drag_copy = True
        self.original_box = box
        self.drag_start_pos = self.widget_to_image(pos)
        
        # Store original box selection state
        box.selected = False
        
        # Add the copy to boxes list immediately
        self.boxes.append(self.drag_copy_box)
        self.drag_copy_box.selected = True
        self.selected_box = self.drag_copy_box
        
        self.update()
        return True

    def update_drag_copy(self, pos):
        """Update position of dragged copy"""
        if self.drag_copy and self.drag_copy_box:
            image_x, image_y = self.widget_to_image(pos)
            
            # Update position (centered at mouse)
            self.drag_copy_box.x = image_x / self.image_width
            self.drag_copy_box.y = image_y / self.image_height
            
            self.update()

    def finish_drag_copy(self):
        """Finish dragging copy"""
        if self.drag_copy and self.drag_copy_box:
            print("✅ Drag copy completed")
            
            # Keep the copied box but make sure it's not selected
            if self.drag_copy_box:
                self.drag_copy_box.selected = False
            
            # Clear all drag-copy related states
            self.drag_copy = False
            self.drag_copy_box = None
            self.drag_start_pos = None
            self.original_box = None
            
            # Clear any lingering selection if needed
            if self.selected_box and not self.selected_box.selected:
                self.selected_box = None
                
            # Reset all interaction states
            self.resizing = False
            self.resizing_handle = None
            self.pasting = False
            
            print("✅ Ready to draw new boxes")
            self.update()
            return True
        return False

    def cancel_drag_copy(self):
        """Cancel drag copy"""
        if self.drag_copy and self.drag_copy_box:
            print("❌ Drag copy cancelled")
            
            # Remove the temporary drag copy box from boxes list
            if self.drag_copy_box in self.boxes:
                self.boxes.remove(self.drag_copy_box)
            
            # Restore original box selection
            if self.original_box:
                self.original_box.selected = True
                self.selected_box = self.original_box
            else:
                self.selected_box = None
            
            # Clear all drag-copy related states
            self.drag_copy = False
            self.drag_copy_box = None
            self.drag_start_pos = None
            self.original_box = None
            
            # Reset all interaction states
            self.resizing = False
            self.resizing_handle = None
            self.pasting = False
            
            print("✅ Ready to draw new boxes")
            self.update()
            return True
        return False
    
    def debug_state(self):
        """Print current state for debugging"""
        print(f"🔍 Canvas State - Drawing: {self.drawing}, DragCopy: {self.drag_copy}, "
            f"Resizing: {self.resizing}, Pasting: {self.pasting}, "
            f"Selected Box: {self.selected_box is not None}")
        
    def force_reset_for_drawing(self):
        """Force reset all states to allow drawing"""
        print("🔄 Force resetting all states for drawing")
        self.drawing = False
        self.drag_copy = False
        self.drag_copy_box = None
        self.resizing = False
        self.resizing_handle = None
        self.pasting = False
        self.paste_box = None
        # Don't clear selected_box here - let that be handled by click on empty area    