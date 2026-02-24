# gui/shape_toolbar.py
from PyQt5.QtWidgets import QToolBar, QAction
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPainter, QColor, QPen, QBrush, QPixmap, QPolygon
from PyQt5.QtCore import QPoint

class ShapeToolbar(QToolBar):
    """Toolbar for selecting shape types"""
    
    # Signal emitted when shape type changes
    shape_changed = pyqtSignal(str)
    # Signal emitted when no shape is selected
    shape_deselected = pyqtSignal()
    
    def __init__(self):
        super().__init__("Shape Tools")
        
        self.setIconSize(QSize(32, 32))
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        # Create shape actions with icons
        self.shapes = [
            ('box', self.create_box_icon(), 'Box', 'Draw bounding boxes'),
            ('polygon', self.create_polygon_icon(), 'Polygon', 'Draw polygons'),
            ('circle', self.create_circle_icon(), 'Circle', 'Draw circles')
        ]
        
        self.actions = {}
        self.current_shape = None  # Start with no shape selected
        
        for shape_id, icon, text, tooltip in self.shapes:
            action = QAction(icon, text, self)
            action.setToolTip(tooltip)
            action.setCheckable(True)
            action.setData(shape_id)
            action.triggered.connect(lambda checked, s=shape_id: self.on_shape_selected(s))
            self.addAction(action)
            self.actions[shape_id] = action
        
        # Add a separator
        self.addSeparator()
        
        # Add a "None" action to deselect all
        self.none_action = QAction("⬜ None", self)
        self.none_action.setToolTip("Deselect all shapes")
        self.none_action.setCheckable(True)
        self.none_action.triggered.connect(self.on_none_selected)
        self.addAction(self.none_action)
        
        # Start with none selected
        self.none_action.setChecked(True)
        
    def create_box_icon(self):
        """Create icon for box shape"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw a rectangle
        painter.setPen(QPen(QColor(100, 150, 255), 2))
        painter.setBrush(QBrush(QColor(100, 150, 255, 50)))
        painter.drawRect(6, 6, 20, 20)
        
        painter.end()
        return QIcon(pixmap)
    
    def create_polygon_icon(self):
        """Create icon for polygon shape"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw a pentagon
        painter.setPen(QPen(QColor(255, 150, 100), 2))
        painter.setBrush(QBrush(QColor(255, 150, 100, 50)))
        
        # Draw a filled pentagon
        points = [
            QPoint(16, 6),   # Top
            QPoint(26, 12),  # Top-right
            QPoint(22, 24),  # Bottom-right
            QPoint(10, 24),  # Bottom-left
            QPoint(6, 12)    # Top-left
        ]
        
        polygon = QPolygon(points)
        painter.drawPolygon(polygon)
        
        painter.end()
        return QIcon(pixmap)
    
    def create_circle_icon(self):
        """Create icon for circle shape"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw a circle
        painter.setPen(QPen(QColor(100, 255, 150), 2))
        painter.setBrush(QBrush(QColor(100, 255, 150, 50)))
        painter.drawEllipse(6, 6, 20, 20)
        
        painter.end()
        return QIcon(pixmap)
    
    def on_shape_selected(self, shape_id):
        """Handle shape selection - manually uncheck others"""
        # Uncheck the none action
        self.none_action.setChecked(False)
        
        # Manually uncheck all other shape actions
        for sid, action in self.actions.items():
            if sid != shape_id:
                action.setChecked(False)
        
        # Set current shape
        self.current_shape = shape_id
        self.shape_changed.emit(shape_id)
        print(f"🔷 Shape selected: {shape_id}")
    
    def on_none_selected(self):
        """Handle none selection - deselect all shapes"""
        # Uncheck all shape actions
        for action in self.actions.values():
            action.setChecked(False)
        
        self.current_shape = None
        self.shape_deselected.emit()
        print("⬜ No shape selected")
    
    def get_current_shape(self):
        """Get currently selected shape (returns None if none selected)"""
        return self.current_shape
    
    def set_selected_shape(self, shape_type):
        """Set the selected shape programmatically (from canvas selection)"""
        if shape_type in self.actions and shape_type != "none":
            # Uncheck none action
            self.none_action.setChecked(False)
            
            # Manually set checked state
            for sid, action in self.actions.items():
                if sid == shape_type:
                    action.setChecked(True)
                else:
                    action.setChecked(False)
            self.current_shape = shape_type
        elif shape_type == "none":
            # Deselect all
            self.on_none_selected()