# gui/main_window.py
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QMenu, QToolBar, 
    QStatusBar, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QMessageBox, QFileDialog, QSplitter,
    QListWidget, QListWidgetItem, QAbstractItemView, 
    QComboBox, QFrame, QPushButton, QShortcut, QTextEdit, QLineEdit, QScrollArea
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon, QKeySequence, QFont, QColor, QPalette, QPolygonF
import os

from gui.canvas import AnnotationCanvas
from gui.class_panel import ClassPanel
from core.class_manager import ClassManager

class ToolButton(QPushButton):
    """Custom tool button with QPainter-drawn icons for consistency"""
    def __init__(self, icon_name, tooltip=None):
        super().__init__("")
        self.icon_name = icon_name
        self.setFixedSize(50, 50)
        self.setCheckable(True)
        self.setToolTip(tooltip or "")
        self.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border: 1px solid #5a5a5a;
            }
            QPushButton:checked {
                background-color: #2a4a6a;
                border: 2px solid #8ab4f8;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
            QPushButton:disabled {
                color: #666666;
                background-color: #1e1e1e;
                border: 1px solid #2a2a2a;
            }
        """)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        from PyQt5.QtGui import QPainter, QPen, QBrush, QPainterPath
        from PyQt5.QtCore import QRectF, QPointF
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # All icons: white lines, 2px, on the button center
        pen = QPen(QColor(220, 220, 220), 2)
        if self.isChecked():
            pen = QPen(QColor(138, 180, 248), 2)  # blue when checked
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        cx, cy = self.width() // 2, self.height() // 2
        s = 12  # half-size of icon area
        name = self.icon_name
        
        if name == "box":
            painter.drawRect(cx - s, cy - s, s * 2, int(s * 1.4))
        
        elif name == "polygon":
            pts = [QPointF(cx, cy - s), QPointF(cx + s, cy - 2),
                   QPointF(cx + 7, cy + s), QPointF(cx - 7, cy + s),
                   QPointF(cx - s, cy - 2)]
            painter.drawPolygon(QPolygonF(pts))
        
        elif name == "circle":
            painter.drawEllipse(QPointF(cx, cy), s, s)
        
        elif name == "ellipse":
            painter.drawEllipse(QPointF(cx, cy), s, int(s * 0.65))
        
        elif name == "frame":
            # Outer rect
            painter.drawRect(cx - s, cy - s, s * 2, s * 2)
            # Inner rect (smaller)
            i = 5
            painter.drawRect(cx - s + i, cy - s + i, s * 2 - i * 2, s * 2 - i * 2)
        
        elif name == "donut":
            painter.drawEllipse(QPointF(cx, cy), s, s)
            painter.drawEllipse(QPointF(cx, cy), s * 0.5, s * 0.5)
        
        elif name == "hollow_ellipse":
            painter.drawEllipse(QPointF(cx, cy), s, int(s * 0.65))
            painter.drawEllipse(QPointF(cx, cy), s * 0.5, int(s * 0.65 * 0.5))
        
        elif name == "none":
            # Arrow cursor icon
            path = QPainterPath()
            path.moveTo(cx - 6, cy - s)
            path.lineTo(cx - 6, cy + 8)
            path.lineTo(cx - 1, cy + 3)
            path.lineTo(cx + 5, cy + 10)
            path.lineTo(cx + 8, cy + 7)
            path.lineTo(cx + 2, cy + 0)
            path.lineTo(cx + 7, cy - 3)
            path.closeSubpath()
            painter.setBrush(QBrush(pen.color()))
            painter.drawPath(path)
        
        elif name == "template":
            # Scissors icon — two overlapping circles + lines
            painter.drawLine(cx - 4, cy - s, cx + 4, cy + 2)
            painter.drawLine(cx + 4, cy - s, cx - 4, cy + 2)
            painter.drawEllipse(QPointF(cx - 5, cy + 7), 5, 5)
            painter.drawEllipse(QPointF(cx + 5, cy + 7), 5, 5)
        
        elif name == "yolo":
            # Grid/detection icon — rectangle with crosshair
            painter.drawRect(cx - s, cy - s + 2, s * 2, s * 2 - 4)
            painter.drawLine(cx, cy - s + 2, cx, cy + s - 2)
            painter.drawLine(cx - s, cy, cx + s, cy)
        
        elif name == "unet":
            # Mask/segmentation icon — filled blob
            path = QPainterPath()
            path.addEllipse(QPointF(cx - 3, cy - 2), 8, 10)
            path.addEllipse(QPointF(cx + 4, cy + 1), 6, 7)
            painter.setBrush(QBrush(QColor(pen.color().red(), pen.color().green(), pen.color().blue(), 60)))
            painter.drawPath(path)
        
        elif name == "zoom_in":
            # Magnifying glass with +
            painter.drawEllipse(QPointF(cx - 2, cy - 2), 8, 8)
            painter.drawLine(cx + 4, cy + 4, cx + s, cy + s)
            painter.drawLine(cx - 6, cy - 2, cx + 2, cy - 2)
            painter.drawLine(cx - 2, cy - 6, cx - 2, cy + 2)
        
        elif name == "zoom_out":
            # Magnifying glass with -
            painter.drawEllipse(QPointF(cx - 2, cy - 2), 8, 8)
            painter.drawLine(cx + 4, cy + 4, cx + s, cy + s)
            painter.drawLine(cx - 6, cy - 2, cx + 2, cy - 2)
        
        elif name == "fit":
            # Four corners expanding
            c = 5
            # Top-left corner
            painter.drawLine(cx - s, cy - s, cx - s + c, cy - s)
            painter.drawLine(cx - s, cy - s, cx - s, cy - s + c)
            # Top-right
            painter.drawLine(cx + s, cy - s, cx + s - c, cy - s)
            painter.drawLine(cx + s, cy - s, cx + s, cy - s + c)
            # Bottom-left
            painter.drawLine(cx - s, cy + s, cx - s + c, cy + s)
            painter.drawLine(cx - s, cy + s, cx - s, cy + s - c)
            # Bottom-right
            painter.drawLine(cx + s, cy + s, cx + s - c, cy + s)
            painter.drawLine(cx + s, cy + s, cx + s, cy + s - c)
        
        elif name == "prev":
            # Left arrow
            painter.drawLine(cx + 5, cy - s + 2, cx - 5, cy)
            painter.drawLine(cx - 5, cy, cx + 5, cy + s - 2)
        
        elif name == "next":
            # Right arrow
            painter.drawLine(cx - 5, cy - s + 2, cx + 5, cy)
            painter.drawLine(cx + 5, cy, cx - 5, cy + s - 2)
        
        else:
            # Fallback: draw the text
            painter.drawText(self.rect(), Qt.AlignCenter, name)
        
        painter.end()

class ShortcutBar(QFrame):
    """Horizontal bar showing keyboard shortcuts"""
    def __init__(self):
        super().__init__()
        self.setFixedHeight(30)
        self.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-bottom: 1px solid #3a3a3a;
            }
            QLabel {
                color: #aaa;
                font-size: 11px;
                padding: 0 10px;
            }
            QLabel#shortcut {
                color: #ffffff;
                font-weight: bold;
                font-size: 12px;
                background-color: #3b82f6; /* Modern Blue */
                padding: 2px 6px;
                border-radius: 4px;
                margin-right: 2px;
            }
            QLabel#desc {
                color: #a1a1aa;
                font-size: 12px;
                font-weight: 500;
                margin-right: 15px;
            }
        """)
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 0, 10, 0)
        self.layout.setSpacing(5)
        
        self.shortcut_items = {} # key -> (key_label, desc_label, separator_label)
        
        shortcuts = [
            ("B", "Box"),
            ("P", "Polygon"),
            ("Q", "Bezier"),
            ("C", "Circle"),
            ("E", "Ellipse"),
            ("F", "Frame"),
            ("O", "Donut"),
            ("H", "Hollow Ellipse"),
            ("T", "Stamp")
        ]
        
        for i, (key, desc) in enumerate(shortcuts):
            key_label = QLabel(key)
            key_label.setObjectName("shortcut")
            self.layout.addWidget(key_label)
            
            desc_label = QLabel(desc)
            desc_label.setObjectName("desc")
            self.layout.addWidget(desc_label)
            
            sep = None
            # Add subtle separator
            if i < len(shortcuts) - 1:
                sep = QLabel("|")
                sep.setStyleSheet("color: #3f3f46; font-size: 10px; margin-right: 10px;")
                self.layout.addWidget(sep)
            
            self.shortcut_items[desc.lower().replace(" ", "_")] = (key_label, desc_label, sep)
        
        self.layout.addStretch()

    def set_item_visible(self, item_key, visible):
        """Toggle visibility of a shortcut item group"""
        if item_key in self.shortcut_items:
            key_label, desc_label, sep = self.shortcut_items[item_key]
            key_label.setVisible(visible)
            desc_label.setVisible(visible)
            if sep:
                sep.setVisible(visible)


class MainWindow(QMainWindow):
    """Main application window with redesigned layout"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Dual Annotator - New Project")
        self.setGeometry(100, 100, 1600, 1000)
        self.setMinimumSize(1200, 700)
        
        # Set dark theme for the entire application
        self.set_dark_theme()
        
        # Application state
        self.current_file = None
        self.is_modified = False
        self.image_folder = None
        self.image_files = []
        self.current_image_index = -1
        
        # Initialize class manager
        self.class_manager = ClassManager()
        self.setup_default_classes()
        
        # Initialize UI
        self.setup_menu_bar()
        self.setup_shortcut_bar()
        self.setup_status_bar()
        self.setup_central_widget()
        
        # Set keyboard shortcuts for navigation
        QShortcut(QKeySequence('A'), self, self.prev_image)
        QShortcut(QKeySequence('D'), self, self.next_image)   

        # Initialize with YOLO mode constraints
        self.switch_mode('yolo')

        
    def set_dark_theme(self):
        """Set dark theme for the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QMenuBar {
                background-color: #2b2b2b;
                color: #ffffff;
                border-bottom: 1px solid #3a3a3a;
            }
            QMenuBar::item {
                background-color: transparent;
                color: #ffffff;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #3a3a3a;
            }
            QMenu {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3a3a3a;
            }
            QMenu::item {
                background-color: transparent;
                color: #ffffff;
                padding: 5px 20px;
            }
            QMenu::item:selected {
                background-color: #3a3a3a;
            }
            QStatusBar {
                background-color: #2b2b2b;
                color: #ffffff;
                border-top: 1px solid #3a3a3a;
            }
            QLabel {
                color: #ffffff;
            }
            QListWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #3a3a3a;
            }
            QListWidget::item:selected {
                background-color: #2a4a6a;
                color: #ffffff;
                border-left: 3px solid #8ab4f8;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
            QPushButton {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
            padding: 5px;
        }
        QPushButton:hover {
                background-color: #3a3a3a;
            }
            QFrame {
                color: #ffffff;
            }
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLineEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 4px;
            }
        """)
        
    def setup_default_classes(self):
        """Add some default classes for testing"""
        try:
            self.class_manager.add_class("Car", "#FF6B6B")
            self.class_manager.add_class("Person", "#4ECDC4")
            self.class_manager.add_class("Bicycle", "#45B7D1")
            self.class_manager.add_class("Dog", "#96CEB4")
            print("Success: Default classes added")
        except Exception as e:
            print(f"Error: Could not add default classes: {e}")
        
    def setup_menu_bar(self):
        """Create the menu bar with all menus and actions"""
        menubar = self.menuBar()
        
        # ===== FILE MENU =====
        file_menu = menubar.addMenu('&File')
        
        # New Project
        new_action = QAction('&New Project', self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        # Open Project
        open_action = QAction('&Open Project', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        # Save Project
        save_action = QAction('&Save Project', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Open Image Folder
        open_folder_action = QAction('&Open Image Folder...', self)
        open_folder_action.setShortcut('Ctrl+Shift+O')
        open_folder_action.triggered.connect(self.open_image_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # ===== EDIT MENU =====
        edit_menu = menubar.addMenu('&Edit')
        
        undo_action = QAction('&Undo', self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction('&Redo', self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        copy_action = QAction('&Copy', self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_selected)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction('&Paste', self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(self.paste_shape)
        edit_menu.addAction(paste_action)
        
        delete_action = QAction('&Delete', self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self.delete_selected)
        edit_menu.addAction(delete_action)
        
        # ===== VIEW MENU =====
        view_menu = menubar.addMenu('&View')
        
        zoom_in_action = QAction('Zoom &In', self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction('Zoom &Out', self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        fit_action = QAction('&Fit to Window', self)
        fit_action.setShortcut('Ctrl+F')
        fit_action.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_action)
        
        # ===== MODE MENU =====
        mode_menu = menubar.addMenu('&Mode')
        
        self.yolo_mode_action = QAction('&YOLO Detection', self)
        self.yolo_mode_action.setCheckable(True)
        self.yolo_mode_action.setChecked(True)
        self.yolo_mode_action.triggered.connect(lambda: self.switch_mode('yolo'))
        mode_menu.addAction(self.yolo_mode_action)
        
        self.unet_mode_action = QAction('&U-Net Segmentation', self)
        self.unet_mode_action.setCheckable(True)
        self.unet_mode_action.triggered.connect(lambda: self.switch_mode('unet'))
        mode_menu.addAction(self.unet_mode_action)
        
        # ===== SHORTCUTS MENU =====
        shortcuts_menu = menubar.addMenu('&Shortcuts')
        
        # Navigation Section
        nav_title = shortcuts_menu.addAction('🔍 NAVIGATION')
        nav_title.setEnabled(False)
        shortcuts_menu.addSeparator()
        

        

        
        pan_action = QAction('Toggle Pan Mode', self)
        pan_action.setShortcut('Space')
        pan_action.triggered.connect(self.toggle_pan_mode)
        shortcuts_menu.addAction(pan_action)
        
        zoom_in_shortcut = QAction('Zoom In', self)
        zoom_in_shortcut.setShortcut('+')
        zoom_in_shortcut.triggered.connect(self.zoom_in)
        shortcuts_menu.addAction(zoom_in_shortcut)
        
        zoom_out_shortcut = QAction('Zoom Out', self)
        zoom_out_shortcut.setShortcut('-')
        zoom_out_shortcut.triggered.connect(self.zoom_out)
        shortcuts_menu.addAction(zoom_out_shortcut)
        
        fit_shortcut = QAction('Fit to Window', self)
        fit_shortcut.setShortcut('Ctrl+F')
        fit_shortcut.triggered.connect(self.fit_to_window)
        shortcuts_menu.addAction(fit_shortcut)
        
        shortcuts_menu.addSeparator()
        
        # Shape Tools Section
        shape_title = shortcuts_menu.addAction('🖌️ SHAPE TOOLS')
        shape_title.setEnabled(False)
        shortcuts_menu.addSeparator()
        
        box_shortcut = QAction('Box Tool', self)
        box_shortcut.setShortcut('B')
        box_shortcut.triggered.connect(lambda: self.set_shape_type('box'))
        shortcuts_menu.addAction(box_shortcut)
        
        polygon_shortcut = QAction('Polygon Tool', self)
        polygon_shortcut.setShortcut('P')
        polygon_shortcut.triggered.connect(lambda: self.set_shape_type('polygon'))
        shortcuts_menu.addAction(polygon_shortcut)
        
        bezier_shortcut = QAction('Bezier Curve', self)
        bezier_shortcut.setShortcut('Q')
        bezier_shortcut.triggered.connect(lambda: self.set_shape_type('bezier_polygon'))
        shortcuts_menu.addAction(bezier_shortcut)
        
        circle_shortcut = QAction('Circle Tool', self)
        circle_shortcut.setShortcut('C')
        circle_shortcut.triggered.connect(lambda: self.set_shape_type('circle'))
        shortcuts_menu.addAction(circle_shortcut)
        
        ellipse_shortcut = QAction('Ellipse Tool', self)
        ellipse_shortcut.setShortcut('E')
        ellipse_shortcut.triggered.connect(lambda: self.set_shape_type('ellipse'))
        shortcuts_menu.addAction(ellipse_shortcut)
        
        self.frame_action = QAction('Frame Tool', self)
        self.frame_action.setShortcut('F')
        self.frame_action.triggered.connect(lambda: self.set_shape_type('frame'))
        shortcuts_menu.addAction(self.frame_action)
        
        self.donut_action = QAction('Donut Tool', self)
        self.donut_action.setShortcut('O')
        self.donut_action.triggered.connect(lambda: self.set_shape_type('donut'))
        shortcuts_menu.addAction(self.donut_action)
        
        self.hollow_ellipse_action = QAction('Hollow Ellipse Tool', self)
        self.hollow_ellipse_action.setShortcut('H')
        self.hollow_ellipse_action.triggered.connect(lambda: self.set_shape_type('hollow_ellipse'))
        shortcuts_menu.addAction(self.hollow_ellipse_action)

        
        template_shortcut = QAction('Template Tool', self)
        template_shortcut.setShortcut('T')
        template_shortcut.triggered.connect(lambda: self.set_shape_type('template'))
        shortcuts_menu.addAction(template_shortcut)
        
        none_shortcut = QAction('None (Selection Mode)', self)
        none_shortcut.setShortcut('N')
        none_shortcut.triggered.connect(lambda: self.set_shape_type(None))
        shortcuts_menu.addAction(none_shortcut)
        
        finish_polygon_shortcut = QAction('Finish Polygon', self)
        finish_polygon_shortcut.setShortcut('Enter')
        finish_polygon_shortcut.triggered.connect(lambda: self.canvas.finish_polygon() if hasattr(self, 'canvas') else None)
        shortcuts_menu.addAction(finish_polygon_shortcut)
        
        shortcuts_menu.addSeparator()
        
        # Editing Section
        edit_title = shortcuts_menu.addAction('✏️ EDITING')
        edit_title.setEnabled(False)
        shortcuts_menu.addSeparator()
        
        delete_shortcut = QAction('Delete Selected', self)
        delete_shortcut.setShortcut('Del')
        delete_shortcut.triggered.connect(self.delete_selected)
        shortcuts_menu.addAction(delete_shortcut)
        
        copy_shortcut = QAction('Copy', self)
        copy_shortcut.setShortcut('Ctrl+C')
        copy_shortcut.triggered.connect(self.copy_selected)
        shortcuts_menu.addAction(copy_shortcut)
        
        paste_shortcut = QAction('Paste', self)
        paste_shortcut.setShortcut('Ctrl+V')
        paste_shortcut.triggered.connect(self.paste_shape)
        shortcuts_menu.addAction(paste_shortcut)
        
        undo_shortcut = QAction('Undo', self)
        undo_shortcut.setShortcut('Ctrl+Z')
        undo_shortcut.triggered.connect(self.undo)
        shortcuts_menu.addAction(undo_shortcut)
        
        redo_shortcut = QAction('Redo', self)
        redo_shortcut.setShortcut('Ctrl+Y')
        redo_shortcut.triggered.connect(self.redo)
        shortcuts_menu.addAction(redo_shortcut)
        
        cancel_shortcut = QAction('Cancel Operation', self)
        cancel_shortcut.setShortcut('Esc')
        cancel_shortcut.triggered.connect(self.cancel_operation)
        shortcuts_menu.addAction(cancel_shortcut)
        
        shortcuts_menu.addSeparator()
        
        # New Feature Shortcuts
        feature_title = shortcuts_menu.addAction('🆕 NEW FEATURES')
        feature_title.setEnabled(False)
        shortcuts_menu.addSeparator()
        
        scale_help = QAction('Scale Pattern/Shape (Shift+Scroll)', self)
        scale_help.setEnabled(False)
        shortcuts_menu.addAction(scale_help)
        
        drag_copy_help = QAction('Clone Shape (Ctrl + Drag)', self)
        drag_copy_help.setEnabled(False)
        shortcuts_menu.addAction(drag_copy_help)
        
        undo_point_help = QAction('Undo Last Point (Right-Click Menu)', self)
        undo_point_help.setEnabled(False)
        shortcuts_menu.addAction(undo_point_help)
        
        # ===== HELP MENU =====
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_shortcut_bar(self):
        """Create horizontal shortcut bar below menu"""
        self.shortcut_bar = ShortcutBar()
        
    def setup_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Mode label
        self.mode_label = QLabel("Mode: YOLO")
        self.mode_label.setStyleSheet("color: #ffffff;")
        self.status_bar.addPermanentWidget(self.mode_label)
        
        # Image info label
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setStyleSheet("color: #ffffff;")
        self.status_bar.addPermanentWidget(self.image_info_label)
        
        # Position label
        self.position_label = QLabel("X: 0, Y: 0")
        self.position_label.setStyleSheet("color: #ffffff;")
        self.status_bar.addPermanentWidget(self.position_label)
        
        # Image counter
        self.counter_label = QLabel("0/0")
        self.counter_label.setStyleSheet("color: #ffffff;")
        self.status_bar.addPermanentWidget(self.counter_label)
        
        self.status_bar.showMessage("Ready")
        
    def setup_central_widget(self):
        """Create the redesigned central widget"""
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main vertical layout
        main_vertical = QVBoxLayout(central)
        main_vertical.setContentsMargins(0, 0, 0, 0)
        main_vertical.setSpacing(0)
        
        # Add shortcut bar
        main_vertical.addWidget(self.shortcut_bar)
        
        # Main horizontal layout for content
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # ===== LEFT SIDEBAR - VERTICAL TOOLBAR =====
        left_toolbar = self.create_vertical_toolbar()
        left_toolbar.setFixedWidth(70)
        content_layout.addWidget(left_toolbar)
        
        # ===== CENTER - CANVAS =====
        self.canvas = AnnotationCanvas()
        self.canvas.set_parent_window(self)  # IMPORTANT: Set parent reference
        self.canvas.set_class_manager(self.class_manager)
        self.canvas.position_changed.connect(self.update_position)
        self.canvas.shape_selected.connect(self.on_canvas_shape_selected)
        content_layout.addWidget(self.canvas, 7)  # 70% stretch factor
        
        # ===== RIGHT PANEL - CLASSES + FILE BROWSER =====
        right_panel = self.create_right_panel()
        right_panel.setFixedWidth(280)
        content_layout.addWidget(right_panel)
        
        main_vertical.addLayout(content_layout)
        
    def create_vertical_toolbar(self):
        """Create vertical toolbar on the left side with icons only"""
        toolbar_widget = QWidget()
        toolbar_widget.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border-right: 1px solid #333;
            }
            QLabel {
                color: #aaa;
                font-size: 10px;
                font-weight: bold;
                margin-top: 10px;
                margin-bottom: 5px;
            }
        """)
        
        layout = QVBoxLayout(toolbar_widget)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignTop)
        
        # Mode section
        mode_label = QLabel("MODE")
        mode_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(mode_label)
        
        self.mode_btn_yolo = ToolButton("yolo", "YOLO Detection Mode")
        self.mode_btn_yolo.setChecked(True)
        self.mode_btn_yolo.clicked.connect(lambda: self.switch_mode('yolo'))
        layout.addWidget(self.mode_btn_yolo)
        
        self.mode_btn_unet = ToolButton("unet", "U-Net Segmentation Mode")
        self.mode_btn_unet.clicked.connect(lambda: self.switch_mode('unet'))
        layout.addWidget(self.mode_btn_unet)
        
        layout.addWidget(self.create_separator())
        
        # Shape tools section
        shape_label = QLabel("SHAPES")
        shape_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(shape_label)
        
        self.shape_btn_box = ToolButton("box", "Box (B)")
        self.shape_btn_box.setChecked(True)
        self.shape_btn_box.clicked.connect(lambda: self.set_shape_type('box'))
        layout.addWidget(self.shape_btn_box)
        
        self.shape_btn_polygon = ToolButton("polygon", "Polygon (P)")
        self.shape_btn_polygon.clicked.connect(lambda: self.set_shape_type('polygon'))
        layout.addWidget(self.shape_btn_polygon)
        
        self.shape_btn_bezier = ToolButton("polygon", "Bezier Curve (Q)") # reuse polygon icon for now
        self.shape_btn_bezier.clicked.connect(lambda: self.set_shape_type('bezier_polygon'))
        layout.addWidget(self.shape_btn_bezier)
        
        self.shape_btn_circle = ToolButton("circle", "Circle (C)")
        self.shape_btn_circle.clicked.connect(lambda: self.set_shape_type('circle'))
        layout.addWidget(self.shape_btn_circle)
        
        self.shape_btn_ellipse = ToolButton("ellipse", "Ellipse (E)")
        self.shape_btn_ellipse.clicked.connect(lambda: self.set_shape_type('ellipse'))
        layout.addWidget(self.shape_btn_ellipse)
        
        # Ring shapes
        self.shape_btn_frame = ToolButton("frame", "Frame (F)")
        self.shape_btn_frame.clicked.connect(lambda: self.set_shape_type('frame'))
        layout.addWidget(self.shape_btn_frame)
        
        self.shape_btn_donut = ToolButton("donut", "Donut (O)")
        self.shape_btn_donut.clicked.connect(lambda: self.set_shape_type('donut'))
        layout.addWidget(self.shape_btn_donut)
        
        self.shape_btn_hollow_ellipse = ToolButton("hollow_ellipse", "Hollow Ellipse (H)")
        self.shape_btn_hollow_ellipse.clicked.connect(lambda: self.set_shape_type('hollow_ellipse'))
        layout.addWidget(self.shape_btn_hollow_ellipse)
        
        self.shape_btn_none = ToolButton("none", "Selection (N)")
        self.shape_btn_none.clicked.connect(lambda: self.set_shape_type(None))
        layout.addWidget(self.shape_btn_none)
        
        layout.addWidget(self.create_separator())
        
        # Template section
        template_label = QLabel("TEMPLATE")
        template_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(template_label)
        
        self.shape_btn_template = ToolButton("template", "Place Template / Stamp")
        self.shape_btn_template.setCheckable(True)
        self.shape_btn_template.clicked.connect(lambda: self.set_shape_type('stamp'))
        layout.addWidget(self.shape_btn_template)
        
        # Template dropdown
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(50)
        self.template_combo.setMinimumHeight(32)
        self.template_combo.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 12px;
                min-height: 28px;
            }
            QComboBox:hover {
                border: 1px solid #4a7ab5;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: #ffffff;
                selection-background-color: #3a5a8a;
                font-size: 12px;
                padding: 4px;
                min-width: 180px;
            }
        """)
        self.template_combo.addItem("-- No template --")
        self.template_combo.currentIndexChanged.connect(self.on_template_selected)
        layout.addWidget(self.template_combo)
        
        layout.addWidget(self.create_separator())
        
        # View tools section
        view_label = QLabel("VIEW")
        view_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(view_label)
        
        btn_zoom_in = ToolButton("zoom_in", "Zoom In (+)")
        btn_zoom_in.setCheckable(False)
        btn_zoom_in.clicked.connect(self.zoom_in)
        layout.addWidget(btn_zoom_in)
        
        btn_zoom_out = ToolButton("zoom_out", "Zoom Out (-)")
        btn_zoom_out.setCheckable(False)
        btn_zoom_out.clicked.connect(self.zoom_out)
        layout.addWidget(btn_zoom_out)
        
        btn_fit = ToolButton("fit", "Fit to Window (Ctrl+F)")
        btn_fit.setCheckable(False)
        btn_fit.clicked.connect(self.fit_to_window)
        layout.addWidget(btn_fit)
        
        layout.addWidget(self.create_separator())
        
        # Navigation section
        nav_label = QLabel("NAVIGATE")
        nav_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(nav_label)
        
        btn_prev = ToolButton("prev", "Previous Image (A)")
        btn_prev.setCheckable(False)
        btn_prev.clicked.connect(self.prev_image)
        layout.addWidget(btn_prev)
        
        btn_next = ToolButton("next", "Next Image (D)")
        btn_next.setCheckable(False)
        btn_next.clicked.connect(self.next_image)
        layout.addWidget(btn_next)
        
        layout.addStretch()
        
        return toolbar_widget
    
    def create_separator(self):
        """Create a horizontal separator line"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #333; max-height: 1px; margin: 5px;")
        return line
    
    def create_right_panel(self):
        """Create right panel with classes and file browser"""
        panel = QWidget()
        panel.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border-left: 1px solid #333;
            }
            QLabel {
                color: #ffffff;
                font-weight: bold;
                padding: 5px;
                font-size: 12px;
            }
            QLabel#section {
                color: #aaa;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(10)
        
        # ===== CLASSES SECTION =====
        classes_label = QLabel("CLASSES")
        classes_label.setObjectName("section")
        classes_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(classes_label)
        
        self.class_panel = ClassPanel(self.class_manager)
        layout.addWidget(self.class_panel)
        
        layout.addWidget(self.create_separator())
        
        # ===== IMAGE FILES SECTION =====
        files_label = QLabel("IMAGE FILES")
        files_label.setObjectName("section")
        files_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(files_label)
        
        # File list widget
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.itemClicked.connect(self.on_file_selected)
        layout.addWidget(self.file_list)
        
        # Import button
        import_btn = QPushButton("📂 Import Images")
        import_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """)
        import_btn.clicked.connect(self.open_image_folder)
        layout.addWidget(import_btn)
        
        return panel
    
    def set_shape_type(self, shape_type):
        """Set the current shape type and update button states"""
        if hasattr(self, 'canvas'):
            self.canvas.set_shape_type(shape_type)
            
            # Update button states
            self.shape_btn_box.setChecked(shape_type == 'box')
            self.shape_btn_polygon.setChecked(shape_type == 'polygon')
            self.shape_btn_bezier.setChecked(shape_type == 'bezier_polygon')
            self.shape_btn_circle.setChecked(shape_type == 'circle')
            self.shape_btn_ellipse.setChecked(shape_type == 'ellipse')
            self.shape_btn_frame.setChecked(shape_type == 'frame')
            self.shape_btn_donut.setChecked(shape_type == 'donut')
            self.shape_btn_hollow_ellipse.setChecked(shape_type == 'hollow_ellipse')
            self.shape_btn_template.setChecked(shape_type in ('template', 'stamp'))
            self.shape_btn_none.setChecked(shape_type == 'none' or shape_type is None)
            
            if shape_type and shape_type != 'none':
                self.status_bar.showMessage(f"Drawing tool: {shape_type}", 1000)
                # if switching to stamp mode, make sure a template name is loaded
                if shape_type == 'stamp' and hasattr(self, 'template_combo') and hasattr(self, 'canvas'):
                    text = self.template_combo.currentText()
                    if "No template" not in text:
                        name = text.replace("📋 ", "")
                        self.canvas.stamp_template_name = name
                        self.status_bar.showMessage(f"Stamp mode: '{name}' — click center + drag to scale", 3000)
                    else:
                        self.status_bar.showMessage("Select a template from the dropdown to use the Stamp tool", 3000)
            else:
                self.status_bar.showMessage("Selection mode - click on shapes to select them", 1000)
    
    def update_template_dropdown(self):
        """Refresh the template dropdown from canvas template_manager"""
        if not hasattr(self, 'canvas') or not hasattr(self.canvas, 'template_manager'):
            return
        
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItem("-- No template --")
        
        for name in self.canvas.template_manager.list_templates():
            self.template_combo.addItem(f"📋 {name}")
        
        self.template_combo.blockSignals(False)
    
    def on_template_selected(self, index):
        """Handle template selection from dropdown"""
        if index <= 0:
            # "No template" selected
            if hasattr(self, 'canvas'):
                self.canvas.stamp_template_name = None
            return
        
        # Get template name (strip emoji prefix)
        text = self.template_combo.currentText()
        name = text.replace("📋 ", "")
        
        if hasattr(self, 'canvas'):
            self.canvas.stamp_template_name = name
            self.set_shape_type('stamp')
            self.status_bar.showMessage(f"Stamp mode: '{name}' — click center + drag to scale", 3000)
    
    def on_class_selected(self, class_id):
        """Handle class selection"""
        cls = self.class_manager.get_class(class_id)
        if cls:
            self.status_bar.showMessage(f"Selected class: {cls.name}", 2000)
    
    def on_canvas_shape_selected(self, shape_type):
        """Handle shape selection from canvas to update toolbar"""
        # Always switch to 'none' mode when any shape is selected
        self.shape_btn_box.setChecked(False)
        self.shape_btn_polygon.setChecked(False)
        self.shape_btn_circle.setChecked(False)
        self.shape_btn_ellipse.setChecked(False)
        self.shape_btn_frame.setChecked(False)
        self.shape_btn_donut.setChecked(False)
        self.shape_btn_hollow_ellipse.setChecked(False)
        self.shape_btn_template.setChecked(False)
        self.shape_btn_none.setChecked(True)
        
        # Update canvas to 'none' mode
        if hasattr(self, 'canvas'):
            self.canvas.set_shape_type('none')
        
        if shape_type != "none":
            self.status_bar.showMessage(f"{shape_type} selected - 'None' mode activated", 2000)
    
    def switch_mode(self, mode):
        """Switch between YOLO and U-Net modes"""
        is_unet = (mode == 'unet')
        
        # Update menu actions
        self.yolo_mode_action.setChecked(not is_unet)
        self.unet_mode_action.setChecked(is_unet)
        
        # Update mode buttons
        self.mode_btn_yolo.setChecked(not is_unet)
        self.mode_btn_unet.setChecked(is_unet)
        
        # Update status bar
        self.mode_label.setText(f"Mode: {'U-Net' if is_unet else 'YOLO'}")
        
        # Update canvas mode
        if hasattr(self, 'canvas'):
            self.canvas.set_mode(mode)
        
        # Show/Hide hollow shape options
        # Buttons
        self.shape_btn_frame.setVisible(is_unet)
        self.shape_btn_donut.setVisible(is_unet)
        self.shape_btn_hollow_ellipse.setVisible(is_unet)
        
        # Menu Shortcuts
        if hasattr(self, 'frame_action'):
            self.frame_action.setVisible(is_unet)
        if hasattr(self, 'donut_action'):
            self.donut_action.setVisible(is_unet)
        if hasattr(self, 'hollow_ellipse_action'):
            self.hollow_ellipse_action.setVisible(is_unet)
            
        # Shortcut Bar Items
        if hasattr(self, 'shortcut_bar'):
            self.shortcut_bar.set_item_visible('frame', is_unet)
            self.shortcut_bar.set_item_visible('donut', is_unet)
            self.shortcut_bar.set_item_visible('hollow_ellipse', is_unet)
            
        # If in YOLO and a hollow shape was selected, switch back to Box
        if not is_unet and hasattr(self, 'canvas'):
            current_tool = getattr(self.canvas, 'current_shape_type', None)
            if current_tool in ('frame', 'donut', 'hollow_ellipse'):
                self.set_shape_type('box')
                
        self.status_bar.showMessage(f"Switched to {mode.upper()} mode", 2000)

    
    def open_image_folder(self):
        """Open a folder containing images"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if folder_path:
            self.image_folder = folder_path
            self.load_images_from_folder(folder_path)
            self.status_bar.showMessage(f"Loaded images from: {folder_path}")
    
    def load_images_from_folder(self, folder_path):
        """Load all image files from the selected folder"""
        self.file_list.clear()
        self.image_files = []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        
        try:
            all_files = os.listdir(folder_path)
            
            for file in sorted(all_files):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    self.image_files.append(file)
                    
            for file in self.image_files:
                item = QListWidgetItem(file)
                self.file_list.addItem(item)
                
            self.update_image_counter()
            
            if self.image_files:
                self.load_image(0)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load images: {str(e)}")
    
    def load_image(self, index):
        """Load image at specified index"""
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            image_path = os.path.join(self.image_folder, self.image_files[index])
            
            self.canvas.load_image(image_path)
            self.file_list.setCurrentRow(index)
            self.update_image_counter()
            self.image_info_label.setText(self.image_files[index])
            self.setWindowTitle(f"Dual Annotator - {self.image_files[index]}")
            self.status_bar.showMessage(f"Loaded: {self.image_files[index]}", 2000)
    
    def on_file_selected(self, item):
        """Handle file selection from list"""
        row = self.file_list.row(item)
        self.load_image(row)
    
    def next_image(self):
        """Load next image"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.load_image(self.current_image_index + 1)
        else:
            self.status_bar.showMessage("Already at last image", 1000)
    
    def prev_image(self):
        """Load previous image"""
        if self.image_files and self.current_image_index > 0:
            self.load_image(self.current_image_index - 1)
        else:
            self.status_bar.showMessage("Already at first image", 1000)
    
    def update_image_counter(self):
        """Update the image counter in status bar"""
        if self.image_files:
            current = self.current_image_index + 1
            total = len(self.image_files)
            self.counter_label.setText(f"{current}/{total}")
        else:
            self.counter_label.setText("0/0")
    
    def update_position(self, x, y):
        """Update cursor position in status bar"""
        self.position_label.setText(f"X: {x}, Y: {y}")
    
    # ===== DELEGATION METHODS =====
    def zoom_in(self):
        if hasattr(self, 'canvas'):
            self.canvas.zoom_in()
    
    def zoom_out(self):
        if hasattr(self, 'canvas'):
            self.canvas.zoom_out()
    
    def fit_to_window(self):
        if hasattr(self, 'canvas'):
            self.canvas.fit_to_window()
    
    def copy_selected(self):
        if hasattr(self, 'canvas'):
            self.canvas.copy_selected()
    
    def paste_shape(self):
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'pixmap') and self.canvas.pixmap:
            # Paste at cursor location
            from PyQt5.QtGui import QCursor
            pos = self.canvas.mapFromGlobal(QCursor.pos())
            # Ensure it's inside the canvas area roughly
            if not self.canvas.rect().contains(pos):
                pos = self.canvas.rect().center()
            self.canvas.start_paste(pos)
    
    def delete_selected(self):
        if hasattr(self, 'canvas'):
            self.canvas.delete_selected()
    
    def undo(self):
        if hasattr(self, 'canvas'):
            self.canvas.undo()
        self.status_bar.showMessage("Undo", 1000)
    
    def redo(self):
        if hasattr(self, 'canvas'):
            self.canvas.redo()
        self.status_bar.showMessage("Redo", 1000)
    
    def toggle_pan_mode(self):
        """Toggle pan mode in canvas"""
        if hasattr(self, 'canvas'):
            self.canvas.pan_mode = not self.canvas.pan_mode
            if self.canvas.pan_mode:
                self.canvas.original_cursor = self.canvas.cursor()
                self.canvas.setCursor(Qt.OpenHandCursor)
                self.status_bar.showMessage("Pan mode ON", 2000)
            else:
                self.canvas.setCursor(self.canvas.original_cursor or Qt.ArrowCursor)
                self.status_bar.showMessage("Pan mode OFF", 2000)

    def cancel_operation(self):
        """Cancel current operation in canvas"""
        if hasattr(self, 'canvas'):
            from PyQt5.QtGui import QKeyEvent
            from PyQt5.QtCore import QEvent
            event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
            self.canvas.keyPressEvent(event)
    
    # ===== PROJECT METHODS =====
    def new_project(self):
        self.status_bar.showMessage("Creating new project...")
    
    def open_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Project Files (*.json)"
        )
        if file_path:
            self.status_bar.showMessage(f"Opened: {file_path}")
    
    def save_project(self):
        if self.current_file:
            self.status_bar.showMessage(f"Saving to: {self.current_file}")
        else:
            self.save_project_as()
    
    def save_project_as(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "Project Files (*.json)"
        )
        if file_path:
            self.current_file = file_path
            self.status_bar.showMessage(f"Saved to: {file_path}")
    
    def show_about(self):
        QMessageBox.about(
            self,
            "About Dual Annotator",
            "<h2>Dual Annotator v1.0.0</h2>"
            "<p>A unified annotation tool for YOLO and U-Net datasets.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>YOLO mode: Bounding box annotation</li>"
            "<li>U-Net mode: Segmentation masks</li>"
            "<li>Copy-paste with resize</li>"
            "<li>Polygon and circle shapes</li>"
            "</ul>"
            "<p>Built with PyQt5 and Python 3.11</p>"
        )