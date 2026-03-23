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
            ("T", "Stamp"),
            ("Ctrl+E", "Export")
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
        
        # Initialize UI
        self.setup_menu_bar()
        self.setup_shortcut_bar()
        self.setup_status_bar()
        self.setup_central_widget()
        
        # Initialize toolbar visibility for default YOLO mode
        # Do NOT call switch_mode() here — canvas.mode is already 'yolo'
        # and the guard `if mode == current_mode: return` would skip it.
        # Instead, set visibility directly since all buttons exist now.
        self.shape_btn_frame.setVisible(False)
        self.shape_btn_donut.setVisible(False)
        self.shape_btn_hollow_ellipse.setVisible(False)
        if hasattr(self, 'frame_action'):
            self.frame_action.setVisible(False)
        if hasattr(self, 'donut_action'):
            self.donut_action.setVisible(False)
        if hasattr(self, 'hollow_ellipse_action'):
            self.hollow_ellipse_action.setVisible(False)
        if hasattr(self, 'shortcut_bar'):
            self.shortcut_bar.set_item_visible('frame', False)
            self.shortcut_bar.set_item_visible('donut', False)
            self.shortcut_bar.set_item_visible('hollow_ellipse', False)

        # AUTOSAVE INTEGRATION
        from core.project_manager import ProjectManager
        self.project_manager = ProjectManager()
        self.canvas.annotation_changed.connect(self._on_annotation_changed)

    def _on_annotation_changed(self):
        """Called every time a shape is added, moved, deleted, or modified."""
        if not self.image_folder or self.current_image_index < 0:
            return
        filename = self.image_files[self.current_image_index]
        # Copy shapes to avoid mutations
        shapes_copy = list(self.canvas.shapes)
        self.project_manager.schedule_autosave(
            filename,
            shapes_copy,
            self.canvas.mode,
            self.canvas.image_width,
            self.canvas.image_height
        )
        
    def closeEvent(self, event):
        """Handle application close"""
        if hasattr(self, "project_manager"):
            self.project_manager.flush_autosave()
        super().closeEvent(event)
        
    # EXPORT INTEGRATION
    def open_export_dialog(self) -> None:
        if not self.image_folder or not self.image_files:
            return
        if hasattr(self, 'project_manager'):
            self.project_manager.flush_autosave()
            
        from core.export_manager import ExportManager
        from gui.export_dialog import ExportDialog
        
        manager = ExportManager(self.project_manager, self.class_manager)
        dlg = ExportDialog(self.project_manager, self.class_manager, self.image_folder, self.image_files, self.canvas.mode, self)
        dlg.set_export_manager(manager)
        dlg.exec_()
        
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
        
    def setup_menu_bar(self):
        """Create the menu bar with all menus and actions"""
        menubar = self.menuBar()
        
        # ===== FILE MENU =====
        file_menu = menubar.addMenu('&File')
        
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
        
        prev_action = QAction('Previous Image', self)
        prev_action.setShortcut('A')
        prev_action.triggered.connect(self.prev_image)
        shortcuts_menu.addAction(prev_action)
        
        next_action = QAction('Next Image', self)
        next_action.setShortcut('D')
        next_action.triggered.connect(self.next_image)
        shortcuts_menu.addAction(next_action)
        
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
        
        self.export_action = QAction('Export Annotations', self)
        self.export_action.setShortcut('Ctrl+E')
        self.export_action.triggered.connect(self.open_export_dialog)
        shortcuts_menu.addAction(self.export_action)
        
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
        
        # ===== EXPORT MENU =====
        export_menu = menubar.addMenu('E&xport')
        export_menu.addAction(self.export_action)

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
        self.class_panel.class_added.connect(self._on_classes_changed)
        self.class_panel.class_removed.connect(self._on_classes_changed)
        self.class_panel.class_edited.connect(self._on_classes_changed)
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
    
    def update_template_dropdown(self, select_name=None):
        """Refresh the template dropdown from canvas template_manager"""
        if not hasattr(self, 'canvas') or not hasattr(self.canvas, 'template_manager'):
            return
        
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItem("-- No template --")
        
        for name in self.canvas.template_manager.list_templates():
            self.template_combo.addItem(f"📋 {name}")
        
        self.template_combo.blockSignals(False)
        
        # Auto-select newly saved template so the user can see it was saved
        if select_name:
            idx = self.template_combo.findText(f"📋 {select_name}")
            if idx >= 0:
                self.template_combo.setCurrentIndex(idx)
    
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
        if not hasattr(self, 'canvas'): return
        current_mode = self.canvas.mode
        if mode == current_mode:
            return
            
        # AUTOSAVE INTEGRATION - check if current image has annotations at mode switch
        has_annotations = bool(self.canvas.shapes)
        coexist = False
        if has_annotations:
            from gui.mode_switch_dialog import ModeSwitchDialog
            from PyQt5.QtWidgets import QDialog
            dlg = ModeSwitchDialog(current_mode, mode, len(self.canvas.shapes), self)
            if dlg.exec_() != QDialog.Accepted:
                # Revert ratio buttons visually
                is_unet = (current_mode == 'unet')
                self.yolo_mode_action.setChecked(not is_unet)
                self.unet_mode_action.setChecked(is_unet)
                self.mode_btn_yolo.setChecked(not is_unet)
                self.mode_btn_unet.setChecked(is_unet)
                return
            coexist = dlg.coexist
            
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
            
        # After switching: update layer visibility in saved JSON
        if hasattr(self, "project_manager") and self.current_image_index >= 0:
            filename = self.image_files[self.current_image_index]
            self.project_manager.set_layer_visibility(filename, current_mode, coexist)
            self.project_manager.set_layer_visibility(filename, mode, True)
            
        # Reload annotations for the new target mode
        if hasattr(self, "project_manager") and self.current_image_index >= 0:
            self._restore_annotations(self.image_files[self.current_image_index])
            
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
        if not folder_path:
            return
            
        # AUTOSAVE INTEGRATION - flush current before switching folders
        if hasattr(self, "project_manager") and self.current_image_index >= 0:
            self.project_manager.flush_autosave()
            
        # Check if this folder already has a project
        project_data = self.project_manager.open_folder(folder_path, parent_widget=self)
        if project_data is None:
            return # user cancelled the resume dialog
            
        self.image_folder = folder_path
        
        # Restore classes FIRST, before loading images, so that when _restore_annotations
        # runs for the first image the class_manager is already populated.
        if project_data.get("classes"):
            self._restore_classes(project_data["classes"])
        else:
            # New project or empty classes - push current default classes to project
            self._on_classes_changed()
        
        self.load_images_from_folder(folder_path)
            
        # Restore mode
        saved_mode = project_data.get("active_mode", "yolo")
        self.switch_mode(saved_mode)
        
        self.status_bar.showMessage(f"Loaded: {folder_path}")
        
    def _restore_classes(self, classes_data):
        """Restore classes from saved project data. Completely replaces the class manager state."""
        self.class_manager.classes = {}
        self.class_manager.current_class_id = None
        for c in classes_data:
            from core.class_manager import ClassCategory
            cls = ClassCategory(name=c["name"], color=c["color"], class_id=c["id"])
            self.class_manager.classes[c["id"]] = cls
        # Auto-select the first class so the user can draw immediately
        all_cls = self.class_manager.get_all_classes()
        if all_cls:
            self.class_manager.current_class_id = all_cls[0].id
        if hasattr(self, 'class_panel'):
            self.class_panel.refresh_list()
        print(f"🎨 Restored {len(classes_data)} classes from project")
    
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
        # AUTOSAVE INTEGRATION - flush before loading new image
        if hasattr(self, "project_manager") and self.current_image_index >= 0:
            self.project_manager.flush_autosave()
            
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            image_path = os.path.join(self.image_folder, self.image_files[index])
            
            self.canvas.load_image(image_path) # this clears canvas.shapes
            
            # AUTOSAVE INTEGRATION - restore annotations after image loads
            if hasattr(self, "project_manager"):
                self._restore_annotations(self.image_files[index])
                
            self.file_list.setCurrentRow(index)
            self.update_image_counter()
            self.image_info_label.setText(self.image_files[index])
            self.setWindowTitle(f"Dual Annotator - {self.image_files[index]}")
            self.status_bar.showMessage(f"Loaded: {self.image_files[index]}", 2000)

    def _restore_annotations(self, filename):
        """Load saved annotations for an image and place them on canvas."""
        data = self.project_manager.load_image_annotations(filename)
        if data is None:
            return  # no saved annotations - canvas stays empty

        if data.get("hash_mismatch"):
            self._show_hash_warning()

        # Determine which layer to show based on current mode
        mode = self.canvas.mode
        layers = data.get("layers", {})
        
        # Load the visible layers
        w, h = self.canvas.image_width, self.canvas.image_height
        shapes = []
        for l_mode, layer in layers.items():
            if layer.get("visible", False):
                annotations = layer.get("annotations", [])
                for ann in annotations:
                    shape = self.project_manager.deserialize_shape(ann, w, h)
                    if shape:
                        shapes.append(shape)
                        
        self.canvas.shapes = shapes
        self.canvas.update()

    def _show_hash_warning(self):
        """Show a temporary hash mismatch warning label or banner"""
        print("⚠️ Image has changed since last annotation.")
        # Just update status bar for now, could be improved with an embedded QLabel widget
        self.status_bar.showMessage("WARNING: Image file has changed since last annotation.", 5000)
    
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
    
    def _on_classes_changed(self):
        """Called when classes are added, removed, or edited to keep project.json in sync."""
        if hasattr(self, "project_manager") and self.project_manager.project_data:
            classes = []
            for cls in self.class_manager.get_all_classes():
                classes.append({
                    "id": cls.id,
                    "name": cls.name,
                    "color": cls.color
                })
            self.project_manager.update_project_classes(classes)

    def show_about(self):
        """Show detailed About dialog"""
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QScrollArea,
                                     QWidget, QLabel, QPushButton)
        from PyQt5.QtCore import Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("About DualAnnotator")
        dlg.setFixedWidth(620)
        dlg.setMinimumHeight(500)
        dlg.setMaximumHeight(800)
        dlg.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #ffffff; }
            QScrollArea { border: none; background-color: #1e1e1e; }
            QWidget#scroll_content { background-color: #1e1e1e; }
            QLabel { color: #ffffff; background-color: transparent; }
            QPushButton {
                background-color: #2a4a6a;
                color: #ffffff;
                border: 1px solid #8ab4f8;
                border-radius: 4px;
                padding: 6px 24px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #3a5a7a; }
        """)

        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(0, 0, 0, 10)
        outer.setSpacing(0)

        # ── Scrollable content ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("scroll_content")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 24, 28, 16)
        layout.setSpacing(0)

        html = """
<style>
  body  { font-family: Arial, sans-serif; font-size: 13px;
          color: #e0e0e0; background: #1e1e1e; margin: 0; padding: 0; }
  h1    { font-size: 20px; color: #8ab4f8; margin: 0 0 4px 0; }
  h2    { font-size: 14px; color: #8ab4f8; margin: 18px 0 6px 0;
          border-bottom: 1px solid #333; padding-bottom: 4px; }
  h3    { font-size: 13px; color: #aac4f0; margin: 12px 0 4px 0; }
  p     { margin: 4px 0 8px 0; line-height: 1.6; color: #cccccc; }
  ul    { margin: 2px 0 8px 0; padding-left: 20px; }
  li    { margin-bottom: 3px; line-height: 1.6; color: #cccccc; }
  code  { background: #2a2a2a; color: #8ab4f8; padding: 1px 5px;
          border-radius: 3px; font-size: 12px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 10px; }
  th    { background: #2a3a5a; color: #8ab4f8; padding: 6px 10px;
          text-align: left; font-size: 12px; }
  td    { padding: 5px 10px; border-bottom: 1px solid #2a2a2a;
          font-size: 12px; color: #cccccc; }
  tr:hover td { background: #252525; }
  .tag  { background: #2a4a2a; color: #6fcf97; padding: 1px 6px;
          border-radius: 3px; font-size: 11px; }
  .warn { color: #f0b429; }
  .dim  { color: #888888; font-size: 11px; }
</style>

<h1>DualAnnotator &nbsp;<span class="dim">v1.0.0</span></h1>
<p>A desktop annotation tool for building YOLO object detection and
U-Net segmentation datasets. Built with Python and PyQt5.</p>

<!-- ═══════════════════════════════════════════════════════ -->
<h2>Two Modes</h2>

<h3>YOLO Mode — Object Detection</h3>
<p>Draw bounding boxes around objects. Each box is saved with a class label
and exported as a normalised <code>cx cy w h</code> line in a <code>.txt</code>
file — the standard format for YOLOv5, YOLOv8, and YOLOv11 training.</p>

<h3>UNet Mode — Segmentation</h3>
<p>Draw precise shapes that follow object boundaries. Supports solid shapes
(polygon, bezier, circle, ellipse) and hollow ring shapes (frame, donut,
hollow ellipse) for annotating ring-like objects such as seals, caps, and
gaskets. Switch modes via the <b>Mode</b> menu or the toolbar buttons.</p>

<p class="warn">⚠ &nbsp;Switching modes on an image that already has
annotations shows a dialog — choose <b>Keep both</b> to show both layers
simultaneously, or <b>Hide previous</b> to work in one mode at a time.
Both layers are always saved — nothing is deleted when you switch.</p>

<!-- ═══════════════════════════════════════════════════════ -->
<h2>Shape Tools</h2>
<table>
  <tr><th>Key</th><th>Tool</th><th>How to draw</th><th>Mode</th></tr>
  <tr><td><code>B</code></td><td>Box</td>
      <td>Click and drag</td><td>YOLO + UNet</td></tr>
  <tr><td><code>P</code></td><td>Polygon</td>
      <td>Click points, Enter or click first point to close</td>
      <td>UNet</td></tr>
  <tr><td><code>Q</code></td><td>Bezier curve</td>
      <td>Click anchor points, Enter to close. Drag midpoint handles to curve edges</td>
      <td>UNet</td></tr>
  <tr><td><code>C</code></td><td>Circle</td>
      <td>Click centre, drag outward</td><td>UNet</td></tr>
  <tr><td><code>E</code></td><td>Ellipse</td>
      <td>Click centre, drag to set both radii</td><td>UNet</td></tr>
  <tr><td><code>F</code></td><td>Frame (hollow rect)</td>
      <td>Click and drag. Drag inner handles to adjust wall thickness</td>
      <td>UNet</td></tr>
  <tr><td><code>O</code></td><td>Donut (hollow circle)</td>
      <td>Click centre, drag outward. Drag inner handles to adjust hole size</td>
      <td>UNet</td></tr>
  <tr><td><code>H</code></td><td>Hollow ellipse</td>
      <td>Click centre, drag. Drag inner handles independently</td>
      <td>UNet</td></tr>
  <tr><td><code>N</code></td><td>Selection</td>
      <td>Click to select, drag to move</td><td>YOLO + UNet</td></tr>
  <tr><td><code>T</code></td><td>Stamp template</td>
      <td>Select template from dropdown, click to place</td>
      <td>YOLO + UNet</td></tr>
</table>

<!-- ═══════════════════════════════════════════════════════ -->
<h2>Hollow Shapes — Inner / Outer Offset</h2>
<p>Any drawn shape can be made hollow via right-click:</p>
<ul>
  <li><b>Right-click a shape → Create Inner Shape</b> — adds an inner cutout
      inset from the outer boundary. A slider controls the gap with a live
      dashed preview.</li>
  <li><b>Right-click a shape → Create Outer Shape</b> — expands a new outer
      boundary around the existing shape.</li>
</ul>
<p>After creating the hollow pair, both boundaries have independent resize
handles. The inner shape is always constrained to stay inside the outer
with a minimum gap. For YOLO export, hollow shapes export the
<b>outer bounding box only</b>.</p>

<!-- ═══════════════════════════════════════════════════════ -->
<h2>Templates (Stamp Tool)</h2>
<p>Save any shape — including hollow pairs — as a reusable template:</p>
<ul>
  <li>Draw and configure a shape, then <b>right-click → Save as Template</b></li>
  <li>Give it a name. The template thumbnail shows the actual shape.</li>
  <li>Select it from the template dropdown, press <code>T</code>, and
      click the canvas to stamp it at the original size.</li>
  <li>Templates are session-only — they reset when the app closes.</li>
</ul>

<!-- ═══════════════════════════════════════════════════════ -->
<h2>Autosave &amp; Project Files</h2>
<p>Every annotation change is saved automatically — there is no Save button.
Saves happen 800 ms after the last action so they never interrupt drawing.</p>
<p>Opening an image folder creates a hidden <code>.dualannotator/</code>
folder inside it:</p>
<ul>
  <li><code>project.json</code> — class definitions, mode, image order</li>
  <li><code>annotations/image_001.jpg.json</code> — one file per image
      containing both YOLO and UNet layers</li>
</ul>
<p>When you reopen the same folder the app detects the existing project and
asks whether to <b>Resume</b> (restore all annotations) or
<b>Start Fresh</b> (delete everything and begin again).</p>
<p class="warn">⚠ &nbsp;If a source image file is replaced or renamed after
annotation, the status bar will show a hash mismatch warning when that image
is loaded. Annotations are preserved — they may just not align to the new image.</p>

<!-- ═══════════════════════════════════════════════════════ -->
<h2>Exporting (File → Export Annotations, Ctrl+E)</h2>
<p>Exports the saved annotations — never the live canvas — so you can export
at any time without interrupting your work.</p>

<h3>YOLO format</h3>
<p>Produces one <code>.txt</code> label file per image, a <code>data.yaml</code>
training config, and a <code>classes.txt</code> index. All coordinates are
normalised to <code>[0.0, 1.0]</code>. Format per line:
<code>&lt;class_id&gt; &lt;cx&gt; &lt;cy&gt; &lt;w&gt; &lt;h&gt;</code></p>

<h3>COCO format</h3>
<p>Produces <code>instances_train.json</code> (and val/test) with pixel-space
bounding boxes in standard COCO annotation format.</p>

<h3>Pascal VOC format</h3>
<p>Produces one <code>.xml</code> file per image with pixel bounding boxes in
Pascal VOC format, compatible with older training pipelines.</p>

<h3>Delta export</h3>
<p>When <b>Only export changes since last export</b> is checked, images that
have not changed since the previous export run are skipped. Only new or
modified images are processed. The output folder uses a timestamped subfolder
name (e.g. <code>yolo_2026-03-14_18-44/</code>) so each export run is
preserved separately.</p>

<h3>Train / Val / Test split</h3>
<p>Annotated images are randomly distributed across train, val, and test
folders according to the configured percentages. A fixed random seed
(default 42) ensures the same image lands in the same split every time you
export, which is essential for reproducible training results.</p>

<!-- ═══════════════════════════════════════════════════════ -->
<h2>All Keyboard Shortcuts</h2>
<table>
  <tr><th>Key</th><th>Action</th><th>Key</th><th>Action</th></tr>
  <tr><td><code>B</code></td><td>Box tool</td>
      <td><code>Ctrl+Z</code></td><td>Undo</td></tr>
  <tr><td><code>P</code></td><td>Polygon tool</td>
      <td><code>Ctrl+Y</code></td><td>Redo</td></tr>
  <tr><td><code>Q</code></td><td>Bezier tool</td>
      <td><code>Ctrl+C</code></td><td>Copy shape</td></tr>
  <tr><td><code>C</code></td><td>Circle tool</td>
      <td><code>Ctrl+V</code></td><td>Paste shape</td></tr>
  <tr><td><code>E</code></td><td>Ellipse tool</td>
      <td><code>Del</code></td><td>Delete selected</td></tr>
  <tr><td><code>F</code></td><td>Frame tool</td>
      <td><code>Ctrl+E</code></td><td>Export annotations</td></tr>
  <tr><td><code>O</code></td><td>Donut tool</td>
      <td><code>Ctrl+Shift+O</code></td><td>Open image folder</td></tr>
  <tr><td><code>H</code></td><td>Hollow ellipse</td>
      <td><code>Ctrl+F</code></td><td>Fit to window</td></tr>
  <tr><td><code>N</code></td><td>Selection mode</td>
      <td><code>+</code></td><td>Zoom in</td></tr>
  <tr><td><code>T</code></td><td>Stamp tool</td>
      <td><code>-</code></td><td>Zoom out</td></tr>
  <tr><td><code>A</code></td><td>Previous image</td>
      <td><code>Space</code></td><td>Toggle pan mode</td></tr>
  <tr><td><code>D</code></td><td>Next image</td>
      <td><code>Enter</code></td><td>Finish polygon / bezier</td></tr>
  <tr><td><code>Ctrl+drag</code></td><td>Clone shape</td>
      <td><code>Esc</code></td><td>Cancel operation</td></tr>
</table>

<p class="dim" style="margin-top: 12px;">
Built with Python 3.9+ and PyQt5 &nbsp;·&nbsp;
Pillow &nbsp;·&nbsp; Shapely &nbsp;·&nbsp; PyYAML
</p>
"""

        label = QLabel()
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setText(html)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(label)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        # ── Close button ──
        btn = QPushButton("Close")
        btn.setFixedWidth(100)
        btn.clicked.connect(dlg.accept)
        btn_row = QWidget()
        btn_layout = QVBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 10, 0)
        btn_layout.addWidget(btn, alignment=Qt.AlignRight)
        outer.addWidget(btn_row)

        dlg.exec_()