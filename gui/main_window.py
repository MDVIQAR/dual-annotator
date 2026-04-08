# gui/main_window.py
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QMenu, QToolBar, 
    QStatusBar, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QMessageBox, QFileDialog, QSplitter,
    QListWidget, QListWidgetItem, QAbstractItemView, 
    QComboBox, QFrame, QPushButton, QShortcut, QTextEdit, QLineEdit, QScrollArea,
    QTabWidget
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
        
        elif name == "bezier":
            path = QPainterPath()
            path.moveTo(cx - s + 2, cy + s - 2)
            path.cubicTo(cx - s + 2, cy - s, cx + s - 2, cy + s, cx + s - 2, cy - s + 2)
            painter.drawPath(path)
        
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
        
        elif name == "concentric":  # CONCENTRIC INTEGRATION
            # Three concentric rings icon
            painter.drawEllipse(QPointF(cx, cy), s, s)
            painter.drawEllipse(QPointF(cx, cy), int(s * 0.65), int(s * 0.65))
            painter.drawEllipse(QPointF(cx, cy), int(s * 0.3), int(s * 0.3))
        
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
            
        elif name == "delete_template":
            # Trash can icon
            painter.setPen(QPen(QColor(255, 138, 138), 2))
            painter.drawLine(cx - 5, cy - s + 4, cx + 5, cy - s + 4) # lid
            painter.drawLine(cx - 3, cy - s + 4, cx - 4, cy + s - 2) # left edge
            painter.drawLine(cx + 3, cy - s + 4, cx + 4, cy + s - 2) # right edge
            painter.drawLine(cx - 4, cy + s - 2, cx + 4, cy + s - 2) # bottom
            painter.drawLine(cx - 2, cy - s + 4, cx - 2, cy - s + 1) # handle part 1
            painter.drawLine(cx + 2, cy - s + 4, cx + 2, cy - s + 1) # handle part 2
            painter.drawLine(cx - 2, cy - s + 1, cx + 2, cy - s + 1) # handle top
        
        elif name == "auto_detect":
            # Magnifying glass with plus (auto-detect icon)
            painter.drawEllipse(cx - 8, cy - 8, 13, 13)
            painter.drawLine(cx + 4, cy + 4, cx + s, cy + s)
            painter.drawLine(cx - 4, cy - 1, cx + 4, cy - 1)
            painter.drawLine(cx, cy - 5, cx, cy + 3)
            
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
            ("Tab", "Next Unannotated"),
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


        # AUTOSAVE INTEGRATION
        from core.project_manager import ProjectManager
        self.project_manager = ProjectManager()
        self.canvas.annotation_changed.connect(self._on_annotation_changed)

        # Keyboard shortcuts (Tab handled by menu action to avoid ambiguity)

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
        # Update live shape count in status bar
        count = len([s for s in self.canvas.shapes if getattr(s, 'hollow_role', None) != 'inner'])
        self.shape_count_label.setText(f"✏️ {count} Shape{'s' if count != 1 else ''}")
        # Auto-update status badge to in_progress if shapes exist but annotated wasn't manually set
        current_status = self.project_manager.get_image_status(filename)
        if shapes_copy and current_status == "unannotated":
            self.project_manager.set_image_status(filename, "in_progress")
            self.update_ui_status_badge(filename, "in_progress")
        # Update annotation statistics panel
        self.update_annotation_stats()

    def update_annotation_stats(self):
        """Update the annotation stats label in the right panel."""
        if not hasattr(self, 'stats_label') or not hasattr(self, 'project_manager'):
            return
        stats = self.project_manager.get_project_stats()
        total = len(self.image_files) if self.image_files else 0
        annotated = stats.get("annotated_images", 0)
        in_prog = stats.get("in_progress_images", 0)
        skipped = stats.get("skipped_images", 0)
        
        # Per-class shape counts on the current image
        class_counts = ""
        if hasattr(self, 'canvas') and self.canvas.shapes:
            counts = {}
            for s in self.canvas.shapes:
                if getattr(s, 'hollow_role', None) == 'inner':
                    continue
                cid = getattr(s, 'class_id', None)
                cls_obj = self.class_manager.get_class(cid) if cid else None
                name = cls_obj.name if cls_obj else "?"
                counts[name] = counts.get(name, 0) + 1
            if counts:
                parts = [f"{name}: {n}" for name, n in counts.items()]
                class_counts = " \u00b7 ".join(parts)
        
        text = f"\ud83d\udcca {annotated}/{total} done \u00b7 {in_prog} wip \u00b7 {skipped} skip"
        if class_counts:
            text += f"\n\ud83c\udfaf {class_counts}"
        self.stats_label.setText(text)

    def closeEvent(self, event):
        """Handle application close and save templates"""
        if hasattr(self, "project_manager"):
            self.project_manager.flush_autosave()
            
            # Persist templates to the project folder
            if self.project_manager.project_dir:
                tmpl_path = os.path.join(self.project_manager.project_dir, 'templates.json')
                self.canvas.template_manager.save_to_file(tmpl_path)
                
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
            /* ---- Top-level Tab Bar ---- */
            QTabWidget::pane {
                border: none;
                background-color: #1e1e1e;
            }
            QTabBar {
                background-color: #16161e;
                border-bottom: 1px solid #2a2a3a;
            }
            QTabBar::tab {
                background-color: #1e1e2e;
                color: #888;
                padding: 10px 24px;
                margin: 0px;
                border: none;
                border-bottom: 3px solid transparent;
                font-size: 13px;
                font-weight: bold;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                color: #8ab4f8;
                background-color: #1e1e1e;
                border-bottom: 3px solid #8ab4f8;
            }
            QTabBar::tab:hover:!selected {
                color: #aaa;
                background-color: #252535;
                border-bottom: 3px solid #444;
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
        
        # Initialize persistent editing actions used by the Shortcuts menu
        self.undo_action = QAction('&Undo', self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self.undo)
        
        self.redo_action = QAction('&Redo', self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.triggered.connect(self.redo)
        
        self.copy_action = QAction('&Copy', self)
        self.copy_action.setShortcut(QKeySequence.Copy)
        self.copy_action.triggered.connect(self.copy_selected)
        
        self.paste_action = QAction('&Paste', self)
        self.paste_action.setShortcut(QKeySequence.Paste)
        self.paste_action.triggered.connect(self.paste_shape)
        
        self.delete_action = QAction('&Delete', self)
        self.delete_action.setShortcut(QKeySequence.Delete)
        self.delete_action.triggered.connect(self.delete_selected)
        
        self.fit_action = QAction('&Fit to Window', self)
        self.fit_action.setShortcut('Ctrl+F')
        self.fit_action.triggered.connect(self.fit_to_window)
        
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
        
        # CONCENTRIC INTEGRATION
        self.concentric_mode_action = QAction('&Concentric Zones', self)
        self.concentric_mode_action.setCheckable(True)
        self.concentric_mode_action.triggered.connect(
            lambda: self.switch_mode('concentric'))
        mode_menu.addAction(self.concentric_mode_action)
        
        # ===== SHORTCUTS MENU =====
        shortcuts_menu = menubar.addMenu('&Shortcuts')
        
        # 1. Navigation & View Sub-menu
        nav_menu = shortcuts_menu.addMenu('🔍 Navigation & View')
        
        prev_action = QAction('Previous Image', self)
        prev_action.setShortcut('A')
        prev_action.triggered.connect(self.prev_image)
        nav_menu.addAction(prev_action)
        
        next_action = QAction('Next Image', self)
        next_action.setShortcut('D')
        next_action.triggered.connect(self.next_image)
        nav_menu.addAction(next_action)
        
        next_un_action = QAction('Next Unannotated', self)
        next_un_action.setShortcut('Tab')
        next_un_action.setShortcutContext(Qt.ApplicationShortcut)
        next_un_action.triggered.connect(self.next_unannotated)
        nav_menu.addAction(next_un_action)
        
        pan_action = QAction('Toggle Pan Mode', self)
        pan_action.setShortcut('Space')
        pan_action.triggered.connect(self.toggle_pan_mode)
        nav_menu.addAction(pan_action)
        
        zoom_in_shortcut = QAction('Zoom In', self)
        zoom_in_shortcut.setShortcut('+')
        zoom_in_shortcut.triggered.connect(self.zoom_in)
        nav_menu.addAction(zoom_in_shortcut)
        
        zoom_out_shortcut = QAction('Zoom Out', self)
        zoom_out_shortcut.setShortcut('-')
        zoom_out_shortcut.triggered.connect(self.zoom_out)
        nav_menu.addAction(zoom_out_shortcut)
        
        nav_menu.addAction(self.fit_action)
        
        
        # 2. Drawing Tools Sub-menu
        shape_menu = shortcuts_menu.addMenu('🖌️ Drawing Tools')
        
        box_shortcut = QAction('Box Tool', self)
        box_shortcut.setShortcut('B')
        box_shortcut.triggered.connect(lambda: self.set_shape_type('box'))
        shape_menu.addAction(box_shortcut)
        
        polygon_shortcut = QAction('Polygon Tool', self)
        polygon_shortcut.setShortcut('P')
        polygon_shortcut.triggered.connect(lambda: self.set_shape_type('polygon'))
        shape_menu.addAction(polygon_shortcut)
        
        bezier_shortcut = QAction('Bezier Curve', self)
        bezier_shortcut.setShortcut('Q')
        bezier_shortcut.triggered.connect(lambda: self.set_shape_type('bezier_polygon'))
        shape_menu.addAction(bezier_shortcut)
        
        circle_shortcut = QAction('Circle Tool', self)
        circle_shortcut.setShortcut('C')
        circle_shortcut.triggered.connect(lambda: self.set_shape_type('circle'))
        shape_menu.addAction(circle_shortcut)
        
        ellipse_shortcut = QAction('Ellipse Tool', self)
        ellipse_shortcut.setShortcut('E')
        ellipse_shortcut.triggered.connect(lambda: self.set_shape_type('ellipse'))
        shape_menu.addAction(ellipse_shortcut)
        
        self.frame_action = QAction('Frame Tool (Hollow)', self)
        self.frame_action.setShortcut('F')
        self.frame_action.triggered.connect(lambda: self.set_shape_type('frame'))
        shape_menu.addAction(self.frame_action)
        
        self.donut_action = QAction('Donut Tool (Hollow)', self)
        self.donut_action.setShortcut('O')
        self.donut_action.triggered.connect(lambda: self.set_shape_type('donut'))
        shape_menu.addAction(self.donut_action)
        
        self.hollow_ellipse_action = QAction('Hollow Ellipse Tool', self)
        self.hollow_ellipse_action.setShortcut('H')
        self.hollow_ellipse_action.triggered.connect(lambda: self.set_shape_type('hollow_ellipse'))
        shape_menu.addAction(self.hollow_ellipse_action)
        
        template_shortcut = QAction('Template Stamp', self)
        template_shortcut.setShortcut('T')
        template_shortcut.triggered.connect(lambda: self.set_shape_type('template'))
        shape_menu.addAction(template_shortcut)
        
        none_shortcut = QAction('Pointer (Selection Mode)', self)
        none_shortcut.setShortcut('N')
        none_shortcut.triggered.connect(lambda: self.set_shape_type(None))
        shape_menu.addAction(none_shortcut)
        
        # 3. Canvas Editing Sub-menu
        edit_tab = shortcuts_menu.addMenu('✏️ Canvas Editing')
        
        finish_polygon_shortcut = QAction('Finish Polygon Shape', self)
        finish_polygon_shortcut.setShortcut('Enter')
        finish_polygon_shortcut.triggered.connect(lambda: self.canvas.finish_polygon() if hasattr(self, 'canvas') else None)
        edit_tab.addAction(finish_polygon_shortcut)
        
        cancel_shortcut = QAction('Cancel Current Operation', self)
        cancel_shortcut.setShortcut('Esc')
        cancel_shortcut.triggered.connect(self.cancel_operation)
        edit_tab.addAction(cancel_shortcut)
        
        edit_tab.addSeparator()
        
        edit_tab.addAction(self.undo_action)
        edit_tab.addAction(self.redo_action)
        
        edit_tab.addSeparator()
        
        edit_tab.addAction(self.copy_action)
        edit_tab.addAction(self.paste_action)
        
        edit_tab.addSeparator()
        
        edit_tab.addAction(self.delete_action)
        
        delete_all_shortcut = QAction('Wipe Entire Canvas', self)
        delete_all_shortcut.setShortcut('Ctrl+Del')
        delete_all_shortcut.triggered.connect(self.delete_all_annotations)
        edit_tab.addAction(delete_all_shortcut)
        
        # 4. Advanced Interaction & Missing Features
        feature_tab = shortcuts_menu.addMenu('⚙️ Advanced Binding References')
        
        class_mgr_help = QAction('Open Class Manager (Shift+C)', self)
        class_mgr_help.setEnabled(False)
        feature_tab.addAction(class_mgr_help)
        
        class_equip_help = QAction('Quick Equip Class (Numbers 1-9)', self)
        class_equip_help.setEnabled(False)
        feature_tab.addAction(class_equip_help)
        
        scale_help = QAction('Scale Shape (Shift + Scroll Wheel)', self)
        scale_help.setEnabled(False)
        feature_tab.addAction(scale_help)
        
        drag_copy_help = QAction('Clone Selection (Ctrl + Drag Shape)', self)
        drag_copy_help.setEnabled(False)
        feature_tab.addAction(drag_copy_help)
        
        undo_point_help = QAction('Undo Mid-Draw Point (Ctrl + Z)', self)
        undo_point_help.setEnabled(False)
        feature_tab.addAction(undo_point_help)
        
        multiselect_help = QAction('Multi-Select Shapes (Shift + Left Click)', self)
        multiselect_help.setEnabled(False)
        feature_tab.addAction(multiselect_help)
        
        nudge_help = QAction('Macro Nudge Shape (Shift + Arrows)', self)
        nudge_help.setEnabled(False)
        feature_tab.addAction(nudge_help)
        
        cutout_help = QAction('Create Hollow Inner Shape (Right-Click)', self)
        cutout_help.setEnabled(False)
        feature_tab.addAction(cutout_help)
        
        feature_tab.addSeparator()
        
        self.export_action = QAction('Open Data Exporter (Ctrl+E)', self)
        self.export_action.setShortcut('Ctrl+E')
        self.export_action.triggered.connect(self.open_export_dialog)
        feature_tab.addAction(self.export_action)
        
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
        
        # Live shape count label
        self.shape_count_label = QLabel("✏️ 0 Shapes")
        self.shape_count_label.setStyleSheet("color: #8ab4f8; font-weight: bold; padding: 0 8px;")
        self.status_bar.addPermanentWidget(self.shape_count_label)
        
        # Image counter
        self.counter_label = QLabel("0/0")
        self.counter_label.setStyleSheet("color: #ffffff;")
        self.status_bar.addPermanentWidget(self.counter_label)

        # Nudge step control
        from PyQt5.QtWidgets import QSpinBox
        nudge_container = QWidget()
        nudge_layout = QHBoxLayout(nudge_container)
        nudge_layout.setContentsMargins(6, 0, 6, 0)
        nudge_layout.setSpacing(4)
        nudge_lbl = QLabel("Nudge:")
        nudge_lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.nudge_step_spin = QSpinBox()
        self.nudge_step_spin.setRange(1, 100)
        self.nudge_step_spin.setValue(1)
        self.nudge_step_spin.setSuffix(" px")
        self.nudge_step_spin.setFixedWidth(72)
        self.nudge_step_spin.setToolTip(
            "Arrow key nudge step (px).\nShift+Arrow = 10x this value."
        )
        self.nudge_step_spin.setStyleSheet("""
            QSpinBox {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 11px;
            }
            QSpinBox:hover { border: 1px solid #4a7ab5; }
            QSpinBox::up-button, QSpinBox::down-button { width: 14px; }
        """)
        self.nudge_step_spin.valueChanged.connect(self._on_nudge_step_changed)
        nudge_layout.addWidget(nudge_lbl)
        nudge_layout.addWidget(self.nudge_step_spin)
        self.status_bar.addPermanentWidget(nudge_container)

        self.status_bar.showMessage("Ready")
        
    def setup_central_widget(self):
        """Create the tabbed central widget"""
        # ── Top-level tab widget ──
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)  # cleaner look
        self.setCentralWidget(self.tab_widget)
        
        # ── Tab 1: Annotate ──
        self.annotate_tab = QWidget()
        self._build_annotate_tab(self.annotate_tab)
        self.tab_widget.addTab(self.annotate_tab, "✏️  Annotate")
        
        # ── Tab 2: RAW → PNG ──
        from gui.tabs.raw_to_png_tab import RawToPngTab
        self.raw_to_png_tab = RawToPngTab()
        self.tab_widget.addTab(self.raw_to_png_tab, "🖼️  RAW → PNG")

    def _build_annotate_tab(self, container):
        """Build the annotation workspace inside *container*."""
        # Main vertical layout
        main_vertical = QVBoxLayout(container)
        main_vertical.setContentsMargins(0, 0, 0, 0)
        main_vertical.setSpacing(0)
        
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
        
        # Link canvas reference into class panel for shape reassignment
        self.class_panel._canvas = self.canvas
        
        main_vertical.addLayout(content_layout)
        
    def _open_template_matching(self, shape):
        if not hasattr(self, '_auto_panel'):
            from gui.auto_annotate_panel import AutoAnnotatePanel
            self._auto_panel = AutoAnnotatePanel(self.canvas, self.canvas)
        self._auto_panel.template_shapes = [shape]
        self._auto_panel._update_template_ui()
        self._auto_panel.show_panel()

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
        
        # CONCENTRIC INTEGRATION
        self.mode_btn_concentric = ToolButton("concentric", "Concentric Zone Mode")
        self.mode_btn_concentric.clicked.connect(lambda: self.switch_mode('concentric'))
        layout.addWidget(self.mode_btn_concentric)
        
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
        
        self.shape_btn_bezier = ToolButton("bezier", "Bezier Curve (Q)")
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
        
        # Delete selected template button
        self.btn_delete_template = ToolButton("delete_template", "Delete Selected Template")
        self.btn_delete_template.setCheckable(False)
        self.btn_delete_template.clicked.connect(self.delete_current_template)
        layout.addWidget(self.btn_delete_template)
        
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
            QSplitter::handle {
                background-color: #333;
                height: 2px;
                margin: 4px 0;
            }
        """)
        
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(8)
        
        # ===== UPPER SECTION (CLASSES + STATS) =====
        upper_container = QWidget()
        upper_container.setStyleSheet("border-left: none;") # remove global border-left inside splitter
        upper_layout = QVBoxLayout(upper_container)
        upper_layout.setContentsMargins(8, 10, 8, 5)
        upper_layout.setSpacing(10)
        
        self.class_panel = ClassPanel(self.class_manager)
        self.class_panel.class_added.connect(self._on_classes_changed)
        self.class_panel.class_removed.connect(self._on_classes_changed)
        self.class_panel.class_edited.connect(self._on_classes_changed)
        self.class_panel.class_selected.connect(self.on_class_selected)
        self.class_panel.classes_reordered.connect(self._on_classes_changed)
        upper_layout.addWidget(self.class_panel, 1)
        
        upper_layout.addWidget(self.create_separator())
        
        # ANNOTATION STATS
        self.stats_label = QLabel("📊 0/0 annotated")
        self.stats_label.setStyleSheet("color: #8ab4f8; font-size: 12px; font-weight: bold; padding: 8px 4px; border: none;")
        self.stats_label.setWordWrap(True)
        self.stats_label.setMinimumHeight(50)
        upper_layout.addWidget(self.stats_label)
        
        # ===== LOWER SECTION (IMAGE FILES) =====
        lower_container = QWidget()
        lower_container.setStyleSheet("border-left: none;")
        lower_layout = QVBoxLayout(lower_container)
        lower_layout.setContentsMargins(8, 5, 8, 10)
        lower_layout.setSpacing(10)
        
        lower_layout.addWidget(self.create_separator())
        
        files_label = QLabel("IMAGE FILES")
        files_label.setObjectName("section")
        files_label.setAlignment(Qt.AlignLeft)
        files_label.setStyleSheet("border: none;")
        lower_layout.addWidget(files_label)
        
        # Search Box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 Search images...")
        self.search_box.textChanged.connect(self.filter_image_list)
        self.search_box.setStyleSheet("""
            QLineEdit { background-color: #2b2b2b; color: #ffffff; border: 1px solid #3a3a3a; border-radius: 3px; padding: 4px; font-size: 11px; }
            QLineEdit:focus { border: 1px solid #8ab4f8; }
        """)
        lower_layout.addWidget(self.search_box)
        
        # File list widget
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.itemClicked.connect(self.on_file_selected)
        lower_layout.addWidget(self.file_list, 1) # Set stretch to 1 here
        
        # Image Action Buttons
        image_actions_layout = QHBoxLayout()
        image_actions_layout.setSpacing(5)
        
        self.skip_btn = QPushButton("⏭️ Skip")
        self.skip_btn.setStyleSheet("""
            QPushButton { background-color: #3b2b2b; color: #ff8a8a; border: 1px solid #4a3a3a; border-radius: 3px; padding: 6px; font-weight: bold; font-size: 11px; }
            QPushButton:hover { background-color: #5a3a3a; }
        """)
        self.skip_btn.clicked.connect(self.mark_image_skipped)
        image_actions_layout.addWidget(self.skip_btn)
        
        self.mark_done_btn = QPushButton("✅ Mark Done")
        self.mark_done_btn.setStyleSheet("""
            QPushButton { background-color: #2b3b2b; color: #8aff8a; border: 1px solid #3a4a3a; border-radius: 3px; padding: 6px; font-weight: bold; font-size: 11px; }
            QPushButton:hover { background-color: #3a5a3a; }
        """)
        self.mark_done_btn.clicked.connect(self.mark_image_done)
        image_actions_layout.addWidget(self.mark_done_btn)
        
        lower_layout.addLayout(image_actions_layout)
        
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
        lower_layout.addWidget(import_btn)
        
        splitter.addWidget(upper_container)
        splitter.addWidget(lower_container)
        splitter.setStretchFactor(0, 1) # Upper section starts smaller
        splitter.setStretchFactor(1, 2) # Lower section starts larger
        
        main_layout.addWidget(splitter)
        
        return panel
        

    def on_class_selected(self, class_id):
        """Handle class selection from down the component tree."""
        cls = self.class_manager.get_class(class_id)
        if hasattr(self, 'status_bar') and cls:
            if hasattr(self, 'canvas') and self.canvas.selected_shape:
                self.status_bar.showMessage(f"Reassigned selected shape → {cls.name}", 2000)
            else:
                self.status_bar.showMessage(f"Selected Class: {cls.name}", 2000)
    
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

    def delete_current_template(self):
        """Delete the currently selected template."""
        if self.template_combo.currentIndex() <= 0:
            QMessageBox.information(self, "No Template", "Please select a template from the dropdown to delete.")
            return
            
        text = self.template_combo.currentText()
        name = text.replace("📋 ", "")
        
        reply = QMessageBox.question(
            self, "Confirm Delete", 
            f"Are you sure you want to completely delete the template '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        if not hasattr(self, 'canvas') or not hasattr(self.canvas, 'template_manager'):
            QMessageBox.warning(self, "Error", "Canvas or template manager not found.")
            return
            
        tm = self.canvas.template_manager
        if tm.delete_template(name):
            # Remove from dropdown
            self.template_combo.removeItem(self.template_combo.currentIndex())
            # Switch to 'No template'
            self.template_combo.setCurrentIndex(0)
            self.status_bar.showMessage(f"Template '{name}' deleted.", 3000)
            if hasattr(self, 'canvas'):
                self.canvas.stamp_template_name = None
                if self.canvas.current_shape_type == 'stamp':
                    self.set_shape_type('none')
        else:
            QMessageBox.warning(self, "Error", f"Failed to delete template '{name}'.")
    
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
        """Switch between YOLO, U-Net, and Concentric modes"""  # CONCENTRIC INTEGRATION
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
                # Revert buttons visually  # CONCENTRIC INTEGRATION
                self.yolo_mode_action.setChecked(current_mode == 'yolo')
                self.unet_mode_action.setChecked(current_mode == 'unet')
                if hasattr(self, 'concentric_mode_action'):
                    self.concentric_mode_action.setChecked(
                        current_mode == 'concentric')
                self.mode_btn_yolo.setChecked(current_mode == 'yolo')
                self.mode_btn_unet.setChecked(current_mode == 'unet')
                if hasattr(self, 'mode_btn_concentric'):
                    self.mode_btn_concentric.setChecked(
                        current_mode == 'concentric')
                return
            coexist = dlg.coexist
            
        is_unet        = (mode == 'unet')  # CONCENTRIC INTEGRATION
        is_concentric  = (mode == 'concentric')
        is_unet_like   = is_unet or is_concentric  # both show UNet shape tools
        
        # Update menu actions
        self.yolo_mode_action.setChecked(mode == 'yolo')
        self.unet_mode_action.setChecked(mode == 'unet')
        if hasattr(self, 'concentric_mode_action'):
            self.concentric_mode_action.setChecked(is_concentric)
        
        # Update mode buttons
        self.mode_btn_yolo.setChecked(mode == 'yolo')
        self.mode_btn_unet.setChecked(mode == 'unet')
        if hasattr(self, 'mode_btn_concentric'):
            self.mode_btn_concentric.setChecked(is_concentric)
        
        # Update status bar
        mode_names = {'yolo': 'YOLO', 'unet': 'U-Net', 'concentric': 'Concentric'}
        self.mode_label.setText(f"Mode: {mode_names.get(mode, mode.upper())}")
        
        # Update canvas mode
        self.canvas.set_mode(mode)
        
        # Show/Hide hollow shape options (available in UNet AND Concentric)
        # Buttons
        self.shape_btn_frame.setVisible(is_unet_like)
        self.shape_btn_donut.setVisible(is_unet_like)
        self.shape_btn_hollow_ellipse.setVisible(is_unet_like)
        
        # Menu Shortcuts
        if hasattr(self, 'frame_action'):
            self.frame_action.setVisible(is_unet_like)
        if hasattr(self, 'donut_action'):
            self.donut_action.setVisible(is_unet_like)
        if hasattr(self, 'hollow_ellipse_action'):
            self.hollow_ellipse_action.setVisible(is_unet_like)
            

            
        # After switching: update layer visibility in saved JSON
        if hasattr(self, "project_manager") and self.current_image_index >= 0:
            filename = self.image_files[self.current_image_index]
            self.project_manager.set_layer_visibility(filename, current_mode, coexist)
            self.project_manager.set_layer_visibility(filename, mode, True)
            
        # Reload annotations for the new target mode
        if hasattr(self, "project_manager") and self.current_image_index >= 0:
            self._restore_annotations(self.image_files[self.current_image_index])
            
        # If in YOLO and a hollow shape was selected, switch back to Box
        if mode == 'yolo' and hasattr(self, 'canvas'):
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
        
        # Load persisted templates
        if self.project_manager.project_dir:
            tmpl_path = os.path.join(self.project_manager.project_dir, 'templates.json')
            self.canvas.template_manager.load_from_file(tmpl_path)
            self.update_template_dropdown()
        
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
                    
            STATUS_COLORS = {
                "annotated": "#8aff8a", "skipped": "#8ab4f8",
                "in_progress": "#ffd54f", "unannotated": "#aaaaaa"
            }
            for file in self.image_files:
                status = self.project_manager.get_image_status(file)
                badge = "✅" if status == "annotated" else "⏭️" if status == "skipped" else "🟡" if status == "in_progress" else "  "
                item = QListWidgetItem(f"{badge} {file}")
                item.setData(Qt.UserRole, file)
                item.setForeground(QColor(STATUS_COLORS.get(status, "#aaaaaa")))
                item.setToolTip(f"Status: {status.replace('_', ' ').title()}")
                self.file_list.addItem(item)
                
            self.update_image_counter()
            self.filter_image_list(self.search_box.text())
            
            if self.image_files:
                self.load_image(0)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load images: {str(e)}")

    def filter_image_list(self, text):
        """Filter the image list based on search text"""
        text = text.lower()
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            filename = item.data(Qt.UserRole)
            if filename:
                item.setHidden(text not in filename.lower())
    
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
                
            # Find and select the matching item in the list
            filename = self.image_files[index]
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.data(Qt.UserRole) == filename:
                    self.file_list.setCurrentItem(item)
                    break
                    
            self.update_image_counter()
            self.image_info_label.setText(self.image_files[index])
            self.setWindowTitle(f"Dual Annotator - {self.image_files[index]}")
            self.status_bar.showMessage(f"Loaded: {self.image_files[index]}", 2000)
            # Reset shape count for new image (will be updated by _restore_annotations)
            if hasattr(self, 'shape_count_label'):
                count = len([s for s in self.canvas.shapes if getattr(s, 'hollow_role', None) != 'inner'])
                self.shape_count_label.setText(f"✏️ {count} Shape{'s' if count != 1 else ''}")
            self.update_annotation_stats()

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
        filename = item.data(Qt.UserRole)
        if filename and filename in self.image_files:
            index = self.image_files.index(filename)
            self.load_image(index)
            
    def mark_image_skipped(self):
        if not self.image_files or getattr(self, 'current_image_index', -1) < 0: return
        filename = self.image_files[self.current_image_index]
        self.project_manager.set_image_status(filename, "skipped")
        self.update_ui_status_badge(filename, "skipped")
        self.next_image()
        
    def mark_image_done(self):
        if not self.image_files or getattr(self, 'current_image_index', -1) < 0: return
        filename = self.image_files[self.current_image_index]
        self.project_manager.set_image_status(filename, "annotated")
        self.update_ui_status_badge(filename, "annotated")
        self.next_image()
        
    def update_ui_status_badge(self, filename, status):
        STATUS_MAP = {
            "annotated":   ("\u2705", "#8aff8a"),
            "skipped":     ("\u23ed\ufe0f",  "#8ab4f8"),
            "in_progress": ("\U0001f7e1", "#ffd54f"),
            "unannotated": ("  ",     "#aaaaaa"),
        }
        badge, color = STATUS_MAP.get(status, ("  ", "#aaaaaa"))
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == filename:
                item.setText(f"{badge} {filename}")
                item.setForeground(QColor(color))
                # Update tooltip
                count = len([s for s in self.canvas.shapes if getattr(s, 'hollow_role', None) != 'inner'])
                if filename == (self.image_files[self.current_image_index] if self.current_image_index >= 0 else None):
                    item.setToolTip(f"Status: {status.replace('_',' ').title()}\nShapes: {count}")
                else:
                    item.setToolTip(f"Status: {status.replace('_',' ').title()}")
                break
    
    def next_image(self):
        """Load next image"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.load_image(self.current_image_index + 1)
        else:
            self.status_bar.showMessage("Already at last image", 1000)
            
    def next_unannotated(self):
        """Jump forward to the next image with no annotations."""
        if not self.image_files or not hasattr(self, 'project_manager') or self.current_image_index < 0:
            return
        import os, json as _json
        start = self.current_image_index + 1
        # Loop through all images starting from the next one
        order = list(range(start, len(self.image_files))) + list(range(0, start))
        for i in order:
            filename = self.image_files[i]
            json_path = os.path.join(
                self.project_manager.annotations_dir, f"{filename}.json")
            
            # If JSON doesn't exist, it's definitely unannotated
            if not os.path.exists(json_path):
                self.load_image(i)
                self.status_bar.showMessage(f"→ Next unannotated: {filename}", 2000)
                return
                
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                    
                status = data.get('status', 'unannotated')
                
                # Check if it has any actual shapes saved
                is_empty = True
                if 'layers' in data:
                    total_shapes = 0
                    for layer in data['layers'].values():
                        total_shapes += len(layer.get('annotations', []))
                    if total_shapes > 0:
                        is_empty = False
                        
                if status == 'unannotated' or is_empty:
                    self.load_image(i)
                    self.status_bar.showMessage(f"→ Next unannotated: {filename}", 2000)
                    return
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue
        
        self.status_bar.showMessage("✓ All images annotated!", 3000)
    
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

    def _on_nudge_step_changed(self, value):
        """Sync nudge step spinbox value into the canvas."""
        if hasattr(self, 'canvas'):
            self.canvas.nudge_step = value

    def _show_nudge_step_dialog(self):
        """Open a small dialog to set the nudge step value."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle("Nudge Step")
        dlg.setFixedWidth(260)
        dlg.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #d8e0f8; }
            QLabel  { color: #d8e0f8; }
            QSpinBox {
                background-color: #161c2e; color: #ffffff;
                border: 1px solid #2e3a58; border-radius: 4px; padding: 4px;
            }
            QPushButton {
                background-color: #1e2640; border: 1px solid #2e3a58;
                border-radius: 4px; padding: 6px 12px; color: #ffffff;
            }
            QPushButton:hover { background-color: #2e3a58; }
        """)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Arrow key nudge distance (px):"))
        layout.addWidget(QLabel("Shift+Arrow moves 10× this value."))
        spin = QSpinBox()
        spin.setRange(1, 100)
        current = getattr(self.canvas, 'nudge_step', 1) if hasattr(self, 'canvas') else 1
        spin.setValue(current)
        spin.setSuffix(" px")
        layout.addWidget(spin)
        btns = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setStyleSheet("background-color: #4a6bff; border: none; font-weight: bold;")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btns.addWidget(cancel_btn)
        btns.addWidget(ok_btn)
        layout.addLayout(btns)
        if dlg.exec_() == QDialog.Accepted:
            new_val = spin.value()
            if hasattr(self, 'canvas'):
                self.canvas.nudge_step = new_val
            if hasattr(self, 'nudge_step_spin'):
                self.nudge_step_spin.setValue(new_val)
    
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
            
    def delete_all_annotations(self):
        """Delete all annotations on the current image with confirmation."""
        if not hasattr(self, 'canvas') or not self.canvas.shapes:
            return
            
        # Count non-inner shapes for a more accurate user count
        count = len([s for s in self.canvas.shapes 
                     if getattr(s, 'hollow_role', None) != 'inner'])
                     
        reply = QMessageBox.question(
            self, "Delete All Annotations",
            f"Are you sure you want to delete all {count} annotation(s) on this image?\n"
            "Autosave will immediately save the empty state.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.canvas.delete_all()
            self.status_bar.showMessage(f"Deleted all {count} annotations", 3000)
    
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
        """Show detailed Help / About dialog"""
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                     QWidget, QPushButton, QListWidget, QSplitter, QTextBrowser)
        from PyQt5.QtCore import Qt, QTimer

        dlg = QDialog(self)
        dlg.setWindowTitle("DualAnnotator Help & Reference")
        dlg.resize(1100, 800) # Maximum detailed window size
        dlg.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #ffffff; }
            QListWidget { background-color: #161616; color: #aaa; border: 1px solid #333; font-size: 14px; padding: 5px; outline: none; }
            QListWidget::item { padding: 12px; border-radius: 4px; margin-bottom: 2px;}
            QListWidget::item:selected { background-color: #2a4a6a; color: white; font-weight: bold; }
            QListWidget::item:hover { background-color: #2a2a2a; }
            QTextBrowser { background-color: #1e1e1e; color: #e0e0e0; border: none; font-size: 14px; padding: 20px; line-height: 1.6;}
            QPushButton { background-color: #2a4a6a; color: #ffffff; border: 1px solid #8ab4f8; border-radius: 4px; padding: 8px 30px; font-size: 14px; font-weight: bold;}
            QPushButton:hover { background-color: #3a5a7a; }
        """)

        outer = QVBoxLayout(dlg)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Left navigation panel
        nav_list = QListWidget()
        nav_list.setFixedWidth(280)
        nav_items = [
            "0. ℹ️ About DualAnnotator",
            "1. 📚 Getting Started",
            "2. 🏷️ Class Management",
            "3. 🖱️ Canvas Navigation",
            "4. 🎨 Drawing Tools",
            "5. ⭕ Hollow Shapes",
            "6. 💾 Autosave & Projects",
            "7. 📋 Auto-Annotation",
            "8. 📤 Exporting Data",
            "9. 🔥 RAW → PNG",
            "10. ⌨️ Keyboard Shortcuts"
        ]
        nav_list.addItems(nav_items)
        
        # Right content panel
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        # Prevent tracking issues during programmatic clicks
        browser._programmatic_scroll = False
        
        html_content = """
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; }
            h1 { color: #8ab4f8; font-size: 24px; border-bottom: 1px solid #333; padding-bottom: 6px; margin-top: 35px; margin-bottom: 10px;}
            h2 { color: #aac4f0; font-size: 17px; margin-top: 20px; margin-bottom: 8px;}
            p, li { font-size: 14px; line-height: 1.5; color: #cccccc; margin-bottom: 8px; margin-top: 4px;}
            code { background-color: #2a2a2a; color: #8ab4f8; padding: 2px 5px; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 13px;}
            .ui-btn { background-color: #333; color: white; border: 1px solid #555; padding: 1px 6px; border-radius: 3px; font-size: 12px; font-weight: bold; }
            .important { color: #f0b429; font-weight: bold; }
            ul { margin-top: 0px; padding-left: 20px;}
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th { border-bottom: 2px solid #555; padding: 8px; text-align: left; color: #8ab4f8; font-size: 14px;}
            td { border-bottom: 1px solid #333; padding: 8px; color: #ccc;}
        </style>
        
        <h1 id="about" style="margin-top: 0px;">0. About DualAnnotator</h1>
        <p>DualAnnotator is an advanced, high-performance desktop annotation tool designed specifically for building top-tier <b>YOLO Object Detection</b> and <b>U-Net Segmentation</b> datasets. It was created to solve the need for pixel-perfect contours and precise hollow topologies (like seals and gaskets) that web-based annotators struggle with.</p>
        <p>Built purely in Python and PyQt5, it natively handles seamless UI manipulation across massive images, scientific thermal processing, and AI template matching.</p>

        <h1 id="getting-started">1. Getting Started & The Three Modes</h1>
        <p>Welcome! DualAnnotator runs three entirely distinct rendering modes simultaneously. You can switch between them using the top toolbar: [Mode: YOLO / UNet / Concentric].</p>
        
        <h2>YOLO Mode (Object Detection)</h2>
        <p>Purely for creating rapid bounding boxes. When exporting, it flattens everything and outputs identical <code>[Class X Y W H]</code> text files used natively by YOLO AI architectures.</p>
        
        <h2>U-Net Mode (Segmentation)</h2>
        <p>For drawing pixel-perfect polygons and hollow geometric shapes. Outputs exact binary black-and-white (or multi-color) PNG Masks representing the real topography of the shapes.</p>

        <h2>Concentric Zones Mode (Advanced Segmentation)</h2>
        <p>A specialized rendering mode leveraging an <b>inner-wins cutout rule</b>. Any shape drawn will automatically "cut away" or subtract the territory of all previously-drawn shapes inside of it. It operates identical to U-Net's masking visually, but you can layer massive overlapping zones and the system mathematically handles the physical segmentation borders for you!</p>
        
        <p class="important">Note: Switching modes on an image asks if you want to 'Keep Both' or 'Hide Previous'. No data is lost! It simply hides irrelevant shapes so you can visually focus on what you're doing.</p>

        <h1 id="class-management">2. Class Management Deep Dive</h1>
        <p>Before you pick up a drawing tool immediately, you must specify what class you are targeting using the <b>Class Sidebar Panel</b>!</p>

        <h2>Assigning & Reassigning Shapes</h2>
        <ul>
            <li><b>To Start Drawing:</b> Simply click a Class in the left sidebar. Any new polygons or boxes you draw will now belong to that class instantly.</li>
            <li><b>To Reassign a Shape:</b> If you trace a huge shape and realize it's under the wrong name, select the shape with the pointer tool (<code>N</code>), and then literally <b>click the correct class in the left panel</b>. The shape's color will swap immediately, reassigning its memory!</li>
        </ul>

        <h2>Managing the List</h2>
        <ul>
            <li><b>Edit & Colors:</b> Hitting the [Edit] button dynamically updates all shapes mapped to that class across entirely active canvas seamlessly.</li>
            <li><b>Dangerous Deletes:</b> If you delete a class in the manager, the app scans everything. Any existing geometries using that identity will become 'unassigned' and must be corrected!</li>
            <li><b>Drag and Drop Priority:</b> Classes can be click-and-dragged up and down in the list to reorder them logically.</li>
            <li><b>Automatic Hotkeys (1-9):</b> The system automatically maps keyboard numbers <code>1</code> through <code>9</code> to the first 9 positions in your list. Pressing <code>1</code> on your keyboard instantly equips the first class without mouse movement!</li>
        </ul>

        <h1 id="canvas-navigation">3. Canvas Navigation & Selection Secrets</h1>
        <p>There are several hidden features for manipulating the workspace completely fluidly without clicking UI buttons.</p>

        <h2>Pan Mode (Moving the Image)</h2>
        <p>If you are zoomed into a massive 4K image, you don't need to use the scrollbars:</p>
        <ul>
            <li><b>Middle Mouse Button:</b> Simply click and hold your <b>Scroll Wheel (Mouse 3)</b> anywhere on the canvas and drag your mouse to pan around instantly!</li>
            <li><b>Spacebar:</b> Hold <code>Spacebar</code> on your keyboard, then Left-Click and drag.</li>
        </ul>

        <h2>Advanced Selection Tool (N)</h2>
        <p>Select the pointer arrow from the toolbar. You can click on objects to highlight them.</p>
        <ul>
            <li><b>Shift + Click:</b> Hold <code>Shift</code> and click multiple different shapes. They will all highlight at once, allowing you to move or delete the entire group simultaneously.</li>
            <li><b>Marquee Select:</b> Left-click on an empty part of the background and drag your mouse to cast a dotted "Selection Box". Any shape caught inside the box will be added to your group selection!</li>
            <li><b>Cloning:</b> Hold <code>Ctrl</code>, click an active shape, and drag it. An identical clone will peel off instantly!</li>
        </ul>

        <h2>Nudging & Scaling (Keyboard Precision)</h2>
        <p>Don't use your mouse if you need pixel-perfect accuracy.</p>
        <ul>
            <li><b>Micro Nudge:</b> Select a shape and tap the <code>Arrow Keys</code>. It will move exactly 1 pixel per tap.</li>
            <li><b>Macro Nudge:</b> Hold <code>Shift + Arrow Keys</code> to jump 10 pixels per tap.</li>
            <li><b>Live Scaling:</b> Select a shape, hold <code>Shift</code>, and scroll your <b>Mouse Wheel</b>. The shape will artificially inflate or deflate smoothly!</li>
        </ul>

        <h1 id="drawing-tools">4. Drawing Tools</h1>
        <h2>The Active Toolset</h2>
        <ul>
            <li><b>Box (B):</b> Standard click-and-drag bounding box.</li>
            <li><b>Polygon (P):</b> Drop continuous anchor points. Press <code>Enter</code> or click the starting point to lock and fill it.</li>
            <li><b>Bezier (Q):</b> Drop anchors, hit <code>Enter</code> to lock the skeleton, and then freely drag the Yellow Diamond mid-point handles to bend smooth, perfect contours!</li>
            <li><b>Circle (C):</b> Click the center, drag outwards for radius.</li>
            <li><b>Ellipse (E):</b> Click the center, drag outwards for width/height.</li>
        </ul>

        <h2>Point-Level Undo / Redo</h2>
        <p>If you are drawing a complex 50-point polygon and you mess up the 49th point, you do not need to restart!</p>
        <p>Simply press <code>Ctrl + Z</code> gracefully <b>while you are actively drawing</b>. It will instantly delete the last point you dropped. If you went too far, press <code>Ctrl + Y</code> to redo the point!</p>

        <h1 id="hollow-shapes">5. Hollow Shapes (Cutouts)</h1>
        <p>Perfect for seals, mechanical gaskets, donuts, or picture frames.</p>
        <h2>Creation</h2>
        <p>You can hollow out <b>any</b> solid shape via the right-click context menu:</p>
        <ul>
            <li><b>Create Inner Shape:</b> Cores out the insides. A popup thickness slider window will appear allowing you to visually adjust the inner-diameter wall interactively across the canvas!</li>
            <li><b>Create Outer Shape:</b> Same concept, but leaves the current shape as the core and expands the boundaries outwards instead.</li>
        </ul>
        <h2>Modification</h2>
        <p>You can delete a hollow cavity at any time by right-clicking the object and selecting <b>Remove Inner Cutout</b>.</p>
        <p class="important">Note: YOLO does not natively understand true transparent holes. If you export a Hollow Shape under YOLO Mode, it will realistically just compute a flat bounding box framing the outermost perimeter.</p>

        <h1 id="autosave-projects">6. Autosave & Projects</h1>
        <h2>Continuous Autosave</h2>
        <p>You never need to remember to press "Save"! The application secretly saves the exact state of your canvas <b>800ms</b> after your last mouse release, ensuring zero interruptions to your workflow.</p>
        
        <h2>The Hidden Folder Structure</h2>
        <p>When you open an image folder, the app generates a hidden <code>.dualannotator</code> directory. <b>Do not delete this if you wish to keep your project.</b></p>
        <ul>
            <li><code>project.json</code>: Contains all UI states, class lists, and colors.</li>
            <li><code>annotations/*.json</code>: Stores lossless exact JSON mathematical vector data of every annotation you drew.</li>
        </ul>
        <h2>Project Recovery</h2>
        <p>Re-opening a folder automatically restores everything down to the exact classes and shapes. If an image is renamed directly in Windows Explorer, the app will flag a mismatch warning on the bottom status bar, but it preserves your data safely!</p>

        <h1 id="templates-auto">7. Templates & Auto-Annotation</h1>
        <h2>Template Stamping</h2>
        <p>Tired of drawing the exact same complex mechanical widget repeatedly?</p>
        <ol>
            <li>Draw it perfectly once.</li>
            <li>Right-click the shape and select <b>Save as Template</b>.</li>
            <li>Select the <b>Stamp Tool (T)</b> from the toolbar. Ensure your template is selected in the dropdown.</li>
            <li>Simply click anywhere on the canvas to instantly drop identical, fully-formed shapes!</li>
        </ol>

        <h2>Template Matching (Computer Vision Automation)</h2>
        <p>You can leverage OpenCV to automate your annotation completely!</p>
        <p>Right-click a shape and select <b>Trigger Template Matching</b>. A sidebar panel will slide open. Click <b>Run Detection</b>, and the app will actively scan the massive current image and attempt to auto-paint templates across the rest of the matching visual signatures. If it misses objects or predicts false-positives, simply drag the <b>Threshold Slider</b> to recalibrate the AI block matches!</p>

        <h1 id="exporting">8. Exporting Data</h1>
        <p>Press <code>Ctrl + E</code> to open the Universal Mass Exporter menu. You can export safely at any time, it does not interrupt your active work!</p>
        <h2>The Formats</h2>
        <ul>
            <li><b>YOLO (Txt):</b> Flattens everything into minimal bounding boxes. Generates <code>classes.txt</code> and <code>data.yaml</code>.</li>
            <li><b>UNet / Concentric (Mask PNG):</b> Flattens everything into solid binary masks. Handles true hollow topologies perfectly! Choose whether you want flat Black/White or Multi-Class Grayscale representations.</li>
            <li><b>Pascal VOC (XML):</b> Classic historical format mapping coordinates to physical pixels.</li>
            <li><b>COCO (Json):</b> Translates thousands of annotations into the massive industry standard <code>instances_train.json</code> COCO structure.</li>
        </ul>
        <h2>Time-Saving Export Deltas</h2>
        <p>The app dynamically organizes your data into Train/Val/Test folders based on slider percentages. 
        If you check <b>"Only export changes"</b>, the app ensures that unchanged images simply reuse existing heavy computations, saving you massive amounts of waiting time on huge datasets!</p>

        <h1 id="raw-png">9. RAW → PNG Converter Tool</h1>
        <p>A built-in heavy-duty Thermal Image converter to process raw 16-bit binary <code>.raw</code> data directly from specialized camera hardware!</p>
        <ul>
            <li><b>NumPy Thread Acceleration:</b> Converts thousands of 16-bit Little-Endian binaries to standard 8-bit PNGs instantly via background processing.</li>
            <li><b>Interactive ROI Crop:</b> Look at the preview canvas! You can use your mouse to draw an active Region of Interest (Crop Box) over the sample image. Hitting "Convert" will physically enforce that crop limit across all folders!</li>
            <li><b>Advanced Normalizations:</b> 
                <ul>
                    <li><b>Original (Defect Padding):</b> Mathematically matches older C++ software parameters. Enforces a strict padding threshold (up to 1000 values) on both extreme ends to prevent thermal outliers from destroying the image contrast.</li>
                    <li><b>Robust Percentile:</b> Mathematically cuts off the upper and lower 1% of extreme outlier pixels, natively hiding broken dead sensor pixels.</li>
                </ul>
            </li>
            <li><b>Colormaps:</b> Remaps standard grayscale heat values out into scientific visual filters like Inferno or Viridis directly during conversion!</li>
        </ul>

        <h1 id="shortcuts">10. Master Keyboard Shortcuts</h1>
        <table>
            <tr><th>Key</th><th>Function</th><th>Key</th><th>Function</th></tr>
            <tr><td><code>1-9</code></td><td>Auto-select Class</td><td><code>Ctrl+Z</code></td><td>Undo Shape / <b>Mid-Draw Point</b></td></tr>
            <tr><td><code>P</code></td><td>Draw Polygon</td><td><code>Ctrl+Y</code></td><td>Redo Shape / <b>Mid-Draw Point</b></td></tr>
            <tr><td><code>Q</code></td><td>Draw Bezier Curve</td><td><code>Ctrl+C / V</code></td><td>Copy & Paste Selection</td></tr>
            <tr><td><code>C / E</code></td><td>Draw Circle / Ellipse</td><td><code>Del / Backspace</code></td><td>Delete Selection</td></tr>
            <tr><td><code>F / O / H</code></td><td>Draw Hollow Variants</td><td><code>Ctrl+Del</code></td><td>Wipe Entire Canvas</td></tr>
            <tr><td><code>B</code></td><td>Draw Bounding Box</td><td><code>Ctrl+Drag</code></td><td>Clone Selection</td></tr>
            <tr><td><code>N</code></td><td>Selection Pointer</td><td><code>Ctrl+A</code></td><td>Select All</td></tr>
            <tr><td><code>T</code></td><td>Template Stamp Tool</td><td><code>Ctrl+F</code></td><td>Fit Image to Screen</td></tr>
            <tr><td><code>A / D</code></td><td>Flip Prev/Next Image</td><td><code>+/-</code></td><td>Zoom In/Out</td></tr>
            <tr><td><code>Tab</code></td><td>Jump to Unannotated</td><td><code>Middle Click / Space</code></td><td>Toggle Hand Pan Mode</td></tr>
            <tr><td><code>Ctrl+E</code></td><td>Mass Exporter Menu</td><td><code>Arrows</code></td><td>Micro Nudge (1px)</td></tr>
            <tr><td><code>Enter</code></td><td>Finish Polygons</td><td><code>Shift+Arrows</code></td><td>Macro Nudge (10px)</td></tr>
            <tr><td><code>Shift+Scroll</code></td><td>Scale/Resize Shape</td><td><code>Shift+Click</code></td><td>Multi-select Shapes</td></tr>
        </table>
        
        <br>
        <p style="text-align:center; color:#555; font-size: 12px; margin-top:30px;">DualAnnotator Core Engine &nbsp;·&nbsp; Python 3.9+ &nbsp;·&nbsp; PyQt5 &nbsp;·&nbsp; OpenCV</p>
        """
        browser.setHtml(html_content)

        # Pre-calculated scroll sync variables
        nav_anchors = [
            "about", "getting-started", "class-management", "canvas-navigation", "drawing-tools",
            "hollow-shapes", "autosave-projects", "templates-auto",
            "exporting", "raw-png", "shortcuts"
        ]
        
        # When user clicks the left list, scroll to exactly that anchor.
        def scroll_to_section(idx):
            if idx < len(nav_anchors):
                browser._programmatic_scroll = True
                browser.scrollToAnchor(nav_anchors[idx])
                # Reset programmatic lock shortly after to allow natural scroll tracking to resume
                QTimer.singleShot(100, lambda: setattr(browser, '_programmatic_scroll', False))

        nav_list.currentRowChanged.connect(scroll_to_section)
        
        # Bi-directional sync: As you scroll, update the left list highlights.
        y_positions = []
        
        def track_scroll(val):
            # Do not track if the scroll was commanded by the QListWidget click
            if getattr(browser, '_programmatic_scroll', False):
                return
                
            nonlocal y_positions
            # Calculate the literal document Y bounds lazily on first scroll
            if not y_positions:
                doc = browser.document()
                layout = doc.documentLayout()
                headers = [
                    "0. About DualAnnotator", "1. Getting Started", "2. Class Management", "3. Canvas Navigation",
                    "4. Drawing Tools", "5. Hollow Shapes", "6. Autosave",
                    "7. Templates", "8. Exporting", "9. RAW", "10. Master Keyboard"
                ]
                for h in headers:
                    cursor = doc.find(h)
                    if not cursor.isNull():
                        y_positions.append(layout.blockBoundingRect(cursor.block()).y())
                    else:
                        y_positions.append(-1)
                        
            # Determine which header we are currently visually underneath
            current_idx = 0
            for i, pos_y in enumerate(y_positions):
                if pos_y != -1 and val >= (pos_y - 20):  # 20px overlap buffer
                    current_idx = i
                    
            if nav_list.currentRow() != current_idx:
                nav_list.blockSignals(True)
                nav_list.setCurrentRow(current_idx)
                nav_list.blockSignals(False)
                
        browser.verticalScrollBar().valueChanged.connect(track_scroll)
        
        splitter.addWidget(nav_list)
        splitter.addWidget(browser)
        splitter.setSizes([280, 820])
        
        outer.addWidget(splitter)
        
        # Bottom Close Button
        btn_lyt = QHBoxLayout()
        btn = QPushButton("Close Reference")
        btn.clicked.connect(dlg.accept)
        btn_lyt.addStretch()
        btn_lyt.addWidget(btn)
        outer.addLayout(btn_lyt)

        dlg.exec_()