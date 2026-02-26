# gui/main_window.py
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QMenu, QToolBar, 
    QStatusBar, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QMessageBox, QFileDialog, QSplitter,
    QListWidget, QListWidgetItem, QAbstractItemView, 
    QComboBox, QFrame, QPushButton, QButtonGroup, QGridLayout
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon, QKeySequence, QFont, QColor, QPalette
import os

from gui.canvas import AnnotationCanvas
from gui.class_panel import ClassPanel
from core.class_manager import ClassManager

class ToolButton(QPushButton):
    """Custom tool button for vertical toolbar with icon only"""
    def __init__(self, icon_text, tooltip=None):
        super().__init__(icon_text)
        self.setFixedSize(50, 50)
        self.setCheckable(True)
        self.setToolTip(tooltip or "")
        self.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                font-size: 20px;
                font-weight: bold;
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
                color: #8ab4f8;
                font-weight: bold;
                background-color: #1e1e1e;
                padding: 3px 8px;
                border-radius: 3px;
                margin: 2px;
            }
            QLabel#desc {
                color: #ddd;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(5)
        
        # Shortcuts
        shortcuts = [
            ("A/D", "Prev/Next Image"),
            ("Del", "Delete Box"),
            ("Ctrl+C", "Copy"),
            ("Ctrl+V", "Paste"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            ("Esc", "Cancel"),
            ("Enter", "Finish Polygon"),
            ("Space", "Pan Mode"),
            ("+/-", "Zoom")
        ]
        
        for key, desc in shortcuts:
            key_label = QLabel(key)
            key_label.setObjectName("shortcut")
            layout.addWidget(key_label)
            
            desc_label = QLabel(desc)
            desc_label.setObjectName("desc")
            layout.addWidget(desc_label)
            
            # Add separator
            if shortcuts.index((key, desc)) < len(shortcuts) - 1:
                sep = QLabel("|")
                sep.setStyleSheet("color: #444;")
                layout.addWidget(sep)
        
        layout.addStretch()

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
        self.setup_shortcut_bar()  # New shortcut bar
        self.setup_status_bar()
        self.setup_central_widget()
        
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
        """)
        
    def setup_default_classes(self):
        """Add some default classes for testing"""
        try:
            self.class_manager.add_class("Car", "#FF6B6B")
            self.class_manager.add_class("Person", "#4ECDC4")
            self.class_manager.add_class("Bicycle", "#45B7D1")
            self.class_manager.add_class("Dog", "#96CEB4")
            print("✅ Default classes added")
        except Exception as e:
            print(f"⚠️ Could not add default classes: {e}")
        
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
        
        # ===== HELP MENU =====
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_shortcut_bar(self):
        """Create horizontal shortcut bar below menu"""
        self.shortcut_bar = ShortcutBar()
        # Add to main window layout (we'll add it in setup_central_widget)
        
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
        
        # Main vertical layout (menu bar + shortcut bar + content)
        main_vertical = QVBoxLayout(central)
        main_vertical.setContentsMargins(0, 0, 0, 0)
        main_vertical.setSpacing(0)
        
        # Add shortcut bar
        main_vertical.addWidget(self.shortcut_bar)
        
        # Main horizontal layout for content
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # ===== LEFT SIDEBAR - VERTICAL TOOLBAR (ICONS ONLY) =====
        left_toolbar = self.create_vertical_toolbar()
        left_toolbar.setFixedWidth(70)
        content_layout.addWidget(left_toolbar)
        
        # ===== CENTER - CANVAS (70%) =====
        self.canvas = AnnotationCanvas()
        self.canvas.set_class_manager(self.class_manager)
        self.canvas.position_changed.connect(self.update_position)
        self.canvas.shape_selected.connect(self.on_canvas_shape_selected)
        content_layout.addWidget(self.canvas, 7)  # 70% stretch factor
        
        # ===== RIGHT PANEL - CLASSES + FILE BROWSER (20%) =====
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
        
        self.mode_btn_yolo = ToolButton("🎯", "YOLO Detection Mode")
        self.mode_btn_yolo.setChecked(True)
        self.mode_btn_yolo.clicked.connect(lambda: self.switch_mode('yolo'))
        layout.addWidget(self.mode_btn_yolo)
        
        self.mode_btn_unet = ToolButton("🔬", "U-Net Segmentation Mode")
        self.mode_btn_unet.clicked.connect(lambda: self.switch_mode('unet'))
        layout.addWidget(self.mode_btn_unet)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # Shape tools section
        shape_label = QLabel("SHAPES")
        shape_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(shape_label)
        
        self.shape_btn_box = ToolButton("⬜", "Draw bounding boxes")
        self.shape_btn_box.setChecked(True)
        self.shape_btn_box.clicked.connect(lambda: self.set_shape_type('box'))
        layout.addWidget(self.shape_btn_box)
        
        self.shape_btn_polygon = ToolButton("🔷", "Draw polygons")
        self.shape_btn_polygon.clicked.connect(lambda: self.set_shape_type('polygon'))
        layout.addWidget(self.shape_btn_polygon)
        
        self.shape_btn_circle = ToolButton("⭕", "Draw circles")
        self.shape_btn_circle.clicked.connect(lambda: self.set_shape_type('circle'))
        layout.addWidget(self.shape_btn_circle)

        self.shape_btn_ellipse = ToolButton("🟢", "Draw ellipses")
        self.shape_btn_ellipse.clicked.connect(lambda: self.set_shape_type('ellipse'))
        layout.addWidget(self.shape_btn_ellipse)
        
        self.shape_btn_none = ToolButton("🚫", "No shape selected")
        self.shape_btn_none.clicked.connect(lambda: self.set_shape_type(None))
        layout.addWidget(self.shape_btn_none)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # Edit tools section
        edit_label = QLabel("EDIT")
        edit_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(edit_label)
        
        btn_undo = ToolButton("↩", "Undo (Ctrl+Z)")
        btn_undo.setCheckable(False)
        btn_undo.clicked.connect(self.undo)
        layout.addWidget(btn_undo)
        
        btn_redo = ToolButton("↪", "Redo (Ctrl+Y)")
        btn_redo.setCheckable(False)
        btn_redo.clicked.connect(self.redo)
        layout.addWidget(btn_redo)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # View tools section
        view_label = QLabel("VIEW")
        view_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(view_label)
        
        btn_zoom_in = ToolButton("➕", "Zoom In (+)")
        btn_zoom_in.setCheckable(False)
        btn_zoom_in.clicked.connect(self.zoom_in)
        layout.addWidget(btn_zoom_in)
        
        btn_zoom_out = ToolButton("➖", "Zoom Out (-)")
        btn_zoom_out.setCheckable(False)
        btn_zoom_out.clicked.connect(self.zoom_out)
        layout.addWidget(btn_zoom_out)
        
        btn_fit = ToolButton("⬜", "Fit to Window (Ctrl+F)")
        btn_fit.setCheckable(False)
        btn_fit.clicked.connect(self.fit_to_window)
        layout.addWidget(btn_fit)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # Navigation section
        nav_label = QLabel("NAVIGATE")
        nav_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(nav_label)
        
        btn_prev = ToolButton("◀", "Previous Image (A)")
        btn_prev.setCheckable(False)
        btn_prev.clicked.connect(self.prev_image)
        layout.addWidget(btn_prev)
        
        btn_next = ToolButton("▶", "Next Image (D)")
        btn_next.setCheckable(False)
        btn_next.clicked.connect(self.next_image)
        layout.addWidget(btn_next)
        
        # Stretch at bottom
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
        # Make sure ClassPanel has proper styling (we'll update it separately)
        layout.addWidget(self.class_panel)
        
        # Separator
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
            self.shape_btn_circle.setChecked(shape_type == 'circle')
            self.shape_btn_ellipse.setChecked(shape_type == 'ellipse')
            self.shape_btn_none.setChecked(shape_type == 'none' or shape_type is None)
            
            if shape_type and shape_type != 'none':
                self.status_bar.showMessage(f"Drawing tool: {shape_type}", 1000)
            else:
                self.status_bar.showMessage("Selection mode - click on shapes to select them", 1000)
    
    def on_canvas_shape_selected(self, shape_type):
        """Handle shape selection from canvas to update toolbar"""
        # When a shape is selected (any shape), automatically switch to 'none' mode
        # This prevents accidental drawing when trying to select shapes
        
        if shape_type == "none":
            # No shape selected - just update button states
            self.shape_btn_box.setChecked(False)
            self.shape_btn_polygon.setChecked(False)
            self.shape_btn_circle.setChecked(False)
            self.shape_btn_ellipse.setChecked(False)
            self.shape_btn_none.setChecked(True)
            
            # Update canvas to 'none' mode (selection only)
            if hasattr(self, 'canvas'):
                self.canvas.set_shape_type('none')
        else:
            # A shape was selected - switch to 'none' mode automatically
            self.shape_btn_box.setChecked(False)
            self.shape_btn_polygon.setChecked(False)
            self.shape_btn_circle.setChecked(False)
            self.shape_btn_ellipse.setChecked(False)  # ← THIS WAS MISSING
            self.shape_btn_none.setChecked(True)
            
            # Update canvas to 'none' mode (selection only)
            if hasattr(self, 'canvas'):
                self.canvas.set_shape_type('none')
            
            # Show message to user
            shape_name = shape_type.capitalize() if shape_type != "none" else "No shape"
            self.status_bar.showMessage(f"{shape_name} selected - 'None' mode activated", 2000)
    
    def switch_mode(self, mode):
        """Switch between YOLO and U-Net modes"""
        if mode == 'yolo':
            self.yolo_mode_action.setChecked(True)
            self.unet_mode_action.setChecked(False)
            self.mode_label.setText("Mode: YOLO")
            self.canvas.set_mode('yolo')
            self.mode_btn_yolo.setChecked(True)
            self.mode_btn_unet.setChecked(False)
        else:
            self.yolo_mode_action.setChecked(False)
            self.unet_mode_action.setChecked(True)
            self.mode_label.setText("Mode: U-Net")
            self.canvas.set_mode('unet')
            self.mode_btn_yolo.setChecked(False)
            self.mode_btn_unet.setChecked(True)
            
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
    
    def prev_image(self):
        """Load previous image"""
        if self.image_files and self.current_image_index > 0:
            self.load_image(self.current_image_index - 1)
    
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
            center = self.canvas.rect().center()
            self.canvas.start_paste(center)
    
    def delete_selected(self):
        if hasattr(self, 'canvas'):
            self.canvas.delete_selected()
    
    def undo(self):
        if hasattr(self, 'canvas'):
            self.canvas.undo()  # You'll need to implement undo in canvas
        self.status_bar.showMessage("Undo", 1000)
    
    def redo(self):
        if hasattr(self, 'canvas'):
            self.canvas.redo()  # You'll need to implement redo in canvas
        self.status_bar.showMessage("Redo", 1000)
    
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