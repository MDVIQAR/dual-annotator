# gui/main_window.py
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QMenu, QToolBar, 
    QStatusBar, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QMessageBox, QFileDialog, QSplitter,
    QListWidget, QListWidgetItem, QAbstractItemView, QComboBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QKeySequence
import os

from gui.canvas import AnnotationCanvas
from gui.class_panel import ClassPanel
from gui.shape_toolbar import ShapeToolbar
from core.class_manager import ClassManager

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Dual Annotator - New Project")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(800, 600)
        
        # Application state
        self.current_file = None
        self.is_modified = False
        self.image_folder = None
        self.image_files = []
        self.current_image_index = -1
        
        # Initialize class manager FIRST
        self.class_manager = ClassManager()
        
        # Add some default classes for testing
        self.setup_default_classes()
        
        # Initialize UI components
        self.setup_menu_bar()
        self.setup_toolbar()  # This now includes shape toolbar setup
        self.setup_status_bar()
        self.setup_central_widget()
        
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
        new_action.setStatusTip('Create a new annotation project')
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        # Open Project
        open_action = QAction('&Open Project', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setStatusTip('Open an existing project')
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        # Save Project
        save_action = QAction('&Save Project', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.setStatusTip('Save the current project')
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        # Save As
        save_as_action = QAction('Save &As...', self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.setStatusTip('Save project with a new name')
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # Open Image Folder
        open_folder_action = QAction('&Open Image Folder...', self)
        open_folder_action.setShortcut('Ctrl+Shift+O')
        open_folder_action.setStatusTip('Open a folder of images')
        open_folder_action.triggered.connect(self.open_image_folder)
        file_menu.addAction(open_folder_action)
        
        # Import Images
        import_action = QAction('&Add Images...', self)
        import_action.setStatusTip('Add more images to current folder')
        import_action.triggered.connect(self.import_images)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        # Export
        export_menu = file_menu.addMenu('&Export')
        
        export_yolo = QAction('YOLO Format', self)
        export_yolo.setStatusTip('Export annotations in YOLO format')
        export_yolo.triggered.connect(lambda: self.export_data('yolo'))
        export_menu.addAction(export_yolo)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip('Exit the application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # ===== EDIT MENU =====
        edit_menu = menubar.addMenu('&Edit')
        
        # Undo
        undo_action = QAction('&Undo', self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.setStatusTip('Undo last action')
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)
        
        # Redo
        redo_action = QAction('&Redo', self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.setStatusTip('Redo last undone action')
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        # Copy
        copy_action = QAction('&Copy', self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.setStatusTip('Copy selected shape')
        copy_action.triggered.connect(self.copy_selected)
        edit_menu.addAction(copy_action)
        
        # Paste
        paste_action = QAction('&Paste', self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.setStatusTip('Paste copied shape')
        paste_action.triggered.connect(self.paste_shape)
        edit_menu.addAction(paste_action)
        
        edit_menu.addSeparator()
        
        # Delete
        delete_action = QAction('&Delete', self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.setStatusTip('Delete selected item')
        delete_action.triggered.connect(self.delete_selected)
        edit_menu.addAction(delete_action)
        
        # ===== VIEW MENU =====
        view_menu = menubar.addMenu('&View')
        
        # Zoom In
        zoom_in_action = QAction('Zoom &In', self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.setStatusTip('Zoom in on image')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        # Zoom Out
        zoom_out_action = QAction('Zoom &Out', self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.setStatusTip('Zoom out of image')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        # Fit to Window
        fit_action = QAction('&Fit to Window', self)
        fit_action.setShortcut('Ctrl+F')
        fit_action.setStatusTip('Fit image to window')
        fit_action.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_action)
        
        view_menu.addSeparator()
        
        # Show/Hide File Browser
        self.toggle_browser_action = QAction('Show File Browser', self)
        self.toggle_browser_action.setCheckable(True)
        self.toggle_browser_action.setChecked(True)
        self.toggle_browser_action.setStatusTip('Show/hide the file browser panel')
        self.toggle_browser_action.triggered.connect(self.toggle_file_browser)
        view_menu.addAction(self.toggle_browser_action)
        
        # ===== MODE MENU =====
        mode_menu = menubar.addMenu('&Mode')
        
        # YOLO Mode
        self.yolo_mode_action = QAction('&YOLO Detection', self)
        self.yolo_mode_action.setCheckable(True)
        self.yolo_mode_action.setChecked(True)
        self.yolo_mode_action.setStatusTip('Switch to YOLO bounding box mode')
        self.yolo_mode_action.triggered.connect(lambda: self.switch_mode('yolo'))
        mode_menu.addAction(self.yolo_mode_action)
        
        # U-Net Mode
        self.unet_mode_action = QAction('&U-Net Segmentation', self)
        self.unet_mode_action.setCheckable(True)
        self.unet_mode_action.setStatusTip('Switch to U-Net segmentation mode')
        self.unet_mode_action.triggered.connect(lambda: self.switch_mode('unet'))
        mode_menu.addAction(self.unet_mode_action)
        
        # ===== HELP MENU =====
        help_menu = menubar.addMenu('&Help')
        
        # About
        about_action = QAction('&About', self)
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_toolbar(self):
        """Create the main toolbar and shape toolbar"""
        # Main toolbar for common actions
        main_toolbar = QToolBar("Main Toolbar")
        main_toolbar.setIconSize(QSize(24, 24))
        main_toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(main_toolbar)
        
        # Add mode selector to main toolbar
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["YOLO Detection", "U-Net Segmentation"])
        self.mode_selector.currentIndexChanged.connect(self.on_mode_selected)
        main_toolbar.addWidget(QLabel("Mode: "))
        main_toolbar.addWidget(self.mode_selector)
        
        main_toolbar.addSeparator()
        
        # Add actions with icons (text only for now)
        save_action = QAction("💾 Save", self)
        save_action.triggered.connect(self.save_project)
        main_toolbar.addAction(save_action)
        
        main_toolbar.addSeparator()
        
        undo_action = QAction("↩ Undo", self)
        undo_action.triggered.connect(self.undo)
        main_toolbar.addAction(undo_action)
        
        redo_action = QAction("↪ Redo", self)
        redo_action.triggered.connect(self.redo)
        main_toolbar.addAction(redo_action)
        
        main_toolbar.addSeparator()
        
        zoom_in_action = QAction("➕ Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        main_toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("➖ Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        main_toolbar.addAction(zoom_out_action)
        
        fit_action = QAction("⬜ Fit", self)
        fit_action.triggered.connect(self.fit_to_window)
        main_toolbar.addAction(fit_action)
        
        main_toolbar.addSeparator()
        
        # Previous/Next image buttons
        prev_action = QAction("◀ Prev", self)
        prev_action.setShortcut('A')
        prev_action.triggered.connect(self.prev_image)
        main_toolbar.addAction(prev_action)
        
        next_action = QAction("▶ Next", self)
        next_action.setShortcut('D')
        next_action.triggered.connect(self.next_image)
        main_toolbar.addAction(next_action)
        
        # Create shape toolbar (visible in both modes)
        self.setup_shape_toolbar()
        
    def setup_shape_toolbar(self):
        """Create the shape toolbar for selecting shape types"""
        self.shape_toolbar = ShapeToolbar()
        self.addToolBar(Qt.TopToolBarArea, self.shape_toolbar)
        
        # Connect shape selection to canvas
        self.shape_toolbar.shape_changed.connect(self.on_shape_selected)
        self.shape_toolbar.shape_deselected.connect(self.on_shape_deselected)
        
        # Show shape toolbar in both modes
        self.shape_toolbar.show()
        
    def setup_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Mode label
        self.mode_label = QLabel("Mode: YOLO")
        self.status_bar.addPermanentWidget(self.mode_label)
        
        # Image info label
        self.image_info_label = QLabel("No image loaded")
        self.status_bar.addPermanentWidget(self.image_info_label)
        
        # Position label
        self.position_label = QLabel("X: 0, Y: 0")
        self.status_bar.addPermanentWidget(self.position_label)
        
        # Image counter
        self.counter_label = QLabel("0/0")
        self.status_bar.addPermanentWidget(self.counter_label)
        
        self.status_bar.showMessage("Ready")
        
    def setup_central_widget(self):
        """Create the central widget with splitter for file browser and canvas"""
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create splitter
        self.splitter = QSplitter(Qt.Horizontal)
        
        # ===== LEFT PANEL - FILE BROWSER =====
        self.file_browser = QWidget()
        self.file_browser.setMinimumWidth(250)
        self.file_browser.setMaximumWidth(400)
        
        browser_layout = QVBoxLayout(self.file_browser)
        browser_layout.setContentsMargins(5, 5, 5, 5)
        
        # Label for file browser
        browser_label = QLabel("📁 Image Files")
        browser_label.setStyleSheet("font-weight: bold; padding: 5px;")
        browser_layout.addWidget(browser_label)
        
        # List widget for files
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.itemClicked.connect(self.on_file_selected)
        browser_layout.addWidget(self.file_list)
        
        # ===== MIDDLE PANEL - CANVAS =====
        self.canvas = AnnotationCanvas()
        self.canvas.set_class_manager(self.class_manager)
        
        # Connect the canvas signal to update position
        self.canvas.position_changed.connect(self.update_position)
        
        # Connect canvas shape selection to update toolbar
        self.canvas.shape_selected.connect(self.on_canvas_shape_selected)
        
        # ===== RIGHT PANEL - CLASSES =====
        self.class_panel = ClassPanel(self.class_manager)
        
        # Connect class selection to canvas
        self.class_panel.class_selected.connect(self.on_class_selected)
        
        # Add widgets to splitter
        self.splitter.addWidget(self.file_browser)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.class_panel)
        
        # Set initial sizes (20% browser, 60% canvas, 20% classes)
        self.splitter.setSizes([250, 700, 250])
        
        # Add splitter to layout
        layout.addWidget(self.splitter)

    def on_shape_deselected(self):
        """Handle when no shape is selected in toolbar"""
        if hasattr(self, 'canvas'):
            # Tell canvas that no shape is selected
            self.canvas.set_shape_type(None)
            self.status_bar.showMessage("No shape selected - click on shapes to select them", 2000)

    def on_canvas_shape_selected(self, shape_type):
        """Handle shape selection from canvas to update toolbar"""
        if hasattr(self, 'shape_toolbar'):
            if shape_type != "none":
                self.shape_toolbar.set_selected_shape(shape_type)    
        
    def on_class_selected(self, class_id):
        """Handle class selection"""
        cls = self.class_manager.get_class(class_id)
        if cls:
            self.status_bar.showMessage(f"Selected class: {cls.name}", 2000)
        
    def on_shape_selected(self, shape_type):
        """Handle shape selection from toolbar"""
        if hasattr(self, 'canvas'):
            self.canvas.set_shape_type(shape_type)
            self.status_bar.showMessage(f"Shape: {shape_type}", 1000)

    def on_mode_selected(self, index):
        """Handle mode selection from combobox"""
        if index == 0:
            self.switch_mode('yolo')
        else:
            self.switch_mode('unet')
        
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
        # Clear current list
        self.file_list.clear()
        self.image_files = []
        
        # Supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        
        # Get all files in folder
        try:
            all_files = os.listdir(folder_path)
            
            # Filter for image files
            for file in sorted(all_files):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    self.image_files.append(file)
                    
            # Add to list widget
            for file in self.image_files:
                item = QListWidgetItem(file)
                self.file_list.addItem(item)
                
            # Update counter
            self.update_image_counter()
            
            # Load first image if available
            if self.image_files:
                self.load_image(0)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load images: {str(e)}")
            
    def load_image(self, index):
        """Load image at specified index"""
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            image_path = os.path.join(self.image_folder, self.image_files[index])
            
            # Load image in canvas
            self.canvas.load_image(image_path)
            
            # Update UI
            self.file_list.setCurrentRow(index)
            self.update_image_counter()
            
            # Update status bar with image info
            self.image_info_label.setText(self.image_files[index])
            
            # Update window title
            self.setWindowTitle(f"Dual Annotator - {self.image_files[index]}")
            
            # Show success message
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
            
    def toggle_file_browser(self):
        """Show or hide the file browser panel"""
        if self.toggle_browser_action.isChecked():
            self.file_browser.show()
        else:
            self.file_browser.hide()
            
    def switch_mode(self, mode):
        """Switch between YOLO and U-Net modes"""
        if mode == 'yolo':
            self.yolo_mode_action.setChecked(True)
            self.unet_mode_action.setChecked(False)
            self.mode_label.setText("Mode: YOLO")
            self.canvas.set_mode('yolo')
            # Shape toolbar stays visible in both modes
            if hasattr(self, 'shape_toolbar'):
                self.shape_toolbar.show()
            self.mode_selector.setCurrentIndex(0)
        else:
            self.yolo_mode_action.setChecked(False)
            self.unet_mode_action.setChecked(True)
            self.mode_label.setText("Mode: U-Net")
            self.canvas.set_mode('unet')
            # Shape toolbar stays visible in both modes
            if hasattr(self, 'shape_toolbar'):
                self.shape_toolbar.show()
            self.mode_selector.setCurrentIndex(1)
            
        self.status_bar.showMessage(f"Switched to {mode.upper()} mode", 2000)
        
    def update_position(self, x, y):
        """Update cursor position in status bar (called from canvas)"""
        self.position_label.setText(f"X: {x}, Y: {y}")
        
    # ===== CANVAS DELEGATION METHODS =====
    def zoom_in(self):
        """Zoom in on the canvas"""
        if hasattr(self, 'canvas'):
            self.canvas.zoom_in()
        
    def zoom_out(self):
        """Zoom out of the canvas"""
        if hasattr(self, 'canvas'):
            self.canvas.zoom_out()
        
    def fit_to_window(self):
        """Fit image to window"""
        if hasattr(self, 'canvas'):
            self.canvas.fit_to_window()
    
    def copy_selected(self):
        """Copy selected shape to clipboard"""
        if hasattr(self, 'canvas'):
            self.canvas.copy_selected()
        
    def paste_shape(self):
        """Paste copied shape"""
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'pixmap') and self.canvas.pixmap and not self.canvas.pixmap.isNull():
            # Trigger paste at center of view
            center = self.canvas.rect().center()
            self.canvas.start_paste(center)
        
    def delete_selected(self):
        """Delete selected shape"""
        if hasattr(self, 'canvas'):
            self.canvas.delete_selected()
        
    # ===== OTHER METHODS =====
    def new_project(self):
        """Create a new project"""
        self.status_bar.showMessage("Creating new project...")
        print("New project created")
        
    def open_project(self):
        """Open an existing project"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Project Files (*.json)"
        )
        if file_path:
            self.status_bar.showMessage(f"Opened: {file_path}")
            
    def save_project(self):
        """Save the current project"""
        if self.current_file:
            self.status_bar.showMessage(f"Saving to: {self.current_file}")
        else:
            self.save_project_as()
            
    def save_project_as(self):
        """Save project with a new name"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "Project Files (*.json)"
        )
        if file_path:
            self.current_file = file_path
            self.status_bar.showMessage(f"Saved to: {file_path}")
            
    def import_images(self):
        """Import additional images"""
        if self.image_folder:
            self.open_image_folder()
        else:
            self.open_image_folder()
            
    def export_data(self, format_type):
        """Export annotations in specified format"""
        self.status_bar.showMessage(f"Exporting in {format_type} format...")
        
    def show_about(self):
        """Show about dialog"""
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
            "<li>Ring shapes for hollow objects</li>"
            "</ul>"
            "<p>Built with PyQt5 and Python 3.11</p>"
        )
        
    def undo(self):
        """Undo last action"""
        print("Undo")
        self.status_bar.showMessage("Undo", 1000)
        
    def redo(self):
        """Redo last undone action"""
        print("Redo")
        self.status_bar.showMessage("Redo", 1000)