# gui/main_window.py
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QMenu, QToolBar, 
    QStatusBar, QLabel, QWidget, QVBoxLayout,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QKeySequence

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Dual Annotator - New Project")
        self.setGeometry(100, 100, 1200, 800)  # x, y, width, height
        self.setMinimumSize(800, 600)  # Minimum window size
        
        # Initialize UI components
        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_status_bar()
        self.setup_central_widget()
        
        # Application state
        self.current_file = None
        self.is_modified = False
        
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
        
        file_menu.addSeparator()  # Line between items
        
        # Import Images
        import_action = QAction('&Import Images...', self)
        import_action.setStatusTip('Import images from folder')
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
        
        # Cut
        cut_action = QAction('Cu&t', self)
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.setStatusTip('Cut selected item')
        cut_action.triggered.connect(self.cut)
        edit_menu.addAction(cut_action)
        
        # Copy
        copy_action = QAction('&Copy', self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.setStatusTip('Copy selected item')
        copy_action.triggered.connect(self.copy)
        edit_menu.addAction(copy_action)
        
        # Paste
        paste_action = QAction('&Paste', self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.setStatusTip('Paste copied item')
        paste_action.triggered.connect(self.paste)
        edit_menu.addAction(paste_action)
        
        edit_menu.addSeparator()
        
        # Delete
        delete_action = QAction('&Delete', self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.setStatusTip('Delete selected item')
        delete_action.triggered.connect(self.delete)
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
        
        # ===== MODE MENU =====
        mode_menu = menubar.addMenu('&Mode')
        
        # YOLO Mode (default)
        self.yolo_mode_action = QAction('&YOLO Detection', self)
        self.yolo_mode_action.setCheckable(True)
        self.yolo_mode_action.setChecked(True)  # Default selected
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
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add some basic actions (icons will be added later)
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_project)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        undo_action = QAction("Undo", self)
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)
        
        toolbar.addSeparator()
        
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
    def setup_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add permanent widgets
        self.mode_label = QLabel("Mode: YOLO")
        self.status_bar.addPermanentWidget(self.mode_label)
        
        self.position_label = QLabel("X: 0, Y: 0")
        self.status_bar.addPermanentWidget(self.position_label)
        
        self.image_info_label = QLabel("No image loaded")
        self.status_bar.addPermanentWidget(self.image_info_label)
        
        # Show ready message
        self.status_bar.showMessage("Ready", 3000)
        
    def setup_central_widget(self):
        """Create the central widget (placeholder for canvas)"""
        central = QWidget()
        self.setCentralWidget(central)
        
        # Simple layout with placeholder text
        layout = QVBoxLayout(central)
        
        placeholder = QLabel("Canvas will go here")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("""
            QLabel {
                color: gray;
                font-size: 18px;
                border: 2px dashed #ccc;
                padding: 20px;
            }
        """)
        layout.addWidget(placeholder)
        
    # ===== ACTION METHODS =====
    
    def new_project(self):
        """Create a new annotation project"""
        self.status_bar.showMessage("Creating new project...")
        # We'll implement this later
        print("New project created")
        
    def open_project(self):
        """Open an existing project"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Project", 
            "", 
            "Project Files (*.json)"
        )
        if file_path:
            self.status_bar.showMessage(f"Opened: {file_path}")
            print(f"Opening project: {file_path}")
            
    def save_project(self):
        """Save the current project"""
        if self.current_file:
            self.status_bar.showMessage(f"Saving to: {self.current_file}")
            print(f"Saving project to: {self.current_file}")
        else:
            self.save_project_as()
            
    def save_project_as(self):
        """Save project with a new name"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "Project Files (*.json)"
        )
        if file_path:
            self.current_file = file_path
            self.status_bar.showMessage(f"Saved to: {file_path}")
            print(f"Saving project as: {file_path}")
            
    def import_images(self):
        """Import images from a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder"
        )
        if folder_path:
            self.status_bar.showMessage(f"Importing images from: {folder_path}")
            print(f"Importing images from: {folder_path}")
            
    def export_data(self, format_type):
        """Export annotations in specified format"""
        self.status_bar.showMessage(f"Exporting in {format_type} format...")
        print(f"Exporting in {format_type} format")
        
    def switch_mode(self, mode):
        """Switch between YOLO and U-Net modes"""
        if mode == 'yolo':
            self.yolo_mode_action.setChecked(True)
            self.unet_mode_action.setChecked(False)
            self.mode_label.setText("Mode: YOLO")
            self.status_bar.showMessage("Switched to YOLO mode", 2000)
        else:
            self.yolo_mode_action.setChecked(False)
            self.unet_mode_action.setChecked(True)
            self.mode_label.setText("Mode: U-Net")
            self.status_bar.showMessage("Switched to U-Net mode", 2000)
            
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
            "<li>Ring shapes for hollow objects</li>"
            "</ul>"
            "<p>Built with PyQt5 and Python 3.11</p>"
        )
        
    # ===== EDIT METHODS (Temporary placeholders) =====
    def undo(self):
        print("Undo")
        self.status_bar.showMessage("Undo", 1000)
        
    def redo(self):
        print("Redo")
        self.status_bar.showMessage("Redo", 1000)
        
    def cut(self):
        print("Cut")
        self.status_bar.showMessage("Cut", 1000)
        
    def copy(self):
        print("Copy")
        self.status_bar.showMessage("Copy", 1000)
        
    def paste(self):
        print("Paste")
        self.status_bar.showMessage("Paste", 1000)
        
    def delete(self):
        print("Delete")
        self.status_bar.showMessage("Delete", 1000)
        
    # ===== VIEW METHODS =====
    def zoom_in(self):
        print("Zoom in")
        self.status_bar.showMessage("Zoom in", 1000)
        
    def zoom_out(self):
        print("Zoom out")
        self.status_bar.showMessage("Zoom out", 1000)
        
    def fit_to_window(self):
        print("Fit to window")
        self.status_bar.showMessage("Fit to window", 1000)