# gui/class_panel.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem,
    QInputDialog, QMessageBox, QColorDialog,
    QAbstractItemView, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QFont, QPixmap, QPainter

from core.class_manager import ClassManager, ClassCategory

class ClassPanel(QWidget):
    """Panel for managing annotation classes"""
    
    # Signals
    class_selected = pyqtSignal(str)
    class_added = pyqtSignal()
    class_removed = pyqtSignal()
    
    def __init__(self, class_manager: ClassManager):
        super().__init__()
        
        self.class_manager = class_manager
        
        # Set up UI with clean, modern design
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("📋 CLASSES")
        header.setStyleSheet("""
            QLabel {
                color: #aaa;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
                padding: 4px 0;
            }
        """)
        layout.addWidget(header)
        
        # Class list with clean styling
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.class_list.itemClicked.connect(self.on_class_clicked)
        self.class_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                outline: none;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 0px;
                border-bottom: 1px solid #2a2a2a;
            }
            QListWidget::item:selected {
                background-color: #1e3a5a;
                border-left: 3px solid #8ab4f8;
            }
            QListWidget::item:hover {
                background-color: #2a2a2a;
            }
        """)
        layout.addWidget(self.class_list)
        
        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        
        # Add button
        self.add_btn = QPushButton("➕ Add")
        self.add_btn.setCursor(Qt.PointingHandCursor)
        self.add_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
        """)
        self.add_btn.clicked.connect(self.add_class)
        btn_layout.addWidget(self.add_btn)
        
        # Edit button
        self.edit_btn = QPushButton("✏️ Edit")
        self.edit_btn.setCursor(Qt.PointingHandCursor)
        self.edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
        """)
        self.edit_btn.clicked.connect(self.edit_class)
        btn_layout.addWidget(self.edit_btn)
        
        # Delete button
        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: #ff8a8a;
                border: 1px solid #5a3a3a;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #5a3a3a;
                color: #ffffff;
                border-color: #ff8a8a;
            }
            QPushButton:pressed {
                background-color: #3a2a2a;
            }
        """)
        self.delete_btn.clicked.connect(self.delete_class)
        btn_layout.addWidget(self.delete_btn)
        
        layout.addLayout(btn_layout)
        
        # Initial refresh
        self.refresh_list()
        
    def refresh_list(self):
        """Refresh the class list"""
        self.class_list.clear()
        for cls in self.class_manager.get_all_classes():
            self.add_class_to_list(cls)
            
    def add_class_to_list(self, cls: ClassCategory):
        """Add a class to the list with clean layout"""
        item = QListWidgetItem()
        item.setData(Qt.UserRole, cls.id)
        item.setSizeHint(QSize(200, 36))
        
        # Create custom widget for the item
        widget = QWidget()
        widget.setStyleSheet("background-color: transparent;")
        
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Color square (bigger and more visible)
        color_box = QLabel()
        color_box.setFixedSize(20, 20)
        color_box.setStyleSheet(f"""
            background-color: {cls.color};
            border: 1px solid #4a4a4a;
            border-radius: 3px;
        """)
        layout.addWidget(color_box)
        
        # Class name with good contrast
        name_label = QLabel(cls.name)
        name_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-weight: 500;
                font-size: 12px;
                background-color: transparent;
            }
        """)
        layout.addWidget(name_label)
        
        layout.addStretch()
        
        self.class_list.addItem(item)
        self.class_list.setItemWidget(item, widget)
        
        # Highlight if current
        if cls.id == self.class_manager.current_class_id:
            item.setSelected(True)
            self.class_list.setCurrentItem(item)
    
    def add_class(self):
        """Add a new class"""
        name, ok = QInputDialog.getText(
            self, "Add Class", "Enter class name:",
            text=""
        )
        
        if ok and name.strip():
            # Choose color
            color = QColorDialog.getColor(
                QColor("#FF6B6B"),
                self,
                "Choose class color"
            )
            
            if color.isValid():
                try:
                    cls = self.class_manager.add_class(name.strip(), color.name())
                    self.add_class_to_list(cls)
                    self.class_added.emit()
                    self.select_class(cls.id)
                except ValueError as e:
                    QMessageBox.warning(self, "Error", str(e))
    
    def edit_class(self):
        """Edit selected class"""
        current = self.class_list.currentItem()
        if not current:
            QMessageBox.information(self, "Info", "Please select a class to edit")
            return
            
        class_id = current.data(Qt.UserRole)
        cls = self.class_manager.get_class(class_id)
        
        if cls:
            # Edit name
            name, ok = QInputDialog.getText(
                self, "Edit Class", "Enter new class name:",
                text=cls.name
            )
            
            if ok and name.strip():
                # Edit color
                color = QColorDialog.getColor(
                    QColor(cls.color),
                    self,
                    "Choose new class color"
                )
                
                if color.isValid():
                    cls.name = name.strip()
                    cls.color = color.name()
                    self.refresh_list()
    
    def delete_class(self):
        """Delete selected class"""
        current = self.class_list.currentItem()
        if not current:
            return
            
        class_id = current.data(Qt.UserRole)
        cls = self.class_manager.get_class(class_id)
        
        if cls:
            reply = QMessageBox.question(
                self,
                "Confirm Delete",
                f"Delete class '{cls.name}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.class_manager.remove_class(class_id)
                self.refresh_list()
                self.class_removed.emit()
    
    def on_class_clicked(self, item):
        """Handle class selection"""
        class_id = item.data(Qt.UserRole)
        self.select_class(class_id)
    
    def select_class(self, class_id):
        """Select a class by ID"""
        self.class_manager.set_current_class(class_id)
        
        # Update selection
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.data(Qt.UserRole) == class_id:
                item.setSelected(True)
                self.class_list.setCurrentItem(item)
            else:
                item.setSelected(False)
        
        self.class_selected.emit(class_id)