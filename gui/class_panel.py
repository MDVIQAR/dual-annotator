# gui/class_panel.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem,
    QInputDialog, QMessageBox, QColorDialog,
    QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QFont

from core.class_manager import ClassManager, ClassCategory, CLASS_COLORS

class ClassPanel(QWidget):
    """Panel for managing annotation classes"""
    
    # Signals
    class_selected = pyqtSignal(str)  # Emits class ID when selected
    class_added = pyqtSignal()  # Emitted when class added
    class_removed = pyqtSignal()  # Emitted when class removed
    
    def __init__(self, class_manager: ClassManager):
        super().__init__()
        
        self.class_manager = class_manager
        
        # Set up UI
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("📋 Classes")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Class list
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.class_list.itemClicked.connect(self.on_class_clicked)
        layout.addWidget(self.class_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("➕ Add")
        self.add_btn.clicked.connect(self.add_class)
        button_layout.addWidget(self.add_btn)
        
        self.edit_btn = QPushButton("✏️ Edit")
        self.edit_btn.clicked.connect(self.edit_class)
        button_layout.addWidget(self.edit_btn)
        
        self.delete_btn = QPushButton("❌ Delete")
        self.delete_btn.clicked.connect(self.delete_class)
        button_layout.addWidget(self.delete_btn)
        
        layout.addLayout(button_layout)
        
        # Refresh the list
        self.refresh_list()
        
    def refresh_list(self):
        """Refresh the class list from manager"""
        self.class_list.clear()
        
        for cls in self.class_manager.get_all_classes():
            self.add_class_to_list(cls)
            
    def add_class_to_list(self, cls: ClassCategory):
        """Add a class to the list widget"""
        item = QListWidgetItem()
        
        # Create widget for the item
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Color indicator
        color_label = QLabel()
        color_label.setFixedSize(16, 16)
        color_label.setStyleSheet(f"background-color: {cls.color}; border: 1px solid gray;")
        layout.addWidget(color_label)
        
        # Class name
        name_label = QLabel(cls.name)
        name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(name_label)
        
        layout.addStretch()
        
        # Set item properties
        item.setSizeHint(widget.sizeHint())
        item.setData(Qt.UserRole, cls.id)
        
        self.class_list.addItem(item)
        self.class_list.setItemWidget(item, widget)
        
        # Highlight if this is the current class
        if cls.id == self.class_manager.current_class_id:
            item.setSelected(True)
            self.class_list.setCurrentItem(item)
            
    def add_class(self):
        """Add a new class"""
        name, ok = QInputDialog.getText(
            self, "Add Class", "Enter class name:"
        )
        
        if ok and name.strip():
            try:
                # Choose color - use a default red if CLASS_COLORS not available
                initial_color = QColor("#FF6B6B")  # Default red
                color = QColorDialog.getColor(
                    initial_color, 
                    self, 
                    "Choose class color"
                )
                
                if color.isValid():
                    cls = self.class_manager.add_class(
                        name.strip(), 
                        color.name()
                    )
                    self.add_class_to_list(cls)
                    self.class_added.emit()
                    
                    # Select the new class
                    self.select_class(cls.id)
                    
            except ValueError as e:
                QMessageBox.warning(self, "Error", str(e))
                
    def edit_class(self):
        """Edit selected class"""
        current_item = self.class_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "Info", "Please select a class to edit")
            return
            
        class_id = current_item.data(Qt.UserRole)
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
        current_item = self.class_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "Info", "Please select a class to delete")
            return
            
        class_id = current_item.data(Qt.UserRole)
        cls = self.class_manager.get_class(class_id)
        
        if cls:
            reply = QMessageBox.question(
                self, 
                "Confirm Delete",
                f"Are you sure you want to delete class '{cls.name}'?",
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
        self.class_selected.emit(class_id)
        
    def get_selected_class(self):
        """Get the currently selected class"""
        current_item = self.class_list.currentItem()
        if current_item:
            class_id = current_item.data(Qt.UserRole)
            return self.class_manager.get_class(class_id)
        return None