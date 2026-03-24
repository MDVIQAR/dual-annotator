# gui/class_panel.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem,
    QInputDialog, QMessageBox, QColorDialog,
    QAbstractItemView, QFrame, QShortcut
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QFont, QKeySequence

from core.class_manager import ClassManager, ClassCategory

class ClassPanel(QWidget):
    """Panel for managing annotation classes"""
    
    class_selected = pyqtSignal(str)
    class_added = pyqtSignal()
    class_removed = pyqtSignal()
    class_edited = pyqtSignal()  # AUTOSAVE INTEGRATION
    classes_reordered = pyqtSignal()
    
    def __init__(self, class_manager: ClassManager):
        super().__init__()
        
        self.class_manager = class_manager
        self._canvas = None  # Set externally by main_window
        
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("📋 CLASSES")
        header.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold; letter-spacing: 1px; padding: 4px 0;")
        layout.addWidget(header)
        
        # Class list — drag-to-reorder enabled
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.class_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.class_list.setDefaultDropAction(Qt.MoveAction)
        self.class_list.itemClicked.connect(self.on_class_clicked)
        self.class_list.model().rowsMoved.connect(self._on_rows_moved)
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
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        
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
        
        self.refresh_list()
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        """Set up keyboard shortcuts 1-9 to select classes"""
        self.shortcuts = []
        for i in range(1, 10):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            # Use default argument to capture current value of i in lambda
            shortcut.activated.connect(lambda idx=i: self.select_class_by_index(idx - 1))
            self.shortcuts.append(shortcut)
            
    def select_class_by_index(self, index):
        """Select a class by its list index (0-based)"""
        if index < self.class_list.count():
            item = self.class_list.item(index)
            class_id = item.data(Qt.UserRole)
            self.select_class(class_id)
            # The class_selected signal and UI update are handled by select_class
        
    def refresh_list(self):
        self.class_list.clear()
        for cls in self.class_manager.get_all_classes():
            self.add_class_to_list(cls)
            
    def add_class_to_list(self, cls: ClassCategory):
        item = QListWidgetItem()
        item.setData(Qt.UserRole, cls.id)
        item.setSizeHint(QSize(200, 36))
        
        widget = QWidget()
        widget.setStyleSheet("background-color: transparent;")
        
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Color square
        color_box = QLabel()
        color_box.setFixedSize(20, 20)
        color_box.setStyleSheet(f"""
            background-color: {cls.color};
            border: 1px solid #4a4a4a;
            border-radius: 3px;
        """)
        layout.addWidget(color_box)
        
        # Shortcut number badge
        idx = self.class_list.count() + 1
        if idx <= 9:
            num_label = QLabel(str(idx))
            num_label.setStyleSheet("color: #8ab4f8; font-weight: bold; background-color: rgba(138, 180, 248, 0.2); padding: 1px 4px; border-radius: 3px; font-size: 10px;")
            layout.addWidget(num_label)
        
        # Class name
        name_label = QLabel(cls.name)
        name_label.setStyleSheet("color: #ffffff; font-weight: 500; font-size: 12px; background-color: transparent;")
        layout.addWidget(name_label)
        
        layout.addStretch()
        
        self.class_list.addItem(item)
        self.class_list.setItemWidget(item, widget)
        
        if cls.id == self.class_manager.current_class_id:
            item.setSelected(True)
            self.class_list.setCurrentItem(item)
    
    def add_class(self):
        name, ok = QInputDialog.getText(self, "Add Class", "Enter class name:", text="")
        if ok and name.strip():
            # Check duplicate names before showing color picker
            existing = self.class_manager.get_class_by_name(name.strip())
            if existing:
                QMessageBox.warning(self, "Duplicate Class",
                    f"A class named '{name.strip()}' already exists.\nPlease choose a different name.")
                return
            color = QColorDialog.getColor(QColor("#FF6B6B"), self, "Choose class color")
            if color.isValid():
                try:
                    cls = self.class_manager.add_class(name.strip(), color.name())
                    self.add_class_to_list(cls)
                    self.class_added.emit()
                    self.select_class(cls.id)
                except ValueError as e:
                    QMessageBox.warning(self, "Error", str(e))
    
    def edit_class(self):
        current = self.class_list.currentItem()
        if not current:
            QMessageBox.information(self, "Info", "Please select a class to edit")
            return
        class_id = current.data(Qt.UserRole)
        cls = self.class_manager.get_class(class_id)
        if cls:
            name, ok = QInputDialog.getText(self, "Edit Class", "Enter new class name:", text=cls.name)
            if ok and name.strip():
                # Check duplicate names (excluding the current class itself)
                existing = self.class_manager.get_class_by_name(name.strip())
                if existing and existing.id != class_id:
                    QMessageBox.warning(self, "Duplicate Class",
                        f"A class named '{name.strip()}' already exists.\nPlease choose a different name.")
                    return
                color = QColorDialog.getColor(QColor(cls.color), self, "Choose new class color")
                if color.isValid():
                    cls.name = name.strip()
                    cls.color = color.name()
                    self.refresh_list()
                    self.class_edited.emit()  # AUTOSAVE INTEGRATION
    
    def delete_class(self):
        current = self.class_list.currentItem()
        if not current:
            return
        class_id = current.data(Qt.UserRole)
        cls = self.class_manager.get_class(class_id)
        if cls:
            # Count how many shapes use this class across the current canvas
            affected_count = 0
            if self._canvas:
                affected_count = sum(1 for s in self._canvas.shapes if getattr(s, 'class_id', None) == class_id)
            
            msg = f"Delete class '{cls.name}'?"
            if affected_count > 0:
                msg += f"\n\n⚠️ {affected_count} shape(s) on the current image use this class."
                msg += "\nThose shapes will become unassigned."
            
            reply = QMessageBox.question(self, "Confirm Delete", msg,
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.class_manager.remove_class(class_id)
                self.refresh_list()
                self.class_removed.emit()
    
    def on_class_clicked(self, item):
        class_id = item.data(Qt.UserRole)
        self.select_class(class_id)
    
    def select_class(self, class_id):
        self.class_manager.set_current_class(class_id)
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.data(Qt.UserRole) == class_id:
                item.setSelected(True)
                self.class_list.setCurrentItem(item)
            else:
                item.setSelected(False)
        
        # If a shape is currently selected on the canvas, reassign it
        if self._canvas and self._canvas.selected_shape:
            old_class_id = self._canvas.selected_shape.class_id
            if old_class_id != class_id:
                self._canvas.selected_shape.class_id = class_id
                self._canvas.update()
                if hasattr(self._canvas, 'annotation_changed'):
                    self._canvas.annotation_changed.emit()
        
        self.class_selected.emit(class_id)

    def _on_rows_moved(self, *args):
        """Called when the user drags to reorder classes."""
        ordered_ids = []
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            cid = item.data(Qt.UserRole)
            if cid:
                ordered_ids.append(cid)
        self.class_manager.reorder_classes(ordered_ids)
        # Rebuild the number badges
        self.refresh_list()
        self.classes_reordered.emit()
        self.class_edited.emit()  # Triggers autosave to persist new order