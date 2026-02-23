# core/class_manager.py
import json
import uuid
from typing import List, Dict, Optional
from PyQt5.QtGui import QColor

# Predefined colors for classes
CLASS_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEEAD",  # Yellow
    "#D4A5A5",  # Pink
    "#9B59B6",  # Purple
    "#3498DB",  # Light Blue
    "#E67E22",  # Orange
    "#2ECC71",  # Emerald
]

class ClassCategory:
    """Represents a single class/category"""
    
    def __init__(self, name: str, color: str = None, class_id: str = None):
        self.id = class_id or str(uuid.uuid4())[:8]  # Short ID
        self.name = name
        self.color = color or CLASS_COLORS[hash(name) % len(CLASS_COLORS)]
        self.description = ""
        self.created_at = None
        self.is_active = True
        
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'id': self.id,
            'name': self.name,
            'color': self.color,
            'description': self.description,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        category = cls(data['name'], data['color'], data['id'])
        category.description = data.get('description', '')
        category.is_active = data.get('is_active', True)
        return category
    
    def get_qcolor(self):
        """Get Qt QColor object"""
        return QColor(self.color)

class ClassManager:
    """Manages all classes/categories"""
    
    def __init__(self):
        self.classes: Dict[str, ClassCategory] = {}
        self.current_class_id: Optional[str] = None
        
    def add_class(self, name: str, color: str = None) -> ClassCategory:
        """Add a new class"""
        # Check for duplicate names
        if self.get_class_by_name(name):
            raise ValueError(f"Class '{name}' already exists")
        
        category = ClassCategory(name, color)
        self.classes[category.id] = category
        return category
    
    def remove_class(self, class_id: str):
        """Remove a class"""
        if class_id in self.classes:
            del self.classes[class_id]
            
        # Clear current class if it was removed
        if self.current_class_id == class_id:
            self.current_class_id = None
    
    def get_class(self, class_id: str) -> Optional[ClassCategory]:
        """Get class by ID"""
        return self.classes.get(class_id)
    
    def get_class_by_name(self, name: str) -> Optional[ClassCategory]:
        """Get class by name"""
        for cls in self.classes.values():
            if cls.name.lower() == name.lower():
                return cls
        return None
    
    def get_all_classes(self) -> List[ClassCategory]:
        """Get all active classes"""
        return list(self.classes.values())
    
    def set_current_class(self, class_id: str):
        """Set the currently active class"""
        if class_id in self.classes:
            self.current_class_id = class_id
            
    def get_current_class(self) -> Optional[ClassCategory]:
        """Get the currently active class"""
        if self.current_class_id:
            return self.classes.get(self.current_class_id)
        return None
    
    def save_to_file(self, filepath: str):
        """Save classes to JSON file"""
        data = {
            'classes': [cls.to_dict() for cls in self.classes.values()],
            'current_class': self.current_class_id
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load classes from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.classes.clear()
        for cls_data in data['classes']:
            cls = ClassCategory.from_dict(cls_data)
            self.classes[cls.id] = cls
        
        self.current_class_id = data.get('current_class')