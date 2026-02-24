# core/shape_base.py
import uuid
from abc import ABC, abstractmethod

class Shape(ABC):
    """Base class for all annotation shapes"""
    
    def __init__(self, class_id=None, image_size=(1, 1)):
        self.id = str(uuid.uuid4())[:8]
        self.class_id = class_id
        self.image_width, self.image_height = image_size
        self.selected = False
        self.created_at = None
        
    @abstractmethod
    def contains_point(self, x, y):
        """Check if point is inside the shape"""
        pass
    
    @abstractmethod
    def to_pixels(self):
        """Convert to pixel coordinates for drawing"""
        pass
    
    @abstractmethod
    def move(self, dx, dy):
        """Move the shape by delta x, y"""
        pass
    
    @abstractmethod
    def get_resize_handles(self):
        """Get resize handle positions"""
        pass
    
    @abstractmethod
    def resize_from_handle(self, handle_name, dx, dy):
        """Resize from a specific handle"""
        pass
    
    def copy(self):
        """Create a copy of this shape"""
        new_shape = self.__class__.__new__(self.__class__)
        new_shape.__dict__ = self.__dict__.copy()
        new_shape.id = str(uuid.uuid4())[:8]
        return new_shape