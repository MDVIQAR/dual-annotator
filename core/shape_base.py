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
        
        # Hollow pair support
        self.inner_shape = None
        self.hollow_role = None  # 'outer' | 'inner' | None
        self.group_id = None
        self.is_hollow = False

        
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
        """Create a deep copy of this shape, preserving the full hollow structure."""
        new_shape = self.__class__.__new__(self.__class__)
        new_shape.__dict__ = {}

        for k, v in self.__dict__.items():
            if k == 'inner_shape':
                # Will be handled below after the loop — skip for now
                continue
            elif isinstance(v, list):
                # Deep-copy list: each element that is itself a list/tuple
                # is converted to tuple; nested Shape objects are copied recursively
                copied_list = []
                for item in v:
                    if isinstance(item, Shape):
                        copied_list.append(item.copy())
                    elif isinstance(item, (list, tuple)):
                        copied_list.append(tuple(item))
                    else:
                        copied_list.append(item)
                new_shape.__dict__[k] = copied_list
            else:
                new_shape.__dict__[k] = v

        # Give the new shape a fresh id
        new_shape.id = str(uuid.uuid4())[:8]

        # Deep-copy inner_shape and re-attach with a fresh group_id so the
        # new hollow pair is completely independent of the original.
        orig_inner = self.__dict__.get('inner_shape')
        if orig_inner is not None and hasattr(orig_inner, 'copy'):
            new_inner = orig_inner.copy()
            new_gid = str(uuid.uuid4())[:8]
            # Wire up the relationship on the NEW shape only
            new_shape.inner_shape = new_inner
            new_shape.is_hollow = True
            new_shape.hollow_role = 'outer'
            new_shape.group_id = new_gid
            new_inner.hollow_role = 'inner'
            new_inner.group_id = new_gid
        else:
            new_shape.inner_shape = None

        return new_shape
        
    def attach_inner(self, inner_shape):
        """Link inner_shape to this shape as the hollow cutout."""
        gid = str(uuid.uuid4())[:8]
        self.inner_shape = inner_shape
        self.is_hollow = True
        self.hollow_role = 'outer'
        self.group_id = gid
        inner_shape.hollow_role = 'inner'
        inner_shape.group_id = gid

    def detach_inner(self):
        """Detach inner shape and reset hollow state."""
        if self.inner_shape:
            self.inner_shape.hollow_role = None
            self.inner_shape.group_id = None
        self.inner_shape = None
        self.is_hollow = False
        self.hollow_role = None
        
    def to_dict(self):
        """Return base dictionary representation"""
        d = {
            'id': self.id,
            'class_id': self.class_id,
            'is_hollow': self.is_hollow,
            'group_id': self.group_id,
            'hollow_role': self.hollow_role
        }
        if self.inner_shape and hasattr(self.inner_shape, 'to_dict'):
            d['inner_shape'] = self.inner_shape.to_dict()
        return d
