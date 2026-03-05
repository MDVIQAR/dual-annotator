# core/__init__.py
from .class_manager import ClassManager, ClassCategory
from .annotation import BoundingBox
from .polygon_shape import PolygonShape
from .bezier_shape import BezierPolygonShape
from .circle_shape import CircleShape
from .ellipse_shape import EllipseShape
from .ring_shape import FrameShape, DonutShape, HollowEllipseShape
from .template_manager import TemplateManager