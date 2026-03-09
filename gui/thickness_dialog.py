from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QApplication
from PyQt5.QtCore import Qt, pyqtSignal
import math

class ThicknessDialog(QWidget):
    """
    Floating panel to set the thickness constraint for hollow shapes. 
    Appears near the shape.
    """
    offset_changed = pyqtSignal(str, float)
    applied = pyqtSignal(str, float)
    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint)
        self.setStyleSheet("""
            QWidget {
                background-color: #161c2e;
                color: #d8e0f8;
                border: 1px solid #2e3a58;
                border-radius: 8px;
            }
            QLabel { border: none; }
            QPushButton {
                background-color: #1e2640;
                border: 1px solid #2e3a58;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover { background-color: #2e3a58; }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px; 
                background: #1e2640;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a6bff;
                border: 1px solid #d8e0f8;
                width: 18px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 9px;
            }
        """)

        layout = QVBoxLayout(self)
        self.title_label = QLabel("⬡ Create Hollow Shape")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self.title_label)

        # Slider and Offset label
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Offset:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(50)
        self.slider.setValue(20)
        self.slider.valueChanged.connect(self.on_slider_changed)
        h_layout.addWidget(self.slider)

        self.val_label = QLabel("20px")
        h_layout.addWidget(self.val_label)
        layout.addLayout(h_layout)

        self.info_label = QLabel("Wall: 20px")
        self.info_label.setStyleSheet("color: #7a88b0; font-size: 11px;")
        layout.addWidget(self.info_label)

        self.warning_label = QLabel("⚠ Very thin — may be noisy")
        self.warning_label.setStyleSheet("color: #f55b7c; font-size: 11px;")
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        # Buttons
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.on_cancel)
        self.apply_btn = QPushButton("Apply ✓")
        self.apply_btn.setStyleSheet("background-color: #4a6bff; color: white; border: none;")
        self.apply_btn.clicked.connect(self.on_apply)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(self.apply_btn)
        layout.addLayout(btn_layout)

        self.current_shape = None
        self.current_mode = 'inner'

    def show_near_shape(self, shape, canvas_widget, mode='inner'):
        self.current_shape = shape
        self.current_mode = mode
        
        # Determine title
        if mode == 'inner':
            self.title_label.setText("⬡ Create Inner Shape")
        else:
            self.title_label.setText("⬢ Create Outer Shape")
            
        # Optional: compute a reasonable max for slider based on shape
        # max_offset = compute_max_offset(shape)
        # self.slider.setMaximum(int(max_offset))
        
        self.update_info_label(self.slider.value())
        
        # Position Dialog (dummy logic fallback if bounding_rect is unavail)
        pos = canvas_widget.mapToGlobal(canvas_widget.rect().center())
        if hasattr(shape, 'bounding_rect'):
            try:
                bbox = shape.bounding_rect()
                pos = canvas_widget.mapToGlobal(bbox.topRight().toPoint())
                pos.setX(pos.x() + 12)
            except:
                pass
        
        try:
            screen = QApplication.screenAt(pos).availableGeometry()
            if pos.x() + self.width() > screen.right():
                pos.setX(screen.right() - self.width() - 8)
        except:
            pass
            
        self.move(pos)
        self.show()
        self.on_slider_changed(self.slider.value()) # trigger initial preview

    def on_slider_changed(self, val):
        self.val_label.setText(f"{val}px")
        if val < 8:
            self.warning_label.show()
        else:
            self.warning_label.hide()
            
        self.update_info_label(val)
        self.offset_changed.emit(self.current_mode, float(val))

    def update_info_label(self, offset):
        shape = self.current_shape
        if not shape: return
        t = getattr(shape, 'type', 'unknown')
        if t == 'box':
            try:
                xw, yw, xh, yh = shape.to_pixels()
                w = abs(xh - xw)
                h = abs(yh - yw)
                if self.current_mode == 'inner':
                    iw = max(0, w - 2*offset)
                    ih = max(0, h - 2*offset)
                else:
                    iw = w + 2*offset
                    ih = h + 2*offset
                self.info_label.setText(f"Wall: {int(offset)}px  Shape: {int(iw)}×{int(ih)}")
            except:
                self.info_label.setText(f"Wall: {int(offset)}px")
        elif t == 'circle':
            try:
                cx, cy, r = shape.to_pixels()
                ir = max(0, r - offset) if self.current_mode == 'inner' else r + offset
                self.info_label.setText(f"Wall: {int(offset)}px  Radius: {int(ir)}")
            except:
                self.info_label.setText(f"Wall: {int(offset)}px")
        else:
            self.info_label.setText(f"Wall: {int(offset)}px")

    def on_cancel(self):
        self.cancelled.emit()
        self.close()

    def on_apply(self):
        self.applied.emit(self.current_mode, float(self.slider.value()))
        self.close()
