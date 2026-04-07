from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QFrame, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from gui.template_engine import TemplateEngine

class DetectionWorker(QThread):
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, engine, template_boxes, existing_boxes):
        super().__init__()
        self.engine         = engine
        self.template_boxes = template_boxes
        self.existing_boxes = existing_boxes

    def run(self):
        try:
            results = self.engine.detect(self.template_boxes, self.existing_boxes)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class AutoAnnotatePanel(QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent or canvas)
        self.canvas = canvas
        self.engine = TemplateEngine()
        self.template_shapes = []   # list of canvas shape objects
        self.worker = None
        self._drag_pos = None
        self._minimized = False

        # Debounce timer for live updates (400ms after last slider move)
        self._live_timer = QTimer()
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(400)
        self._live_timer.timeout.connect(self.run_detection)

        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setFixedWidth(280)
        self._build_ui()
        self._apply_style()
        self.hide()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Title bar
        self.title_bar = QFrame()
        self.title_bar.setObjectName("title_bar")
        self.title_bar.setFixedHeight(30)
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(8, 0, 8, 0)
        
        lbl_drag = QLabel("⣿")
        lbl_drag.setStyleSheet("color: #7a88b0;")
        
        lbl_title = QLabel("Auto-Detect")
        lbl_title.setObjectName("title_lbl")
        
        self.btn_minimize = QPushButton("─")
        self.btn_minimize.setObjectName("btn_minimize")
        self.btn_minimize.setFixedSize(24, 24)
        self.btn_minimize.clicked.connect(self._toggle_minimize)
        
        btn_close = QPushButton("✕")
        btn_close.setObjectName("btn_close")
        btn_close.setFixedSize(24, 24)
        btn_close.clicked.connect(self.hide)
        
        title_layout.addWidget(lbl_drag)
        title_layout.addWidget(lbl_title, 1)
        title_layout.addWidget(self.btn_minimize)
        title_layout.addWidget(btn_close)

        main_layout.addWidget(self.title_bar)

        # Body
        self.body_widget = QWidget()
        body_layout = QVBoxLayout(self.body_widget)
        body_layout.setContentsMargins(12, 8, 12, 12)

        # Templates Group
        self.templates_group = QGroupBox("Templates")
        tmpl_layout = QVBoxLayout(self.templates_group)
        self.lbl_templates = QLabel("No templates — right-click a shape to add")
        self.lbl_templates.setWordWrap(True)
        btn_clear_tmpl = QPushButton("Clear Templates")
        btn_clear_tmpl.clicked.connect(self._clear_templates)
        tmpl_layout.addWidget(self.lbl_templates)
        tmpl_layout.addWidget(btn_clear_tmpl)
        body_layout.addWidget(self.templates_group)

        # Settings Group
        self.settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout(self.settings_group)
        
        # Threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(30, 99)
        self.slider_threshold.setValue(70)
        self.lbl_thresh_val = QLabel("0.70")
        self.slider_threshold.valueChanged.connect(self._on_thresh_changed)
        thresh_layout.addWidget(self.slider_threshold)
        thresh_layout.addWidget(self.lbl_thresh_val)
        settings_layout.addLayout(thresh_layout)

        # NMS
        nms_layout = QHBoxLayout()
        nms_layout.addWidget(QLabel("NMS:"))
        self.slider_nms = QSlider(Qt.Horizontal)
        self.slider_nms.setRange(50, 99)
        self.slider_nms.setValue(85)
        self.lbl_nms_val = QLabel("0.85")
        self.slider_nms.valueChanged.connect(self._on_nms_changed)
        nms_layout.addWidget(self.slider_nms)
        nms_layout.addWidget(self.lbl_nms_val)
        settings_layout.addLayout(nms_layout)

        self.chk_live = QCheckBox("Live updates")
        settings_layout.addWidget(self.chk_live)
        body_layout.addWidget(self.settings_group)

        # Run Button
        self.btn_run = QPushButton("▶ Run Detection")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.clicked.connect(self.run_detection)
        body_layout.addWidget(self.btn_run)

        # Status Group
        self.status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(self.status_group)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setObjectName("status_lbl")
        self.counts_lbl = QLabel("New: 0  Skipped: 0  Total: 0")
        self.counts_lbl.setObjectName("counts_lbl")
        status_layout.addWidget(self.status_lbl)
        status_layout.addWidget(self.counts_lbl)
        body_layout.addWidget(self.status_group)

        main_layout.addWidget(self.body_widget)

    def _apply_style(self):
        self.setStyleSheet("""
            AutoAnnotatePanel, QWidget { background-color: #1e1e2e; color: #d8e0f8; }
            QGroupBox {
                border: 1px solid #2e3a58; border-radius: 4px;
                margin-top: 8px; padding-top: 12px; color: #d8e0f8;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
            QPushButton {
                background-color: #1e2640; border: 1px solid #2e3a58;
                border-radius: 4px; padding: 5px 10px; color: #ffffff;
            }
            QPushButton:hover { background-color: #2e3a58; }
            QPushButton#btn_close, QPushButton#btn_minimize {
                padding: 0px; border-radius: 12px; background-color: transparent; border: none; font-size: 14px;
                color: #7a88b0;
            }
            QPushButton#btn_close:hover { background-color: #ff4a4a; color: white; border-radius: 12px; }
            QPushButton#btn_minimize:hover { background-color: #4a6bff; color: white; border-radius: 12px; }
            QPushButton#btn_run {
                background-color: #4a6bff; border: none; font-weight: bold; padding: 8px;
            }
            QPushButton#btn_run:hover { background-color: #3a5bef; }
            QPushButton#btn_run:disabled { background-color: #2a3a6a; color: #7a88b0; }
            QSlider::groove:horizontal {
                height: 4px; background: #2e3a58; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; background: #4a6bff;
                border-radius: 7px; margin: -5px 0;
            }
            QSlider::sub-page:horizontal { background: #4a6bff; border-radius: 2px; }
            QCheckBox { color: #d8e0f8; }
            QCheckBox::indicator:checked { background: #4a6bff; border: 1px solid #4a6bff; }
            QLabel#title_lbl { font-weight: bold; font-size: 13px; color: #ffffff; }
            QLabel#status_lbl { color: #8ab4f8; font-size: 11px; }
            QLabel#counts_lbl { color: #8b949e; font-size: 11px; }
            QFrame#title_bar { background-color: #161c2e; border-bottom: 1px solid #2e3a58; }
        """)

    def _on_thresh_changed(self, v):
        self.lbl_thresh_val.setText(f"{v/100:.2f}")
        if self.chk_live.isChecked() and self.template_shapes:
            self._live_timer.start()

    def _on_nms_changed(self, v):
        self.lbl_nms_val.setText(f"{v/100:.2f}")
        if self.chk_live.isChecked() and self.template_shapes:
            self._live_timer.start()

    def _clear_templates(self):
        self.template_shapes = []
        self._update_template_ui()

    def add_template_shape(self, shape):
        if shape not in self.template_shapes:
            self.template_shapes.append(shape)
            self._update_template_ui()
            if self.chk_live.isChecked() and self.canvas.image_path:
                self._live_timer.start()

    def _update_template_ui(self):
        cnt = len(self.template_shapes)
        if cnt == 0:
            self.lbl_templates.setText("No templates — right-click a shape to add")
        else:
            types = [getattr(s, 'type', 'unknown') for s in self.template_shapes]
            type_str = ", ".join(types)
            self.lbl_templates.setText(f"{cnt} templates: {type_str}")

    def show_panel(self):
        if self.canvas.image_path:
            self.engine.load_image(self.canvas.image_path)
        self.move(self.canvas.width() - self.width() - 16, 16)
        self.show()
        self.raise_()

    def _get_template_boxes_px(self):
        iw, ih = self.canvas.image_width, self.canvas.image_height
        boxes = []
        for shape in self.template_shapes:
            # Remove shapes that no longer exist on canvas
            if shape not in self.canvas.shapes:
                continue
            bbox = TemplateEngine.shape_to_pixel_bbox(shape, iw, ih)
            if bbox:
                boxes.append(bbox)
        return boxes

    def _get_existing_boxes_px(self):
        iw, ih = self.canvas.image_width, self.canvas.image_height
        result = []
        for shape in self.canvas.shapes:
            if getattr(shape, 'hollow_role', None) == 'inner':
                continue
            if shape in self.template_shapes:
                continue  # don't suppress near the template itself
            bbox = TemplateEngine.shape_to_pixel_bbox(shape, iw, ih)
            if bbox:
                result.append(bbox)
        return result

    def _set_status(self, text):
        self.status_lbl.setText(text)

    def _update_counts(self, new, skipped, total):
        self.counts_lbl.setText(f"New: {new}  Skipped: {skipped}  Total: {total}")

    def run_detection(self):
        template_boxes = self._get_template_boxes_px()
        if not template_boxes:
            self._set_status("Add at least one template shape first")
            return
        if not self.canvas.image_path:
            self._set_status("No image loaded")
            return

        self.engine.load_image(self.canvas.image_path)
        self.engine.threshold    = self.slider_threshold.value() / 100.0
        self.engine.nms_fraction = self.slider_nms.value() / 100.0
        existing_boxes = self._get_existing_boxes_px()

        self.btn_run.setEnabled(False)
        self._set_status("Detecting...")

        self.worker = DetectionWorker(
            self.engine, template_boxes, existing_boxes)
        self.worker.finished.connect(self._on_detection_done)
        self.worker.error.connect(self._on_detection_error)
        self.worker.start()

    def _on_detection_done(self, results):
        self.btn_run.setEnabled(True)
        iw, ih = self.canvas.image_width, self.canvas.image_height

        class_id = None
        for s in self.template_shapes:
            if s in self.canvas.shapes:
                class_id = getattr(s, 'class_id', None)
                if class_id:
                    break
        if not class_id and self.canvas.class_manager:
            cur = self.canvas.class_manager.get_current_class()
            if cur:
                class_id = cur.id

        if not results:
            self._set_status("No new detections found")
            total = len([s for s in self.canvas.shapes
                         if getattr(s, 'hollow_role', None) != 'inner'])
            self._update_counts(new=0, skipped=0, total=total)
            return

        from core.annotation import BoundingBox
        self.canvas.save_state()
        added = 0
        for (x, y, w, h) in results:
            box = BoundingBox(image_size=(iw, ih))
            box.from_pixels(x, y, x + w, y + h, iw, ih)
            box.class_id = class_id
            self.canvas.shapes.append(box)
            added += 1

        self.canvas.annotation_changed.emit()
        self.canvas.update()

        total = len([s for s in self.canvas.shapes
                     if getattr(s, 'hollow_role', None) != 'inner'])
        self._set_status(f"Done — added {added} annotation(s)")
        self._update_counts(new=added, skipped=0, total=total)

    def _on_detection_error(self, msg):
        self.btn_run.setEnabled(True)
        self._set_status(f"Error: {msg}")

    def _toggle_minimize(self):
        if not self._minimized:
            # minimize it
            self._pre_min_pos = self.pos()
            self._minimized = True
            self.body_widget.setVisible(False)
            self.btn_minimize.setText("□")
            self.adjustSize()
            
            # Snap to bottom right of parent
            if self.parentWidget():
                parent_rect = self.parentWidget().rect()
                self.move(parent_rect.width() - self.width() - 16, parent_rect.height() - self.height() - 16)
        else:
            # restore it
            self._minimized = False
            self.body_widget.setVisible(True)
            self.btn_minimize.setText("─")
            self.adjustSize()
            
            if hasattr(self, '_pre_min_pos'):
                # Also ensure we don't restore it off screen
                self.move(self._pre_min_pos)

    # Drag support using the title_bar area
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_bar.geometry().contains(event.pos()):
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
