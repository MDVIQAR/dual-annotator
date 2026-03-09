from PyQt5.QtWidgets import QMenu, QAction
from PyQt5.QtCore import pyqtSignal, QPoint

class ShapeContextMenu(QMenu):
    request_inner = pyqtSignal(object)
    request_outer = pyqtSignal(object)
    request_save_template = pyqtSignal(object)
    request_duplicate = pyqtSignal(object)
    request_delete = pyqtSignal(object)
    request_remove_hollow = pyqtSignal(object)
    request_undo = pyqtSignal()
    request_redo = pyqtSignal()
    request_paste = pyqtSignal(QPoint)
    
    # Polygon drawing signals
    request_undo_pt = pyqtSignal()
    request_redo_pt = pyqtSignal()
    request_finish = pyqtSignal()
    request_cancel = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = parent
        self.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #3a5a8a;
            }
            QMenu::separator {
                height: 1px;
                background: #3a3a3a;
                margin: 4px 8px;
            }
            QMenu::item:disabled { color: #888888; }
        """)

    def build_for_canvas(self, canvas, global_pos, local_pos=None):
        self.clear()
        
        # --- While drawing polygon ---
        if canvas.drawing_polygon or canvas.polygon_points:
            self._add_drawing_options(canvas)
            self.exec_(global_pos)
            return

        # --- While drawing bezier polygon ---
        if canvas.drawing_bezier or canvas.bezier_points:
            self._add_drawing_options(canvas, is_bezier=True)
            self.exec_(global_pos)
            return

        # --- Shape selected ---
        if canvas.selected_shape:
            shape = canvas.selected_shape
            shape_type = getattr(shape, 'type', 'unknown').title()
            
            header = self.addAction(f"{shape_type} #{shape.id[:4]}")
            header.setDisabled(True)
            self.addSeparator()

            # Delete
            delete_act = self.addAction("🗑 Delete Shape (Del)")
            delete_act.triggered.connect(lambda: self.request_delete.emit(shape))
            
            # Copy
            copy_act = self.addAction("📋 Copy Shape (Ctrl+C)")
            copy_act.triggered.connect(lambda: self.request_duplicate.emit(shape))
            
            self.addSeparator()

            # Hollow shape options
            if not getattr(shape, 'inner_shape', None):
                inner_act = self.addAction("⬡ Create Inner Shape")
                inner_act.triggered.connect(lambda: self.request_inner.emit(shape))
                
                outer_act = self.addAction("◎ Create Outer Shape")
                outer_act.triggered.connect(lambda: self.request_outer.emit(shape))
            else:
                inner_adj_act = self.addAction("⬡ Adjust Inner Shape")
                inner_adj_act.triggered.connect(lambda: self.request_inner.emit(shape))
                
                remove_hollow = self.addAction("⊘ Remove Hollow (make solid)")
                remove_hollow.triggered.connect(lambda: self.request_remove_hollow.emit(shape))
            
            self.addSeparator()
            
            # Save as Template
            save_tmpl_act = self.addAction("💾 Save as Template")
            save_tmpl_act.triggered.connect(lambda: self.request_save_template.emit(shape))
            if not getattr(shape, 'is_hollow', False):
                save_tmpl_act.setDisabled(True)
                save_tmpl_act.setToolTip("Create inner/outer first")

            self.addSeparator()
        
        # --- General actions ---
        if canvas.clipboard_shape and local_pos:
            paste_act = self.addAction("📋 Paste Shape (Ctrl+V)")
            paste_act.triggered.connect(lambda: self.request_paste.emit(local_pos))
        
        if canvas.undo_stack:
            undo_act = self.addAction("⟲ Undo (Ctrl+Z)")
            undo_act.triggered.connect(self.request_undo.emit)
        
        if canvas.redo_stack:
            redo_act = self.addAction("⟳ Redo (Ctrl+Y)")
            redo_act.triggered.connect(self.request_redo.emit)
        
        if self.actions():
            self.exec_(global_pos)

    def _add_drawing_options(self, canvas, is_bezier=False):
        points = canvas.bezier_points if is_bezier else canvas.polygon_points
        if len(points) > 0:
            undo_pt = self.addAction("⟲ Undo Last Point")
            undo_pt.triggered.connect(self.request_undo_pt.emit)
            
            undone_points = canvas.undone_bezier_points if is_bezier else getattr(canvas, 'undone_polygon_points', [])
            if undone_points:
                redo_pt = self.addAction("⟳ Redo Last Point")
                redo_pt.triggered.connect(self.request_redo_pt.emit)
        
        if len(points) >= 3:
            finish = self.addAction(f"✓ Finish {'Bezier ' if is_bezier else ''}Polygon (Enter)")
            finish.triggered.connect(self.request_finish.emit)
        
        cancel = self.addAction("✗ Cancel Drawing (Esc)")
        cancel.triggered.connect(self.request_cancel.emit)
