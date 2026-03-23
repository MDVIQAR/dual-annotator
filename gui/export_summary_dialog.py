import os
import sys
import subprocess
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit

class ExportSummaryDialog(QDialog):
    def __init__(self, summary_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Summary")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.output_dir = summary_dict.get("output_dir", "")
        
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #d8e0f8;
            }
            QLabel {
                color: #d8e0f8;
            }
            QPushButton {
                background-color: #1e2640;
                border: 1px solid #2e3a58;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #2e3a58;
            }
            QTextEdit {
                background-color: #161c2e;
                color: #ffffff;
                border: 1px solid #2e3a58;
                border-radius: 4px;
            }
        """)

        layout = QVBoxLayout(self)

        title = QLabel("<h2>Export Complete</h2>")
        layout.addWidget(title)

        stats = f"""
<b>Format:</b> {summary_dict.get('format', '').upper()}<br>
<b>Exported Images:</b> {summary_dict.get('exported', 0)}<br>
<b>Total Annotations Written:</b> {summary_dict.get('total_annotations', 0)}<br>
<b>Skipped (Unchanged):</b> {summary_dict.get('skipped_unchanged', 0)}<br>
<b>Skipped (No Class):</b> {summary_dict.get('skipped_no_class', 0)}<br>
<br>
<b>Splits:</b> Train: {summary_dict.get('train_count', 0)} | Val: {summary_dict.get('val_count', 0)} | Test: {summary_dict.get('test_count', 0)}
"""
        lbl_stats = QLabel(stats)
        lbl_stats.setTextFormat(1) # RichText
        layout.addWidget(lbl_stats)

        warnings = summary_dict.get("warnings", [])
        if warnings:
            layout.addWidget(QLabel("<b>Warnings:</b>"))
            txt_warn = QTextEdit()
            txt_warn.setReadOnly(True)
            txt_warn.setText("\\n".join(warnings))
            txt_warn.setStyleSheet("color: #f55b7c;")
            layout.addWidget(txt_warn)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_open = QPushButton("Open Folder")
        btn_open.clicked.connect(self.open_folder)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        
        btn_layout.addWidget(btn_open)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

    def open_folder(self):
        if not os.path.exists(self.output_dir):
            return
        
        if sys.platform == "win32":
            os.startfile(self.output_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", self.output_dir])
        else:
            subprocess.Popen(["xdg-open", self.output_dir])
