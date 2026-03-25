from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QPushButton, QHBoxLayout

class ModeSwitchDialog(QDialog):
    def __init__(self, from_mode, to_mode, shape_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Switch to {to_mode.upper()} mode?")
        self.coexist = False
        
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #d8e0f8;
            }
            QLabel, QRadioButton {
                color: #d8e0f8;
                font-size: 13px;
            }
            QPushButton {
                background-color: #1e2640;
                border: 1px solid #2e3a58;
                border-radius: 4px;
                padding: 6px 16px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2e3a58;
            }
            QPushButton:pressed {
                background-color: #4a6bff;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"This image has {shape_count} {from_mode.upper()} annotations."))
        
        self.radio_coexist = QRadioButton("Keep both visible (coexist)")
        self.radio_hide = QRadioButton(f"Hide {from_mode.upper()}, show {to_mode.upper()} only (recommended)")
        self.radio_hide.setChecked(True)
        
        layout.addWidget(self.radio_coexist)
        layout.addWidget(self.radio_hide)
        
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_switch = QPushButton("Switch Mode")
        
        btn_cancel.clicked.connect(self.reject)
        def on_switch():
            self.coexist = self.radio_coexist.isChecked()
            self.accept()
            
        btn_switch.clicked.connect(on_switch)
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_switch)
        layout.addLayout(btn_layout)
