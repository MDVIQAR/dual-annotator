from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QCheckBox, QSpinBox, QPushButton, QProgressBar, QMessageBox, QGroupBox, QButtonGroup
import os
import json
from datetime import datetime

class ExportWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_sig = pyqtSignal(dict)  # 'finished' is a built-in signal name, use finished_sig
    error = pyqtSignal(str)

    def __init__(self, manager, kwargs):
        super().__init__()
        self.manager = manager
        self.kwargs = kwargs

    def run(self):
        try:
            summary = self.manager.run_export(
                **self.kwargs,
                progress_callback=self.on_progress
            )
            self.finished_sig.emit(summary)
        except Exception as e:
            self.error.emit(str(e))

    def on_progress(self, current, total, msg):
        self.progress.emit(current, total, msg)

class ExportDialog(QDialog):
    def __init__(self, project_manager, class_manager, image_folder, image_files, current_mode, parent=None):
        super().__init__(parent)
        self.pm = project_manager
        self.cm = class_manager
        self.image_folder = image_folder
        self.image_files = image_files
        self.current_mode = current_mode
        self.export_manager = None
        self.worker = None

        self.setWindowTitle("Export Annotations")
        self.setMinimumWidth(400)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #d8e0f8;
            }
            QLabel, QCheckBox, QRadioButton, QGroupBox {
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
            QPushButton:disabled {
                background-color: #161c2e;
                color: #7a88b0;
            }
            QSpinBox {
                background-color: #161c2e;
                color: #ffffff;
                border: 1px solid #2e3a58;
                border-radius: 4px;
                padding: 4px;
            }
            QGroupBox {
                border: 1px solid #2e3a58;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QProgressBar {
                border: 1px solid #2e3a58;
                border-radius: 4px;
                background-color: #161c2e;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4a6bff;
                border-radius: 3px;
            }
        """)

        layout = QVBoxLayout(self)

        # 1. Format Group
        fmt_group = QGroupBox("Format")
        fmt_layout = QHBoxLayout()
        self.radio_yolo = QRadioButton("YOLO")
        self.radio_coco = QRadioButton("COCO")
        self.radio_voc = QRadioButton("Pascal VOC")
        self.radio_unet = QRadioButton("UNet Masks")
        self.radio_yolo.setChecked(True)
        self.fmt_group_btn = QButtonGroup(self)
        self.fmt_group_btn.addButton(self.radio_yolo, id=1)
        self.fmt_group_btn.addButton(self.radio_coco, id=2)
        self.fmt_group_btn.addButton(self.radio_voc, id=3)
        self.fmt_group_btn.addButton(self.radio_unet, id=4)
        fmt_layout.addWidget(self.radio_yolo)
        fmt_layout.addWidget(self.radio_coco)
        fmt_layout.addWidget(self.radio_voc)
        fmt_layout.addWidget(self.radio_unet)
        fmt_group.setLayout(fmt_layout)
        layout.addWidget(fmt_group)

        # 1b. UNet mask mode sub-group (hidden unless UNet selected)
        # Only scale: 0/1 (for training) or 0/255 (for visualization).
        # Binary vs multiclass is auto-detected per image at export time.
        self.unet_group = QGroupBox("UNet Mask Scale")
        unet_layout = QHBoxLayout()
        self.radio_bin01  = QRadioButton("0 / 1  (training)")
        self.radio_bin255 = QRadioButton("0 / 255  (visualization)")
        self.radio_bin01.setChecked(True)
        self.mask_mode_group = QButtonGroup(self)
        self.mask_mode_group.addButton(self.radio_bin01,  id=1)
        self.mask_mode_group.addButton(self.radio_bin255, id=2)
        unet_layout.addWidget(self.radio_bin01)
        unet_layout.addWidget(self.radio_bin255)
        self.unet_group.setLayout(unet_layout)
        self.unet_group.setVisible(False)
        layout.addWidget(self.unet_group)

        # Wire format-change signal for all format buttons
        self.fmt_group_btn.buttonClicked.connect(self._on_fmt_changed)

        # 2. Split Group
        split_group = QGroupBox("Dataset Split")
        split_layout = QVBoxLayout()
        self.chk_split = QCheckBox("Enable Train/Val/Test Split")
        self.chk_split.setChecked(True)
        self.chk_split.stateChanged.connect(self._toggle_split)
        split_layout.addWidget(self.chk_split)

        pct_layout = QHBoxLayout()
        pct_layout.addWidget(QLabel("Train %:"))
        self.spin_train = QSpinBox()
        self.spin_train.setRange(0, 100)
        self.spin_train.setValue(80)
        pct_layout.addWidget(self.spin_train)
        
        pct_layout.addWidget(QLabel("Val %:"))
        self.spin_val = QSpinBox()
        self.spin_val.setRange(0, 100)
        self.spin_val.setValue(20)
        pct_layout.addWidget(self.spin_val)
        
        pct_layout.addStretch()
        pct_layout.addWidget(QLabel("Seed:"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        pct_layout.addWidget(self.spin_seed)
        
        split_layout.addLayout(pct_layout)
        split_group.setLayout(split_layout)
        layout.addWidget(split_group)

        # 3. Output Options
        opt_group = QGroupBox("Options")
        opt_layout = QVBoxLayout()
        self.chk_delta = QCheckBox("Incremental/Delta Export (Skip unchanged)")
        self.chk_delta.setChecked(True)
        self.chk_annotated = QCheckBox("Draw annotations on images")
        self.chk_annotated.setChecked(True)   # default for YOLO
        self.chk_timestamp = QCheckBox("Use timestamped subfolder")
        self.chk_timestamp.setChecked(True)
        opt_layout.addWidget(self.chk_delta)
        opt_layout.addWidget(self.chk_annotated)
        opt_layout.addWidget(self.chk_timestamp)
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        # Counts
        self.lbl_counts = QLabel("Calculating...")
        layout.addWidget(self.lbl_counts)

        # Apply initial formatting rules based on default 'YOLO' selection
        self._on_fmt_changed()

        # Progress
        self.progress_lbl = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        self.progress_lbl.hide()
        layout.addWidget(self.progress_lbl)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_export = QPushButton("Export")
        self.btn_export.setStyleSheet("background-color: #4a6bff; border: none; font-weight: bold;")
        self.btn_export.clicked.connect(self.start_export)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_export)
        layout.addLayout(btn_layout)

        self._read_counts()

    def _on_fmt_changed(self, _button=None):
        """Show/hide UNet panel and auto-adjust option checkboxes per format."""
        is_unet = self.radio_unet.isChecked()
        is_yolo = self.radio_yolo.isChecked()
        self.unet_group.setVisible(is_unet)

        if is_unet:
            # UNet: Only timestamp on, rest unchecked
            self.chk_split.setChecked(False)
            self.chk_delta.setChecked(False)
            self.chk_annotated.setChecked(False)
            self.chk_annotated.setEnabled(False)
            self.chk_timestamp.setChecked(True)
        elif is_yolo:
            # YOLO: Draw annotations and timestamp on, rest unchecked
            self.chk_split.setChecked(False)
            self.chk_delta.setChecked(False)
            self.chk_annotated.setChecked(True)
            self.chk_annotated.setEnabled(True)
            self.chk_timestamp.setChecked(True)
        else:
            # COCO / VOC: Only timestamp on, rest unchecked
            self.chk_split.setChecked(False)
            self.chk_delta.setChecked(False)
            self.chk_annotated.setChecked(False)
            self.chk_annotated.setEnabled(True)
            self.chk_timestamp.setChecked(True)

    def _toggle_split(self, state):
        enabled = state != 0
        self.spin_train.setEnabled(enabled)
        self.spin_val.setEnabled(enabled)
        self.spin_seed.setEnabled(enabled)

    def _read_counts(self):
        # Count annotated, unannotated
        annotated = 0
        unannotated = 0
        skipped = 0
        if os.path.exists(self.pm.annotations_dir):
            for filename in os.listdir(self.pm.annotations_dir):
                if not filename.endswith(".json") or filename.endswith(".tmp"):
                    continue
                path = os.path.join(self.pm.annotations_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        st = data.get("status", "")
                        if st in ("annotated", "in_progress"):
                            annotated += 1
                        elif st == "unannotated":
                            unannotated += 1
                        else:
                            skipped += 1
                except:
                    pass
        self.lbl_counts.setText(f"Images to process: {annotated} annotated, {skipped} skipped, {unannotated} unannotated")
        if annotated == 0:
            self.btn_export.setEnabled(False)
            self.btn_export.setToolTip("No annotated images to export.")

    def set_export_manager(self, export_manager):
        self.export_manager = export_manager

    def start_export(self):
        if not self.export_manager:
            QMessageBox.critical(self, "Error", "ExportManager not set.")
            return

        split_en = self.chk_split.isChecked()
        t_pct = self.spin_train.value()
        v_pct = self.spin_val.value()
        te_pct = 100 - t_pct - v_pct  # test gets the remainder

        if split_en and (t_pct + v_pct > 100):
            QMessageBox.warning(self, "Validation Error", "Train % + Val % cannot exceed 100%.")
            return

        fmt = "yolo"
        if self.radio_coco.isChecked():  fmt = "coco"
        elif self.radio_voc.isChecked(): fmt = "voc"
        elif self.radio_unet.isChecked(): fmt = "unet"

        # Output dir selection logic
        ts = ""
        if self.chk_timestamp.isChecked():
            ts = "_" + datetime.now().strftime("%Y-%m-%d_%H-%M")
        dir_name = f"{fmt}{ts}"
        output_dir = os.path.join(self.image_folder, "exports", dir_name)

        kwargs = {
            "output_dir": output_dir,
            "fmt": fmt,
            "split": split_en,
            "train_pct": t_pct,
            "val_pct": v_pct,
            "test_pct": te_pct,
            "seed": self.spin_seed.value(),
            "delta_only": self.chk_delta.isChecked(),
            "draw_annotated": self.chk_annotated.isChecked()
        }

        # UNet mask mode — auto-detect binary vs multiclass per image;
        # the user only picks the pixel scale (0/1 or 0/255).
        if fmt == "unet":
            scale_255 = self.mask_mode_group.checkedId() == 2
            kwargs["unet_mask_mode"] = "auto_255" if scale_255 else "auto_01"
        else:
            kwargs["unet_mask_mode"] = "binary_01"

        self.btn_export.setEnabled(False)
        self.chk_split.setEnabled(False)
        self.fmt_group_btn.button(1).setEnabled(False)
        self.fmt_group_btn.button(2).setEnabled(False)
        self.fmt_group_btn.button(3).setEnabled(False)
        self.fmt_group_btn.button(4).setEnabled(False)
        self.chk_delta.setEnabled(False)
        self.chk_annotated.setEnabled(False)
        self.chk_timestamp.setEnabled(False)
        
        self.progress_bar.show()
        self.progress_lbl.show()

        self.worker = ExportWorker(self.export_manager, kwargs)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished_sig.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def update_progress(self, current, total, msg):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_lbl.setText(msg)

    def on_success(self, summary):
        from gui.export_summary_dialog import ExportSummaryDialog
        dlg = ExportSummaryDialog(summary, self)
        dlg.exec_()
        self.accept()

    def on_error(self, err_msg):
        QMessageBox.critical(self, "Export Failed", f"An error occurred during export:\n{err_msg}")
        self.reject()
