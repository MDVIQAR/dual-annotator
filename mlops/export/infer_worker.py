"""
mlops/export/infer_worker.py

QThread worker that launches infer_{model_type}.py as a subprocess
and streams PROGRESS / RESULT lines to the UI.
"""

import os
import re
import sys
import subprocess
import pathlib

from PyQt5.QtCore import QThread, pyqtSignal

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\r')


class InferWorker(QThread):
    log      = pyqtSignal(str)                    # console line
    progress = pyqtSignal(int)                    # 0-100
    # (image_name, orig_path, annotated_path, overlay_path)
    # overlay_path is "" for YOLO (2-panel), non-empty for UNet (3-panel)
    result   = pyqtSignal(str, str, str, str)
    finished = pyqtSignal(bool, int)              # (success, total_count)

    def __init__(self, model_type: str, onnx_path: str, config_path: str,
                 test_folder: str, out_folder: str, scripts_dir: str,
                 conf: float = 0.25, parent=None):
        super().__init__(parent)
        self._model_type  = model_type
        self._onnx_path   = onnx_path
        self._config_path = config_path
        self._test_folder = test_folder
        self._out_folder  = out_folder
        self._scripts_dir = scripts_dir
        self._conf        = conf
        self._total       = 0

    def run(self):
        try:
            self._execute()
            self.finished.emit(True, self._total)
        except Exception as exc:
            self.log.emit(f"[ERROR] {exc}")
            self.finished.emit(False, 0)

    def _execute(self):
        script = os.path.join(self._scripts_dir, f"infer_{self._model_type}.py")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"Inference script not found: {script}")

        cmd = [
            sys.executable, script,
            "--onnx",        self._onnx_path,
            "--config",      self._config_path,
            "--test-folder", self._test_folder,
            "--out",         self._out_folder,
        ]
        # Confidence threshold only applies to detection models (YOLO)
        if self._model_type == "yolo":
            cmd += ["--conf", str(self._conf)]

        project_root = str(pathlib.Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            cmd,
            stdout   = subprocess.PIPE,
            stderr   = subprocess.STDOUT,
            text     = True,
            encoding = "utf-8",
            errors   = "replace",
            bufsize  = 1,
            env      = env,
        )

        for raw_line in proc.stdout:
            line = _ANSI_RE.sub("", raw_line).rstrip()
            if not line:
                continue
            if line.startswith("PROGRESS "):
                try:
                    self.progress.emit(int(line.split()[1]))
                except (IndexError, ValueError):
                    pass
            elif line.startswith("RESULT "):
                # YOLO:  RESULT name|orig_path|annotated_path
                # UNet:  RESULT name|orig_path|mask_path|overlay_path
                payload = line[len("RESULT "):]
                parts   = payload.split("|")
                if len(parts) >= 3:
                    self._total += 1
                    name      = parts[0].strip()
                    orig      = parts[1].strip()
                    annotated = parts[2].strip()
                    overlay   = parts[3].strip() if len(parts) >= 4 else ""
                    self.result.emit(name, orig, annotated, overlay)
            elif line.startswith("DONE "):
                pass
            else:
                self.log.emit(line)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Inference script exited with code {proc.returncode}")
