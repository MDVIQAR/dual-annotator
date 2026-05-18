"""
mlops/export/infer_worker.py

QThread worker that launches infer_{model_type}.py as a subprocess
and streams PROGRESS / RESULT lines to the UI.
"""

import os
import sys
import subprocess
import pathlib

from PyQt5.QtCore import QThread, pyqtSignal


class InferWorker(QThread):
    log      = pyqtSignal(str)         # console line
    progress = pyqtSignal(int)         # 0-100
    result   = pyqtSignal(str, str)    # (image_name, abs_path_to_combined)
    finished = pyqtSignal(bool, int)   # (success, total_count)

    def __init__(self, model_type: str, onnx_path: str, config_path: str,
                 test_folder: str, out_folder: str, scripts_dir: str, parent=None):
        super().__init__(parent)
        self._model_type  = model_type
        self._onnx_path   = onnx_path
        self._config_path = config_path
        self._test_folder = test_folder
        self._out_folder  = out_folder
        self._scripts_dir = scripts_dir
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
            line = raw_line.rstrip()
            if line.startswith("PROGRESS "):
                try:
                    self.progress.emit(int(line.split()[1]))
                except (IndexError, ValueError):
                    pass
            elif line.startswith("RESULT "):
                # Format: RESULT {name}|{abs_path}
                payload = line[len("RESULT "):]
                parts   = payload.split("|", 1)
                if len(parts) == 2:
                    self._total += 1
                    self.result.emit(parts[0].strip(), parts[1].strip())
            elif line.startswith("DONE "):
                pass
            else:
                self.log.emit(line)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Inference script exited with code {proc.returncode}")
