"""
mlops/export/onnx_worker.py

QThread worker that launches export_onnx.py as a subprocess
and streams PROGRESS / STATUS lines to the UI.
"""
import os
import re
import subprocess
import pathlib

from PyQt5.QtCore import QThread, pyqtSignal
from mlops.utils import find_python

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\r')


class OnnxWorker(QThread):
    log      = pyqtSignal(str)   # one line for the console
    progress = pyqtSignal(int)   # 0–100
    finished = pyqtSignal(bool)  # success

    def __init__(self, version_folder: str, scripts_dir: str, parent=None):
        super().__init__(parent)
        self._version_folder = version_folder
        self._scripts_dir    = scripts_dir
        self._proc           = None

    def run(self):
        try:
            self._execute()
            self.finished.emit(True)
        except Exception as exc:
            self.log.emit(f"[ERROR] {exc}")
            self.finished.emit(False)

    def _execute(self):
        script = os.path.join(self._scripts_dir, "export_onnx.py")
        cmd = [find_python(), script, "--version-folder", self._version_folder]

        project_root = str(pathlib.Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        self._proc = subprocess.Popen(
            cmd,
            stdout   = subprocess.PIPE,
            stderr   = subprocess.STDOUT,
            text     = True,
            encoding = "utf-8",
            errors   = "replace",
            bufsize  = 1,
            env      = env,
        )

        for raw_line in self._proc.stdout:
            line = _ANSI_RE.sub("", raw_line).rstrip()
            if not line:
                continue
            if line.startswith("PROGRESS "):
                try:
                    self.progress.emit(int(line.split()[1]))
                except (IndexError, ValueError):
                    pass
            elif line.startswith("STATUS "):
                self.log.emit(line[len("STATUS "):])
            elif line.startswith("[ERROR]"):
                self.log.emit(line)
            else:
                self.log.emit(line)

        self._proc.wait()
        if self._proc.returncode != 0:
            raise RuntimeError(f"Export script exited with code {self._proc.returncode}")
