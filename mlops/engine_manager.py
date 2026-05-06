# mlops/engine_manager.py
# Phase 2: Lazy AI Engine — check + install PyTorch/Ultralytics on demand.
# The main .exe stays lightweight; the AI engine is downloaded ONLY when the
# user explicitly clicks "Initialize AI Engine" for the first time.

import sys
import subprocess
import importlib.util
from PyQt5.QtCore import QThread, pyqtSignal


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def check_engine() -> bool:
    """
    Returns True if `ultralytics` is importable in the current Python environment.
    This is fast — no actual import, just a spec check.
    """
    try:
        spec = importlib.util.find_spec("ultralytics")
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def get_python_executable() -> str:
    """Return the Python interpreter currently running this app."""
    return sys.executable


# ---------------------------------------------------------------------------
# Engine Installer Worker
# ---------------------------------------------------------------------------

class EngineInstallWorker(QThread):
    """
    Background thread that runs:
        pip install ultralytics torch --upgrade

    Streams installation output line-by-line so the console stays alive.
    Emits finished(True) on success, finished(False) on failure.
    """
    log      = pyqtSignal(str)   # real-time pip output
    progress = pyqtSignal(int)   # rough 0-100 estimate
    finished = pyqtSignal(bool)  # True = success

    def run(self):
        self.log.emit("=" * 52)
        self.log.emit("  Installing AI Engine (PyTorch + Ultralytics)")
        self.log.emit("  This will only happen once.")
        self.log.emit("=" * 52)
        self.progress.emit(5)

        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "ultralytics", "numpy<2"
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                stripped = line.rstrip()
                if stripped:
                    self.log.emit(f"   {stripped}")
            proc.wait()

            if proc.returncode != 0:
                self.log.emit(f"\n[ERROR] pip exited with code {proc.returncode}")
                self.finished.emit(False)
                return

        except Exception as exc:
            self.log.emit(f"[ERROR] Failed to run pip: {exc}")
            self.finished.emit(False)
            return

        self.progress.emit(90)        # Final verification
        self.log.emit("\n[Verify] Checking ultralytics import...")
        if check_engine():
            self.progress.emit(100)
            self.log.emit("\n  AI Engine installed successfully!")
            self.log.emit("  You can now start training.")
            self.finished.emit(True)
        else:
            self.log.emit(
                "\n[ERROR] Installation finished but ultralytics could not be imported.\n"
                "        Try restarting the application."
            )
            self.finished.emit(False)
