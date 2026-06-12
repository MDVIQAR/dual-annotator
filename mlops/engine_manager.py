# mlops/engine_manager.py
"""
Auto-setup AI Engine: creates a Python venv next to the exe and installs
torch + ML packages into it. Runs once on first training attempt.
"""

import os
import sys
import shutil
import subprocess
from PyQt5.QtCore import QThread, pyqtSignal

from mlops.utils import find_python


def needs_setup() -> bool:
    """Return True if we're in a frozen exe and no AI Engine venv exists yet."""
    if not getattr(sys, "frozen", False):
        return False
    return find_python() == sys.executable


def _get_venv_dir() -> str:
    """Return the path where the venv should be created (next to the exe)."""
    return os.path.join(os.path.dirname(sys.executable), "venv")


def _find_system_python() -> str | None:
    """Find a system Python 3.11+ that can create venvs."""
    # Try 'python' on PATH
    py = shutil.which("python")
    if py and os.path.normcase(py) != os.path.normcase(sys.executable):
        try:
            result = subprocess.run(
                [py, "--version"],
                capture_output=True, text=True, timeout=10
            )
            version = result.stdout.strip()
            if result.returncode == 0 and "3.1" in version:
                return py
        except Exception:
            pass

    # Try 'py -3' (Python Launcher for Windows)
    py_launcher = shutil.which("py")
    if py_launcher:
        try:
            result = subprocess.run(
                [py_launcher, "-3", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return f"{py_launcher} -3"
        except Exception:
            pass

    return None


class EngineInstallWorker(QThread):
    """
    Background thread that:
      1. Finds system Python
      2. Creates a venv next to the exe
      3. Installs torch + ML packages via pip
      4. Verifies the installation
    """
    log      = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._do_install()
        except Exception as e:
            self.log.emit(f"\n[ERROR] Setup failed: {e}")
            self.finished.emit(False)

    def _do_install(self):
        venv_dir = _get_venv_dir()

        # ── Step 1: Find system Python ──
        self.log.emit("=" * 52)
        self.log.emit("  AI Engine Setup (one-time, takes 5-10 minutes)")
        self.log.emit("=" * 52)
        self.log.emit("")
        self.log.emit("[1/4] Finding Python installation...")
        self.progress.emit(5)

        system_py = _find_system_python()
        if system_py is None:
            self.log.emit("")
            self.log.emit("[ERROR] Python 3.11+ not found on this system.")
            self.log.emit("        Please install Python from:")
            self.log.emit("        https://www.python.org/downloads/")
            self.log.emit("")
            self.log.emit("        Make sure to check 'Add Python to PATH'")
            self.log.emit("        during installation, then try again.")
            self.finished.emit(False)
            return

        self.log.emit(f"  Found: {system_py}")

        # ── Step 2: Create venv ──
        self.log.emit("")
        self.log.emit("[2/4] Creating virtual environment...")
        self.progress.emit(10)

        if os.path.isdir(venv_dir):
            self.log.emit("  Removing old venv...")
            shutil.rmtree(venv_dir, ignore_errors=True)

        if " " in system_py:
            parts = system_py.split()
            cmd = parts + ["-m", "venv", venv_dir]
        else:
            cmd = [system_py, "-m", "venv", venv_dir]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.log.emit(f"[ERROR] Failed to create venv: {result.stderr}")
            self.finished.emit(False)
            return

        venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
        if not os.path.isfile(venv_py):
            self.log.emit("[ERROR] Venv created but python.exe not found inside it")
            self.finished.emit(False)
            return

        self.log.emit(f"  Created: {venv_dir}")

        # ── Step 3: Install packages ──
        self.log.emit("")
        self.log.emit("[3/4] Installing AI packages (this takes a few minutes)...")
        self.log.emit("")
        self.progress.emit(15)

        # Upgrade pip
        self._run_pip(venv_py, ["install", "--upgrade", "pip"], "pip upgrade")
        if self._cancelled:
            return

        # Install torch CPU
        self.log.emit("  Installing PyTorch (CPU)...")
        self.progress.emit(20)
        ok = self._run_pip(venv_py, [
            "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], "PyTorch")
        if not ok:
            return
        if self._cancelled:
            return
        self.progress.emit(55)

        # Install ML packages
        self.log.emit("")
        self.log.emit("  Installing ML frameworks...")
        self.progress.emit(60)
        ok = self._run_pip(venv_py, [
            "install",
            "pytorch-lightning",
            "segmentation-models-pytorch",
            "albumentations",
            "torchmetrics",
            "ultralytics",
            "onnxruntime",
            "numpy<2",
        ], "ML packages")
        if not ok:
            return
        if self._cancelled:
            return
        self.progress.emit(85)

        # ── Step 4: Verify ──
        self.log.emit("")
        self.log.emit("[4/4] Verifying installation...")
        self.progress.emit(90)

        verify_cmd = [
            venv_py, "-c",
            "import torch; import pytorch_lightning; import ultralytics; "
            "import segmentation_models_pytorch; print('ALL OK')"
        ]
        result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and "ALL OK" in result.stdout:
            self.log.emit("  All packages verified successfully!")
            self.log.emit("")
            self.log.emit("=" * 52)
            self.log.emit("  AI Engine installed successfully!")
            self.log.emit("  Training is ready to use.")
            self.log.emit("=" * 52)
            self.progress.emit(100)
            self.finished.emit(True)
        else:
            self.log.emit(f"[ERROR] Verification failed: {result.stderr}")
            self.finished.emit(False)

    def _run_pip(self, venv_py: str, args: list, label: str) -> bool:
        """Run a pip command in the venv, streaming output."""
        cmd = [venv_py, "-m", "pip"] + args
        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
            for line in proc.stdout:
                stripped = line.rstrip()
                if stripped:
                    self.log.emit(f"   {stripped}")
                if self._cancelled:
                    proc.kill()
                    self.log.emit("[CANCELLED] Setup cancelled by user.")
                    self.finished.emit(False)
                    return False
            proc.wait()

            if proc.returncode != 0:
                self.log.emit(f"\n[ERROR] {label} installation failed (exit {proc.returncode})")
                self.finished.emit(False)
                return False
            return True

        except Exception as e:
            self.log.emit(f"[ERROR] {label} failed: {e}")
            self.finished.emit(False)
            return False
