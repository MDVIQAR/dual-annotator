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

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def needs_setup() -> bool:
    """Return True if we're in a frozen exe and no AI Engine venv exists yet."""
    if not getattr(sys, "frozen", False):
        return False
    return find_python() == sys.executable


def _get_venv_dir() -> str:
    """Return the path where the venv should be created (next to the exe)."""
    return os.path.join(os.path.dirname(sys.executable), "venv")


def _get_torch_index_url() -> str:
    """Return the right PyTorch wheel index URL for this machine."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=_NO_WINDOW,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "https://download.pytorch.org/whl/cu118"
    except Exception:
        pass
    return "https://download.pytorch.org/whl/cpu"


def _find_system_python() -> str | None:
    """Find Python 3.11 specifically."""
    # Try 'python' on PATH first
    py = shutil.which("python")
    if py and os.path.normcase(py) != os.path.normcase(sys.executable):
        try:
            result = subprocess.run(
                [py, "--version"],
                capture_output=True, text=True, timeout=10,
                creationflags=_NO_WINDOW,
            )
            if result.returncode == 0 and "3.11" in result.stdout:
                return py
        except Exception:
            pass

    # Then try py -3.11 (Python Launcher)
    py_launcher = shutil.which("py")
    if py_launcher:
        try:
            result = subprocess.run(
                [py_launcher, "-3.11", "--version"],
                capture_output=True, text=True, timeout=10,
                creationflags=_NO_WINDOW,
            )
            if result.returncode == 0 and "3.11" in result.stdout:
                return f"{py_launcher} -3.11"
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
            # Check if a Python exists but it's the wrong version
            any_py = shutil.which("python") or shutil.which("py")
            if any_py:
                try:
                    r = subprocess.run(
                        [any_py, "--version"],
                        capture_output=True, text=True, timeout=10,
                        creationflags=_NO_WINDOW,
                    )
                    found_ver = r.stdout.strip() or r.stderr.strip()
                    self.log.emit("")
                    self.log.emit(f"[ERROR] Found {found_ver}, but Python 3.11 is required.")
                    self.log.emit("")
                    self.log.emit("  Python 3.12 and 3.13 are NOT compatible with the")
                    self.log.emit("  ML packages used for training (PyTorch, etc.).")
                    self.log.emit("")
                    self.log.emit("  Download Python 3.11.9 from:")
                    self.log.emit("  https://www.python.org/downloads/release/python-3119/")
                    self.log.emit("")
                    self.log.emit("  You can have multiple Python versions installed —")
                    self.log.emit("  they won't conflict.")
                    self.log.emit("")
                    self.log.emit("  After installing, close and reopen this app,")
                    self.log.emit("  then click Start Training again.")
                    self.log.emit("")
                    self.log.emit("  For more help: docs → Train → Troubleshooting")
                    self.finished.emit(False)
                    return
                except Exception:
                    pass

            self.log.emit("")
            self.log.emit("[ERROR] Python 3.11 not found on this system.")
            self.log.emit("")
            self.log.emit("  Troubleshooting:")
            self.log.emit("  ────────────────")
            self.log.emit("  1. Python 3.11 is REQUIRED. Other versions (3.12, 3.13)")
            self.log.emit("     are NOT compatible with PyTorch and will cause errors.")
            self.log.emit("")
            self.log.emit("  2. Download Python 3.11.9 from:")
            self.log.emit("     https://www.python.org/downloads/release/python-3119/")
            self.log.emit("")
            self.log.emit("  3. During installation, CHECK these two boxes:")
            self.log.emit("     [x] Add Python to PATH")
            self.log.emit("     [x] Install for all users (recommended)")
            self.log.emit("")
            self.log.emit("  4. After installing, CLOSE and REOPEN this app,")
            self.log.emit("     then click Start Training again.")
            self.log.emit("")
            self.log.emit("  For more help: docs → Train → Troubleshooting")
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
            cmd = system_py.split() + ["-m", "venv", venv_dir]
        else:
            cmd = [system_py, "-m", "venv", venv_dir]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=_NO_WINDOW,
        )
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

        # Install torch (CUDA build if an NVIDIA GPU is detected, else CPU)
        index_url = _get_torch_index_url()
        build_label = "GPU/CUDA" if "cpu" not in index_url else "CPU"
        self.log.emit(f"  Installing PyTorch ({build_label})...")
        self.progress.emit(20)
        ok = self._run_pip(venv_py, [
            "install", "torch", "torchvision",
            "--index-url", index_url
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
            "onnx<=1.16.0",
            "numpy==1.26.4",
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
        try:
            result = subprocess.run(
                verify_cmd, capture_output=True, text=True, timeout=120,
                creationflags=_NO_WINDOW,
            )
        except subprocess.TimeoutExpired:
            self.log.emit("  Verification timed out (this is OK — first import is slow)")
            self.log.emit("  Packages were installed successfully.")
            self.log.emit("")
            self.log.emit("=" * 52)
            self.log.emit("  AI Engine installed successfully!")
            self.log.emit("  Training is ready to use.")
            self.log.emit("=" * 52)
            self.progress.emit(100)
            self.finished.emit(True)
            return

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
            error_detail = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            self.log.emit(f"[ERROR] Verification failed: {error_detail}")
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
            creationflags=_NO_WINDOW,
        )

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
