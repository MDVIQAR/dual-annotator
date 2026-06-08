"""mlops/utils.py — shared helpers for subprocess-based workers."""

import os
import shutil
import sys


def find_python() -> str:
    """Return a Python executable that has torch installed.

    In a frozen PyInstaller exe, sys.executable is DualAnnotator.exe which
    cannot load torch due to DLL environment differences in the frozen env.
    We look for the venv that engine_manager creates next to the exe, then
    fall back to a verified system Python, then the exe itself.
    """
    if not getattr(sys, "frozen", False):
        return sys.executable

    import subprocess

    # 1. venv created by engine_manager next to the exe
    exe_dir = os.path.dirname(sys.executable)
    venv_py = os.path.join(exe_dir, "venv", "Scripts", "python.exe")
    if os.path.isfile(venv_py):
        return venv_py

    # 2. system Python — only if it has torch
    system_py = shutil.which("python")
    if system_py and os.path.normcase(system_py) != os.path.normcase(sys.executable):
        try:
            subprocess.run(
                [system_py, "-c", "import torch"],
                capture_output=True, timeout=10,
            ).check_returncode()
            return system_py
        except Exception:
            pass

    # 3. fallback — frozen exe (training will likely fail, but surfaces the error)
    return sys.executable
