"""mlops/utils.py — shared helpers for training/export workers."""

import os
import shutil
import subprocess
import sys


def find_python() -> str:
    """Return a Python executable that has torch installed.

    In a frozen PyInstaller exe, sys.executable is DualAnnotator.exe which
    cannot load torch due to DLL environment differences.  Search order:
      1. venv/ next to the exe or in any ancestor directory (up to 5 levels)
      2. System Python verified to have torch
      3. Fallback: sys.executable (surfaces the error)
    """
    if not getattr(sys, "frozen", False):
        return sys.executable

    # 1. Walk up from exe dir and cwd looking for a venv
    for base in [os.path.dirname(sys.executable), os.getcwd()]:
        venv_py = os.path.join(base, "venv", "Scripts", "python.exe")
        if os.path.isfile(venv_py):
            return venv_py
        parent = os.path.dirname(base)
        for _ in range(5):
            if parent == os.path.dirname(parent):
                break
            venv_py = os.path.join(parent, "venv", "Scripts", "python.exe")
            if os.path.isfile(venv_py):
                return venv_py
            parent = os.path.dirname(parent)

    # 2. System Python — only if it has torch
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

    # 3. Fallback — frozen exe (training will likely fail, but surfaces the error)
    return sys.executable


def resolve_device(requested: str) -> str:
    """Return a usable device string for training.

    Falls back to CPU with a warning if a CUDA device was requested but
    isn't actually available on this machine (e.g. no NVIDIA GPU/driver).
    Import of torch is deliberately lazy — this module is also imported
    from the main app process, which may not have torch installed.
    """
    if not requested.startswith("cuda"):
        return requested

    import torch
    if torch.cuda.is_available():
        return requested

    print(
        f"[WARN] Device '{requested}' requested but CUDA is not available "
        "on this machine — falling back to CPU.",
        flush=True,
    )
    return "cpu"
