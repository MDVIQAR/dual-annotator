# hooks/rthook_torch_dlls.py
# ─────────────────────────────────────────────────────────────
# PyInstaller RUNTIME hook – executed before main.py runs.
# Ensures torch/lib DLLs are on the search path even before
# the first "import" statement in user code.
# ─────────────────────────────────────────────────────────────
import os
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    _MEIPASS = sys._MEIPASS
    _torch_lib = os.path.join(_MEIPASS, "torch", "lib")

    if os.path.isdir(_torch_lib):
        # Prepend to PATH so legacy LoadLibrary calls find the DLLs
        os.environ["PATH"] = (
            _torch_lib + os.pathsep +
            _MEIPASS   + os.pathsep +
            os.environ.get("PATH", "")
        )
        # Also register via the modern API (Win 10 1607+)
        if hasattr(os, "add_dll_directory"):
            for d in (_torch_lib, _MEIPASS):
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
