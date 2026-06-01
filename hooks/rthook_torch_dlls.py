# hooks/rthook_torch_dlls.py
# PyInstaller runtime hook — runs BEFORE main.py
# Ensures torch/lib DLLs are discoverable before any other bundled DLLs
# (especially Qt5) can interfere with c10.dll initialisation.

import os
import sys

if getattr(sys, "frozen", False):
    bundle_dir = sys._MEIPASS  # type: ignore[attr-defined]
    torch_lib = os.path.join(bundle_dir, "torch", "lib")
    if os.path.isdir(torch_lib):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(torch_lib)
            except OSError:
                pass
