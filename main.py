# main.py
import os
import sys

# ============================================================
# FROZEN-EXE DLL FIX  (must run before ANY other import)
# ============================================================
# Problem: PyTorch's c10.dll depends on ~15 peer DLLs inside
# torch/lib/.  In a frozen PyInstaller build the normal DLL
# search path doesn't include that directory, so c10.dll finds
# nothing and raises WinError 1114.
#
# Fix: register torch/lib as a DLL-search directory *and*
# pre-load every .dll inside it so they're already in the
# process's module table when c10.dll asks the loader for them.
# ============================================================
if getattr(sys, "frozen", False) and sys.platform == "win32":
    import ctypes
    import ctypes.wintypes
    import glob

    _MEIPASS = sys._MEIPASS                       # PyInstaller's _internal folder
    _torch_lib = os.path.join(_MEIPASS, "torch", "lib")

    # 1. Register DLL search directories (Win 10 1607+)
    if os.path.isdir(_torch_lib):
        os.environ["PATH"] = (
            _torch_lib + os.pathsep +
            _MEIPASS   + os.pathsep +
            os.environ.get("PATH", "")
        )
        if hasattr(os, "add_dll_directory"):
            for d in (_torch_lib, _MEIPASS):
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass

        # 2. Pre-load EVERY .dll inside torch/lib in dependency order.
        #    Loading order matters: VC-runtime -> MKL/OpenMP -> c10 -> torch_cpu -> rest
        _priority = [
            "vcruntime140.dll",
            "vcruntime140_1.dll",
            "msvcp140.dll",
            "msvcp140_1.dll",
            "concrt140.dll",
            "libiomp5md.dll",        # Intel OpenMP
            "libiompstubs5md.dll",
            "asmjit.dll",
            "fbgemm.dll",
            "uv.dll",
            "c10.dll",               # Core C++ lib
            "torch_cpu.dll",         # Must come after c10
            "torch.dll",
            "torch_python.dll",
        ]

        _loaded = set()
        # Phase A - load known DLLs in the right order
        for name in _priority:
            p = os.path.join(_torch_lib, name)
            if os.path.isfile(p):
                try:
                    ctypes.CDLL(p)
                    _loaded.add(name.lower())
                except OSError:
                    pass

        # Phase B - load every remaining .dll we haven't touched yet
        for p in glob.glob(os.path.join(_torch_lib, "*.dll")):
            if os.path.basename(p).lower() not in _loaded:
                try:
                    ctypes.CDLL(p)
                except OSError:
                    pass

    # 3. Also handle torch_global_deps (some torch builds put deps here)
    _torch_global = os.path.join(_MEIPASS, "torch", "_C")
    if os.path.isdir(_torch_global) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_torch_global)
        except OSError:
            pass

# ============================================================
# Warm-import torch so the DLLs are resolved once for the
# whole process lifetime.  If it fails here we still let the
# GUI start (training just won't work until torch is fixed).
# ============================================================
try:
    import torch  # noqa: F401
except Exception as _e:
    print(f"[WARN] torch import failed (training will not work): {_e}",
          file=sys.stderr)


# ============================================================
# Application entry-point
# ============================================================
if __name__ == "__main__":
    # ---------------------------------------------------------
    # Subprocess interceptor for frozen .exe
    # When TrainWorker calls:
    #   DualAnnotator.exe  mlops/scripts/train_unet.py --config ...
    # we detect the .py arg and run it in-process.
    # ---------------------------------------------------------
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".py"):
        import runpy

        script = sys.argv[1]

        # Resolve path: could be relative to _MEIPASS in frozen mode
        if getattr(sys, "frozen", False) and not os.path.isabs(script):
            candidate = os.path.join(sys._MEIPASS, script)
            if os.path.isfile(candidate):
                script = candidate

        sys.argv = [script] + sys.argv[2:]  # clean argv for the child
        runpy.run_path(script, run_name="__main__")
        sys.exit(0)

    # Normal GUI launch
    from PyQt5.QtWidgets import QApplication
    from gui.main_window import MainWindow

    def main():
        app = QApplication(sys.argv)
        app.setApplicationName("Dual Annotator")
        app.setApplicationVersion("1.0.0")
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())

    main()
