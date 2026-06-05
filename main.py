# main.py
import os
import sys

# WinError 1114 fix: c10.dll fails to initialize because its VC++ runtime
# dependencies aren't loaded yet. Fix: explicitly pre-load them via ctypes
# so they're in the process DLL cache before c10.dll initializes.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    import ctypes
    _torch_lib = os.path.join(sys._MEIPASS, "torch", "lib")
    if os.path.isdir(_torch_lib):
        # Pre-load VC++ runtime DLLs so c10.dll finds them already in cache
        for _dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
            _dll_path = os.path.join(_torch_lib, _dll)
            if os.path.isfile(_dll_path):
                try:
                    ctypes.CDLL(_dll_path)
                except OSError:
                    pass
        # Register torch/lib and bundle root as DLL search directories
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_torch_lib)
                os.add_dll_directory(sys._MEIPASS)
            except OSError:
                pass

try:
    import torch  # noqa: F401
except Exception:
    pass
import runpy

if __name__ == '__main__':
    # ---------------------------------------------------------
    # Issue 4 fix: PyInstaller Subprocess Interceptor
    # When compiled to .exe, TrainWorker calls:
    #   DualAnnotator.exe train_unet.py --config ... --out ...
    # Without this interceptor, that command would open a second
    # UI window instead of running the training script.
    # This detects the .py argument and runs it silently.
    # ---------------------------------------------------------
    if len(sys.argv) >= 2 and sys.argv[1].endswith('.py'):
        script_to_run = sys.argv[1]
        sys.argv.pop(1)
        runpy.run_path(script_to_run, run_name="__main__")
        sys.exit(0)

    from PyQt5.QtWidgets import QApplication
    from gui.main_window import MainWindow

    def main():
        app = QApplication(sys.argv)
        app.setApplicationName("Dual Annotator")
        app.setApplicationVersion("1.0.0")
        
        # Create window
        window = MainWindow()
        
        # Show the window
        window.show()
        
        sys.exit(app.exec_())

    main()