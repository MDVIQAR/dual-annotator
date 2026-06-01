# main.py
# NOTE: torch must be imported FIRST on Windows to avoid a DLL initialization
# conflict (WinError 1114) between PyTorch's c10.dll and PyQt5's Qt DLLs.
# When bundled with PyInstaller, the bootloader extracts ALL DLLs (including
# Qt5) into a _MEI temp folder.  c10.dll then fails to initialise because
# conflicting DLLs are already on the search path.  We fix this by explicitly
# adding torch's DLL directory to the OS DLL search order *before* import.
import os, sys

def _fix_torch_dll_path():
    """Prepend torch/lib to PATH and register it with os.add_dll_directory
    so that c10.dll (and friends) can find their dependencies before any
    conflicting Qt DLLs are picked up."""
    # Only needed inside a PyInstaller bundle (frozen exe)
    if not getattr(sys, "frozen", False):
        return
    bundle_dir = sys._MEIPASS  # type: ignore[attr-defined]
    torch_lib = os.path.join(bundle_dir, "torch", "lib")
    if not os.path.isdir(torch_lib):
        return
    # 1. Prepend to PATH so LoadLibrary finds torch DLLs first
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    # 2. On Python ≥3.8 also use os.add_dll_directory (more reliable)
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib)
        except OSError:
            pass

_fix_torch_dll_path()

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