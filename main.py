# main.py
import os
import sys

# On Python 3.8+ Windows changed DLL loading — PATH is no longer searched.
# Inside a PyInstaller bundle torch's own add_dll_directory points to the
# wrong path, so c10.dll fails with WinError 1114.  Fix: register torch/lib
# explicitly before any import touches it.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _bundle = sys._MEIPASS
    # Register bundle root so bundled VC++ runtime DLLs are findable
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_bundle)
        except OSError:
            pass
    # Register torch/lib so c10.dll and friends can find each other
    _torch_lib = os.path.join(_bundle, "torch", "lib")
    if os.path.isdir(_torch_lib):
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_torch_lib)
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