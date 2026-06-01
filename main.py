# main.py
# NOTE: torch must be imported FIRST on Windows to avoid a DLL initialization
# conflict (WinError 1114) between PyTorch's c10.dll and PyQt5's Qt DLLs.
try:
    import torch  # noqa: F401
except Exception:
    pass
import sys
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