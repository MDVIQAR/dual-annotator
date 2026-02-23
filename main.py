# main.py
import sys
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

if __name__ == '__main__':
    main()