# main.py
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    """Main entry point for the application"""
    
    # Create the application object
    app = QApplication(sys.argv)
    
    # Set application information
    app.setApplicationName("Dual Annotator")
    app.setApplicationVersion("1.0.0")
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()