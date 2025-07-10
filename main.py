# ai-photo-processor/main.py

import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == "__main__":
    # Standard boilerplate to run a PyQt application
    app = QApplication(sys.argv)
    
    # Optional: Apply a style for a more modern look across different OSes
    # app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    # Start the application's event loop
    sys.exit(app.exec_())