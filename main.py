# ai-photo-processor/main.py

import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

# The global exception hook is no longer needed and has been removed.

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())