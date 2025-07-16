# ai-photo-processor/main.py

import sys
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox
from main_window import MainWindow

# --- TEMPORARY DEBUGGING ---
# This is a global exception hook to catch errors that would otherwise
# cause a silent crash. It will display the error in a message box.
def global_exception_hook(exctype, value, tb):
    """Catches uncaught exceptions, prints them, and shows a message box."""
    traceback_details = "".join(traceback.format_exception(exctype, value, tb))
    error_msg = f"A critical error occurred:\n\n{traceback_details}"
    
    # Print the full error to the console, which is crucial for debugging
    print("--- UNHANDLED EXCEPTION ---")
    print(error_msg)
    print("---------------------------")
    
    # Show a message box to the user.
    QMessageBox.critical(None, "Unhandled Application Error", error_msg)
    
    # Exit cleanly after showing the error
    sys.exit(1)
# --- END TEMPORARY DEBUGGING ---


if __name__ == "__main__":
    # --- TEMPORARY DEBUGGING ---
    # Set the global hook. This must be done before the QApplication is created.
    sys.excepthook = global_exception_hook
    # --- END TEMPORARY DEBUGGING ---
    
    # Standard boilerplate to run a PyQt application
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    # Start the application's event loop
    sys.exit(app.exec_())