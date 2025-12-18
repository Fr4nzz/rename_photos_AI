# ai-photo-processor/controllers/base_handler.py

from PyQt5.QtCore import QObject, QThread
from PyQt5.QtWidgets import QMainWindow

from app_state import AppState
from utils.logger import SimpleLogger


class BaseTabHandler(QObject):
    """Base class for tab handlers with common worker thread management."""

    def __init__(self, ui, app_state: AppState, logger: SimpleLogger, main_window: QMainWindow):
        super().__init__()
        self.ui = ui
        self.app_state = app_state
        self.logger = logger
        self.main_window = main_window
        self.worker_thread = None
        self.current_worker = None

    def stop_worker(self):
        """Stop the current worker and clean up thread."""
        if self.current_worker:
            self.current_worker.stop()
        self._cleanup_worker_thread()

    def _cleanup_worker_thread(self, timeout_ms: int = 5000):
        """Clean up worker thread with timeout handling."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            if not self.worker_thread.wait(timeout_ms):
                self.logger.warn("Worker thread did not stop gracefully, forcing termination.")
                self.worker_thread.terminate()
                self.worker_thread.wait(1000)
        self.worker_thread = None
        self.current_worker = None
