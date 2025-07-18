# ai-photo-processor/main_window.py

from PyQt5.QtWidgets import QMainWindow, QTabWidget, QMessageBox

from app_state import AppState
from ui.process_tab import ProcessImagesTab
from ui.review_tab import ReviewResultsTab
from ui.api_keys_tab import ApiKeysTab
from controllers.process_tab_handler import ProcessTabHandler
from controllers.review_tab_handler import ReviewTabHandler
from utils.logger import SimpleLogger

class MainWindow(QMainWindow):
    """
    The main application window.
    Owns the core components (UI, State, Handlers) and orchestrates their interactions.
    Delegates tab-specific logic to dedicated handler classes.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Photo Processor")
        self.setGeometry(100, 100, 1600, 900)

        # 1. Initialize Core Components
        self.logger = SimpleLogger()
        self.app_state = AppState(self.logger)

        # 2. Initialize UI Tabs
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        self.process_tab = ProcessImagesTab()
        self.review_tab = ReviewResultsTab()
        self.api_keys_tab = ApiKeysTab()

        self.tab_widget.addTab(self.process_tab, "Process Images")
        self.tab_widget.addTab(self.review_tab, "Review Results")
        self.tab_widget.addTab(self.api_keys_tab, "API Keys")

        # 3. Initialize and Connect Handlers
        self.process_handler = ProcessTabHandler(self.process_tab, self.app_state, self.logger, self)
        self.process_handler.connect_signals()

        self.review_handler = ReviewTabHandler(self.review_tab, self.app_state, self.logger, self)
        self.review_handler.connect_signals()

        # Connect signals that are managed at the main window level
        self._connect_main_signals()

        # 4. Populate UI with initial state
        self.process_handler.populate_initial_ui()
        self.review_handler.populate_initial_ui()
        self.api_keys_tab.api_keys_text_edit.setPlainText("\n".join(self.app_state.api_keys))

        self.logger.info("Application UI is ready.")

    def _connect_main_signals(self):
        """Connects signals for widgets directly managed by the main window."""
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.api_keys_tab.save_api_keys_button.clicked.connect(self.save_api_keys)

    def on_tab_changed(self, index: int):
        """Handle logic for when the user switches tabs."""
        if self.tab_widget.widget(index) == self.review_tab:
            self.logger.info("Switched to Review Results tab.")
            # --- MODIFIED: Pass a flag to select the newest CSV by default ---
            self.review_handler.refresh_csv_dropdown(select_newest=True)

    def save_api_keys(self):
        """Handles saving API keys, a simple enough action to keep here."""
        keys = self.api_keys_tab.api_keys_text_edit.toPlainText().strip().split('\n')
        self.app_state.api_keys = [key.strip() for key in keys if key.strip()]
        self.app_state.save_api_keys()
        self.process_handler.update_models_dropdown() # Trigger model update
        QMessageBox.information(self, "Success", "API keys saved.")

    def closeEvent(self, event):
        """Ensures graceful shutdown of the application."""
        self.logger.info("--- Application closing ---")
        
        self.process_handler._sync_settings_from_ui()
        self.review_handler._sync_settings_from_ui()
        
        self.app_state.save_settings()

        self.process_handler.stop_worker()
        self.review_handler.stop_worker()

        event.accept()