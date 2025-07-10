# ai-photo-processor/ui/api_keys_tab.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit

class ApiKeysTab(QWidget):
    """The UI for the 'API Keys' tab."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_layout = QVBoxLayout(self)
        self._setup_ui()

    def _setup_ui(self):
        """Sets up the widgets for the API keys tab."""
        self.main_layout.addWidget(QLabel("Enter Google AI API keys below, one per line:"))

        self.api_keys_text_edit = QTextEdit()
        self.api_keys_text_edit.setPlaceholderText("g_api_key_1...\ng_api_key_2...")
        self.main_layout.addWidget(self.api_keys_text_edit)

        self.save_api_keys_button = QPushButton("Save API Keys")
        self.main_layout.addWidget(self.save_api_keys_button)
        self.main_layout.addStretch()