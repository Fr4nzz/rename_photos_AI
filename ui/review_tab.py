# ai-photo-processor/ui/review_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QScrollArea, QGridLayout, QLabel, QCheckBox
)
from PyQt5.QtCore import Qt

class ReviewResultsTab(QWidget):
    """The UI for the 'Review Results' tab."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The redundant ui_widgets list has been removed.
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("CSV File:"))
        self.csv_dropdown = QComboBox()
        controls_layout.addWidget(self.csv_dropdown, 1)

        self.crop_review_checkbox = QCheckBox("Enable Cropping")
        controls_layout.addWidget(self.crop_review_checkbox)
        
        self.show_duplicates_checkbox = QCheckBox("Show Mismatches Only")
        controls_layout.addWidget(self.show_duplicates_checkbox)

        controls_layout.addStretch(1)

        self.recalc_names_button = QPushButton("Recalculate Final Names")
        self.rename_files_button = QPushButton("Rename Files")
        self.restore_names_button = QPushButton("Restore Original Names")

        controls_layout.addWidget(self.recalc_names_button)
        controls_layout.addWidget(self.rename_files_button)
        controls_layout.addWidget(self.restore_names_button)
        main_layout.addLayout(controls_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)

        scroll_area.setWidget(self.grid_container)
        main_layout.addWidget(scroll_area)

    def clear_grid(self):
        """Removes all items from the results grid."""
        # The ui_widgets list is gone, so no need to clear it here.
        while self.grid_layout.count():
            if child := self.grid_layout.takeAt(0):
                if child.widget():
                    child.widget().deleteLater()

    def add_item_to_grid(self, row_idx: int, col_idx: int, item_widget: QWidget):
        """Adds a new widget to the grid."""
        self.grid_layout.addWidget(item_widget, row_idx, col_idx)

    def set_grid_message(self, text: str):
        """Clears the grid and displays a single message."""
        self.clear_grid()
        self.grid_layout.addWidget(QLabel(text), 0, 0, 1, 2, Qt.AlignCenter)