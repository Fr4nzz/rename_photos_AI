# ai-photo-processor/ui/review_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QScrollArea, QGridLayout, QLabel
)
from PyQt5.QtCore import Qt

class ReviewResultsTab(QWidget):
    """The UI for the 'Review Results' tab."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui_widgets = []
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        controls_layout = QHBoxLayout()
        
        self.csv_dropdown = QComboBox()
        self.recalc_names_button = QPushButton("Recalculate Final Names")
        self.rename_files_button = QPushButton("Rename Files")
        self.restore_names_button = QPushButton("Restore Original Names")

        controls_layout.addWidget(QLabel("CSV File:"))
        controls_layout.addWidget(self.csv_dropdown, 1)
        controls_layout.addStretch(1)
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
        scroll_area.setWidget(self.grid_container)
        main_layout.addWidget(scroll_area)

    def clear_grid(self):
        """Removes all items from the results grid."""
        self.ui_widgets.clear()
        while self.grid_layout.count():
            if child := self.grid_layout.takeAt(0):
                if child.widget():
                    child.widget().deleteLater()

    def add_item_to_grid(self, row_idx: int, col_idx: int, item_widget: QWidget):
        """Adds a new widget to the grid."""
        self.grid_layout.addWidget(item_widget, row_idx, col_idx)