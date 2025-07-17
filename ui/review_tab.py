# ai-photo-processor/ui/review_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QScrollArea, QGridLayout, QLabel, QCheckBox, QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator

class ReviewResultsTab(QWidget):
    """The UI for the 'Review Results' tab."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # --- Top Controls Toolbar ---
        top_controls_layout = QHBoxLayout()
        top_controls_layout.addWidget(QLabel("CSV File:"))
        self.csv_dropdown = QComboBox()
        top_controls_layout.addWidget(self.csv_dropdown, 1)
        self.refresh_from_disk_button = QPushButton("Refresh from Disk")
        top_controls_layout.addWidget(self.refresh_from_disk_button)
        self.crop_review_checkbox = QCheckBox("Enable Cropping")
        top_controls_layout.addWidget(self.crop_review_checkbox)
        self.show_duplicates_checkbox = QCheckBox("Show Mismatches Only")
        top_controls_layout.addWidget(self.show_duplicates_checkbox)
        
        top_controls_layout.addStretch()
        
        # --- Items per Page Input ---
        top_controls_layout.addWidget(QLabel("Items/Page:"))
        self.items_per_page_input = QLineEdit()
        self.items_per_page_input.setValidator(QIntValidator(1, 9999))
        self.items_per_page_input.setFixedWidth(50)
        top_controls_layout.addWidget(self.items_per_page_input)
        
        # --- NEW: Thumbnail Height Input ---
        top_controls_layout.addWidget(QLabel("Thumb H:"))
        self.thumb_height_input = QLineEdit()
        self.thumb_height_input.setValidator(QIntValidator(100, 4000)) # Height from 100px to 4k
        self.thumb_height_input.setFixedWidth(60)
        top_controls_layout.addWidget(self.thumb_height_input)
        
        main_layout.addLayout(top_controls_layout)

        # --- Pagination Controls Toolbar ---
        pagination_controls_layout = QHBoxLayout()
        self.prev_page_button = QPushButton("<< Previous")
        self.page_label = QLabel("Page 1 of 1")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.next_page_button = QPushButton("Next >>")
        pagination_controls_layout.addWidget(self.prev_page_button)
        pagination_controls_layout.addWidget(self.page_label, 1)
        pagination_controls_layout.addWidget(self.next_page_button)
        main_layout.addLayout(pagination_controls_layout)

        # --- Main Content Scroll Area ---
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

        # --- Bottom Action Buttons Toolbar ---
        bottom_actions_layout = QHBoxLayout()
        bottom_actions_layout.addStretch(1)
        self.save_changes_button = QPushButton("Save Changes")
        self.recalc_names_button = QPushButton("Recalculate Final Names")
        self.rename_files_button = QPushButton("Rename Files")
        self.restore_names_button = QPushButton("Restore Original Names")
        bottom_actions_layout.addWidget(self.save_changes_button)
        bottom_actions_layout.addWidget(self.recalc_names_button)
        bottom_actions_layout.addWidget(self.rename_files_button)
        bottom_actions_layout.addWidget(self.restore_names_button)
        main_layout.addLayout(bottom_actions_layout)

    def clear_grid(self):
        """Removes all items from the results grid."""
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