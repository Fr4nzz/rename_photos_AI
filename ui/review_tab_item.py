# ai-photo-processor/ui/review_tab_item.py

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QGroupBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

from .widgets import ClickableLabel

class ReviewItemWidget(QGroupBox):
    """A widget representing a single item in the review grid."""
    data_changed = pyqtSignal()
    # Signal emitted when CAM or suffix changes, for recalculating To field
    cam_or_suffix_changed = pyqtSignal(object)  # Emits self

    def __init__(self, df_index: int, item_data: dict, main_column: str, count: int, parent=None):
        # --- MODIFIED: Use 'current_path' for the title to show the current filename ---
        super().__init__(os.path.basename(item_data.get('current_path', 'N/A')), parent)
        self.df_index = df_index
        self.main_column = main_column
        self.fields = {}
        self._setup_ui(item_data)

        # Display a warning if the count of items with this ID is not 2
        if count != 2 and item_data.get(main_column):
            self.set_warning(f"Warning: Appears {count} times in total.")
        else:
            self.clear_warning()

        # Display batch number if available
        batch_num = item_data.get('batch_number', '')
        try:
            batch_num_int = int(float(batch_num)) if batch_num else 0
            if batch_num_int > 0:
                self.set_batch_number(batch_num_int)
            else:
                self.batch_label.setVisible(False)
        except (ValueError, TypeError):
            self.batch_label.setVisible(False)

        # Display the runtime status of the file
        status = item_data.get('status', '')
        if status == 'Renamed':
            self.set_status("Already Renamed", "color: #FF8C00;")  # Orange
        elif status == 'New':
            self.set_status("New File", "color: #32CD32;")  # LimeGreen
        elif status == 'Missing':
            self.set_status("File Missing", "color: #F08080;")  # LightCoral
        else:  # 'Original' status
            self.status_label.setVisible(False)

    def _setup_ui(self, item_data: dict):
        main_layout = QVBoxLayout(self)

        # Top bar for status messages
        top_bar_layout = QHBoxLayout()
        self.warning_label = QLabel()
        self.warning_label.setStyleSheet("color: red; font-weight: bold;")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        self.batch_label = QLabel()
        self.batch_label.setStyleSheet("color: #4169E1; font-weight: bold;")  # Royal Blue
        self.batch_label.setVisible(False)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        top_bar_layout.addWidget(self.warning_label)
        top_bar_layout.addWidget(self.batch_label)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(self.status_label)
        main_layout.addLayout(top_bar_layout)

        self.image_label = ClickableLabel("Loading...")
        self.image_label.setMinimumHeight(180)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color:#333; color:white;")
        main_layout.addWidget(self.image_label, 1)

        # CAM field with suffix field on the same row
        self._add_cam_suffix_row(item_data)
        self._add_field_row("To:", 'to', item_data.get('to', ''))
        if 'co' in item_data:
            self._add_field_row("Crossed Out:", 'co', item_data.get('co', ''))
        if 'n' in item_data:
            self._add_field_row("Notes:", 'n', item_data.get('n', ''))

        if 'skip' in item_data:
            self.fields['skip'] = QCheckBox("Skip this file")
            self.fields['skip'].setChecked(item_data.get('skip', '') == 'x')
            self.fields['skip'].stateChanged.connect(self.data_changed.emit)
            main_layout.addWidget(self.fields['skip'])

    def _add_cam_suffix_row(self, item_data: dict):
        """Add the CAM field with a suffix field next to it."""
        row_layout = QHBoxLayout()

        # CAM label and input
        cam_label = QLabel(f"{self.main_column}:")
        cam_label.setFixedWidth(40)
        cam_edit = QLineEdit(str(item_data.get(self.main_column, '')))
        cam_edit.editingFinished.connect(self.data_changed.emit)
        cam_edit.editingFinished.connect(lambda: self.cam_or_suffix_changed.emit(self))
        self.fields['main_column'] = cam_edit

        # Suffix label and input
        suffix_label = QLabel("Suffix:")
        suffix_label.setFixedWidth(40)
        suffix_edit = QLineEdit(str(item_data.get('suffix', '')))
        suffix_edit.setFixedWidth(50)
        suffix_edit.editingFinished.connect(self.data_changed.emit)
        suffix_edit.editingFinished.connect(lambda: self.cam_or_suffix_changed.emit(self))
        self.fields['suffix'] = suffix_edit

        row_layout.addWidget(cam_label)
        row_layout.addWidget(cam_edit)
        row_layout.addWidget(suffix_label)
        row_layout.addWidget(suffix_edit)
        self.layout().addLayout(row_layout)

    def _add_field_row(self, label_text: str, field_key: str, initial_value: str):
        row_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(80)
        line_edit = QLineEdit(str(initial_value))
        line_edit.editingFinished.connect(self.data_changed.emit)
        self.fields[field_key] = line_edit
        row_layout.addWidget(label)
        row_layout.addWidget(line_edit)
        self.layout().addLayout(row_layout)

    def set_image(self, pixmap: QPixmap):
        self.image_label.setPixmap(pixmap)
    
    def set_warning(self, text: str):
        self.warning_label.setText(text)
        self.warning_label.setVisible(True)

    def clear_warning(self):
        self.warning_label.setVisible(False)

    def set_batch_number(self, batch_num: int):
        self.batch_label.setText(f"Message {batch_num}")
        self.batch_label.setVisible(True)
        
    def set_status(self, text: str, style: str):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)
        self.status_label.setVisible(True)
       
    def get_data(self) -> dict:
        data = {'df_index': self.df_index}
        for key, widget in self.fields.items():
            if isinstance(widget, QLineEdit):
                data[key] = widget.text().strip()
            elif isinstance(widget, QCheckBox):
                data[key] = 'x' if widget.isChecked() else ''

        # Special handling for the main column to use the correct key
        if 'main_column' in data:
            data[self.main_column] = data.pop('main_column')

        return data

    def update_to_field(self, new_value: str):
        """Update the To field value programmatically."""
        if 'to' in self.fields:
            self.fields['to'].setText(new_value)