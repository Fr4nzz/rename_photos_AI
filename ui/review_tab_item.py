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

    def __init__(self, df_index: int, item_data: dict, main_column: str, count: int, parent=None):
        super().__init__(os.path.basename(item_data.get('from', 'N/A')), parent)
        self.df_index = df_index
        self.main_column = main_column
        self.fields = {}
        self._setup_ui(item_data)
        
        if count != 2 and item_data.get(main_column): # Only show warning if there's an identifier
            self.set_warning(f"Warning: Appears {count} times in total.")
        else:
            self.clear_warning()
        
        status = item_data.get('status', '')
        if status == 'Renamed':
            self.set_status("Already Renamed", "color: #FF8C00;") # Orange
        elif status == 'New':
            self.set_status("New File", "color: #32CD32;") # LimeGreen
        else:
            self.status_label.setVisible(False)


    def _setup_ui(self, item_data: dict):
        main_layout = QVBoxLayout(self)
        
        # Top bar for status messages
        top_bar_layout = QHBoxLayout()
        self.warning_label = QLabel()
        self.warning_label.setStyleSheet("color: red; font-weight: bold;")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        top_bar_layout.addWidget(self.warning_label)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(self.status_label)
        main_layout.addLayout(top_bar_layout)

        self.image_label = ClickableLabel("Loading...")
        self.image_label.setMinimumHeight(180)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color:#333; color:white;")
        main_layout.addWidget(self.image_label, 1)
        
        self._add_field_row(f"{self.main_column}:", 'main_column', item_data.get(self.main_column, ''))
        self._add_field_row("To:", 'to', item_data.get('to', ''))
        if 'co' in item_data: self._add_field_row("Crossed Out:", 'co', item_data.get('co', ''))
        if 'skip' in item_data:
            self.fields['skip'] = QCheckBox("Skip this file")
            self.fields['skip'].setChecked(item_data.get('skip', '') == 'x')
            self.fields['skip'].stateChanged.connect(self.data_changed.emit)
            main_layout.addWidget(self.fields['skip'])

    def _add_field_row(self, label_text: str, field_key: str, initial_value: str):
        row_layout = QHBoxLayout()
        label = QLabel(label_text); label.setFixedWidth(80)
        line_edit = QLineEdit(str(initial_value))
        line_edit.editingFinished.connect(self.data_changed.emit)
        self.fields[field_key] = line_edit
        row_layout.addWidget(label); row_layout.addWidget(line_edit)
        self.layout().addLayout(row_layout)

    def set_image(self, pixmap: QPixmap):
        self.image_label.setPixmap(pixmap)
    
    def set_warning(self, text: str):
        self.warning_label.setText(text)
        self.warning_label.setVisible(True)

    def clear_warning(self):
        self.warning_label.setVisible(False)
        
    def set_status(self, text: str, style: str):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)
        self.status_label.setVisible(True)
       
    def get_data(self) -> dict:
        data = {'df_index': self.df_index}
        for key, widget in self.fields.items():
            if isinstance(widget, QLineEdit): data[key] = widget.text().strip()
            elif isinstance(widget, QCheckBox): data[key] = 'x' if widget.isChecked() else ''
        if 'main_column' in data: data[self.main_column] = data.pop('main_column')
        return data