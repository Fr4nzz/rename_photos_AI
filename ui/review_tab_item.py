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

    def __init__(self, df_index: int, item_data: dict, main_column: str, parent=None):
        super().__init__(os.path.basename(item_data.get('from', 'N/A')), parent)
        self.df_index = df_index
        self.main_column = main_column
        self.fields = {}
        self._setup_ui(item_data)

    def _setup_ui(self, item_data: dict):
        main_layout = QVBoxLayout(self)
        self.setMinimumWidth(350)
        self.image_label = ClickableLabel("Loading...")
        self.image_label.setFixedSize(320, 180)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color:#333; color:white;")
        main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        
        self._add_field_row(f"{self.main_column}:", 'main_column', item_data.get(self.main_column, ''))
        self._add_field_row("To:", 'to', item_data.get('to', ''))
        if 'co' in item_data: self._add_field_row("Crossed Out:", 'co', item_data.get('co', ''))
        if 'skip' in item_data:
            self.fields['skip'] = QCheckBox("Skip this file")
            self.fields['skip'].setChecked(item_data.get('skip', '') == 'x')
            self.fields['skip'].stateChanged.connect(self.data_changed.emit)
            main_layout.addWidget(self.fields['skip'])
        main_layout.addStretch()

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
       
    def get_data(self) -> dict:
        data = {'df_index': self.df_index}
        for key, widget in self.fields.items():
            if isinstance(widget, QLineEdit): data[key] = widget.text().strip()
            elif isinstance(widget, QCheckBox): data[key] = 'x' if widget.isChecked() else ''
        if 'main_column' in data: data[self.main_column] = data.pop('main_column')
        return data