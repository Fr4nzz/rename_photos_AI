# ai-photo-processor/ui/process_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QComboBox, QCheckBox, QTextEdit, QProgressBar, QGroupBox, QFormLayout,
    QSplitter
)
from PyQt5.QtCore import Qt

from .widgets import ClickableLabel

class ProcessImagesTab(QWidget):
    """The UI for the 'Process Images' tab, featuring a redesigned, resizable layout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_ui()

    def create_preview_label(self, text: str) -> ClickableLabel:
        """Helper to create a standard, styled, clickable preview label."""
        label = ClickableLabel(text)
        label.setFixedSize(400, 225)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background-color: #333; color: white; border: 1px solid #555;")
        return label

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        top_bar_layout = QHBoxLayout()
        
        folder_group = QGroupBox("Input Folder")
        folder_layout = QHBoxLayout()
        self.dir_path_label = ClickableLabel("(No folder selected)")
        self.browse_button = QPushButton("Browse...")
        folder_layout.addWidget(self.dir_path_label, 1)
        folder_layout.addWidget(self.browse_button)
        folder_group.setLayout(folder_layout)

        process_group = QGroupBox("Process")
        process_layout = QHBoxLayout()
        self.start_processing_button = QPushButton("Ask Gemini (Start)")
        self.stop_processing_button = QPushButton("Stop")
        self.stop_processing_button.setEnabled(False)
        process_layout.addWidget(self.start_processing_button)
        process_layout.addWidget(self.stop_processing_button)
        process_group.setLayout(process_layout)

        top_bar_layout.addWidget(folder_group, 2)
        top_bar_layout.addWidget(process_group, 1)
        main_layout.addLayout(top_bar_layout)

        splitter = QSplitter(Qt.Horizontal)
        previews_panel = QGroupBox("Previews")
        previews_layout = QFormLayout()
        previews_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        preview_selection_hbox = QHBoxLayout()
        self.preview_image_dropdown = QComboBox()
        self.preview_raw_checkbox = QCheckBox("Preview RAW")
        preview_selection_hbox.addWidget(self.preview_image_dropdown, 1)
        preview_selection_hbox.addWidget(self.preview_raw_checkbox)
        self.original_preview_label = self.create_preview_label("Original")
        self.rotated_preview_label = self.create_preview_label("Rotated")
        self.processed_preview_label = self.create_preview_label("Processed")
        self.batch_preview_dropdown = QComboBox()
        self.combined_preview_label = self.create_preview_label("Batch Preview")
        previews_layout.addRow("Image:", preview_selection_hbox)
        previews_layout.addRow("Original:", self.original_preview_label)
        previews_layout.addRow("Rotated:", self.rotated_preview_label)
        previews_layout.addRow("Processed:", self.processed_preview_label)
        previews_layout.addRow("Preview Batch:", self.batch_preview_dropdown)
        previews_layout.addRow("Batch Preview:", self.combined_preview_label)
        previews_panel.setLayout(previews_layout)
        splitter.addWidget(previews_panel)

        settings_panel = QWidget()
        settings_main_layout = QHBoxLayout(settings_panel)
        settings_left_vbox = QVBoxLayout()
        settings_right_vbox = QVBoxLayout()

        rotation_group = QGroupBox("Image Rotation")
        rotation_layout = QFormLayout()
        self.rotation_dropdown = QComboBox()
        self.use_exif_checkbox = QCheckBox("Use EXIF rotation")
        self.apply_rotation_button = QPushButton("Apply Rotation to Files")
        rotation_layout.addRow("Rotation:", self.rotation_dropdown)
        rotation_layout.addRow(self.use_exif_checkbox)
        rotation_layout.addRow(self.apply_rotation_button)
        rotation_group.setLayout(rotation_layout)

        crop_group = QGroupBox("Cropping & Filters")
        crop_layout = QFormLayout()
        self.zoom_checkbox = QCheckBox("Enable Cropping")
        self.grayscale_checkbox = QCheckBox("Convert to Grayscale")
        self.crop_top_input = QLineEdit()
        self.crop_bottom_input = QLineEdit()
        self.crop_left_input = QLineEdit()
        self.crop_right_input = QLineEdit()
        crop_layout.addRow(self.zoom_checkbox)
        crop_layout.addRow(self.grayscale_checkbox)
        crop_layout.addRow("Top %:", self.crop_top_input)
        crop_layout.addRow("Bottom %:", self.crop_bottom_input)
        crop_layout.addRow("Left %:", self.crop_left_input)
        crop_layout.addRow("Right %:", self.crop_right_input)
        crop_group.setLayout(crop_layout)

        settings_left_vbox.addWidget(rotation_group)
        settings_left_vbox.addWidget(crop_group)
        settings_left_vbox.addStretch()

        exiftool_group = QGroupBox("ExifTool Configuration")
        exiftool_layout = QFormLayout()
        exiftool_hbox = QHBoxLayout()
        self.exiftool_path_input = QLineEdit()
        self.exiftool_browse_button = QPushButton("...")
        exiftool_hbox.addWidget(self.exiftool_path_input)
        exiftool_hbox.addWidget(self.exiftool_browse_button)
        exiftool_layout.addRow("ExifTool Path:", exiftool_hbox)
        exiftool_group.setLayout(exiftool_layout)

        api_group = QGroupBox("API Settings")
        api_layout = QFormLayout()
        self.model_dropdown = QComboBox()
        self.batch_size_input = QLineEdit()
        self.merged_img_height_input = QLineEdit()
        self.main_column_input = QLineEdit()
        self.prompt_text_edit = QTextEdit()
        self.prompt_text_edit.setAcceptRichText(False)
        self.save_prompt_button = QPushButton("Save All Settings")
        self.restore_prompt_button = QPushButton("Restore Default Prompt")
        prompt_buttons_layout = QHBoxLayout()
        prompt_buttons_layout.addWidget(self.save_prompt_button)
        prompt_buttons_layout.addWidget(self.restore_prompt_button)
        api_layout.addRow("Model:", self.model_dropdown)
        api_layout.addRow("Batch Size:", self.batch_size_input)
        api_layout.addRow("Merged Height:", self.merged_img_height_input)
        api_layout.addRow("Main Column:", self.main_column_input)
        api_layout.addRow(QLabel("Prompt:"))
        api_layout.addRow(self.prompt_text_edit)
        api_layout.addRow(prompt_buttons_layout)
        api_group.setLayout(api_layout)

        settings_right_vbox.addWidget(exiftool_group)
        settings_right_vbox.addWidget(api_group)

        settings_main_layout.addLayout(settings_left_vbox)
        settings_main_layout.addLayout(settings_right_vbox)
        splitter.addWidget(settings_panel)

        splitter.setSizes([450, 600])
        main_layout.addWidget(splitter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        main_layout.addWidget(self.progress_bar)