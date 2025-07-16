# ai-photo-processor/ui/process_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QComboBox, QCheckBox, QTextEdit, QProgressBar, QGroupBox, QFormLayout,
    QGridLayout, QSplitter
)
from PyQt5.QtCore import Qt

from .widgets import ClickableLabel

class ProcessImagesTab(QWidget):
    """The UI for the 'Process Images' tab, featuring a fully resizable splitter layout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Left Panel: ALL Settings ---
        settings_panel = QWidget()
        settings_vbox = QVBoxLayout(settings_panel)
        settings_vbox.setContentsMargins(0, 0, 5, 0)

        folder_group = QGroupBox("Input Folder")
        folder_layout = QHBoxLayout(folder_group)
        self.dir_path_label = ClickableLabel("(No folder selected)")
        self.dir_path_label.setWordWrap(True)
        self.browse_button = QPushButton("Browse...")
        folder_layout.addWidget(self.dir_path_label, 1)
        folder_layout.addWidget(self.browse_button)
        settings_vbox.addWidget(folder_group)
        
        exiftool_group = QGroupBox("ExifTool Configuration")
        exiftool_layout = QHBoxLayout(exiftool_group)
        self.exiftool_path_input = QLineEdit()
        self.exiftool_browse_button = QPushButton("...")
        exiftool_layout.addWidget(QLabel("Path:"))
        exiftool_layout.addWidget(self.exiftool_path_input, 1)
        exiftool_layout.addWidget(self.exiftool_browse_button)
        settings_vbox.addWidget(exiftool_group)

        selection_group = QGroupBox("Preview Selection")
        selection_layout = QFormLayout(selection_group)
        self.preview_image_dropdown = QComboBox()
        self.preview_raw_checkbox = QCheckBox("Preview RAW")
        self.batch_preview_dropdown = QComboBox()
        selection_layout.addRow("Image:", self.preview_image_dropdown)
        selection_layout.addRow("Batch:", self.batch_preview_dropdown)
        selection_layout.addRow(self.preview_raw_checkbox)
        settings_vbox.addWidget(selection_group)

        rotation_group = QGroupBox("Image Rotation")
        rotation_layout = QFormLayout(rotation_group)
        self.rotation_dropdown = QComboBox()
        self.use_exif_checkbox = QCheckBox("Use EXIF rotation")
        self.apply_rotation_button = QPushButton("Apply Rotation to Files")
        rotation_layout.addRow("Rotation:", self.rotation_dropdown)
        rotation_layout.addRow(self.use_exif_checkbox)
        rotation_layout.addRow(self.apply_rotation_button)
        settings_vbox.addWidget(rotation_group)

        crop_group = QGroupBox("Cropping & Filters")
        crop_layout = QFormLayout(crop_group)
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
        settings_vbox.addWidget(crop_group)

        api_group = QGroupBox("API Settings")
        api_layout = QFormLayout(api_group)
        self.model_dropdown = QComboBox()
        self.batch_size_input = QLineEdit()
        self.merged_img_height_input = QLineEdit()
        self.main_column_input = QLineEdit()
        self.save_prompt_button = QPushButton("Save All Settings")
        self.restore_prompt_button = QPushButton("Restore Default Prompt")
        prompt_buttons_layout = QHBoxLayout()
        prompt_buttons_layout.addStretch()
        prompt_buttons_layout.addWidget(self.save_prompt_button)
        prompt_buttons_layout.addWidget(self.restore_prompt_button)
        api_layout.addRow("Model:", self.model_dropdown)
        api_layout.addRow("Batch Size:", self.batch_size_input)
        api_layout.addRow("Merged Height:", self.merged_img_height_input)
        api_layout.addRow("Main Column:", self.main_column_input)
        api_layout.addRow(prompt_buttons_layout)
        settings_vbox.addWidget(api_group)
        settings_vbox.addStretch()

        # --- Right Panel: Previews and Prompt ---
        right_panel = QSplitter(Qt.Vertical)
        right_panel.setContentsMargins(5, 0, 0, 0)
        
        previews_container = QWidget()
        previews_grid_layout = QGridLayout(previews_container)
        
        self.original_preview_label = ClickableLabel()
        self.rotated_preview_label = ClickableLabel()
        self.processed_preview_label = ClickableLabel()
        self.combined_preview_label = ClickableLabel()

        previews_data = [
            ("Original", self.original_preview_label), ("Rotated", self.rotated_preview_label),
            ("Processed (for Gemini)", self.processed_preview_label), ("Batch Preview", self.combined_preview_label)
        ]
        
        for i, (title, label) in enumerate(previews_data):
            label.setStyleSheet("background-color: #333;")
            group_box = QGroupBox(title)
            group_layout = QHBoxLayout(group_box)
            group_layout.addWidget(label)
            previews_grid_layout.addWidget(group_box, i // 2, i % 2)
        
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_text_edit = QTextEdit()
        self.prompt_text_edit.setAcceptRichText(False)
        prompt_layout.addWidget(self.prompt_text_edit)
        
        right_panel.addWidget(previews_container)
        right_panel.addWidget(prompt_group)

        # --- Main Splitter Setup ---
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(settings_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([450, 800])
        right_panel.setSizes([600, 200])

        main_layout.addWidget(main_splitter)

        # --- Bottom Bar ---
        bottom_bar_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.start_processing_button = QPushButton("Ask Gemini (Start)")
        self.stop_processing_button = QPushButton("Stop")
        self.stop_processing_button.setEnabled(False)
        bottom_bar_layout.addWidget(self.progress_bar, 1)
        bottom_bar_layout.addWidget(self.start_processing_button)
        bottom_bar_layout.addWidget(self.stop_processing_button)
        main_layout.addLayout(bottom_bar_layout)