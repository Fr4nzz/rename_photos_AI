# ai-photo-processor/controllers/process_tab_handler.py

import os
import pandas as pd
import re
from pathlib import Path
from PIL import Image

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow
from PyQt5.QtCore import Qt, QUrl, QThread
from PyQt5.QtGui import QPixmap, QImage, QDesktopServices

from app_state import AppState, DEFAULT_PROMPT
from controllers.base_handler import BaseTabHandler
from ui.process_tab import ProcessImagesTab
from workers import RotationWorker, GeminiWorker, PreviewWorker
from utils.file_management import get_image_files, SUPPORTED_RAW_EXTENSIONS
from utils.image_processing import (
    preprocess_image, merge_images, fix_orientation, decode_raw_image
)
from utils.helpers import safe_int, safe_float
from utils.logger import SimpleLogger


class ProcessTabHandler(BaseTabHandler):

    def __init__(self, ui: ProcessImagesTab, app_state: AppState, logger: SimpleLogger, main_window: QMainWindow):
        super().__init__(ui, app_state, logger, main_window)
        self.preview_worker = None
        self.preview_thread = None

    def connect_signals(self):
        self.ui.browse_button.clicked.connect(self.select_directory)
        self.ui.dir_path_label.clicked.connect(self.on_folder_path_clicked)
        self.ui.exiftool_browse_button.clicked.connect(self.select_exiftool_path)
        self.ui.preview_image_dropdown.currentIndexChanged.connect(self.update_previews)
        self.ui.preview_raw_checkbox.stateChanged.connect(self.on_preview_mode_changed)
        self.ui.rotation_dropdown.currentIndexChanged.connect(self.update_all_previews)
        self.ui.use_exif_checkbox.stateChanged.connect(self.update_previews)
        self.ui.apply_rotation_button.clicked.connect(self.start_rotation)
        self.ui.batch_preview_dropdown.currentIndexChanged.connect(self.update_batch_preview)
        # Crop/filter checkboxes update both individual and batch previews
        for widget in [self.ui.zoom_checkbox, self.ui.grayscale_checkbox, self.ui.prerotate_checkbox]:
            widget.stateChanged.connect(self.update_all_previews)
        # Crop inputs update both individual and batch previews
        for widget in [self.ui.crop_top_input, self.ui.crop_bottom_input,
                       self.ui.crop_left_input, self.ui.crop_right_input]:
            widget.editingFinished.connect(self.update_all_previews)
        for widget in [self.ui.images_per_prompt_input, self.ui.batch_size_input, self.ui.merged_img_height_input]:
             widget.editingFinished.connect(self.update_batch_preview)
        self.ui.main_column_input.editingFinished.connect(self._sync_settings_from_ui)
        self.ui.model_dropdown.currentIndexChanged.connect(self._sync_settings_from_ui)
        self.ui.prompt_text_edit.textChanged.connect(self._sync_settings_from_ui)
        self.ui.save_prompt_button.clicked.connect(self.save_settings)
        self.ui.restore_prompt_button.clicked.connect(self.restore_default_prompt)
        self.ui.start_processing_button.clicked.connect(self.start_gemini_processing)
        self.ui.stop_processing_button.clicked.connect(self.stop_worker)
        for label in [self.ui.original_preview_label, self.ui.rotated_preview_label,
                      self.ui.processed_preview_label, self.ui.combined_preview_label]:
            label.clicked.connect(self.on_preview_label_clicked)

    def populate_initial_ui(self):
        path = self.app_state.input_directory
        self.ui.dir_path_label.setText(path or "(No folder selected)")
        self.ui.dir_path_label.setFilePath(path)
        self.ui.exiftool_path_input.setText(self.app_state.settings.get('exiftool_path', ''))
        
        orientations = {"0° (No Change)": 0, "90° CCW": 90, "180°": 180, "90° CW": 270}
        self.ui.rotation_dropdown.blockSignals(True)
        self.ui.rotation_dropdown.clear()
        for text, angle in orientations.items():
            self.ui.rotation_dropdown.addItem(text, angle)
        
        saved_angle = self.app_state.settings.get('rotation_angle', 180)
        index = self.ui.rotation_dropdown.findData(saved_angle)
        self.ui.rotation_dropdown.setCurrentIndex(index if index >= 0 else 2)
        self.ui.rotation_dropdown.blockSignals(False)
        
        self.ui.use_exif_checkbox.setChecked(self.app_state.settings['use_exif'])
        self.ui.preview_raw_checkbox.setChecked(self.app_state.settings['preview_raw'])
        
        cs = self.app_state.settings['crop_settings']
        self.ui.zoom_checkbox.setChecked(cs['zoom'])
        self.ui.grayscale_checkbox.setChecked(cs['grayscale'])
        self.ui.prerotate_checkbox.setChecked(cs.get('prerotate', False))
        self.ui.crop_top_input.setText(str(cs['top']))
        self.ui.crop_bottom_input.setText(str(cs['bottom']))
        self.ui.crop_left_input.setText(str(cs['left']))
        self.ui.crop_right_input.setText(str(cs['right']))
        
        self.ui.images_per_prompt_input.setText(str(self.app_state.settings.get('images_per_prompt', 10)))
        self.ui.batch_size_input.setText(str(self.app_state.settings['batch_size']))
        self.ui.merged_img_height_input.setText(str(self.app_state.settings['merged_img_height']))
        self.ui.main_column_input.setText(str(self.app_state.settings['main_column']))
        
        prompt_text = self.app_state.settings.get('prompt_text', '').strip()
        if not prompt_text:
            self.logger.info("Prompt is empty, loading default.")
            prompt_text = DEFAULT_PROMPT
            self.app_state.settings['prompt_text'] = prompt_text
        
        self.ui.prompt_text_edit.setPlainText(prompt_text)
        
        self.update_models_dropdown()
        self.refresh_file_dependent_ui()

    def select_directory(self):
        start_dir = self.app_state.input_directory or os.path.expanduser("~")
        if directory := QFileDialog.getExistingDirectory(self.main_window, "Select Folder", start_dir):
            self.app_state.set_input_directory(directory)
            self.ui.dir_path_label.setText(directory)
            self.ui.dir_path_label.setFilePath(directory)
            self.refresh_file_dependent_ui()

    def select_exiftool_path(self):
        if path := QFileDialog.getOpenFileName(self.main_window, "Select exiftool executable")[0]:
            self.ui.exiftool_path_input.setText(path)
            self._sync_settings_from_ui()
            self.app_state.save_settings()

    def on_folder_path_clicked(self, path: str):
        if path and os.path.isdir(path):
            self.logger.info(f"Opening folder in system explorer: {path}")
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def on_preview_label_clicked(self, path: str):
        if path and os.path.exists(path):
            self.logger.info(f"Opening image in default viewer: {path}")
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else: self.logger.warn(f"Cannot open file. Invalid path: {path}")

    def save_settings(self):
        self._sync_settings_from_ui()
        self.app_state.save_settings()
        QMessageBox.information(self.main_window, "Success", "Settings have been saved.")

    def restore_default_prompt(self):
        self.app_state.settings['prompt_text'] = DEFAULT_PROMPT
        self.ui.prompt_text_edit.setPlainText(DEFAULT_PROMPT)
        self.save_settings()

    def _sync_settings_from_ui(self):
        s, ui = self.app_state.settings, self.ui
        s['images_per_prompt'] = safe_int(ui.images_per_prompt_input.text(), default=5)
        s['batch_size'] = safe_int(ui.batch_size_input.text(), default=9)
        s['merged_img_height'] = safe_int(ui.merged_img_height_input.text(), default=1080)
        s['main_column'] = ui.main_column_input.text() or 'CAM'
        s['model_name'] = ui.model_dropdown.currentText()
        s['prompt_text'] = ui.prompt_text_edit.toPlainText()
        s['exiftool_path'] = ui.exiftool_path_input.text()
        s['rotation_angle'] = ui.rotation_dropdown.currentData(Qt.UserRole)
        s['use_exif'] = ui.use_exif_checkbox.isChecked()
        s['preview_raw'] = ui.preview_raw_checkbox.isChecked()

        # Update crop settings - safe_float handles empty/invalid input gracefully
        cs = s['crop_settings']
        cs['zoom'] = ui.zoom_checkbox.isChecked()
        cs['grayscale'] = ui.grayscale_checkbox.isChecked()
        cs['prerotate'] = ui.prerotate_checkbox.isChecked()
        cs['top'] = safe_float(ui.crop_top_input.text(), default=cs.get('top', 0.0))
        cs['bottom'] = safe_float(ui.crop_bottom_input.text(), default=cs.get('bottom', 0.0))
        cs['left'] = safe_float(ui.crop_left_input.text(), default=cs.get('left', 0.0))
        cs['right'] = safe_float(ui.crop_right_input.text(), default=cs.get('right', 0.0))

    def on_preview_mode_changed(self):
        self.refresh_file_dependent_ui()

    def update_all_previews(self):
        """Update both individual previews and batch preview in a single worker."""
        self._sync_settings_from_ui()

        # Get individual image path
        img_path = None
        if selected_file := self.ui.preview_image_dropdown.currentText():
            img_path = Path(self.app_state.input_directory) / selected_file
            if img_path.exists():
                for label in [self.ui.original_preview_label, self.ui.rotated_preview_label, self.ui.processed_preview_label]:
                    label.setText("Loading...")
                    label.setFilePath("")
            else:
                img_path = None

        # Get batch files and update dropdown
        jpg_files = get_image_files(self.app_state.input_directory, 'compressed')
        batch_start_idx = 0

        if jpg_files:
            s = self.app_state.settings
            batch_size = s['batch_size']
            if batch_size > 0:
                num_batches = (len(jpg_files) + batch_size - 1) // batch_size
                current_idx = self.ui.batch_preview_dropdown.currentIndex()
                self.ui.batch_preview_dropdown.blockSignals(True)
                self.ui.batch_preview_dropdown.clear()
                if num_batches > 0:
                    self.ui.batch_preview_dropdown.addItems([f"Batch {i+1}" for i in range(num_batches)])
                    if 0 <= current_idx < num_batches:
                        self.ui.batch_preview_dropdown.setCurrentIndex(current_idx)
                self.ui.batch_preview_dropdown.blockSignals(False)

                if self.ui.batch_preview_dropdown.count() > 0:
                    batch_start_idx = self.ui.batch_preview_dropdown.currentIndex() * batch_size
                    self.ui.combined_preview_label.clear()
                    self.ui.combined_preview_label.setText("Loading...")
                    self.ui.combined_preview_label.setFilePath("")

        # Start single worker for both
        if img_path or jpg_files:
            self._start_preview_worker(img_path, jpg_files, batch_start_idx)

    def refresh_file_dependent_ui(self):
        self._sync_settings_from_ui()
        if not self.app_state.input_directory: return
        self.populate_continue_dropdown() # Populate the continue dropdown
        file_type = 'raw' if self.app_state.settings['preview_raw'] else 'compressed'
        image_files = get_image_files(self.app_state.input_directory, file_type)
        self.ui.preview_image_dropdown.blockSignals(True)
        self.ui.preview_image_dropdown.clear()
        if image_files: self.ui.preview_image_dropdown.addItems([p.name for p in image_files])
        self.ui.preview_image_dropdown.blockSignals(False)
        if image_files:
            self.update_all_previews()
        else: self.clear_all_previews()

    def clear_all_previews(self):
        for label in [self.ui.original_preview_label, self.ui.rotated_preview_label,
                      self.ui.processed_preview_label, self.ui.combined_preview_label]:
            label.clear()
            label.setText("No Images Found")
            label.setFilePath("")

    def start_rotation(self):
        if not self.app_state.input_directory:
            QMessageBox.warning(self.main_window, "No Folder", "Please select an input folder first.")
            return
        self._sync_settings_from_ui()
        settings_copy = self.app_state.settings.copy()
        settings_copy['folder_path'] = self.app_state.input_directory
        settings_copy['file_type'] = 'raw' if settings_copy['preview_raw'] else 'compressed'
        self.current_worker = RotationWorker(settings_copy, self.logger)
        self._start_worker_thread()

    def start_gemini_processing(self):
        self._sync_settings_from_ui()
        run_mode = self.ui.continue_dropdown.currentText()
        start_batch, total_batches, df = 1, 0, pd.DataFrame()

        if not self.app_state.api_keys:
            QMessageBox.warning(self.main_window, "No API Keys", "Please add one or more API keys in the 'API Keys' tab.")
            return

        if run_mode.startswith("Continue from"):
            csv_name = self.ui.continue_dropdown.currentData()
            csv_path = os.path.join(self.app_state.rename_files_dir, csv_name)
            if os.path.exists(csv_path):
                self.logger.info(f"Attempting to continue from {csv_name}")
                df = pd.read_csv(csv_path)
                match = re.search(r'_b(\d+)of(\d+)\.csv$', csv_name)
                if match:
                    last_completed = int(match.group(1))
                    total_batches = int(match.group(2))
                    start_batch = last_completed + 1
                    self.logger.info(f"Resuming from batch {start_batch} of {total_batches}.")
                else:
                    self.logger.warn(f"Could not parse batch numbers from {csv_name}. Starting over.")
                    run_mode = "Start Over"
            else:
                self.logger.warn(f"Selected CSV {csv_name} not found. Starting over.")
                run_mode = "Start Over"
        
        if run_mode == "Start Over":
            self.logger.info("Starting a new processing run.")
            image_paths = get_image_files(self.app_state.input_directory, 'compressed')
            if not image_paths:
                QMessageBox.warning(self.main_window, "No Images", "No compressed images found to process.")
                return

            num_files = len(image_paths)
            df_data = {
                'from': [str(p) for p in image_paths],
                'photo_ID': range(1, num_files + 1),
                'CAM': [''] * num_files, 'co': [''] * num_files,
                'n': [''] * num_files, 'skip': [''] * num_files,
            }
            df = pd.DataFrame(df_data)

        if df.empty:
            QMessageBox.warning(self.main_window, "No Data", "Could not prepare data for processing.")
            return

        self.app_state.current_df = df
        settings_for_worker = self.app_state.settings.copy()
        settings_for_worker['api_keys'] = self.app_state.api_keys
        
        self.current_worker = GeminiWorker(
            df.copy(), settings_for_worker, self.app_state.rename_files_dir, self.logger,
            start_batch=start_batch, total_batches=total_batches
        )
        self._start_worker_thread()

    def _start_worker_thread(self):
        self.worker_thread = QThread()
        self.current_worker.moveToThread(self.worker_thread)
        self.current_worker.progress.connect(self.update_progress_bar)
        self.current_worker.finished.connect(self.on_worker_finished)
        self.current_worker.error.connect(self.on_worker_error)
        if isinstance(self.current_worker, GeminiWorker):
            self.current_worker.batch_completed.connect(self.on_gemini_batch_complete)
        self.worker_thread.started.connect(self.current_worker.run)
        self.worker_thread.start()
        self.set_ui_processing_state(True)


    def set_ui_processing_state(self, is_processing: bool):
        self.ui.start_processing_button.setEnabled(not is_processing)
        self.ui.stop_processing_button.setEnabled(is_processing)
        self.ui.apply_rotation_button.setEnabled(not is_processing)

    def on_gemini_batch_complete(self, df: pd.DataFrame, batch_num: int, total_batches: int):
        self.app_state.current_df = df
        path = os.path.join(self.app_state.rename_files_dir, f"temp_output_b{batch_num}of{total_batches}.csv")
        df.to_csv(path, index=False)
        self.logger.info(f"Batch {batch_num}/{total_batches} complete. Saved to {path}")

    def on_worker_finished(self):
        QMessageBox.information(self.main_window, "Complete", "The process has finished.")
        self.set_ui_processing_state(False)
        self._cleanup_worker_thread()
        self.update_progress_bar(0, "%p%")
        self.populate_continue_dropdown()  # Refresh options after run

    def on_worker_error(self, message: str):
        QMessageBox.critical(self.main_window, "Error", message)
        self.set_ui_processing_state(False)
        self._cleanup_worker_thread()
        self.update_progress_bar(0, "Error")
        self.populate_continue_dropdown()  # Refresh options after error

    def update_progress_bar(self, value: int, text_format: str):
        self.ui.progress_bar.setFormat(text_format)
        self.ui.progress_bar.setValue(value)

    def pil_to_qpixmap(self, pil_img: Image.Image) -> QPixmap:
        if not pil_img: return QPixmap()
        if pil_img.mode not in ['RGB', 'RGBA']: pil_img = pil_img.convert('RGB')
        
        if pil_img.mode == 'RGB':
            q_img = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, 3 * pil_img.width, QImage.Format_RGB888)
        else: # RGBA
            q_img = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, 4 * pil_img.width, QImage.Format_RGBA8888)
        
        return QPixmap.fromImage(q_img)

    def update_previews(self):
        self._sync_settings_from_ui()
        if not (selected_file := self.ui.preview_image_dropdown.currentText()): return
        img_path = Path(self.app_state.input_directory) / selected_file
        if not img_path.exists(): return

        # Show loading state
        for label in [self.ui.original_preview_label, self.ui.rotated_preview_label, self.ui.processed_preview_label]:
            label.setText("Loading...")
            label.setFilePath("")

        self._start_preview_worker(img_path)

    def update_batch_preview(self):
        self._sync_settings_from_ui()
        jpg_files = get_image_files(self.app_state.input_directory, 'compressed')
        if not jpg_files:
            self.ui.combined_preview_label.clear()
            self.ui.combined_preview_label.setText("No Images")
            return

        s = self.app_state.settings
        batch_size = s['batch_size']
        if batch_size <= 0: return

        num_batches = (len(jpg_files) + batch_size - 1) // batch_size

        current_idx = self.ui.batch_preview_dropdown.currentIndex()
        self.ui.batch_preview_dropdown.blockSignals(True)
        self.ui.batch_preview_dropdown.clear()
        if num_batches > 0:
            self.ui.batch_preview_dropdown.addItems([f"Batch {i+1}" for i in range(num_batches)])
            if 0 <= current_idx < num_batches: self.ui.batch_preview_dropdown.setCurrentIndex(current_idx)
        self.ui.batch_preview_dropdown.blockSignals(False)

        if self.ui.batch_preview_dropdown.count() == 0: return

        # Show loading state and start worker
        self.ui.combined_preview_label.clear()
        self.ui.combined_preview_label.setText("Loading...")
        self.ui.combined_preview_label.setFilePath("")

        start_idx = self.ui.batch_preview_dropdown.currentIndex() * batch_size
        self._start_preview_worker(None, jpg_files, start_idx)

    def _stop_preview_worker(self):
        """Stop any running preview worker."""
        if self.preview_worker:
            self.preview_worker.stop()
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.quit()
            self.preview_thread.wait(1000)
        self.preview_worker = None
        self.preview_thread = None

    def _start_preview_worker(self, img_path=None, jpg_files=None, batch_start_idx=0):
        """Start the preview worker thread."""
        self._stop_preview_worker()

        s = self.app_state.settings
        config = {
            'image_path': str(img_path) if img_path else '',
            'jpg_files': jpg_files or [],
            'batch_start_idx': batch_start_idx,
            'use_exif': s.get('use_exif', True),
            'rotation_angle': s.get('rotation_angle', 0),
            'crop_settings': s.get('crop_settings', {}),
            'batch_size': s.get('batch_size', 9),
            'merged_img_height': s.get('merged_img_height', 1080),
            'temp_dir': self.app_state.rename_files_dir or '',
        }

        self.preview_worker = PreviewWorker(config, self.logger)
        self.preview_thread = QThread()
        self.preview_worker.moveToThread(self.preview_thread)

        self.preview_worker.preview_ready.connect(self._on_preview_ready)
        self.preview_worker.batch_preview_ready.connect(self._on_batch_preview_ready)
        self.preview_worker.finished.connect(self._on_preview_finished)
        self.preview_thread.started.connect(self.preview_worker.run)

        self.preview_thread.start()

    def _on_preview_ready(self, preview_type: str, img_bytes: bytes, width: int, height: int, file_path: str):
        """Handle individual preview ready signal."""
        q_img = QImage(img_bytes, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        label_map = {
            'original': self.ui.original_preview_label,
            'rotated': self.ui.rotated_preview_label,
            'processed': self.ui.processed_preview_label,
        }
        if label := label_map.get(preview_type):
            label.setPixmap(pixmap)
            label.setFilePath(file_path)

    def _on_batch_preview_ready(self, img_bytes: bytes, width: int, height: int, file_path: str):
        """Handle batch preview ready signal."""
        q_img = QImage(img_bytes, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.ui.combined_preview_label.setPixmap(pixmap)
        self.ui.combined_preview_label.setFilePath(file_path)

    def _on_preview_finished(self):
        """Clean up after preview worker finishes.

        Note: Do NOT call _stop_preview_worker() here! If a new worker was
        started while this one was running, it would kill the new worker.
        Cleanup happens in _start_preview_worker when a new worker starts.
        """
        pass

    def populate_continue_dropdown(self):
        self.ui.continue_dropdown.clear()
        self.ui.continue_dropdown.addItem("Start Over", "")

        if not self.app_state.rename_files_dir or not os.path.isdir(self.app_state.rename_files_dir):
            return

        try:
            partials = [
                f for f in os.listdir(self.app_state.rename_files_dir)
                if f.startswith("temp_output_") and f.endswith('.csv')
            ]
            # Sort by modification time, newest first
            partials.sort(
                key=lambda f: os.path.getmtime(os.path.join(self.app_state.rename_files_dir, f)),
                reverse=True
            )

            for p in partials:
                self.ui.continue_dropdown.addItem(f"Continue from {p}", p) # Store filename in UserData

            if len(partials) > 0:
                self.ui.continue_dropdown.setCurrentIndex(1) # Select the newest partial by default
        except FileNotFoundError:
            self.logger.warn("Could not populate continue dropdown, rename_files directory not found.")

    def update_models_dropdown(self):
        """Fetch Gemini models (2.5+), Flash before Pro."""
        self.ui.model_dropdown.clear()

        if not self.app_state.api_keys:
            self.ui.model_dropdown.addItem("No API Key Set")
            return

        try:
            from google import genai

            client = genai.Client(api_key=self.app_state.api_keys[0])

            # Patterns to exclude (non-text-generation models)
            EXCLUDED_PATTERNS = [
                '-tts', '-image', '-audio', '-video', 'embedding',
                'aqa', 'bisheng', 'learnlm', 'imagen', 'veo'
            ]

            usable_models = []

            for model in client.models.list():
                # Get model name, strip "models/" prefix if present
                model_name = model.name
                if '/' in model_name:
                    model_name = model_name.split('/')[-1]

                model_lower = model_name.lower()

                # Skip non-Gemini models
                if 'gemini' not in model_lower:
                    continue

                # Skip excluded patterns
                if any(pattern in model_lower for pattern in EXCLUDED_PATTERNS):
                    continue

                # Parse version - handles "gemini-2.5-flash" and "gemini-3-flash"
                version_match = re.search(r'gemini-(\d+)(?:\.(\d+))?', model_lower)
                if not version_match:
                    continue

                major = int(version_match.group(1))
                minor = int(version_match.group(2)) if version_match.group(2) else 0

                # Filter: version >= 2.5
                if major < 2 or (major == 2 and minor < 5):
                    continue

                usable_models.append({
                    'name': model_name,
                    'major': major,
                    'minor': minor,
                    'is_preview': 'preview' in model_lower,
                    'is_flash': 'flash' in model_lower,
                    'is_pro': 'pro' in model_lower,
                })

            # Sort priority (highest first):
            # 1. Higher version (3.x before 2.x)
            # 2. Flash BEFORE Pro (free tier friendly - Pro often not available on free tier)
            # 3. Stable before preview (but both should work)
            usable_models.sort(
                key=lambda m: (
                    m['major'],           # Higher major version first
                    m['minor'],           # Higher minor version first
                    m['is_flash'],        # Flash BEFORE Pro (True sorts after False, so flash=True goes first with reverse)
                    not m['is_preview'],  # Stable before preview
                ),
                reverse=True
            )

            # Populate dropdown
            if usable_models:
                model_names = [m['name'] for m in usable_models]
                self.app_state.available_models = model_names
                self.ui.model_dropdown.addItems(model_names)

                # Try to restore saved model, otherwise use first (best) option
                saved_model = self.app_state.settings.get('model_name')
                if saved_model and saved_model in model_names:
                    self.ui.model_dropdown.setCurrentText(saved_model)
                else:
                    # Default to first model (should be gemini-3-flash or gemini-3-flash-preview)
                    self.ui.model_dropdown.setCurrentIndex(0)
                    self.app_state.settings['model_name'] = model_names[0]
                    self.logger.info(f"Default model set to: {model_names[0]}")
            else:
                self.ui.model_dropdown.addItem("No compatible models found")

        except Exception as e:
            self.logger.error("Failed to fetch Gemini models", exception=e)
            self.ui.model_dropdown.addItem("Error fetching models")