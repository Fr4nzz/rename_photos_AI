# ai-photo-processor/controllers/process_tab_handler.py

import os
import google.generativeai as genai
import pandas as pd
from pathlib import Path
from PIL import Image

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow
from PyQt5.QtCore import Qt, QUrl, QThread, QObject
from PyQt5.QtGui import QPixmap, QImage, QDesktopServices

from app_state import AppState, DEFAULT_PROMPT
from ui.process_tab import ProcessImagesTab
from workers import RotationWorker, GeminiWorker
from utils.file_management import get_image_files, SUPPORTED_RAW_EXTENSIONS
from utils.image_processing import (
    preprocess_image, merge_images, fix_orientation, decode_raw_with_info
)
from utils.logger import SimpleLogger


class ProcessTabHandler(QObject):
    """Controller for all logic related to the Process Images tab."""

    def __init__(self, ui: ProcessImagesTab, app_state: AppState, logger: SimpleLogger, main_window: QMainWindow):
        super().__init__()
        self.ui = ui
        self.app_state = app_state
        self.logger = logger
        self.main_window = main_window
        self.worker_thread = None
        self.current_worker = None

    def connect_signals(self):
        # Folder and ExifTool
        self.ui.browse_button.clicked.connect(self.select_directory)
        self.ui.dir_path_label.clicked.connect(self.on_folder_path_clicked)
        self.ui.exiftool_browse_button.clicked.connect(self.select_exiftool_path)
        # Previews
        self.ui.preview_image_dropdown.currentIndexChanged.connect(self.update_previews)
        self.ui.preview_raw_checkbox.stateChanged.connect(self.on_preview_mode_changed)
        self.ui.rotation_dropdown.currentIndexChanged.connect(self.update_previews)
        self.ui.use_exif_checkbox.stateChanged.connect(self.update_previews)
        self.ui.apply_rotation_button.clicked.connect(self.start_rotation)
        self.ui.batch_preview_dropdown.currentIndexChanged.connect(self.update_batch_preview)
        # Settings Syncing
        for widget in [self.ui.zoom_checkbox, self.ui.grayscale_checkbox]:
            widget.stateChanged.connect(self._sync_and_update_previews)
        for widget in [
            self.ui.crop_top_input, self.ui.crop_bottom_input, self.ui.crop_left_input, self.ui.crop_right_input,
            self.ui.batch_size_input, self.ui.merged_img_height_input, self.ui.main_column_input]:
            widget.editingFinished.connect(self._sync_and_update_previews)
        self.ui.model_dropdown.currentIndexChanged.connect(self._sync_settings_from_ui)
        self.ui.prompt_text_edit.textChanged.connect(self._sync_settings_from_ui)
        self.ui.save_prompt_button.clicked.connect(self.save_settings)
        self.ui.restore_prompt_button.clicked.connect(self.restore_default_prompt)
        # Primary Actions
        self.ui.start_processing_button.clicked.connect(self.start_gemini_processing)
        self.ui.stop_processing_button.clicked.connect(self.stop_worker)
        # Clickable Previews
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
        for text, angle in orientations.items():
            self.ui.rotation_dropdown.addItem(text, angle)
        self.ui.rotation_dropdown.setCurrentIndex(2)
        self.ui.rotation_dropdown.blockSignals(False)

        self.ui.use_exif_checkbox.setChecked(self.app_state.settings['use_exif'])
        self.ui.preview_raw_checkbox.setChecked(self.app_state.settings['preview_raw'])
        
        cs = self.app_state.settings['crop_settings']
        self.ui.zoom_checkbox.setChecked(cs['zoom'])
        self.ui.grayscale_checkbox.setChecked(cs['grayscale'])
        self.ui.crop_top_input.setText(str(cs['top']))
        self.ui.crop_bottom_input.setText(str(cs['bottom']))
        self.ui.crop_left_input.setText(str(cs['left']))
        self.ui.crop_right_input.setText(str(cs['right']))
        
        self.ui.batch_size_input.setText(str(self.app_state.settings['batch_size']))
        self.ui.merged_img_height_input.setText(str(self.app_state.settings['merged_img_height']))
        self.ui.main_column_input.setText(str(self.app_state.settings['main_column']))
        self.ui.prompt_text_edit.setPlainText(self.app_state.settings['prompt_text'])

        self.update_models_dropdown()
        self.refresh_file_dependent_ui()

    def select_directory(self):
        start_dir = self.app_state.input_directory or os.path.expanduser("~")
        directory = QFileDialog.getExistingDirectory(self.main_window, "Select Folder", start_dir)
        if directory:
            self.app_state.set_input_directory(directory)
            self.ui.dir_path_label.setText(directory)
            self.ui.dir_path_label.setFilePath(directory)
            self.refresh_file_dependent_ui()

    def select_exiftool_path(self):
        path, _ = QFileDialog.getOpenFileName(self.main_window, "Select exiftool executable")
        if path:
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
        else:
            self.logger.warn(f"Cannot open file. Path is invalid or file does not exist: {path}")

    def save_settings(self):
        self._sync_settings_from_ui()
        self.app_state.save_settings()
        QMessageBox.information(self.main_window, "Success", "Settings have been saved.")

    def restore_default_prompt(self):
        self.app_state.settings['prompt_text'] = DEFAULT_PROMPT
        self.ui.prompt_text_edit.setPlainText(DEFAULT_PROMPT)
        self.save_settings()
        QMessageBox.information(self.main_window, "Success", "Default prompt has been restored and saved.")

    def _sync_settings_from_ui(self):
        s, ui = self.app_state.settings, self.ui
        s['batch_size'] = int(ui.batch_size_input.text()) if ui.batch_size_input.text().isdigit() else 9
        s['merged_img_height'] = int(ui.merged_img_height_input.text()) if ui.merged_img_height_input.text().isdigit() else 1080
        s['main_column'] = ui.main_column_input.text() or 'CAM'
        s['model_name'] = ui.model_dropdown.currentText()
        s['prompt_text'] = ui.prompt_text_edit.toPlainText()
        s['exiftool_path'] = ui.exiftool_path_input.text()
        s['rotation_angle'] = ui.rotation_dropdown.currentData(Qt.UserRole)
        s['use_exif'] = ui.use_exif_checkbox.isChecked()
        s['preview_raw'] = ui.preview_raw_checkbox.isChecked()
        cs = s['crop_settings']
        try:
            cs['zoom'] = ui.zoom_checkbox.isChecked()
            cs['grayscale'] = ui.grayscale_checkbox.isChecked()
            cs['top'] = float(ui.crop_top_input.text())
            cs['bottom'] = float(ui.crop_bottom_input.text())
            cs['left'] = float(ui.crop_left_input.text())
            cs['right'] = float(ui.crop_right_input.text())
        except (ValueError, TypeError):
             self.logger.warn("Invalid crop value entered. Ignoring change.")

    def _sync_and_update_previews(self):
        self._sync_settings_from_ui()
        self.update_previews()
        self.update_batch_preview()

    def on_preview_mode_changed(self):
        self._sync_settings_from_ui()
        self.refresh_file_dependent_ui()

    def refresh_file_dependent_ui(self):
        if not self.app_state.input_directory: return
        file_type = 'raw' if self.app_state.settings['preview_raw'] else 'compressed'
        image_files = get_image_files(self.app_state.input_directory, file_type)
        self.ui.preview_image_dropdown.blockSignals(True)
        self.ui.preview_image_dropdown.clear()
        if image_files: self.ui.preview_image_dropdown.addItems([p.name for p in image_files])
        self.ui.preview_image_dropdown.blockSignals(False)
        if image_files: self.update_previews(); self.update_batch_preview()
        else: self.clear_all_previews()

    def clear_all_previews(self):
        for label in [self.ui.original_preview_label, self.ui.rotated_preview_label,
                      self.ui.processed_preview_label, self.ui.combined_preview_label]:
            label.clear(); label.setText("No Images Found"); label.setFilePath("")

    def start_rotation(self):
        if not self.app_state.input_directory:
            QMessageBox.warning(self.main_window, "No Folder", "Please select an input folder first."); return
        self._sync_settings_from_ui()
        self.current_worker = RotationWorker(self.app_state.settings, self.logger)
        self._start_worker_thread()

    def start_gemini_processing(self):
        self._sync_settings_from_ui()
        image_paths = get_image_files(self.app_state.input_directory, 'compressed')
        if not image_paths:
            QMessageBox.warning(self.main_window, "No Images", "No compressed images found for processing."); return
        df_data = {'from': [str(p) for p in image_paths], 'photo_ID': range(1, len(image_paths) + 1)}
        self.app_state.current_df = pd.DataFrame(df_data)
        self.current_worker = GeminiWorker(self.app_state.current_df.copy(), self.app_state.settings, self.app_state.rename_files_dir, self.logger)
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

    def stop_worker(self):
        if self.current_worker: self.current_worker.stop()

    def set_ui_processing_state(self, is_processing: bool):
        # Only enable/disable the action buttons, not the whole UI
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
        self.worker_thread.quit(); self.worker_thread.wait()
        self.worker_thread, self.current_worker = None, None
        self.update_progress_bar(0, "%p%")

    def on_worker_error(self, message: str):
        QMessageBox.critical(self.main_window, "Error", message)
        self.set_ui_processing_state(False)
        if self.worker_thread: self.worker_thread.quit(); self.worker_thread.wait()
        self.worker_thread, self.current_worker = None, None
        self.update_progress_bar(0, "Error")

    def update_progress_bar(self, value: int, text_format: str):
        self.ui.progress_bar.setFormat(text_format)
        self.ui.progress_bar.setValue(value)

    def pil_to_qpixmap(self, pil_img: Image.Image) -> QPixmap:
        if not pil_img: return QPixmap()
        if pil_img.mode not in ['RGB', 'RGBA']: pil_img = pil_img.convert('RGB')
        if pil_img.mode == 'RGB':
            return QPixmap.fromImage(QImage(pil_img.tobytes(), pil_img.width, pil_img.height, 3 * pil_img.width, QImage.Format_RGB888))
        else: # RGBA
            return QPixmap.fromImage(QImage(pil_img.tobytes(), pil_img.width, pil_img.height, 4 * pil_img.width, QImage.Format_RGBA8888))

    def update_previews(self):
        if not (selected_file := self.ui.preview_image_dropdown.currentText()): return
        img_path = Path(self.app_state.input_directory) / selected_file
        if not img_path.exists(): return
        s = self.app_state.settings
        base_img_previews, exif_corrected_img = None, None
        try:
            if img_path.suffix.lower() in SUPPORTED_RAW_EXTENSIONS:
                exif_corrected_img, native_angle = decode_raw_with_info(img_path)
                if not exif_corrected_img: raise IOError("RAW Decode Failed")
                base_img_previews = exif_corrected_img if s['use_exif'] else exif_corrected_img.rotate(-native_angle, expand=True)
            else:
                raw_pil_image = Image.open(img_path)
                exif_corrected_img = fix_orientation(raw_pil_image.copy())
                base_img_previews = exif_corrected_img if s['use_exif'] else raw_pil_image
            
            rotated_image = base_img_previews.rotate(s['rotation_angle'], expand=True)
            processed_image = preprocess_image(exif_corrected_img, "1", s['crop_settings'])

            for label, img, path in [
                (self.ui.original_preview_label, base_img_previews, str(img_path)),
                (self.ui.rotated_preview_label, rotated_image, str(img_path)),
                (self.ui.processed_preview_label, processed_image, str(img_path))
            ]:
                label.setPixmap(self.pil_to_qpixmap(img)); label.setFilePath(path)
        except Exception as e:
            self.logger.error(f"Error generating preview for {img_path.name}", exception=e)
            for label in [self.ui.original_preview_label, self.ui.rotated_preview_label, self.ui.processed_preview_label]:
                label.setText("Preview Error"); label.setFilePath("")

    def update_batch_preview(self):
        jpg_files = get_image_files(self.app_state.input_directory, 'compressed')
        if not jpg_files:
            self.ui.combined_preview_label.clear(); self.ui.combined_preview_label.setText("No Images"); return
        s = self.app_state.settings
        batch_size, num_batches = s['batch_size'], (len(jpg_files) + s['batch_size'] - 1) // s['batch_size']
        
        current_idx = self.ui.batch_preview_dropdown.currentIndex()
        self.ui.batch_preview_dropdown.blockSignals(True); self.ui.batch_preview_dropdown.clear()
        if num_batches > 0:
            self.ui.batch_preview_dropdown.addItems([f"Batch {i+1}" for i in range(num_batches)])
            if 0 <= current_idx < num_batches: self.ui.batch_preview_dropdown.setCurrentIndex(current_idx)
        self.ui.batch_preview_dropdown.blockSignals(False)
        
        if self.ui.batch_preview_dropdown.count() == 0: return
        
        start_idx = self.ui.batch_preview_dropdown.currentIndex() * batch_size
        batch_to_show = jpg_files[start_idx : start_idx + batch_size]
        images_to_merge = [preprocess_image(fix_orientation(Image.open(p)), str(start_idx + i + 1), s['crop_settings']) for i, p in enumerate(batch_to_show)]
        
        if merged := merge_images(images_to_merge, s['merged_img_height']):
            temp_path = Path(self.app_state.rename_files_dir) / "temp_merged_preview.jpg"
            merged.save(temp_path, quality=90)
            self.ui.combined_preview_label.setPixmap(self.pil_to_qpixmap(merged))
            self.ui.combined_preview_label.setFilePath(str(temp_path))
        else:
            self.ui.combined_preview_label.clear(); self.ui.combined_preview_label.setText("Batch Preview"); self.ui.combined_preview_label.setFilePath("")

    def update_models_dropdown(self):
        self.ui.model_dropdown.clear()
        if not self.app_state.api_keys:
            self.ui.model_dropdown.addItem("No API Key Set"); return
        try:
            genai.configure(api_key=self.app_state.api_keys[0])
            all_models = genai.list_models()
            vision_models = sorted([m.name.split('/')[-1] for m in all_models if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name])
            if vision_models:
                self.app_state.available_models = vision_models
                self.ui.model_dropdown.addItems(vision_models)
                if (saved_model := self.app_state.settings.get('model_name')) in vision_models:
                    self.ui.model_dropdown.setCurrentText(saved_model)
                else:
                    self.app_state.settings['model_name'] = vision_models[0]
            else: self.ui.model_dropdown.addItem("Could not fetch models")
        except Exception as e:
            self.logger.error("Failed to fetch Gemini models", exception=e)
            self.ui.model_dropdown.addItem("Error fetching models")