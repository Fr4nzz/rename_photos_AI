# ai-photo-processor/controllers/review_tab_handler.py

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow, QWidget
from PyQt5.QtCore import QThread, QObject
from PyQt5.QtGui import QPixmap, QImage

from app_state import AppState
from ui.review_tab import ReviewResultsTab
from ui.review_tab_item import ReviewItemWidget
from workers import ImageLoadWorker
from utils.name_calculator import calculate_final_names
from utils.file_management import get_image_files, SUPPORTED_RAW_EXTENSIONS
from utils.logger import SimpleLogger

class ReviewTabHandler(QObject):
    """Controller for all logic related to the Review Results tab."""

    def __init__(self, ui: ReviewResultsTab, app_state: AppState, logger: SimpleLogger, main_window: QMainWindow):
        super().__init__()
        self.ui = ui
        self.app_state = app_state
        self.logger = logger
        self.main_window = main_window
        self.image_load_worker = None
        self.image_load_thread = None
        self.path_to_widget_map = {}

    def connect_signals(self):
        self.ui.csv_dropdown.currentIndexChanged.connect(self.refresh_view)
        self.ui.refresh_from_disk_button.clicked.connect(self.refresh_view)
        self.ui.crop_review_checkbox.stateChanged.connect(self._handle_ui_change)
        self.ui.show_duplicates_checkbox.stateChanged.connect(self._handle_ui_change)
        self.ui.save_changes_button.clicked.connect(self.save_manual_changes)
        self.ui.recalc_names_button.clicked.connect(self.recalculate_names)
        self.ui.rename_files_button.clicked.connect(self.rename_files)
        self.ui.restore_names_button.clicked.connect(self.restore_file_names)
    
    def populate_initial_ui(self):
        self.ui.crop_review_checkbox.setChecked(self.app_state.settings.get('review_crop_enabled', True))
    
    def stop_worker(self):
        if self.image_load_thread and self.image_load_thread.isRunning():
            self.image_load_worker.stop()
            self.image_load_thread.quit()
            self.image_load_thread.wait()

    def _handle_ui_change(self):
        self.app_state.settings['review_crop_enabled'] = self.ui.crop_review_checkbox.isChecked()
        self._populate_review_grid()

    def refresh_csv_dropdown(self):
        """Populates the CSV dropdown and triggers a full view refresh."""
        self.ui.csv_dropdown.blockSignals(True)
        current_selection = self.ui.csv_dropdown.currentText()
        self.ui.csv_dropdown.clear()
        
        if self.app_state.rename_files_dir and os.path.exists(self.app_state.rename_files_dir):
            csv_files = sorted(
                [f for f in os.listdir(self.app_state.rename_files_dir) if f.endswith('.csv') and not f.startswith('rename_log')],
                key=lambda f: os.path.getmtime(os.path.join(self.app_state.rename_files_dir, f)),
                reverse=True
            )
            self.ui.csv_dropdown.addItems(csv_files)
            if current_selection in csv_files:
                self.ui.csv_dropdown.setCurrentText(current_selection)

        self.ui.csv_dropdown.blockSignals(False)
        self.refresh_view()

    def refresh_view(self):
        """The main method to refresh the entire review tab, using the filesystem as the source of truth."""
        self.logger.info("Refreshing Review Results view...")
        self.stop_worker()

        if not self.app_state.input_directory or not os.path.exists(self.app_state.input_directory):
            self.ui.set_grid_message("Select an input directory in the 'Process Images' tab first.")
            return

        disk_files = get_image_files(self.app_state.input_directory, 'compressed')
        if not disk_files:
            self.ui.set_grid_message("No compressed image files (JPG, PNG, etc.) found in the selected directory.")
            return

        main_col = self.app_state.settings['main_column']
        df_cols = ['from', 'to', 'skip', 'co', 'status', main_col]
        
        csv_df = pd.DataFrame()
        csv_name = self.ui.csv_dropdown.currentText()
        if csv_name:
            path = os.path.join(self.app_state.rename_files_dir, csv_name)
            if os.path.exists(path):
                try:
                    csv_df = pd.read_csv(path, dtype=str).fillna('')
                    for col in df_cols:
                        if col not in csv_df.columns: csv_df[col] = ''
                except Exception as e:
                    self.logger.error(f"Could not load or parse CSV: {csv_name}", exception=e)

        reconciled_data = []
        for file_path in disk_files:
            file_path_str = str(file_path)
            new_row = {col: '' for col in df_cols}
            new_row.update({'from': file_path_str, 'status': 'New'})

            if not csv_df.empty:
                match = csv_df[csv_df['from'] == file_path_str]
                if not match.empty:
                    new_row.update(match.iloc[0].to_dict())
                    new_row['status'] = 'Original'
                else:
                    match = csv_df[csv_df['to'] == file_path.name]
                    if not match.empty:
                        new_row.update(match.iloc[0].to_dict())
                        new_row['from'] = file_path_str
                        new_row['status'] = 'Renamed'
            reconciled_data.append(new_row)

        final_df = pd.DataFrame(reconciled_data)
        if 'photo_ID' not in final_df.columns or final_df['photo_ID'].isnull().any():
            final_df['photo_ID'] = range(1, len(final_df) + 1)
        
        self.app_state.current_df = final_df
        self._populate_review_grid()

    def _populate_review_grid(self):
        try:
            self.stop_worker()
            self.ui.clear_grid()
            self.path_to_widget_map.clear()

            df = self.app_state.current_df
            main_col = self.app_state.settings['main_column']

            if df.empty or main_col not in df.columns:
                self.ui.set_grid_message("No data or main column not found.")
                return

            display_df = df
            if self.ui.show_duplicates_checkbox.isChecked():
                id_counts = df.loc[df[main_col] != '', main_col].value_counts()
                mismatched_ids = id_counts[id_counts != 2].index
                display_df = df[df[main_col].isin(mismatched_ids)]
                if display_df.empty:
                    self.ui.set_grid_message("No items with mismatched pair counts found.")
                    return

            grid_row, grid_col = 0, 0
            image_paths_to_load = []
            id_counts_total = df.loc[df[main_col] != '', main_col].value_counts().to_dict()

            grouped = display_df.groupby(main_col, sort=False, dropna=False)
            for identifier, group in grouped:
                for idx, row in group.iterrows():
                    total_count = id_counts_total.get(identifier, 0)
                    
                    item_widget = ReviewItemWidget(idx, row.to_dict(), main_col, total_count)
                    item_widget.data_changed.connect(lambda w=item_widget: self._sync_df_from_review_item(w))
                    self.ui.add_item_to_grid(grid_row, grid_col, item_widget)
                    
                    img_path = Path(row['from'])
                    if img_path.exists():
                        self.path_to_widget_map[str(img_path)] = item_widget
                        item_widget.image_label.setFilePath(str(img_path))
                        item_widget.image_label.clicked.connect(self.main_window.process_handler.on_preview_label_clicked)
                        image_paths_to_load.append(img_path)
                    
                    grid_col += 1
                    if grid_col >= 2: grid_col, grid_row = 0, grid_row + 1
                
                if grid_col == 1: self.ui.add_item_to_grid(grid_row, 1, QWidget())
                if grid_col != 0: grid_col, grid_row = 0, grid_row + 1
            
            if image_paths_to_load:
                self.start_image_load_worker(image_paths_to_load)

        except Exception as e:
            error_message = f"Error populating review grid: {e}\n{traceback.format_exc()}"
            self.logger.error(error_message)
            QMessageBox.critical(self.main_window, "Application Error", f"{error_message}")

    def _sync_df_from_review_item(self, sender_widget: ReviewItemWidget):
        data = sender_widget.get_data()
        idx = data.pop('df_index')
        for col, value in data.items():
            if col in self.app_state.current_df.columns:
                self.app_state.current_df.loc[idx, col] = value
               
    def start_image_load_worker(self, paths):
        active_crop_settings = self.app_state.settings['crop_settings'].copy()
        active_crop_settings['zoom'] = self.ui.crop_review_checkbox.isChecked()
        
        self.image_load_thread = QThread()
        self.image_load_worker = ImageLoadWorker(paths, self.logger, active_crop_settings)
        self.image_load_worker.moveToThread(self.image_load_thread)
        self.image_load_worker.image_loaded.connect(self.on_review_image_loaded)
        self.image_load_worker.finished.connect(self.image_load_thread.quit)
        self.image_load_thread.started.connect(self.image_load_worker.run)
        self.image_load_thread.start()
       
    def on_review_image_loaded(self, path_str: str, img_bytes: bytes, width: int, height: int):
        q_image = QImage(img_bytes, width, height, width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        if widget := self.path_to_widget_map.get(path_str):
            widget.set_image(pixmap)

    def save_manual_changes(self):
        """
        Saves any edits in the review grid. If no CSV is selected, it creates a new one.
        """
        if self.app_state.current_df.empty:
            QMessageBox.warning(self.main_window, "No Data", "There is no data to save.")
            return

        try:
            self._save_current_df()
            # After saving, the dropdown is refreshed and the correct file is selected.
            QMessageBox.information(self.main_window, "Success", f"Changes saved to {self.ui.csv_dropdown.currentText()}.")
        except Exception as e:
            self.logger.error("Failed during manual save.", exception=e)
            QMessageBox.critical(self.main_window, "Save Error", f"Could not save changes to the CSV file.\n\nError: {e}")

    def recalculate_names(self):
        if self.app_state.current_df.empty: return
        self.app_state.current_df = calculate_final_names(self.app_state.current_df, self.app_state.settings['main_column'])
        self._save_current_df()
        self._populate_review_grid()
        QMessageBox.information(self.main_window, "Success", "'To' column recalculated and saved.")

    def rename_files(self):
        if self.app_state.current_df.empty or 'to' not in self.app_state.current_df.columns:
            return QMessageBox.warning(self.main_window, "No Data", "No data loaded or 'to' column is missing.")
        if QMessageBox.question(self.main_window, "Confirm Rename", "This will rename files on your disk. Are you sure?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No:
            return
       
        self.logger.info("--- Starting File Rename Process ---")
        log_entries, renamed_count, error_count = [], 0, 0
        df = self.app_state.current_df

        for idx, row in df.iterrows():
            if row.get('skip') == 'x' or not row.get('to'): continue
            
            src_path, dst_name = Path(row['from']), row['to']
            dst_path = src_path.parent / dst_name
            if src_path == dst_path: continue

            if not src_path.exists():
                self.logger.warn(f"Skipping. Source not found: {src_path.name}"); continue
            
            try:
                src_path.rename(dst_path)
                self.logger.info(f"Renamed: '{src_path.name}' -> '{dst_path.name}'")
                df.at[idx, 'from'] = str(dst_path) 
                df.at[idx, 'to'] = '' 
                log_entries.append({'original_path': str(src_path), 'new_path': str(dst_path)})
                renamed_count += 1

                original_base_name = os.path.splitext(os.path.basename(str(src_path)))[0]
                new_base_name = os.path.splitext(dst_name)[0]
                for raw_file in Path(src_path.parent).glob(f"{original_base_name}.*"):
                    if raw_file.suffix.lower() in SUPPORTED_RAW_EXTENSIONS:
                        raw_dst_path = raw_file.with_name(f"{new_base_name}{raw_file.suffix}")
                        raw_file.rename(raw_dst_path)
                        self.logger.info(f"Renamed RAW: '{raw_file.name}' -> '{raw_dst_path.name}'")
                        log_entries.append({'original_path': str(raw_file), 'new_path': str(raw_dst_path)})
                        renamed_count += 1
                        break

            except Exception as e:
                self.logger.error(f"Could not rename {src_path.name} or its counterpart", exception=e); error_count += 1
       
        if log_entries:
            log_path = Path(self.app_state.rename_files_dir) / "rename_log.csv"
            pd.DataFrame(log_entries).to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
        
        self.app_state.current_df = df
        self._save_current_df()
        self.refresh_view()
        QMessageBox.information(self.main_window, "Rename Complete", f"Renamed {renamed_count} files.\n{error_count} errors occurred.")
    
    def restore_file_names(self):
        log_path = Path(self.app_state.rename_files_dir) / "rename_log.csv"
        if not log_path.exists(): return QMessageBox.warning(self.main_window, "Not Found", "rename_log.csv not found.")
        if QMessageBox.question(self.main_window, "Confirm Restore", "Restore filenames based on the log?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
        
        log_df = pd.read_csv(log_path)
        restored_count, error_count = 0, 0
        
        for idx, row in log_df.iloc[::-1].iterrows():
            src_path, dst_path = Path(row['new_path']), Path(row['original_path'])
            if not src_path.exists(): continue
            try:
                if dst_path.exists():
                     self.logger.error(f"Cannot restore {src_path.name}, destination {dst_path.name} already exists.")
                     error_count += 1; continue
                src_path.rename(dst_path)
                restored_count += 1
            except Exception as e: 
                self.logger.error(f"Could not restore {src_path.name}", exception=e); error_count += 1
        
        if restored_count > 0:
            archive_name = f"rename_log_{datetime.now():%Y%m%d_%H%M%S}.csv.restored"
            log_path.rename(log_path.with_name(archive_name))
        
        self.refresh_view()
        QMessageBox.information(self.main_window, "Restore Complete", f"Restored {restored_count} files.\n{error_count} errors occurred.")

    def _save_current_df(self):
        """Saves the current DataFrame to the selected CSV or a new one."""
        if self.app_state.current_df.empty: return

        csv_name = self.ui.csv_dropdown.currentText()
        if not csv_name:
            csv_name = f"review_state_{datetime.now():%Y%m%d_%H%M%S}.csv"
        
        save_path = Path(self.app_state.rename_files_dir) / csv_name
        try:
            df_to_save = self.app_state.current_df.drop(columns=['status'], errors='ignore')
            df_to_save.to_csv(save_path, index=False)
            self.logger.info(f"DataFrame state saved to {save_path.name}")
            
            # This is a critical step to ensure the UI is in sync after saving
            current_text = self.ui.csv_dropdown.currentText()
            self.refresh_csv_dropdown()
            # After refresh, try to set the dropdown to the file we just saved to.
            # This handles both updating an existing file and creating a new one.
            new_index = self.ui.csv_dropdown.findText(csv_name)
            if new_index > -1:
                 self.ui.csv_dropdown.setCurrentIndex(new_index)

        except Exception as e:
            self.logger.error(f"Failed to save DataFrame to {save_path.name}", exception=e)