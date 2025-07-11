# ai-photo-processor/controllers/review_tab_handler.py

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow
from PyQt5.QtCore import QThread, QObject
from PyQt5.QtGui import QPixmap, QImage

from app_state import AppState
from ui.review_tab import ReviewResultsTab
from ui.review_tab_item import ReviewItemWidget
from workers import ImageLoadWorker
from utils.name_calculator import calculate_final_names
from utils.file_management import SUPPORTED_RAW_EXTENSIONS
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

    def connect_signals(self):
        self.ui.csv_dropdown.currentIndexChanged.connect(self.load_selected_csv)
        self.ui.recalc_names_button.clicked.connect(self.recalculate_names)
        self.ui.rename_files_button.clicked.connect(self.rename_files)
        self.ui.restore_names_button.clicked.connect(self.restore_file_names)
    
    def stop_worker(self):
        if self.image_load_worker: self.image_load_worker.stop()

    def refresh_csv_dropdown(self):
        self.ui.csv_dropdown.blockSignals(True)
        self.ui.csv_dropdown.clear()
        if self.app_state.rename_files_dir and os.path.exists(self.app_state.rename_files_dir):
            csv_files = sorted(
                [f for f in os.listdir(self.app_state.rename_files_dir) if f.endswith('.csv')],
                key=lambda f: os.path.getmtime(os.path.join(self.app_state.rename_files_dir, f)),
                reverse=True
            )
            self.ui.csv_dropdown.addItems(csv_files)
        self.ui.csv_dropdown.blockSignals(False)
        if self.ui.csv_dropdown.count() > 0: self.load_selected_csv(0)
        else: self.ui.clear_grid()

    def load_selected_csv(self, index: int):
        if index < 0 or not (csv_name := self.ui.csv_dropdown.currentText()):
            self.ui.clear_grid(); return
        path = os.path.join(self.app_state.rename_files_dir, csv_name)
        try:
            self.app_state.current_df = pd.read_csv(path).fillna('')
            self._populate_review_grid()
        except Exception as e:
            self.logger.error(f"Could not load or parse CSV file: {csv_name}", exception=e)
            QMessageBox.critical(self.main_window, "Error", f"Could not load or parse CSV file: {e}")

    def _populate_review_grid(self):
        self.ui.clear_grid()
        df = self.app_state.current_df
        main_col = self.app_state.settings['main_column']
        if df.empty: return
        if main_col not in df.columns:
            QMessageBox.warning(self.main_window, "Column Not Found", f"The main column '{main_col}' was not found in the CSV."); return

        image_paths_to_load = []
        for i, (idx, row) in enumerate(df.iterrows()):
            item_widget = ReviewItemWidget(idx, row.to_dict(), main_col)
            item_widget.data_changed.connect(lambda w=item_widget: self._sync_df_from_review_item(w))
            self.ui.add_item_to_grid(i // 2, i % 2, item_widget)
            self.ui.ui_widgets.append(item_widget)
            img_path = Path(row['from'])
            item_widget.image_label.setFilePath(str(img_path))
            item_widget.image_label.clicked.connect(self.main_window.process_handler.on_preview_label_clicked)
            if img_path.exists(): image_paths_to_load.append(img_path)
        if image_paths_to_load: self.start_image_load_worker(image_paths_to_load)

    def _sync_df_from_review_item(self, sender_widget: ReviewItemWidget):
        data = sender_widget.get_data()
        idx = data.pop('df_index')
        for col, value in data.items():
            if col in self.app_state.current_df.columns:
                self.app_state.current_df.at[idx, col] = value
               
    def start_image_load_worker(self, paths):
        if self.image_load_thread and self.image_load_thread.isRunning(): self.image_load_worker.stop()
        self.image_load_thread = QThread()
        self.image_load_worker = ImageLoadWorker(paths, (320, 180), self.logger)
        self.image_load_worker.moveToThread(self.image_load_thread)
        self.image_load_worker.image_loaded.connect(self.on_review_image_loaded)
        self.image_load_worker.finished.connect(self.image_load_thread.quit)
        self.image_load_thread.started.connect(self.image_load_worker.run)
        self.image_load_thread.start()
       
    def on_review_image_loaded(self, path_str: str, q_image: QImage):
        pixmap = QPixmap.fromImage(q_image)
        for widget in self.ui.ui_widgets:
            if self.app_state.current_df.loc[widget.df_index, 'from'] == path_str:
                widget.set_image(pixmap); break

    def recalculate_names(self):
        if self.app_state.current_df.empty: return
        self.app_state.current_df = calculate_final_names(self.app_state.current_df, self.app_state.settings['main_column'])
        self._populate_review_grid()
        QMessageBox.information(self.main_window, "Success", "'To' column has been recalculated.")

    def rename_files(self):
        if self.app_state.current_df.empty or 'to' not in self.app_state.current_df.columns:
            return
        if QMessageBox.question(self.main_window, "Confirm Rename", "This will rename files on your disk. Are you sure?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No:
            return
       
        self.logger.info("--- Starting File Rename Process ---")
        log_entries, renamed_count, error_count = [], 0, 0
        op_time = datetime.now().isoformat()
        df = self.app_state.current_df

        for idx, row in df.iterrows():
            if row.get('skip') == 'x' or not row.get('to'):
                continue
            
            src_path = Path(row['from'])
            dst_path = src_path.parent / row['to']

            if not src_path.exists():
                self.logger.warn(f"Skipping rename. Source not found: {src_path.name}")
                continue
            
            try:
                # Rename the primary file (e.g., the JPG)
                src_path.rename(dst_path)
                self.logger.info(f"Renamed: '{src_path.name}' -> '{dst_path.name}'")
                renamed_count += 1
                log_entries.append({'timestamp': op_time, 'original_path': str(src_path), 'new_path': str(dst_path)})

                # Find and rename the associated RAW file, preserving its extension case.
                original_base_name = src_path.stem
                for raw_src_path in src_path.parent.glob(f"{original_base_name}.*"):
                    if raw_src_path.suffix.lower() in SUPPORTED_RAW_EXTENSIONS:
                        raw_dst_path = dst_path.with_suffix(raw_src_path.suffix)
                        raw_src_path.rename(raw_dst_path)
                        self.logger.info(f"Renamed RAW: '{raw_src_path.name}' -> '{raw_dst_path.name}'")
                        renamed_count += 1
                        log_entries.append({'timestamp': op_time, 'original_path': str(raw_src_path), 'new_path': str(raw_dst_path)})
                        break # Assume one RAW file per primary file

            except Exception as e:
                self.logger.error(f"Could not rename {src_path.name} or its counterpart", exception=e)
                error_count += 1
       
        if log_entries:
            log_path = Path(self.app_state.rename_files_dir) / "rename_log.csv"
            pd.DataFrame(log_entries).to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
            self.logger.info(f"Appended {len(log_entries)} entries to rename log.")
       
        msg = f"Successfully renamed {renamed_count} files."
        if error_count > 0:
            msg += f"\n\nEncountered {error_count} errors. Check console log."
        QMessageBox.information(self.main_window, "Rename Complete", msg)
    
    def restore_file_names(self):
        log_path = Path(self.app_state.rename_files_dir) / "rename_log.csv"
        if not log_path.exists():
            QMessageBox.warning(self.main_window, "Not Found", "rename_log.csv not found."); return
        if QMessageBox.question(self.main_window, "Confirm Restore", "This will restore filenames based on the log. Are you sure?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No: return
        
        self.logger.info("--- Starting File Restore Process ---")
        log_df = pd.read_csv(log_path)
        restored_count, error_count = 0, 0
        
        for idx, row in log_df.iloc[::-1].iterrows():
            src_path, dst_path = Path(row['new_path']), Path(row['original_path'])
            if not src_path.exists(): self.logger.warn(f"Skipping restore. File not found: {src_path.name}"); continue
            try:
                src_path.rename(dst_path)
                self.logger.info(f"Restored: '{src_path.name}' -> '{dst_path.name}'")
                restored_count += 1
            except Exception as e: self.logger.error(f"Could not restore {src_path.name}", exception=e); error_count += 1
        
        if restored_count > 0:
            restored_log_path = log_path.with_name(f"rename_log_{datetime.now():%Y%m%d_%H%M%S}.csv.restored")
            log_path.rename(restored_log_path)
            self.logger.info(f"Restore complete. Log file archived to: {restored_log_path.name}")
        
        msg = f"Successfully restored {restored_count} files."
        if error_count > 0: msg += f"\n\nEncountered {error_count} errors. Check console log."
        QMessageBox.information(self.main_window, "Restore Complete", msg)