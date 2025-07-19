# ai-photo-processor/controllers/review_tab_handler.py

import os
import re
import math
import pandas as pd
from pathlib import Path
from datetime import datetime

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

QUALITY_TO_HEIGHT = {
    "480p": 480, "540p": 540, "720p": 720, "900p": 900, "1080p": 1080,
    "Original": 0
}

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
        self.current_page = 0
        self.total_pages = 1
        self.filtered_df = pd.DataFrame()

    def connect_signals(self):
        self.ui.csv_dropdown.currentIndexChanged.connect(self.refresh_view)
        self.ui.refresh_from_disk_button.clicked.connect(lambda: self.refresh_csv_dropdown(select_newest=True))
        self.ui.crop_review_checkbox.stateChanged.connect(self._handle_ui_change)
        self.ui.show_duplicates_checkbox.stateChanged.connect(self._handle_ui_change)
        
        self.ui.items_per_page_input.editingFinished.connect(self._handle_ui_change)
        self.ui.image_quality_dropdown.currentIndexChanged.connect(self._handle_ui_change)
        
        self.ui.suffix_mode_dropdown.currentIndexChanged.connect(self._handle_suffix_mode_change)
        self.ui.custom_suffix_input.editingFinished.connect(self._sync_settings_from_ui)

        self.ui.save_changes_button.clicked.connect(self.create_checked_csv) 
        self.ui.recalc_names_button.clicked.connect(self.recalculate_names)
        self.ui.rename_files_button.clicked.connect(self.rename_files)
        self.ui.restore_names_button.clicked.connect(self.restore_file_names)
        self.ui.next_page_button.clicked.connect(self.go_to_next_page)
        self.ui.prev_page_button.clicked.connect(self.go_to_prev_page)
    
    def populate_initial_ui(self):
        self.ui.crop_review_checkbox.setChecked(self.app_state.settings.get('review_crop_enabled', True))
        
        items_per_page = self.app_state.settings.get('review_items_per_page', 50)
        self.ui.items_per_page_input.setText(str(items_per_page))
        
        quality_setting = self.app_state.settings.get('review_thumb_height', '720p')
        self.ui.image_quality_dropdown.setCurrentText(quality_setting)

        suffix_mode = self.app_state.settings.get('suffix_mode', 'Standard')
        self.ui.suffix_mode_dropdown.setCurrentText(suffix_mode)

        custom_suffixes = self.app_state.settings.get('custom_suffixes', 'd,v')
        self.ui.custom_suffix_input.setText(custom_suffixes)
        
        self._update_suffix_widgets_visibility()

    def _sync_settings_from_ui(self):
        """Read values from the UI and update the app_state."""
        self.app_state.settings['review_crop_enabled'] = self.ui.crop_review_checkbox.isChecked()
        
        try:
            items_per_page = int(self.ui.items_per_page_input.text())
            self.app_state.settings['review_items_per_page'] = items_per_page if items_per_page > 0 else 1
        except (ValueError, TypeError):
            self.app_state.settings['review_items_per_page'] = 50
            self.ui.items_per_page_input.setText("50")

        self.app_state.settings['review_thumb_height'] = self.ui.image_quality_dropdown.currentText()
        self.app_state.settings['suffix_mode'] = self.ui.suffix_mode_dropdown.currentText()
        self.app_state.settings['custom_suffixes'] = self.ui.custom_suffix_input.text()

    def stop_worker(self):
        if self.image_load_thread and self.image_load_thread.isRunning():
            self.image_load_worker.stop()
            self.image_load_thread.quit()
            self.image_load_thread.wait()
    
    def _handle_suffix_mode_change(self):
        """Syncs settings and updates widget visibility when the suffix mode changes."""
        self._sync_settings_from_ui()
        self._update_suffix_widgets_visibility()

    def _update_suffix_widgets_visibility(self):
        """Shows or hides the custom suffix input based on the dropdown selection."""
        is_custom_mode = (self.ui.suffix_mode_dropdown.currentText() == "Custom")
        self.ui.custom_suffix_label.setVisible(is_custom_mode)
        self.ui.custom_suffix_input.setVisible(is_custom_mode)

    def _handle_ui_change(self):
        self._sync_settings_from_ui()
        self.refresh_view()

    def refresh_csv_dropdown(self, select_newest: bool = False):
        self.ui.csv_dropdown.blockSignals(True)
        current_selection = self.ui.csv_dropdown.currentText()
        self.ui.csv_dropdown.clear()
        
        rename_dir = self.app_state.rename_files_dir
        if rename_dir and os.path.exists(rename_dir):
            try:
                # Filter out the rename log file from the dropdown
                csv_files = sorted(
                    [f for f in os.listdir(rename_dir) if f.endswith('.csv') and not f.startswith('rename_log')],
                    key=lambda f: os.path.getmtime(os.path.join(rename_dir, f)),
                    reverse=True
                )
                if csv_files:
                    self.ui.csv_dropdown.addItems(csv_files)
                    if select_newest and self.ui.csv_dropdown.count() > 0:
                        self.ui.csv_dropdown.setCurrentIndex(0)
                    elif current_selection in csv_files:
                        self.ui.csv_dropdown.setCurrentText(current_selection)
            except Exception as e:
                self.logger.error(f"Failed to list CSV files in {rename_dir}", exception=e)

        self.ui.csv_dropdown.blockSignals(False)
        self.refresh_view()

    def refresh_view(self):
        """
        The core data reconciliation logic. It links files on disk to their data
        in the selected CSV using the central rename_log.csv as a ledger.
        """
        self.logger.info("Refreshing Review Results view...")
        self.stop_worker()

        if not self.app_state.input_directory or not os.path.exists(self.app_state.input_directory):
            self.ui.set_grid_message("Select an input directory in the 'Process Images' tab first.")
            return

        disk_files = get_image_files(self.app_state.input_directory, 'compressed')
        if not disk_files:
            self.ui.set_grid_message("No compressed image files found.")
            return

        # Load the rename log to map current names back to original names
        log_path = Path(self.app_state.rename_files_dir) / "rename_log.csv"
        rename_log_map = {}
        if log_path.exists():
            try:
                log_df = pd.read_csv(log_path)
                # Create a map from new_path -> original_path for quick lookups
                rename_log_map = pd.Series(log_df.original_path.values, index=log_df.new_path).to_dict()
            except Exception as e:
                self.logger.error(f"Could not read or parse {log_path.name}", exception=e)

        # Load the selected data CSV
        csv_name = self.ui.csv_dropdown.currentText()
        csv_df = pd.DataFrame()
        if csv_name:
            csv_path = os.path.join(self.app_state.rename_files_dir, csv_name)
            if os.path.exists(csv_path):
                try:
                    csv_df = pd.read_csv(csv_path, dtype=str).fillna('')
                except Exception as e:
                    self.logger.error(f"Could not load selected CSV: {csv_name}", exception=e)
        
        # Define the expected columns.
        main_col = self.app_state.settings.get('main_column', 'CAM')
        expected_cols = ['from', 'to', 'skip', 'co', main_col, 'n', 'suffix']

        # Reconcile data
        reconciled_data = []
        seen_original_paths = set()

        for current_path in disk_files:
            current_path_str = str(current_path)
            # Determine the original path using the log file
            original_path = rename_log_map.get(current_path_str, current_path_str)
            seen_original_paths.add(original_path)
            
            # Find the corresponding data row in the CSV using the original 'from' path
            data_row = {}
            status = "New" # Default status if not found in CSV
            if not csv_df.empty:
                match = csv_df[csv_df['from'] == original_path]
                if not match.empty:
                    data_row = match.iloc[0].to_dict()
                    status = "Renamed" if current_path_str != original_path else "Original"

            # Build the final row for the main DataFrame
            final_row = {'current_path': current_path_str, 'status': status}
            for col in expected_cols:
                final_row[col] = data_row.get(col, '')
            final_row['from'] = original_path # Ensure 'from' is always the original path
            
            reconciled_data.append(final_row)
        
        # Add rows for files that are in the CSV but not on disk (e.g., deleted)
        if not csv_df.empty:
            missing_files_df = csv_df[~csv_df['from'].isin(seen_original_paths)]
            for _, row in missing_files_df.iterrows():
                final_row = row.to_dict()
                final_row['current_path'] = "File not found"
                final_row['status'] = "Missing"
                reconciled_data.append(final_row)

        self.app_state.current_df = pd.DataFrame(reconciled_data)
        if 'photo_ID' not in self.app_state.current_df.columns or self.app_state.current_df['photo_ID'].isnull().any():
            self.app_state.current_df['photo_ID'] = range(1, len(self.app_state.current_df) + 1)
        
        self._apply_filters_and_update_pages()
        self._populate_review_grid()

    def _apply_filters_and_update_pages(self):
        """Applies UI filters to the main DataFrame and resets pagination."""
        df = self.app_state.current_df
        if df.empty:
            self.filtered_df, self.total_pages, self.current_page = pd.DataFrame(), 1, 0
            return

        display_df = df[df['status'] != 'Missing'] # Don't show missing files unless specifically asked
        if self.ui.show_duplicates_checkbox.isChecked():
            main_col = self.app_state.settings['main_column']
            if main_col in df.columns:
                valid_ids = df.loc[df[main_col].notna() & (df[main_col] != ''), main_col]
                id_counts = valid_ids.value_counts()
                mismatched_ids = id_counts[id_counts != 2].index
                display_df = display_df[display_df[main_col].isin(mismatched_ids)]

        self.filtered_df = display_df.reset_index(drop=True)
        items_per_page = self.app_state.settings.get('review_items_per_page', 50)
        self.total_pages = math.ceil(len(self.filtered_df) / items_per_page) if items_per_page > 0 else 1
        self.total_pages = max(1, self.total_pages)
        self.current_page = 0

    def _populate_review_grid(self):
        """Populates the grid with items for the CURRENT page."""
        try:
            self.stop_worker()
            self.ui.clear_grid()
            self.path_to_widget_map.clear()
            self._update_navigation_controls()

            main_col = self.app_state.settings['main_column']
            if self.filtered_df.empty or main_col not in self.filtered_df.columns:
                self.ui.set_grid_message("No items to display with current filters.")
                return

            items_per_page = self.app_state.settings.get('review_items_per_page', 50)
            start_idx = self.current_page * items_per_page
            end_idx = start_idx + items_per_page
            page_df = self.filtered_df.iloc[start_idx:end_idx]

            if page_df.empty:
                self.ui.set_grid_message("No items on this page.")
                return

            grid_row, grid_col, image_paths_to_load = 0, 0, []
            id_counts_total = self.app_state.current_df.loc[self.app_state.current_df[main_col].notna() & (self.app_state.current_df[main_col] != ''), main_col].value_counts().to_dict()

            for identifier, group in page_df.groupby(main_col, sort=False, dropna=False):
                for _, row in group.iterrows():
                    total_count = id_counts_total.get(identifier, 0) if pd.notna(identifier) else 0
                    # Note: We now pass the 'current_path' for display and loading
                    item_widget = ReviewItemWidget(row.name, row.to_dict(), main_col, total_count)
                    item_widget.data_changed.connect(lambda w=item_widget: self._sync_df_from_review_item(w))
                    item_widget.data_changed.connect(self._handle_autosave)
                    self.ui.add_item_to_grid(grid_row, grid_col, item_widget)
                    
                    if img_path_str := row.get('current_path'):
                        img_path = Path(img_path_str)
                        if img_path.exists():
                            self.path_to_widget_map[str(img_path)] = item_widget
                            item_widget.image_label.setFilePath(str(img_path))
                            item_widget.image_label.clicked.connect(self.main_window.process_handler.on_preview_label_clicked)
                            image_paths_to_load.append(img_path)
                    
                    grid_col += 1
                    if grid_col >= 2: grid_col, grid_row = 0, grid_row + 1
                
                if grid_col == 1: self.ui.add_item_to_grid(grid_row, 1, QWidget()) # Fill empty cell
                if grid_col != 0: grid_col, grid_row = 0, grid_row + 1
            
            if image_paths_to_load:
                self.start_image_load_worker(image_paths_to_load)

        except Exception as e:
            self.logger.error(f"Error populating review grid: {e}", exception=e)

    def _sync_df_from_review_item(self, sender_widget: ReviewItemWidget):
        data = sender_widget.get_data()
        idx = data.pop('df_index')
        # Sync with both DataFrames to keep UI filters consistent until next refresh
        for df in [self.app_state.current_df, self.filtered_df]:
             for col, value in data.items():
                if col in df.columns and idx < len(df):
                    df.loc[idx, col] = value
               
    def start_image_load_worker(self, paths):
        crop_settings = {**self.app_state.settings['crop_settings'], 'zoom': self.ui.crop_review_checkbox.isChecked()}
        
        quality_str = self.app_state.settings.get('review_thumb_height', '720p')
        thumb_height = QUALITY_TO_HEIGHT.get(quality_str, 720)
        
        self.image_load_thread = QThread()
        self.image_load_worker = ImageLoadWorker(paths, self.logger, crop_settings, thumb_height)
        self.image_load_worker.moveToThread(self.image_load_thread)
        self.image_load_worker.image_loaded.connect(self.on_review_image_loaded)
        self.image_load_worker.finished.connect(self.image_load_thread.quit)
        self.image_load_thread.started.connect(self.image_load_worker.run)
        self.image_load_thread.start()
       
    def on_review_image_loaded(self, path_str: str, img_bytes: bytes, width: int, height: int):
        q_img = QImage(img_bytes, width, height, width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        if widget := self.path_to_widget_map.get(path_str):
            widget.set_image(pixmap)

    def recalculate_names(self):
        """
        Recalculates the 'to' column and saves the result to a new, permanent
        'checked' CSV file, then loads it.
        """
        if self.app_state.current_df.empty:
            return

        self._sync_settings_from_ui()
        settings = self.app_state.settings
        
        self.app_state.current_df = calculate_final_names(
            self.app_state.current_df,
            settings['main_column'],
            settings['suffix_mode'],
            settings['custom_suffixes']
        )
        
        # After calculating names, save them to a new 'checked' file and refresh
        self.create_checked_csv(
            success_message="'To' column recalculated and saved to a new 'checked' file."
        )

    def rename_files(self):
        """
        Renames files on disk based on the 'to' column and logs the changes
        in 'rename_log.csv'. This version correctly handles file name conflicts.
        """
        if self.app_state.current_df.empty or 'to' not in self.app_state.current_df.columns:
            QMessageBox.warning(self.main_window, "No Data", "No data loaded or 'to' column is missing. Please 'Recalculate Final Names' first.")
            return
        
        if QMessageBox.question(self.main_window, "Confirm Rename", "This will rename files on your disk. Are you sure?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No:
            return

        self.logger.info("--- Starting File Rename Process ---")
        log_entries, error_count, renamed_count = [], 0, 0
        df = self.app_state.current_df
        
        # 1. PLAN: Create a list of all rename operations that need to happen.
        rename_plan = []
        rename_df = df[(df['to'].notna()) & (df['to'] != '') & (df['skip'] != 'x')].copy()

        for _, row in rename_df.iterrows():
            src_path = Path(row['current_path'])
            dst_path = src_path.parent / row['to']

            if not src_path.exists():
                self.logger.warn(f"Skipping plan. Source not found: {src_path.name}")
                continue
            if src_path == dst_path:
                continue

            # Add primary file (e.g., JPG) to the plan
            rename_plan.append({'src': src_path, 'dst': dst_path, 'orig': row['from']})
            
            # Find and add RAW counterpart to the plan
            raw_src_path = None
            for ext in SUPPORTED_RAW_EXTENSIONS:
                for suffix in [ext.lower(), ext.upper()]:
                    if (p := src_path.with_suffix(suffix)).exists():
                        raw_src_path = p
                        break
                if raw_src_path: break
            
            if raw_src_path:
                raw_dst_path = raw_src_path.with_name(f"{dst_path.stem}{raw_src_path.suffix}")
                original_raw_path = Path(row['from']).with_suffix(raw_src_path.suffix)
                rename_plan.append({'src': raw_src_path, 'dst': raw_dst_path, 'orig': str(original_raw_path)})

        # 2. EXECUTE: Iteratively perform renames, handling conflicts.
        while rename_plan:
            # Separate operations that can be done now from those that are blocked
            runnable = [op for op in rename_plan if not op['dst'].exists()]
            blocked = [op for op in rename_plan if op['dst'].exists()]
            
            if not runnable and blocked:
                # Deadlock detected. Break the cycle by renaming one file to a temporary name.
                op_to_break = blocked[0]
                temp_name = f"{op_to_break['src'].name}.tmp_rename"
                temp_path = op_to_break['src'].parent / temp_name
                
                self.logger.warn(f"Deadlock detected. Temporarily renaming '{op_to_break['src'].name}' to '{temp_path.name}'")
                op_to_break['src'].rename(temp_path)
                op_to_break['src'] = temp_path # Update the plan to use the temp path as the source now
                
                # The plan for the next loop is all blocked ops, one of which is now modified
                rename_plan = blocked
                continue # Restart the loop; the broken cycle should now be runnable

            if not runnable and not blocked:
                # All done
                break

            # Execute all non-conflicting renames for this pass
            for op in runnable:
                try:
                    op['src'].rename(op['dst'])
                    self.logger.info(f"Renamed: '{op['src'].name}' -> '{op['dst'].name}'")
                    log_entries.append({'original_path': op['orig'], 'new_path': str(op['dst'])})
                    renamed_count += 1
                except Exception as e:
                    self.logger.error(f"Could not rename {op['src'].name}", exception=e)
                    error_count += 1
            
            # The next set of operations to consider is the currently blocked ones
            rename_plan = blocked
        
        # 3. FINALIZE: Log results and update UI
        if log_entries:
            log_path = Path(self.app_state.rename_files_dir) / "rename_log.csv"
            new_log_df = pd.DataFrame(log_entries)
            new_log_df.to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
        
        df.loc[rename_df.index, 'to'] = ''
        self.app_state.current_df = df
        
        self.create_checked_csv(f"Rename complete: {renamed_count} files renamed, {error_count} errors. Saved state to new 'checked' file.")
        self.refresh_view()
        
    def restore_file_names(self):
        log_path = Path(self.app_state.rename_files_dir) / "rename_log.csv"
        if not log_path.exists(): return QMessageBox.warning(self.main_window, "Not Found", "rename_log.csv not found. Nothing to restore.")
        if QMessageBox.question(self.main_window, "Confirm Restore", "This will restore ALL filenames recorded in the log file. This action archives the current log. Are you sure?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
        
        try:
            log_df = pd.read_csv(log_path)
            restored_count, error_count = 0, 0
            
            # Iterate backwards to avoid conflicts if a file was renamed multiple times
            for _, row in log_df.iloc[::-1].iterrows():
                src_path, dst_path = Path(row['new_path']), Path(row['original_path'])
                if not src_path.exists(): continue
                try:
                    if dst_path.exists():
                         self.logger.error(f"Cannot restore '{src_path.name}', destination '{dst_path.name}' already exists.")
                         error_count += 1; continue
                    src_path.rename(dst_path)
                    restored_count += 1
                except Exception as e: 
                    self.logger.error(f"Could not restore '{src_path.name}'", exception=e); error_count += 1
            
            if restored_count > 0:
                # Archive the log file so it's not used again
                archive_name = f"rename_log_{datetime.now():%Y%m%d_%H%M%S}.csv.restored"
                log_path.rename(log_path.with_name(archive_name))
            
            self.refresh_view()
            QMessageBox.information(self.main_window, "Restore Complete", f"Restored {restored_count} files.\n{error_count} errors occurred.")
        except Exception as e:
            self.logger.error("Failed during restore operation.", exception=e)
            QMessageBox.critical(self.main_window, "Error", f"An error occurred during restore: {e}")

    def _get_base_csv_name(self, filename: str) -> str:
        """Strips prefixes and timestamps to get a clean base filename."""
        if not filename:
            # If no file is loaded, create a name based on the input folder
            folder_name = Path(self.app_state.input_directory).name
            return f"{folder_name}_review" if folder_name else "review_state"

        base_name = filename
        if base_name.startswith("autosave_"): base_name = base_name[len("autosave_"):]
        elif base_name.startswith("checked_"): base_name = base_name[len("checked_"):]

        base_name, _ = os.path.splitext(base_name)
        base_name = re.sub(r'_\d{8}_\d{6}', '', base_name, 1) # Remove timestamp
        return base_name

    def _handle_autosave(self):
        """Saves the current DataFrame state to a temporary autosave file."""
        if self.app_state.current_df.empty: return

        loaded_csv = self.ui.csv_dropdown.currentText()
        # Even if no CSV is loaded, we can autosave the current state
        base_name = self._get_base_csv_name(loaded_csv)
        autosave_filename = f"autosave_{base_name}.csv"

        try:
            self._save_df_to_file(autosave_filename)
            self.logger.info(f"Autosaved changes to {autosave_filename}")
        except Exception as e:
            self.logger.warn(f"Autosave failed for {autosave_filename}", exception=e)

    def create_checked_csv(self, success_message=None):
        """Saves the current state to a new, timestamped 'checked' file."""
        if self.app_state.current_df.empty:
            return QMessageBox.warning(self.main_window, "No Data", "There is no data to save.")

        loaded_csv = self.ui.csv_dropdown.currentText()
        base_name = self._get_base_csv_name(loaded_csv)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"checked_{base_name}_{timestamp}.csv"

        try:
            self._save_df_to_file(new_filename)
            self.logger.info(f"Created new checked file: {new_filename}")

            # Refresh the dropdown and select the newly created file
            self.refresh_csv_dropdown()
            new_index = self.ui.csv_dropdown.findText(new_filename)
            if new_index > -1:
                self.ui.csv_dropdown.setCurrentIndex(new_index)
            
            msg = success_message or f"Changes saved to new file:\n{new_filename}"
            QMessageBox.information(self.main_window, "Success", msg)

        except Exception as e:
            self.logger.error(f"Failed to create checked CSV '{new_filename}'", exception=e)
            QMessageBox.critical(self.main_window, "Error", f"Could not save changes: {e}")

    def _save_df_to_file(self, filename: str):
        """Saves the current DataFrame to a specific file, dropping transient columns."""
        if self.app_state.current_df.empty:
            raise ValueError("Cannot save an empty DataFrame.")

        save_path = Path(self.app_state.rename_files_dir) / filename
        
        # Drop columns that are determined at runtime to keep CSVs clean
        df_to_save = self.app_state.current_df.drop(columns=['status', 'current_path'], errors='ignore')
        df_to_save.to_csv(save_path, index=False)
        self.logger.info(f"DataFrame state saved to {save_path.name}")

    def go_to_next_page(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._populate_review_grid()

    def go_to_prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._populate_review_grid()

    def _update_navigation_controls(self):
        """Updates the state and text of pagination controls."""
        self.ui.prev_page_button.setEnabled(self.current_page > 0)
        self.ui.next_page_button.setEnabled(self.current_page < self.total_pages - 1)
        
        items_per_page = self.app_state.settings.get('review_items_per_page', 50)
        total_items = len(self.filtered_df)
        
        if total_items == 0:
            self.ui.page_label.setText("Page 1 of 1 (No items)")
        else:
            start_item = self.current_page * items_per_page + 1
            end_item = min((self.current_page + 1) * items_per_page, total_items)
            self.ui.page_label.setText(
                f"Page {self.current_page + 1} of {self.total_pages} "
                f"({start_item}-{end_item} of {total_items})"
            )