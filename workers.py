# ai-photo-processor/workers.py

import json
import re
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from PyQt5.QtCore import QObject, pyqtSignal
from PIL import Image
from PIL.Image import Resampling

from utils.gemini_handler import GeminiHandler
from utils.image_processing import (
    preprocess_image, merge_images, apply_rotation_to_folder, fix_orientation,
    decode_raw_image, crop_image
)
from utils.logger import SimpleLogger

class RotationWorker(QObject):
    finished, error = pyqtSignal(), pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, config: Dict[str, Any], logger: SimpleLogger):
        super().__init__()
        self.config, self.logger, self.is_stopped = config, logger, False

    def stop(self) -> None:
        self.is_stopped = True

    def run(self) -> None:
        try:
            self.config['progress_callback'] = self.progress.emit
            apply_rotation_to_folder(self.config, self.logger)
        except Exception as e:
            self.error.emit(f"Rotation process failed: {e}")
            self.logger.error("Rotation process failed.", exception=e)
        finally:
            if not self.is_stopped:
                self.finished.emit()


class GeminiWorker(QObject):
    finished, error = pyqtSignal(), pyqtSignal(str)
    batch_completed = pyqtSignal(pd.DataFrame, int, int)
    progress = pyqtSignal(int, str)
    # New signal to report failed batches at the end
    failed_batches_report = pyqtSignal(list)  # List of batch numbers that failed

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any], temp_dir: str, logger: SimpleLogger,
                 start_batch: int = 1, total_batches: int = 0, retry_batches: List[int] = None):
        super().__init__()
        self.df, self.config, self.temp_dir, self.logger = df, config, temp_dir, logger
        self.is_stopped = False
        self.start_batch = start_batch
        self.total_batches = total_batches
        self.retry_batches = retry_batches or []  # Specific batches to retry
        self.failed_batches = []  # Track batches that failed even after retry

    def stop(self) -> None:
        self.is_stopped = True
        self.logger.info("Stop signal received by GeminiWorker.")

    def run(self) -> None:
        try:
            handler = GeminiHandler(self.config['api_keys'], self.config['model_name'], self.logger)
            grid_rows = self.config.get('grid_rows', 3)
            grid_cols = self.config.get('grid_cols', 3)
            grid_size = grid_rows * grid_cols
            images_per_prompt = self.config.get('images_per_prompt', 10)
            photos_per_api_call = images_per_prompt * grid_size

            total_photos = len(self.df)
            if self.total_batches == 0:
                self.total_batches = (total_photos + photos_per_api_call - 1) // photos_per_api_call

            # Add batch_number column if it doesn't exist
            if 'batch_number' not in self.df.columns:
                self.df['batch_number'] = 0

            # Determine which batches to process
            if self.retry_batches:
                # Retry mode: only process specified batches
                batches_to_process = [b - 1 for b in self.retry_batches if 1 <= b <= self.total_batches]
                self.logger.info(f"Retry mode: processing batches {self.retry_batches}")
            else:
                # Normal mode: process from start_batch to end
                batches_to_process = list(range(self.start_batch - 1, self.total_batches))

            # Calculate total steps for progress
            total_steps = len(batches_to_process) * (images_per_prompt + 1)
            current_step = 0

            prerotate = self.config['crop_settings'].get('prerotate', False)
            use_exif = self.config.get('use_exif', True)
            rotation_angle = self.config.get('rotation_angle', 0)

            for api_call_idx in batches_to_process:
                if self.is_stopped:
                    self.logger.warn("Processing stopped by user.")
                    break

                api_call_num = api_call_idx + 1
                api_call_start = api_call_idx * photos_per_api_call
                api_call_end = min(api_call_start + photos_per_api_call, total_photos)

                self.logger.info(f"--- API Call {api_call_num}/{self.total_batches}: photos {api_call_start + 1}-{api_call_end} ---")

                merged_images = []
                label_ranges = []
                batch_photo_indices = []  # Track photo indices for this batch

                for img_idx in range(images_per_prompt):
                    if self.is_stopped:
                        break

                    batch_start = api_call_start + (img_idx * grid_size)
                    batch_end = min(batch_start + grid_size, api_call_end)
                    if batch_start >= api_call_end:
                        break

                    batch_df = self.df.iloc[batch_start:batch_end]
                    first_label = batch_start + 1
                    last_label = batch_end
                    batch_photo_indices.extend(range(batch_start, batch_end))

                    images = []
                    for _, row in batch_df.iterrows():
                        raw_img = Image.open(row['from'])
                        # If prerotate is ON, follow use_exif setting; if OFF, always use EXIF rotation
                        if prerotate:
                            img = fix_orientation(raw_img) if use_exif else raw_img
                            img = img.rotate(rotation_angle, expand=True)
                        else:
                            img = fix_orientation(raw_img)
                        images.append(preprocess_image(img, str(row['photo_ID']), self.config['crop_settings']))

                    if merged_img := merge_images(images, self.config['merged_img_height'], grid_rows, grid_cols):
                        merged_images.append(merged_img)
                        label_ranges.append((first_label, last_label))

                    # Update progress after each merged image
                    current_step += 1
                    pct = int(current_step / total_steps * 100)
                    self.progress.emit(pct, f"Merging image {img_idx + 1}/{images_per_prompt} for batch {api_call_num} - {pct}%")

                if self.is_stopped:
                    break

                if not merged_images:
                    self.logger.warn(f"Skipping API call {api_call_num}: no images merged.")
                    continue

                # Update progress: sending to API
                self.progress.emit(int(current_step / total_steps * 100), f"Sending {len(merged_images)} images to Gemini...")

                label_desc = ", ".join([f"Image {i+1}: labels {r[0]}-{r[1]}" for i, r in enumerate(label_ranges)])
                prompt = self.config['prompt_text'] + f"\n\nAnalyze {len(merged_images)} images. {label_desc}."

                # Try to get response, with retry on empty result
                response_text, success, is_empty = self._send_with_retry(handler, prompt, merged_images, api_call_num)

                if not success:
                    self.error.emit(f"API call {api_call_num} failed: {response_text}")
                    break

                if is_empty:
                    # Response was empty even after retry
                    self.failed_batches.append(api_call_num)
                    self.logger.warn(f"Message {api_call_num} returned empty response even after retry.")

                # Update progress after API response
                current_step += 1
                pct = int(current_step / total_steps * 100)
                self.progress.emit(pct, f"Message {api_call_num}/{self.total_batches} complete - {pct}%")

                # Update batch_number for all photos in this batch
                for idx in batch_photo_indices:
                    if idx < len(self.df):
                        self.df.iloc[idx, self.df.columns.get_loc('batch_number')] = api_call_num

                self._parse_and_update_df(response_text, api_call_num)
                self.batch_completed.emit(self.df.copy(), api_call_num, self.total_batches)

            # Report failed batches at the end
            if self.failed_batches:
                self.failed_batches_report.emit(self.failed_batches)

        except Exception as e:
            self.error.emit(f"An unexpected error occurred in GeminiWorker: {e}")
            self.logger.error("GeminiWorker run failed.", exception=e)
        finally:
            if not self.is_stopped:
                self.finished.emit()

    def _send_with_retry(self, handler, prompt: str, merged_images: List, batch_num: int, max_retries: int = 1):
        """Send request to Gemini with retry on empty response."""
        for attempt in range(max_retries + 1):
            start_time = time.perf_counter()
            response_text, success = handler.send_request(prompt, merged_images)

            if not success:
                return response_text, False, False

            self.logger.info(f"Gemini response in {time.perf_counter() - start_time:.2f}s.")

            # Check if response contains valid data
            if self._response_has_data(response_text):
                return response_text, True, False

            # Empty response
            if attempt < max_retries:
                self.logger.warn(f"Message {batch_num} returned empty response, retrying...")
                time.sleep(2)  # Brief pause before retry
            else:
                self.logger.warn(f"Message {batch_num} still empty after {max_retries} retry(s).")

        return response_text, True, True  # Success but empty

    def _response_has_data(self, response_text: str) -> bool:
        """Check if response contains non-empty data."""
        match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text, re.I)
        if not match:
            return False
        try:
            data = json.loads(match.group(1))
            if not data:
                return False
            # Check if at least one entry has a non-empty CAM value
            main_col = self.config.get('main_column', 'CAM')
            for photo_id, item_data in data.items():
                if item_data.get(main_col, '').strip():
                    return True
            return False
        except (json.JSONDecodeError, AttributeError):
            return False

    def _parse_and_update_df(self, response_text: str, batch_num: int) -> None:
        if not (match := re.search(r'```json\s*([\s\S]+?)\s*```', response_text, re.I)):
            self.logger.warn(f"Could not find JSON block in Gemini response.")
            return
        try:
            data = json.loads(match.group(1))
            for photo_id_str, item_data in data.items():
                try:
                    photo_id = int(photo_id_str)
                    row_indices = self.df.index[self.df['photo_ID'] == photo_id].tolist()
                    if row_indices:
                        valid_data = {k: v for k, v in item_data.items() if k in self.df.columns}
                        self.df.loc[row_indices[0], valid_data.keys()] = valid_data.values()
                except (ValueError, KeyError) as e:
                    self.logger.warn(f"Could not process item '{photo_id_str}' from JSON. Error: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Could not parse JSON from Gemini response.", exception=e)

class PreviewWorker(QObject):
    """Worker for rendering image previews in a background thread."""
    # Signals: (preview_type, image_bytes, width, height, file_path)
    preview_ready = pyqtSignal(str, bytes, int, int, str)
    batch_preview_ready = pyqtSignal(bytes, int, int, str)
    finished = pyqtSignal()

    def __init__(self, config: Dict[str, Any], logger: SimpleLogger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.is_stopped = False

    def stop(self) -> None:
        self.is_stopped = True

    def run(self) -> None:
        try:
            self._render_individual_previews()
            if not self.is_stopped:
                self._render_batch_preview()
        except Exception as e:
            self.logger.error("Preview rendering failed.", exception=e)
        finally:
            self.finished.emit()

    def _render_individual_previews(self):
        image_path_str = self.config.get('image_path', '')
        if not image_path_str:
            return
        img_path = Path(image_path_str)
        if not img_path.is_file():
            return

        s = self.config
        raw_extensions = {'.cr2', '.orf', '.tif', '.tiff', '.nef', '.arw', '.dng', '.raf'}

        try:
            if img_path.suffix.lower() in raw_extensions:
                base_img = decode_raw_image(img_path, use_exif=s.get('use_exif', True))
                exif_corrected = decode_raw_image(img_path, use_exif=True)
            else:
                unrotated = Image.open(img_path)
                exif_corrected = fix_orientation(unrotated.copy())
                base_img = exif_corrected if s.get('use_exif', True) else unrotated

            if not base_img:
                return

            # Original preview
            if not self.is_stopped:
                self._emit_preview('original', base_img, str(img_path))

            # Rotated preview
            if not self.is_stopped:
                rotated = base_img.rotate(s.get('rotation_angle', 0), expand=True)
                self._emit_preview('rotated', rotated, str(img_path))

            # Processed preview
            if not self.is_stopped:
                prerotate = s.get('crop_settings', {}).get('prerotate', False)
                use_exif = s.get('use_exif', True)
                # If prerotate is ON, follow use_exif setting; if OFF, always use EXIF rotation
                if prerotate:
                    img_for_gemini = base_img.copy()  # base_img already respects use_exif
                    img_for_gemini = img_for_gemini.rotate(s.get('rotation_angle', 0), expand=True)
                else:
                    img_for_gemini = exif_corrected  # Always EXIF-corrected when not pre-rotating
                processed = preprocess_image(img_for_gemini, "1", s.get('crop_settings', {}))
                self._emit_preview('processed', processed, str(img_path))

        except Exception as e:
            self.logger.error(f"Error rendering preview for {img_path.name}", exception=e)

    def _render_batch_preview(self):
        jpg_files = self.config.get('jpg_files', [])
        if not jpg_files:
            return

        s = self.config
        grid_rows = s.get('grid_rows', 3)
        grid_cols = s.get('grid_cols', 3)
        grid_size = grid_rows * grid_cols
        start_idx = s.get('batch_start_idx', 0)
        prerotate = s.get('crop_settings', {}).get('prerotate', False)
        use_exif = s.get('use_exif', True)
        rotation_angle = s.get('rotation_angle', 0)

        images_to_merge = []
        for i, p in enumerate(jpg_files[start_idx:start_idx + grid_size]):
            if self.is_stopped:
                return
            try:
                img = Image.open(p)
                # If prerotate is ON, follow use_exif setting; if OFF, always use EXIF rotation
                if prerotate:
                    base_img = fix_orientation(img) if use_exif else img
                    img_for_gemini = base_img.rotate(rotation_angle, expand=True)
                else:
                    img_for_gemini = fix_orientation(img)  # Always EXIF-corrected when not pre-rotating
                processed = preprocess_image(img_for_gemini, str(start_idx + i + 1), s.get('crop_settings', {}))
                images_to_merge.append(processed)
            except Exception as e:
                self.logger.error(f"Failed to process image for batch: {p.name}", exception=e)

        if self.is_stopped or not images_to_merge:
            return

        if merged := merge_images(images_to_merge, s.get('merged_img_height', 1080), grid_rows, grid_cols):
            temp_dir = s.get('temp_dir', '')
            if not temp_dir:
                return
            temp_path = Path(temp_dir) / "temp_merged_preview.jpg"
            merged.save(temp_path, quality=90)

            merged_rgb = merged.convert('RGB')
            img_bytes = merged_rgb.tobytes("raw", "RGB")
            self.batch_preview_ready.emit(img_bytes, merged_rgb.width, merged_rgb.height, str(temp_path))

    def _emit_preview(self, preview_type: str, img: Image.Image, path: str):
        img_rgb = img.convert('RGB') if img.mode != 'RGB' else img
        img_bytes = img_rgb.tobytes("raw", "RGB")
        self.preview_ready.emit(preview_type, img_bytes, img_rgb.width, img_rgb.height, path)


class ImageLoadWorker(QObject):
    image_loaded = pyqtSignal(str, bytes, int, int)
    finished = pyqtSignal()
    
    def __init__(self, image_paths: List[Path], logger: SimpleLogger, crop_settings: Dict[str, Any], thumb_height: int):
        super().__init__()
        self.image_paths = image_paths
        self.logger = logger
        self.crop_settings = crop_settings
        self.thumb_height = thumb_height
        self.is_stopped = False

    def stop(self) -> None:
        self.is_stopped = True

    def run(self) -> None:
        for path in self.image_paths:
            if self.is_stopped: break
            try:
                pil_img = decode_raw_image(path, use_exif=True) if path.suffix.lower() in {'.cr2', '.orf', '.tif', '.tiff'} else Image.open(path)
                if not pil_img: continue
                
                pil_img = fix_orientation(pil_img)
                pil_img = crop_image(pil_img, self.crop_settings)
                
                # --- MODIFIED: Only resize if thumb_height > 0 and image is larger ---
                if self.thumb_height > 0 and pil_img.height > self.thumb_height:
                    aspect_ratio = pil_img.width / pil_img.height
                    new_width = int(self.thumb_height * aspect_ratio)
                    pil_img = pil_img.resize((new_width, self.thumb_height), Resampling.LANCZOS)

                pil_img = pil_img.convert('RGB')
                
                img_bytes = pil_img.tobytes("raw", "RGB")
                width, height = pil_img.size

                self.image_loaded.emit(str(path), img_bytes, width, height)
            except Exception as e:
                self.logger.error(f"Error loading review image {path.name}", exception=e)
        
        self.finished.emit()