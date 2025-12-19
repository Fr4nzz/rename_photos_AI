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

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any], temp_dir: str, logger: SimpleLogger, start_batch: int = 1, total_batches: int = 0):
        super().__init__()
        self.df, self.config, self.temp_dir, self.logger = df, config, temp_dir, logger
        self.is_stopped = False
        self.start_batch = start_batch
        self.total_batches = total_batches

    def stop(self) -> None:
        self.is_stopped = True
        self.logger.info("Stop signal received by GeminiWorker.")

    def run(self) -> None:
        try:
            handler = GeminiHandler(self.config['api_keys'], self.config['model_name'], self.logger)
            batch_size = self.config['batch_size']
            images_per_prompt = self.config.get('images_per_prompt', 10)
            photos_per_api_call = images_per_prompt * batch_size

            # Calculate total API calls needed
            total_photos = len(self.df)
            if self.total_batches == 0:
                self.total_batches = (total_photos + photos_per_api_call - 1) // photos_per_api_call

            prerotate = self.config['crop_settings'].get('prerotate', False)
            rotation_angle = self.config.get('rotation_angle', 0)

            for api_call_idx in range(self.start_batch - 1, self.total_batches):
                if self.is_stopped:
                    self.logger.warn("Processing stopped by user."); break

                api_call_num = api_call_idx + 1
                api_call_start = api_call_idx * photos_per_api_call
                api_call_end = min(api_call_start + photos_per_api_call, total_photos)
                api_call_df = self.df.iloc[api_call_start:api_call_end]

                self.logger.info(f"--- API Call {api_call_num}/{self.total_batches}: photos {api_call_start + 1}-{api_call_end} ---")

                # Create multiple merged images for this API call
                merged_images = []
                label_ranges = []

                for img_idx in range(images_per_prompt):
                    batch_start = api_call_start + (img_idx * batch_size)
                    batch_end = min(batch_start + batch_size, api_call_end)
                    if batch_start >= api_call_end:
                        break

                    batch_df = self.df.iloc[batch_start:batch_end]
                    first_label = batch_start + 1
                    last_label = batch_end

                    images = []
                    for _, row in batch_df.iterrows():
                        img = fix_orientation(Image.open(row['from']))
                        if prerotate:
                            img = img.rotate(rotation_angle, expand=True)
                        images.append(preprocess_image(img, str(row['photo_ID']), self.config['crop_settings']))

                    if merged_img := merge_images(images, self.config['merged_img_height']):
                        merged_images.append(merged_img)
                        label_ranges.append((first_label, last_label))

                if not merged_images:
                    self.logger.warn(f"Skipping API call {api_call_num}: no images merged.")
                    continue

                # Build prompt with label ranges for all merged images
                label_desc = ", ".join([f"Image {i+1}: labels {r[0]}-{r[1]}" for i, r in enumerate(label_ranges)])
                prompt = self.config['prompt_text'] + f"\n\nAnalyze {len(merged_images)} images. {label_desc}."

                start_time = time.perf_counter()
                response_text, success = handler.send_request(prompt, merged_images)

                if not success:
                    self.error.emit(f"API call {api_call_num} failed: {response_text}"); break

                self.logger.info(f"Gemini response in {time.perf_counter() - start_time:.2f}s. Response: {response_text.replace(chr(10), ' ')}")
                self._parse_and_update_df(response_text)

                self.progress.emit(int(api_call_num / self.total_batches * 100), f"API Call {api_call_num}/{self.total_batches} - %p%")
                self.batch_completed.emit(self.df.copy(), api_call_num, self.total_batches)
        except Exception as e:
            self.error.emit(f"An unexpected error occurred in GeminiWorker: {e}")
            self.logger.error("GeminiWorker run failed.", exception=e)
        finally:
            if not self.is_stopped:
                self.finished.emit()

    def _parse_and_update_df(self, response_text: str) -> None:
        if not (match := re.search(r'```json\s*([\s\S]+?)\s*```', response_text, re.I)):
            self.logger.warn(f"Could not find JSON block in Gemini response."); return
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