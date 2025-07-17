# ai-photo-processor/workers.py

import json
import re
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

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

    def stop(self): self.is_stopped = True

    def run(self):
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

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any], temp_dir: str, logger: SimpleLogger):
        super().__init__()
        self.df, self.config, self.temp_dir, self.logger = df, config, temp_dir, logger
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True
        self.logger.info("Stop signal received by GeminiWorker.")

    def run(self):
        try:
            handler = GeminiHandler(self.config['api_keys'], self.config['model_name'], self.logger)
            batch_size = self.config['batch_size']
            total_batches = (len(self.df) + batch_size - 1) // batch_size
            
            for i in range(total_batches):
                if self.is_stopped:
                    self.logger.warn("Processing stopped by user."); break
                
                batch_num, start_idx = i + 1, i * batch_size
                batch_df = self.df.iloc[start_idx : start_idx + batch_size]
                self.logger.info(f"--- Starting Batch {batch_num}/{total_batches} ---")

                images = [preprocess_image(fix_orientation(Image.open(row['from'])), str(row['photo_ID']), self.config['crop_settings']) for _, row in batch_df.iterrows()]
                
                if not (merged_img := merge_images(images, self.config['merged_img_height'])):
                    self.logger.warn(f"Skipping batch {batch_num} due to failure in merging images.")
                    continue

                temp_img_path = Path(self.temp_dir) / f"temp_batch_{batch_num}.jpg"
                merged_img.save(temp_img_path)
                
                prompt = self.config['prompt_text'] + f"\n\nAnalyze images labeled {start_idx + 1} to {start_idx + len(images)}."
                
                start_time = time.perf_counter()
                response_text, success = handler.send_request(prompt, str(temp_img_path))
                
                if not success:
                    self.error.emit(f"API call failed for batch {batch_num}: {response_text}"); break

                self.logger.info(f"Gemini response received in {time.perf_counter() - start_time:.2f}s. Response: {response_text.replace(chr(10), ' ')}")
                self._parse_and_update_df(response_text)
                
                self.progress.emit(int(batch_num / total_batches * 100), f"Batch {batch_num}/{total_batches} - %p%")
                self.batch_completed.emit(self.df.copy(), batch_num, total_batches)
        except Exception as e:
            self.error.emit(f"An unexpected error occurred in GeminiWorker: {e}")
            self.logger.error("GeminiWorker run failed.", exception=e)
        finally:
            if not self.is_stopped:
                self.finished.emit()

    def _parse_and_update_df(self, response_text: str):
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

    def stop(self): self.is_stopped = True

    def run(self):
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