import os
import sys
import re
import math
import json
import time
import random
import datetime
import pandas as pd
import piexif
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QCheckBox, QComboBox,
    QScrollArea, QLineEdit, QGridLayout, QMessageBox, QProgressBar, QGroupBox,
    QTextEdit
)
import google.generativeai as genai
from googleapiclient.errors import HttpError
import numpy as np

# Configuration
SAMPLE_JSON = """{
  "1": {"CAM": "CAM074806", "co": "", "n": "", "skip": ""},
  "2": {"CAM": "CAM074806", "co": "", "n": "", "skip": ""},
  "3": {"CAM": "Empty", "co": "", "n": "CAM missing", "skip": "x"},
  "4": {"CAM": "CAM070555", "co": "CAM072554", "n": "", "skip": ""}
}"""

DEFAULT_PROMPT = """Extract CAM (CAM07xxxx) and notes (n) from the image.
- 2 wing photos (dorsal and ventral) per individual (CAM) are arranged in a grid left to right, top to bottom.
- If no CAMID is visible or image should be skipped, set skip: 'x', else skip: ''
- If CAMID is crossed out, set 'co' to the crossed out CAMID and put the new CAMID in 'CAM'
- CAMIDs have no spaces, remember CAM format (CAM07xxxx)
- Use notes (n) to indicate anything unusual (e.g., repeated, rotated 90°, etc).
- Put skipped reason in notes 'n'
- Double-check numbers are correctly OCRed; consecutive photos might not have consecutive CAMs
- Return JSON as shown in example; always give all keys even if empty. Example:
{sample_json_output}
"""

CONFIG = {
    'api_keys': [],  # Will be loaded from file
    'directory_path': '',
    'compressed_exts': [".jpg", ".jpeg", ".png"],
    'raw_exts': [".cr2", ".orf"],
    'batch_size': 9,
    'merged_img_height': 1080,
    'main_column': 'CAM',
    'prompt_text_base': DEFAULT_PROMPT.format(sample_json_output=SAMPLE_JSON),
    'temp_output_prefix': 'temp_output',
    'output_prefix': 'output',
}

def load_api_keys():
    try:
        with open('API_keys.txt', 'r') as f:
            keys = [line.strip() for line in f if line.strip()]
            CONFIG['api_keys'] = keys
            return keys
    except FileNotFoundError:
        return []

def save_api_keys(keys_text):
    with open('API_keys.txt', 'w') as f:
        f.write(keys_text)
    # Update config with new keys
    CONFIG['api_keys'] = [k.strip() for k in keys_text.splitlines() if k.strip()]

class GeminiWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)
    error = QtCore.pyqtSignal(str)
    batch_completed = QtCore.pyqtSignal(pd.DataFrame, int, int)

    def __init__(self, config, df, start_batch, total_batches, rename_files_dir):
        super().__init__()
        self.config = config
        self.df = df
        self.start_batch = start_batch
        self.total_batches = total_batches
        self.rename_files_dir = rename_files_dir
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def process(self):
        try:
            gemini_handler = GeminiHandler(self.config)
            
            for batch_i in range(self.start_batch - 1, self.total_batches):
                if self.should_stop:
                    break
                    
                bnum = batch_i + 1
                start_idx = batch_i * self.config['batch_size']
                end_idx = start_idx + self.config['batch_size']
                batch_df = self.df.iloc[start_idx:end_idx].copy()
                
                images = [ImageProcessor.preprocess_image(row['from'], row['photo_ID']) 
                         for _, row in batch_df.iterrows()]
                merged = ImageProcessor.merge_images(images, merged_img_height=self.config['merged_img_height'])
                
                if merged and not self.should_stop:
                    temp_file = f"temp_merged_b{bnum}of{self.total_batches}.jpg"
                    temp_path = os.path.join(self.rename_files_dir, temp_file)
                    merged.save(temp_path)
                    
                    # Add batch range information to the prompt
                    batch_range_info = f"\nSending images labeled from {start_idx + 1} to {end_idx}"
                    self.config['prompt_text_base'] += batch_range_info
                    
                    response, _ = gemini_handler.send_request(temp_path)
                    # Remove the batch range info from prompt to keep it clean for next batch
                    self.config['prompt_text_base'] = self.config['prompt_text_base'].replace(batch_range_info, '')
                    
                    resp_text = getattr(response, 'text', str(response))
                    
                    match = re.search(r'```json\s*\n([\s\S]*?)\n```', resp_text)
                    if match and not self.should_stop:
                        jdata = json.loads(match.group(1))
                        
                        # Add new columns to the dataframe if they don't exist
                        for item_data in jdata.values():
                            for key in item_data.keys():
                                if key not in self.df.columns:
                                    self.df[key] = None
                                    
                        # Map the JSON data to the dataframe rows
                        for row_num, item_data in jdata.items():
                            # Find the row where photo_ID matches the Gemini response key
                            matching_rows = self.df[self.df['photo_ID'] == int(row_num)]
                            if not matching_rows.empty:
                                idx = matching_rows.index[0]
                                for key, value in item_data.items():
                                    self.df.at[idx, key] = value
                                    
                        partial_csv = f"{self.config['temp_output_prefix']}_b{bnum}of{self.total_batches}.csv"
                        self.df.to_csv(os.path.join(self.rename_files_dir, partial_csv), index=False)
                        
                        # Emit progress and batch completion
                        progress = int((bnum / self.total_batches) * 100)
                        self.progress.emit(progress)
                        self.batch_completed.emit(self.df.copy(), bnum, self.total_batches)
            
            if not self.should_stop:
                self.finished.emit()
                
        except Exception as e:
            print(f"\nProcessing error: {str(e)}")  # Added console logging
            self.error.emit(str(e))

class ImageLoadWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    image_loaded = QtCore.pyqtSignal(str, QtGui.QImage)
    error = QtCore.pyqtSignal(str, str)

    def __init__(self, image_paths, crop_settings=None):
        super().__init__()
        self.image_paths = image_paths
        self.crop_settings = crop_settings
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def process(self):
        for path in self.image_paths:
            if self.should_stop:
                break
                
            try:
                pil_img = Image.open(path)
                
                # Apply EXIF rotation first if it's a JPEG (PNG doesn't have EXIF)
                if not path.lower().endswith('.png'):
                    pil_img = ImageProcessor.fix_orientation(pil_img)
                
                # Apply crop if settings exist
                if self.crop_settings:
                    pil_img = ImageProcessor.crop_image(pil_img, self.crop_settings)
                
                pil_img = pil_img.convert('RGB')
                
                w, h = pil_img.size
                target_height = 300
                ratio = target_height / h
                target_width = int(w * ratio)
                
                qimg = QtGui.QImage(pil_img.tobytes("raw","RGB"), w, h, QtGui.QImage.Format_RGB888)
                qimg = qimg.scaled(target_width, target_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                
                self.image_loaded.emit(path, qimg)
                
            except Exception as e:
                self.error.emit(path, str(e))
                
        self.finished.emit()

    def on_image_loaded(self, image_path, qimage):
        if image_path in self.image_labels and image_path in self.pending_image_loads:
            label = self.image_labels[image_path]
            label.setPixmap(QtGui.QPixmap.fromImage(qimage))
            del self.pending_image_loads[image_path]

class ImageExtensionHandler:
    COMPRESSED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    RAW_EXTENSIONS = {'.cr2', '.orf'}
    
    @staticmethod
    def is_compressed_image(ext):
        return ext.lower() in ImageExtensionHandler.COMPRESSED_EXTENSIONS
        
    @staticmethod
    def is_raw_image(ext):
        return ext.lower() in ImageExtensionHandler.RAW_EXTENSIONS

class ImageProcessor:
    @staticmethod
    def get_system_font():
        """Get first available system font"""
        system_fonts = {
            'darwin': "Helvetica.ttc",
            'linux': "DejaVuSans.ttf",
            'windows': "arial.ttf"
        }
        
        for font_name in system_fonts.values():
            try:
                return ImageFont.truetype(font_name, 50)
            except:
                continue
        return ImageFont.load_default()

    @staticmethod
    def fix_orientation(img):
        """Apply correct orientation based on EXIF data"""
        try:
            exif = dict(img._getexif().items())
            orientation = exif.get(274, 1)
            rotations = {
                1: lambda x: x,
                2: lambda x: ImageOps.mirror(x),
                3: lambda x: x.rotate(180, expand=True),
                4: lambda x: ImageOps.flip(x),
                5: lambda x: ImageOps.mirror(x.rotate(90, expand=True)),
                6: lambda x: x.rotate(270, expand=True),
                7: lambda x: ImageOps.mirror(x.rotate(270, expand=True)),
                8: lambda x: x.rotate(90, expand=True)
            }
            return rotations.get(orientation, lambda x: x)(img)
        except:
            return img

    @staticmethod
    def crop_image(img, crop_settings=None):
        if crop_settings is None:
            crop_settings = {
                'top': 0.1,
                'bottom': 0.5,
                'left': 0.0,
                'right': 0.5,
                'zoom': True
            }
            
        # If zoom is disabled, return the original image
        if 'zoom' in crop_settings and not crop_settings['zoom']:
            return img
            
        w, h = img.size
            
        crop_upper = int(h * crop_settings['top'])
        crop_lower = int(h * crop_settings['bottom'])
        crop_left = int(w * crop_settings['left'])
        crop_right = int(w * crop_settings['right'])
        
        return img.crop((crop_left, crop_upper, crop_right, crop_lower))

    @staticmethod
    def preprocess_image(image_path, label, crop_settings=None):
        # Open the image
        img = Image.open(image_path)
        
        # Apply EXIF rotation first if it's a JPEG (PNG doesn't have EXIF)
        if not image_path.lower().endswith('.png'):
            img = ImageProcessor.fix_orientation(img)
            
        # Only apply crop if zoom is enabled
        if crop_settings and crop_settings.get('zoom', False):
            img = ImageProcessor.crop_image(img, crop_settings)
            
        # Only apply grayscale if enabled
        if crop_settings and crop_settings.get('grayscale', True):
            img = ImageOps.grayscale(img)
            
        img = ImageOps.expand(img, border=10, fill='black')
        
        # Convert to RGBA to support transparency
        img = img.convert('RGBA')
        draw = ImageDraw.Draw(img)
        font = ImageProcessor.get_system_font()  # This already returns a font with size 50
            
        text = str(label)  # label is now the row number from DataFrame
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position for text and background
        x = (img.width - text_width) // 2
        y = 10
        
        # Draw semi-transparent white background
        padding = 10  # Padding around text
        background_bbox = [
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding
        ]
        draw.rectangle(background_bbox, fill=(255, 255, 255, 128))  # White with 50% opacity
        
        # Draw text
        draw.text((x, y), text, fill='black', font=font)
        
        # Convert back to RGB for final output
        return img.convert('RGB')

    @staticmethod
    def merge_images(images, merged_img_height=1080):
        if not images:
            return None
            
        n = len(images)
        rows = math.ceil(math.sqrt(n))  # Start with square-ish layout
        cols = math.ceil(n/rows)
        
        # Calculate the height for each row
        row_height = merged_img_height // rows
        
        # Calculate the width for each column based on the widest image
        # We'll use the first image as reference and scale it to row_height
        ref_img = images[0]
        ref_width, ref_height = ref_img.size
        scale_ratio = row_height / ref_height
        col_width = int(ref_width * scale_ratio)
        
        # Calculate total grid dimensions
        grid_height = merged_img_height
        grid_width = cols * col_width
        
        # Create the grid
        grid = Image.new('RGB', (grid_width, grid_height), 'white')
        
        # Process each image
        for idx, img in enumerate(images):
            if idx >= n:  # Skip if we've processed all images
                break
                
            # Calculate position in grid
            r, c = divmod(idx, cols)
            
            # Calculate scaling to fit in cell while preserving aspect ratio
            img_width, img_height = img.size
            scale_ratio = min(col_width/img_width, row_height/img_height)
            new_width = int(img_width * scale_ratio)
            new_height = int(img_height * scale_ratio)
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Calculate position to center image in cell
            x = c * col_width + (col_width - new_width) // 2
            y = r * row_height + (row_height - new_height) // 2
            
            # Paste image into grid
            grid.paste(resized_img, (x, y))
            
        return grid

class GeminiHandler:
    def __init__(self, config):
        self.config = config
        self.api_keys = config['api_keys']
        self.current_api_key_index = 0
        self.retry_count = 0
        self.max_retries = len(self.api_keys) * 2  # Try each key twice before giving up
        self._configure_client()

    def _configure_client(self):
        if self.api_keys:
            genai.configure(api_key=self.api_keys[self.current_api_key_index])
            print(f"\nSwitched to API key {self.current_api_key_index + 1}")
        self.model = genai.GenerativeModel(model_name=self.config['model_name'])
        self.chat = self.model.start_chat()

    def update_model(self, model_name):
        self.config['model_name'] = model_name
        self._configure_client()

    def _switch_api_key(self):
        self.current_api_key_index = (self.current_api_key_index + 1) % len(self.api_keys)
        self._configure_client()
        self.retry_count += 1

    def upload_file(self, file_path, max_retries=5):
        last_error = None
        for attempt in range(max_retries):
            try:
                return genai.upload_file(file_path)
            except HttpError as e:
                last_error = e
                if e.resp.status in [429, 500, 503]:
                    if self.retry_count < self.max_retries:
                        print(f"\nUpload error: {str(e)}")
                        print(f"Switching to API key {self.current_api_key_index + 2} and retrying...")
                        self._switch_api_key()
                        continue
                raise
        raise last_error if last_error else RuntimeError(f"Failed to upload {file_path}")

    def send_request(self, merged_file_path, max_retries=6):
        last_error = None
        for attempt in range(max_retries):
            try:
                uploaded_file = self.upload_file(merged_file_path)
                prompt = [
                    self.config['prompt_text_base'],
                    "\n\nDo the same for the following images:",
                    uploaded_file
                ]
                
                try:
                    response = self.chat.send_message(prompt)
                    print(f"\nGemini Response: {getattr(response, 'text', str(response))}")
                    # Reset retry count on successful request
                    self.retry_count = 0
                    return response, uploaded_file
                except Exception as e:
                    last_error = e
                    print(f"\nError from Gemini: {str(e)}")
                    if "quota" in str(e).lower() or "429" in str(e):
                        if self.retry_count < self.max_retries:
                            print(f"Switching to API key {self.current_api_key_index + 2} and retrying...")
                            self._switch_api_key()
                            continue
                    raise

            except HttpError as e:
                last_error = e
                print(f"\nHTTP Error: {str(e)}")
                if e.resp.status in [429, 500, 503] or "quota" in str(e).lower():
                    if self.retry_count < self.max_retries:
                        print(f"Switching to API key {self.current_api_key_index + 2} and retrying...")
                        self._switch_api_key()
                        continue
                raise

            except Exception as e:
                last_error = e
                print(f"\nUnexpected error: {str(e)}")
                if self.retry_count < self.max_retries:
                    print(f"Switching to API key {self.current_api_key_index + 2} and retrying...")
                    self._switch_api_key()
                    continue
                raise

        raise last_error if last_error else RuntimeError(f"Failed after {max_retries} attempts")

class FileManager:
    @staticmethod
    def get_last_dir_file():
        return os.path.join(os.path.dirname(__file__), 'last_dir.txt')

    @staticmethod
    def load_last_dir():
        path = FileManager.get_last_dir_file()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                last_dir = f.read().strip()
                if last_dir and os.path.isdir(last_dir):
                    return last_dir, os.path.dirname(last_dir)
        return '', ''

    @staticmethod
    def save_last_dir(directory):
        with open(FileManager.get_last_dir_file(), 'w', encoding='utf-8') as f:
            f.write(directory)

    @staticmethod
    def ensure_rename_files_dir(directory):
        rename_dir = os.path.join(directory, "rename_files")
        os.makedirs(rename_dir, exist_ok=True)
        return rename_dir

    @staticmethod
    def get_all_files(directory, valid_exts):
        """Get all files with valid extensions. For Gemini processing, only return standard image files."""
        files = []
        for f in os.listdir(directory):
            ext = os.path.splitext(f)[1]
            if any(ext.upper().endswith(valid_ext.upper()) for valid_ext in valid_exts):
                # For Gemini processing, only include standard image files
                if ImageExtensionHandler.is_compressed_image(ext):
                    files.append(os.path.join(directory, f))
                else:
                    print(f"Skipping raw file for processing: {f}")
        return files

    @staticmethod
    def get_image_pairs(directory):
        """Get pairs of compressed and raw images"""
        files = os.listdir(directory)
        pairs = []
        
        for f in files:
            base, ext = os.path.splitext(f)
            if ImageExtensionHandler.is_compressed_image(ext):
                # Look for matching raw file
                for raw_ext in CONFIG['raw_exts']:
                    raw_file = base + raw_ext
                    if raw_file in files:
                        pairs.append((os.path.join(directory, f), os.path.join(directory, raw_file)))
                        break
                        
        return pairs

class NameCalculator:
    @staticmethod
    def calculate_suffixes_for_cam(cam_df, suffix_mode, main_column='CAM'):
        non_skipped = cam_df[cam_df['skip'] != 'x'].copy()
        
        if suffix_mode == 'clips':
            sufs = [f"v{i+1}" for i in range(len(non_skipped))]
        else:
            pattern = ['d', 'v']
            sufs = []
            for i in range(len(non_skipped)):
                round_i = i // 2
                mod_i = i % 2
                sufs.append(pattern[mod_i] if round_i == 0 else pattern[mod_i] + str(round_i + 1))
                
        cam_df = cam_df.copy()
        suf_idx = 0
        for i in cam_df.index:
            cam_df.at[i, 'suffix'] = '' if cam_df.at[i, 'skip'] == 'x' else sufs[suf_idx]
            if cam_df.at[i, 'skip'] != 'x':
                suf_idx += 1
                
        return cam_df['suffix']

    @staticmethod
    def recalc_final_names(df, suffix_mode, main_column='CAM'):
        if df.empty:
            return df
            
        df = df.copy()
        df['suffix'] = ''
        non_skipped_mask = df['skip'] != 'x'
        
        for cam_name, group in df[non_skipped_mask].groupby(main_column, group_keys=False):
            if not cam_name:
                continue
            suffixes = NameCalculator.calculate_suffixes_for_cam(group, suffix_mode, main_column)
            for idx, suffix in zip(group.index, suffixes):
                df.at[idx, 'suffix'] = suffix
        
        # Get the source file extension for each row
        df['to'] = ''
        for idx, row in df.iterrows():
            if not row['skip']:
                src_file = row['from']
                # Get the original extension with its case
                src_ext = os.path.splitext(src_file)[1]
                base_name = str(row[main_column]) + str(row['suffix'])
                df.at[idx, 'to'] = base_name + src_ext
        
        return df

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Photo Processor")
        self.resize(1400, 800)
        
        # Load API keys at startup
        load_api_keys()
        
        self.input_directory, self.parent_directory = FileManager.load_last_dir()
        self.rename_files_dir = ""
        if self.input_directory and os.path.isdir(self.input_directory):
            self.rename_files_dir = FileManager.ensure_rename_files_dir(self.input_directory)
            
        self.current_df = pd.DataFrame()
        self.suffix_mode = 'wings'
        self.ext_list = [".JPG", ".CR2"]
        
        # Preview update control
        self.combined_preview_updating = False
        self.active_tab = 0  # Track the active tab to prevent updates when not on Process Images tab
        
        # Load crop settings
        self.crop_settings = {
            'top': 0.1,
            'bottom': 0.5,
            'left': 0.0,
            'right': 0.5,
            'zoom': True,  # Default to zoom in
            'grayscale': True  # Default to grayscale
        }
        
        # Load prompt from file or use default
        self.load_prompt()
        
        # Thread management
        self.gemini_thread = None
        self.gemini_worker = None
        self.image_threads = []
        self.image_workers = []
        self.pending_image_loads = {}
        self.image_labels = {}
        
        # Initialize Gemini models list
        self.available_models = []
        if CONFIG['api_keys']:
            try:
                genai.configure(api_key=CONFIG['api_keys'][0])
                # Filter and sort models in one go
                self.available_models = sorted(
                    [m.name.split('/')[-1] for m in genai.list_models() 
                     if 'legacy' not in m.display_name.lower() 
                     and 'gemini' in m.name.lower()
                     and (lambda x: float(x.split('-')[1]) >= 2.0 if x.split('-')[1].replace('.','').isdigit() else False)(m.name.split('/')[-1])],
                    key=lambda x: (-float(x.split('-')[1]) if x.split('-')[1].replace('.','').isdigit() else 0, 
                                -('exp' in x.lower()), x)
                )
            except Exception as e:
                print(f"Error loading Gemini models: {e}")
        
        self.setup_ui()
        
        # Populate model dropdown
        self.populate_model_dropdown()
        
        if self.input_directory and os.path.isdir(self.input_directory):
            self.lab_dir_path.setText(self.input_directory)
            self.refresh_ui()

    def setup_ui(self):
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        self.build_ask_gemini_tab()
        self.build_check_output_tab()
        self.build_api_keys_tab()  # Add new tab
        
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def build_api_keys_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # API Keys text area
        self.api_keys_text = QTextEdit()
        self.api_keys_text.setPlaceholderText("Enter API keys here, one per line")
        
        # Load existing API keys if any
        if CONFIG['api_keys']:
            self.api_keys_text.setPlainText('\n'.join(CONFIG['api_keys']))
        
        layout.addWidget(QLabel("API Keys (one per line):"))
        layout.addWidget(self.api_keys_text)
        
        # Save button
        save_keys_btn = QPushButton("Save API Keys")
        save_keys_btn.clicked.connect(self.on_save_api_keys)
        layout.addWidget(save_keys_btn)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "API Keys")

    def on_save_api_keys(self):
        keys_text = self.api_keys_text.toPlainText()
        save_api_keys(keys_text)
        QMessageBox.information(self, "Success", "API keys saved successfully.")

    def cleanup_threads(self):
        # Clean up Gemini thread
        if self.gemini_worker:
            self.gemini_worker.stop()
        if self.gemini_thread:
            self.gemini_thread.quit()
            self.gemini_thread.wait()
            
        # Clean up image loading threads
        for worker in self.image_workers:
            worker.stop()
        for thread in self.image_threads:
            thread.quit()
            thread.wait()
            
        self.image_threads.clear()
        self.image_workers.clear()
        self.pending_image_loads.clear()

    def start_gemini_processing(self, df, start_batch, total_batches):
        self.cleanup_threads()
        
        # Create thread and worker
        self.gemini_thread = QtCore.QThread()
        self.gemini_worker = GeminiWorker(CONFIG, df, start_batch, total_batches, self.rename_files_dir)
        self.gemini_worker.moveToThread(self.gemini_thread)
        
        # Connect signals
        self.gemini_thread.started.connect(self.gemini_worker.process)
        self.gemini_worker.finished.connect(self.on_gemini_processing_finished)
        self.gemini_worker.progress.connect(self.progress_bar.setValue)
        self.gemini_worker.error.connect(self.on_gemini_error)
        self.gemini_worker.batch_completed.connect(self.on_batch_completed)
        
        # Start processing
        self.gemini_thread.start()

    def start_image_loading(self, image_paths):
        # Clean up any existing image threads
        for worker in self.image_workers:
            worker.stop()
        for thread in self.image_threads:
            thread.quit()
            thread.wait()
            
        self.image_threads.clear()
        self.image_workers.clear()
        self.pending_image_loads.clear()
        
        # Create a new thread and worker for image loading
        thread = QtCore.QThread()
        worker = ImageLoadWorker(image_paths, self.crop_settings)
        worker.moveToThread(thread)
        
        # Connect signals
        thread.started.connect(worker.process)
        worker.finished.connect(thread.quit)
        worker.image_loaded.connect(self.on_image_loaded)
        worker.error.connect(self.on_image_load_error)
        
        # Store thread and worker references
        self.image_threads.append(thread)
        self.image_workers.append(worker)
        
        # Track which images we're waiting for
        for path in image_paths:
            self.pending_image_loads[path] = True
            
        # Start processing
        thread.start()

    def on_gemini_processing_finished(self):
        final_csv = f"{CONFIG['output_prefix']}_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.gemini_worker.df.to_csv(os.path.join(self.rename_files_dir, final_csv), index=False)
        QMessageBox.information(self, "Done", "Processing completed.")
        self.cleanup_threads()
        self.refresh_ui()

    def on_gemini_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Processing error: {error_msg}")
        self.cleanup_threads()

    def on_batch_completed(self, updated_df, batch_num, total_batches):
        self.current_df = updated_df
        print(f"Completed batch {batch_num} of {total_batches}")

    def on_image_loaded(self, image_path, qimage):
        if image_path in self.image_labels and image_path in self.pending_image_loads:
            label = self.image_labels[image_path]
            label.setPixmap(QtGui.QPixmap.fromImage(qimage))
            del self.pending_image_loads[image_path]

    def on_image_load_error(self, image_path, error_msg):
        if image_path in self.image_labels and image_path in self.pending_image_loads:
            label = self.image_labels[image_path]
            label.setText(f"Error: {error_msg}")
            del self.pending_image_loads[image_path]

    def closeEvent(self, event):
        self.cleanup_threads()
        super().closeEvent(event)

    def build_ask_gemini_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # First row: Input folder, Preview Image dropdown, Rotate JPG dropdown, and Rotate button
        row1 = QHBoxLayout()
        self.lab_dir_path = QLabel("(none)")
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.on_browse_directory)
        row1.addWidget(QLabel("Input Folder:"))
        row1.addWidget(self.lab_dir_path)
        row1.addWidget(btn_browse)
        
        self.preview_dropdown = QComboBox()
        self.preview_dropdown.currentIndexChanged.connect(self.on_preview_dropdown_changed)
        row1.addWidget(QLabel("Preview Image:"))
        row1.addWidget(self.preview_dropdown)
        
        self.rotation_dropdown = QComboBox()
        for deg, text in [(0, "0° (No rotation)"), (90, "90° anticlockwise"),
                         (180, "180° anticlockwise"), (270, "270° anticlockwise")]:
            self.rotation_dropdown.addItem(text, deg)
        self.rotation_dropdown.setCurrentIndex(2)
        self.rotation_dropdown.currentIndexChanged.connect(self.update_jpg_preview)
        row1.addWidget(QLabel("Rotate JPG:"))
        row1.addWidget(self.rotation_dropdown)
        
        btn_rotate = QPushButton("Rotate JPGs")
        btn_rotate.clicked.connect(self.on_rotate_jpgs)
        row1.addWidget(btn_rotate)
        layout.addLayout(row1)
        
        # Second row: Original Image and Rotated Image
        row2 = QHBoxLayout()
        self.lbl_before_preview = self.create_preview_label()
        self.lbl_rotated_preview = self.create_preview_label()
        row2.addWidget(QLabel("Original:"))
        row2.addWidget(self.lbl_before_preview)
        row2.addWidget(QLabel("Rotated:"))
        row2.addWidget(self.lbl_rotated_preview)
        layout.addLayout(row2)
        
        # Third row: Zoom In Image and Zoom settings
        row3 = QHBoxLayout()
        
        # Left side: Zoom In Image
        left_side = QVBoxLayout()
        zoomed_row = QHBoxLayout()
        zoomed_row.addWidget(QLabel("Zoom In:"))
        self.lbl_after_preview = self.create_preview_label()
        zoomed_row.addWidget(self.lbl_after_preview)
        left_side.addLayout(zoomed_row)
        
        # Right side: Zoom settings
        right_side = QVBoxLayout()
        
        # Create horizontal layout for checkboxes
        checkbox_layout = QHBoxLayout()
        
        # Zoom checkbox
        self.zoom_checkbox = QCheckBox("Zoom In (proportion to crop from each side)")
        self.zoom_checkbox.setChecked(True)
        self.zoom_checkbox.stateChanged.connect(self.on_zoom_checkbox_changed)
        checkbox_layout.addWidget(self.zoom_checkbox)
        
        # Grayscale checkbox
        self.grayscale_checkbox = QCheckBox("Convert to Grayscale")
        self.grayscale_checkbox.setChecked(True)
        self.grayscale_checkbox.stateChanged.connect(self.on_grayscale_checkbox_changed)
        checkbox_layout.addWidget(self.grayscale_checkbox)
        
        # Add the checkbox layout to the right side
        right_side.addLayout(checkbox_layout)
        
        # Top and Bottom settings
        top_bottom = QHBoxLayout()
        self.crop_inputs = {}
        for key, label in [('top', 'Top:'), ('bottom', 'Bottom:')]:
            top_bottom.addWidget(QLabel(label))
            input_widget = QLineEdit(str(self.crop_settings[key]))
            input_widget.textChanged.connect(self.on_crop_setting_changed)
            self.crop_inputs[key] = input_widget
            top_bottom.addWidget(input_widget)
        right_side.addLayout(top_bottom)
        
        # Left and Right settings
        left_right = QHBoxLayout()
        for key, label in [('left', 'Left:'), ('right', 'Right:')]:
            left_right.addWidget(QLabel(label))
            input_widget = QLineEdit(str(self.crop_settings[key]))
            input_widget.textChanged.connect(self.on_crop_setting_changed)
            self.crop_inputs[key] = input_widget
            left_right.addWidget(input_widget)
        right_side.addLayout(left_right)
        
        # Add left and right sides to row3
        row3.addLayout(left_side)
        row3.addLayout(right_side)
        layout.addLayout(row3)
        
        # Fourth row: Combined Image and Batch settings
        row4 = QHBoxLayout()
        self.lbl_combined_preview = self.create_preview_label()
        row4.addWidget(QLabel("Combined:"))
        row4.addWidget(self.lbl_combined_preview)
        
        # Batch settings on the right
        batch_settings = QVBoxLayout()
        
        # Batch preview dropdown
        self.batch_preview_dropdown = QComboBox()
        self.batch_preview_dropdown.currentIndexChanged.connect(self.on_batch_preview_changed)
        batch_preview_layout = QHBoxLayout()
        batch_preview_layout.addWidget(QLabel("Preview Batch:"))
        batch_preview_layout.addWidget(self.batch_preview_dropdown)
        batch_settings.addLayout(batch_preview_layout)
        
        # Batch size
        batch_size_layout = QHBoxLayout()
        self.batch_size_input = QLineEdit(str(CONFIG['batch_size']))
        self.batch_size_input.textChanged.connect(self.on_batch_size_changed)
        batch_size_layout.addWidget(QLabel("Batch Size:"))
        batch_size_layout.addWidget(self.batch_size_input)
        batch_settings.addLayout(batch_size_layout)
        
        # Total height
        merged_img_height_layout = QHBoxLayout()
        self.merged_img_height_input = QLineEdit(str(CONFIG['merged_img_height']))
        self.merged_img_height_input.textChanged.connect(self.on_merged_img_height_changed)
        merged_img_height_layout.addWidget(QLabel("Merged Image Height:"))
        merged_img_height_layout.addWidget(self.merged_img_height_input)
        batch_settings.addLayout(merged_img_height_layout)
        
        row4.addLayout(batch_settings)
        layout.addLayout(row4)
        
        # Fifth row: Prompt
        row5 = QVBoxLayout()
        self.prompt_text = QTextEdit()
        self.prompt_text.setPlainText(CONFIG['prompt_text_base'])
        self.prompt_text.setMinimumHeight(100)
        row5.addWidget(self.prompt_text)
        layout.addLayout(row5)
        
        # Sixth row: Main column and Prompt buttons
        row6 = QHBoxLayout()
        self.main_column_input = QLineEdit(CONFIG['main_column'])
        self.main_column_input.textChanged.connect(self.on_main_column_changed)
        row6.addWidget(QLabel("Main column (column to display in Review Results):"))
        row6.addWidget(self.main_column_input)
        
        save_prompt_btn = QPushButton("Save Prompt")
        save_prompt_btn.clicked.connect(self.on_save_prompt)
        restore_prompt_btn = QPushButton("Restore Default Prompt")
        restore_prompt_btn.clicked.connect(self.restore_default_prompt)
        row6.addWidget(save_prompt_btn)
        row6.addWidget(restore_prompt_btn)
        layout.addLayout(row6)
        
        # Seventh row: Continue dropdown and Ask Gemini button
        row7 = QHBoxLayout()
        self.continue_dropdown = QComboBox()
        self.continue_dropdown.addItem("Start Over")
        row7.addWidget(self.continue_dropdown)
        
        # Add model dropdown
        self.model_dropdown = QComboBox()
        self.model_dropdown.currentIndexChanged.connect(self.on_model_changed)
        row7.addWidget(QLabel("Model:"))
        row7.addWidget(self.model_dropdown)
        
        ask_btn = QPushButton("Ask Gemini (Start Processing)")
        ask_btn.clicked.connect(self.on_ask_gemini)
        row7.addWidget(ask_btn)
        layout.addLayout(row7)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%")  # Show percentage
        self.progress_bar.valueChanged.connect(self.update_progress_text)
        layout.addWidget(self.progress_bar)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Process Images")

    def update_progress_text(self, value):
        self.progress_bar.setFormat(f"{value}%")

    def build_check_output_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Top controls
        controls = QHBoxLayout()
        
        self.csv_dropdown = QComboBox()
        self.csv_dropdown.addItem("(select CSV)")
        self.csv_dropdown.currentIndexChanged.connect(self.on_csv_dropdown_changed)
        controls.addWidget(self.csv_dropdown)
        
        self.suffix_mode_dropdown = QComboBox()
        self.suffix_mode_dropdown.addItems(["Just Wings", "Wing Clips"])
        self.suffix_mode_dropdown.currentIndexChanged.connect(self.on_suffix_mode_changed)
        controls.addWidget(self.suffix_mode_dropdown)
        
        self.show_dupes_chk = QCheckBox("Show duplicates only")
        self.show_dupes_chk.stateChanged.connect(self.update_display)
        controls.addWidget(self.show_dupes_chk)
        
        for btn_text, handler in [
            ("Open CSV", self.on_open_csv),
            ("Recalculate Final Name", self.on_recalc_final_name),
            ("Rename Files", self.on_rename_files),
            ("Restore Original", self.on_restore_original_filenames)
        ]:
            btn = QPushButton(btn_text)
            btn.clicked.connect(handler)
            controls.addWidget(btn)
            
        layout.addLayout(controls)
        
        # Scroll area for grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container_widget = QWidget()
        self.grid_layout = QGridLayout(self.container_widget)
        self.scroll.setWidget(self.container_widget)
        layout.addWidget(self.scroll)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Review Results")

    def on_open_csv(self):
        current_csv = self.csv_dropdown.currentText()
        if current_csv and current_csv != "(select CSV)":
            csv_path = os.path.join(self.rename_files_dir, current_csv)
            if os.path.exists(csv_path):
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(csv_path))

    def create_preview_label(self):
        label = QLabel()
        # Set size to 16:9 aspect ratio (480x270) - larger size
        label.setFixedSize(480, 270)
        label.setStyleSheet("background-color: #aaa;")
        return label

    def create_clickable_label(self, image_path):
        label = QLabel()
        label.image_path = image_path
        label.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        label.installEventFilter(self)
        return label

    def get_image_path(self, original_path):
        if os.path.exists(original_path):
            return original_path
            
        # Try to find renamed file
        dir_path = os.path.dirname(original_path)
        
        # Find the row in current_df that matches this file
        matching_row = self.current_df[self.current_df['from'] == original_path]
        if not matching_row.empty:
            new_name = matching_row.iloc[0]['to']
            if new_name:
                renamed_path = os.path.join(dir_path, new_name)
                if os.path.exists(renamed_path):
                    return renamed_path
                    
        return original_path

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            if hasattr(obj, 'image_path'):
                image_path = self.get_image_path(obj.image_path)
                if os.path.exists(image_path):
                    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(image_path))
                else:
                    print(f"Error: Could not find image at: {image_path}")
                return True
        return super().eventFilter(obj, event)

    def pil_to_qimage(self, pil_img, w_target, h_target):
        # Convert to RGB if needed
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
            
        # Get image dimensions
        w, h = pil_img.size
        
        # Convert PIL image to bytes in the correct format for QImage
        # QImage expects RGB data in row-major order
        img_data = pil_img.tobytes("raw", "RGB")
        qimg = QtGui.QImage(img_data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        
        # Let Qt handle the aspect ratio scaling
        return qimg.scaled(w_target, h_target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def refresh_ui(self):
        self.populate_continue_dropdown()
        self.populate_checkoutput_csv_dropdown()
        
        # Only update image previews if we're on the first tab
        if self.active_tab == 0:
            self.populate_preview_dropdown()
            self.populate_batch_preview_dropdown()
            
        self.auto_select_latest_csv()

    def on_tab_changed(self, index):
        # Update active tab tracker
        self.active_tab = index
        
        # If switching to Process Images tab (index 0), reload API keys
        if index == 0:
            load_api_keys()
        
        if self.input_directory and os.path.isdir(self.input_directory):
            self.refresh_ui()

    def auto_select_latest_csv(self):
        if self.csv_dropdown.count() > 1:
            self.csv_dropdown.setCurrentIndex(1)

    def on_browse_directory(self):
        init_folder = self.parent_directory if self.parent_directory else os.path.expanduser("~")
        chosen = QFileDialog.getExistingDirectory(self, "Select Folder", init_folder)
        
        if chosen:
            self.input_directory = chosen
            self.parent_directory = os.path.dirname(chosen)
            self.lab_dir_path.setText(chosen)
            FileManager.save_last_dir(chosen)
            
            self.rename_files_dir = FileManager.ensure_rename_files_dir(chosen)
            self.refresh_ui()

    def on_rotate_jpgs(self):
        if not self.input_directory:
            QMessageBox.warning(self, "No folder", "Pick an input folder first.")
            return
            
        # Get all compressed image files
        image_files = [f for f in os.listdir(self.input_directory) 
                      if ImageExtensionHandler.is_compressed_image(os.path.splitext(f)[1])]
        
        if not image_files:
            QMessageBox.information(self, "No images", "No supported image files found in this folder.")
            return
            
        self.progress_bar.setValue(0)
        total_files = len(image_files)
        rotation_degs = self.rotation_dropdown.currentData()
        
        deg_to_orientation = {0: 1, 90: 6, 180: 3, 270: 8}
        target_orientation = deg_to_orientation.get(rotation_degs, 1)
        
        for i, filename in enumerate(image_files):
            img_path = os.path.join(self.input_directory, filename)
            
            try:
                with Image.open(img_path) as img:
                    # Check if it's a PNG file
                    if filename.lower().endswith('.png'):
                        # Rotate the actual image data
                        rotated_img = img.rotate(rotation_degs, expand=True)
                        rotated_img.save(img_path, 'PNG')
                    else:
                        # For JPEG files, modify EXIF data
                        if 'exif' in img.info:
                            exif_dict = piexif.load(img.info['exif'])
                            exif_dict['0th'][piexif.ImageIFD.Orientation] = target_orientation
                            piexif.insert(piexif.dump(exif_dict), img_path)
                            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                
            self.progress_bar.setValue(int((i + 1) / total_files * 100))
            QApplication.processEvents()
            
        QMessageBox.information(self, "Done", "Image rotation completed.")

    def on_ask_gemini(self):
        if not self.input_directory:
            print("No folder selected. Please pick an input folder first.")
            QMessageBox.warning(self, "No folder", "Pick an input folder first.")
            return
            
        choice = self.continue_dropdown.currentText()
        start_batch = 1
        total_batches = 0
        df = pd.DataFrame()
        
        print("\n=== Starting Gemini Processing ===")
        print(f"Input directory: {self.input_directory}")
        print(f"Using model: {self.model_dropdown.currentText()}")
        
        if choice.startswith("Continue from"):
            partial_csv_name = choice.replace("Continue from ", "").strip()
            partial_csv_path = os.path.join(self.rename_files_dir, partial_csv_name)
            
            if os.path.exists(partial_csv_path):
                print(f"Continuing from {partial_csv_name}")
                df = pd.read_csv(partial_csv_path)
                m = re.search(r"_b(\d+)of(\d+)\.csv", partial_csv_name)
                if m:
                    start_batch = int(m.group(1)) + 1
                    total_batches = int(m.group(2))
                    print(f"Resuming from batch {start_batch} of {total_batches}")
            else:
                print(f"Partial CSV not found: {partial_csv_path}. Starting fresh.")
                choice = "Start Over"
                
        if choice == "Start Over":
            print("Starting fresh run.")
            # Get all standard image files for processing
            all_files = FileManager.get_all_files(self.input_directory, [".JPG", ".JPEG", ".PNG"])
            if not all_files:
                print("No standard image files (JPG/JPEG/PNG) found in the input folder.")
                QMessageBox.warning(self, "No files", "No standard image files (JPG/JPEG/PNG) found in the input folder.")
                return
                
            print(f"Found {len(all_files)} standard image files to process")
            
            # Get all raw files for reference
            raw_files = [f for f in os.listdir(self.input_directory) 
                        if ImageExtensionHandler.is_raw_image(os.path.splitext(f)[1])]
            if raw_files:
                print(f"Found {len(raw_files)} raw files (CR2/ORF) - these will be included in renaming")
            
            df = pd.DataFrame({
                'from': all_files,
                'photo_ID': range(1, len(all_files) + 1),
                'CAM': '',
                'note': '',
                'skip': '',
                'co': ''
            })
            
        if total_batches == 0:
            total_batches = (len(df) // CONFIG['batch_size']) + (1 if len(df) % CONFIG['batch_size'] else 0)
            print(f"Will process {len(df)} files in {total_batches} batches of {CONFIG['batch_size']}")
            
        self.start_gemini_processing(df, start_batch, total_batches)

    def on_csv_dropdown_changed(self, idx):
        if idx < 1:
            return
            
        csv_name = self.csv_dropdown.itemText(idx)
        path = os.path.join(self.rename_files_dir, csv_name)
        
        if os.path.exists(path):
            try:
                # Read CSV and handle NaN values
                df = pd.read_csv(path)
                
                # Fill NaN values with empty strings for all string columns
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna('')
                
                # Ensure skip column exists and is properly formatted if it exists
                if 'skip' in df.columns:
                    df['skip'] = df['skip'].apply(lambda x: 'x' if str(x).lower() == 'x' else '')
                    
                # Ensure 'to' column exists 
                if 'to' not in df.columns:
                    df['to'] = ''
                    
                # Check if the main column exists, if not add a warning
                main_column = CONFIG['main_column']
                if main_column not in df.columns:
                    print(f"Warning: Main column '{main_column}' not found in CSV")
                        
                self.current_df = df
                self.update_display()
            except Exception as e:
                print(f"Error loading CSV: {str(e)}")
                QMessageBox.warning(self, "Load Error", f"Error loading CSV: {str(e)}")

    def on_suffix_mode_changed(self, index):
        self.suffix_mode = 'wings' if index == 0 else 'clips'
        self.update_display()

    def on_recalc_final_name(self):
        self.collect_ui_changes_into_df()
        self.current_df = NameCalculator.recalc_final_names(
            self.current_df, self.suffix_mode, CONFIG['main_column'])
        self.update_display()
        QMessageBox.information(self, "Done", "Recalculated final names.")

    def on_rename_files(self):
        if self.current_df.empty:
            print("No CSV loaded. Cannot rename files.")
            QMessageBox.warning(self, "No data", "No CSV loaded.")
            return
            
        # Check if 'to' column is empty or missing
        if 'to' not in self.current_df.columns or self.current_df['to'].isna().all() or (self.current_df['to'] == '').all():
            result = QMessageBox.question(
                self, 
                "Missing Target Names", 
                "No target filenames found. Do you want to recalculate them first?",
                QMessageBox.Yes | QMessageBox.No
            )
            if result == QMessageBox.Yes:
                self.on_recalc_final_name()
            else:
                return
            
        print("\n=== Starting File Rename ===")
        self.collect_ui_changes_into_df()
        
        # Make skip column handling optional
        if 'skip' in self.current_df.columns:
            non_skipped_df = self.current_df[self.current_df['skip'] != 'x']
        else:
            non_skipped_df = self.current_df[self.current_df['to'] != '']
            
        log_entries = []
        renamed_count = 0
        
        print(f"Found {len(non_skipped_df)} files to rename")
        
        # Save the current state of the DataFrame with 'to' column before renaming
        current_csv = self.csv_dropdown.currentText()
        if current_csv and current_csv != "(select CSV)":
            csv_path = os.path.join(self.rename_files_dir, current_csv)
            print(f"Saving current state to: {csv_path}")
            self.current_df.to_csv(csv_path, index=False)
        
        for _, row in non_skipped_df.iterrows():
            if not row['to']:
                continue
                
            src_file = row['from']
            if os.path.exists(src_file):
                # Get the original extension with its case
                src_ext = os.path.splitext(src_file)[1]
                base_name = os.path.splitext(row['to'])[0]
                
                # Use the original file extension with its case
                dst_file = os.path.join(os.path.dirname(src_file), base_name + src_ext)
                
                try:
                    print(f"Renaming: {os.path.basename(src_file)} -> {os.path.basename(dst_file)}")
                    os.rename(src_file, dst_file)
                    log_entries.append({'original': src_file, 'renamed': dst_file})
                    renamed_count += 1
                    
                    # Check for corresponding raw file
                    src_base = os.path.splitext(src_file)[0]
                    for raw_ext in CONFIG['raw_exts']:
                        raw_file = src_base + raw_ext
                        if os.path.exists(raw_file):
                            # Preserve the case of the raw extension
                            raw_dst = os.path.join(os.path.dirname(raw_file), base_name + raw_ext)
                            print(f"Renaming raw: {os.path.basename(raw_file)} -> {os.path.basename(raw_dst)}")
                            os.rename(raw_file, raw_dst)
                            log_entries.append({'original': raw_file, 'renamed': raw_dst})
                            renamed_count += 1
                            
                except Exception as e:
                    print(f"Error renaming {src_file}: {e}")
                    
        if log_entries:
            log_path = os.path.join(self.rename_files_dir, "rename_log.csv")
            old_df = pd.DataFrame()
            if os.path.exists(log_path):
                old_df = pd.read_csv(log_path)
            pd.concat([old_df, pd.DataFrame(log_entries)], ignore_index=True).to_csv(log_path, index=False)
            print(f"\n=== Rename Completed ===")
            print(f"Renamed {renamed_count} files")
            print(f"Log saved to: {log_path}")
            QMessageBox.information(self, "Renamed", f"{renamed_count} files renamed.\nLog -> {log_path}")
        else:
            print("\nNo files were renamed (all skipped or missing)")
            QMessageBox.information(self, "No rename", "No files renamed (all skipped or missing).")

    def on_restore_original_filenames(self):
        if not self.rename_files_dir:
            print("No rename_files directory found")
            return
            
        log_path = os.path.join(self.rename_files_dir, "rename_log.csv")
        if not os.path.exists(log_path):
            print("rename_log.csv not found")
            QMessageBox.warning(self, "No log", "rename_log.csv not found.")
            return
            
        print("\n=== Starting File Restore ===")
        log_df = pd.read_csv(log_path)
        restored_count = 0
        errors = []
        
        print(f"Found {len(log_df)} files to restore")
        
        for _, row in log_df.iterrows():
            if os.path.exists(row['renamed']):
                try:
                    print(f"Restoring: {os.path.basename(row['renamed'])} -> {os.path.basename(row['original'])}")
                    os.rename(row['renamed'], row['original'])
                    restored_count += 1
                except Exception as e:
                    error_msg = f"Error restoring {row['renamed']}: {e}"
                    print(error_msg)
                    errors.append(error_msg)
                    
        if errors:
            error_msg = "\n".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more errors"
            print("\nRestore Errors:")
            print(error_msg)
            QMessageBox.warning(self, "Restore Errors", error_msg)
            
        print(f"\n=== Restore Completed ===")
        print(f"Restored {restored_count} files to original names")
        QMessageBox.information(self, "Restored", f"{restored_count} files restored to original names.")

    def update_display(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self.current_df.empty:
            return

        main_column = CONFIG['main_column']
        if main_column not in self.current_df.columns:
            msg_label = QLabel(f"Column '{main_column}' not found in data. Please select a valid column.")
            msg_label.setStyleSheet("color: red; font-weight: bold;")
            self.grid_layout.addWidget(msg_label, 0, 0)
            return

        df_disp = self.current_df.copy()
        if self.show_dupes_chk.isChecked():
            value_counts = df_disp[main_column].value_counts()
            non_pair_values = value_counts[value_counts != 2].index
            df_disp = df_disp[df_disp[main_column].isin(non_pair_values)]
            
            if df_disp.empty:
                msg_label = QLabel(f"All {main_column} values appear exactly twice (normal pairs)")
                msg_label.setStyleSheet("color: blue; font-weight: bold;")
                self.grid_layout.addWidget(msg_label, 0, 0)
                return

        self.ui_widgets = []
        row_idx = 0
        col_idx = 0
        cols = 2

        current_csv = self.csv_dropdown.currentText()
        csv_path = os.path.join(self.rename_files_dir, current_csv)
        
        # Clear previous image labels
        self.image_labels.clear()
        
        # Collect image paths to load
        image_paths_to_load = []

        for i, rowdata in df_disp.iterrows():
            container = QWidget()
            container.setMinimumWidth(400)
            vlay = QVBoxLayout(container)
            vlay.setSpacing(10)

            # Get value from main column and ensure from_path is set
            main_value = str(rowdata.get(main_column, ''))
            from_path = rowdata.get('from', '')
            
            # Show image if we have a path, regardless of main_value
            if from_path:
                # Count occurrences in filtered display and full dataframe
                if main_value:  # Only show warning if we have a main_value
                    disp_count = len(df_disp[df_disp[main_column] == main_value])
                    total_count = len(self.current_df[self.current_df[main_column] == main_value])
                    if total_count != 2:
                        count_label = QLabel(f"Warning: {main_column} '{main_value}' appears {total_count} times")
                        count_label.setStyleSheet("color: red; font-weight: bold;")
                        vlay.addWidget(count_label)
                
                # Get the correct image path (either original or renamed)
                image_path = self.get_image_path(from_path)
                if os.path.exists(image_path):
                    # Create placeholder label
                    lbl_img = self.create_clickable_label(image_path)
                    lbl_img.setText("Loading...")
                    lbl_img.setAlignment(QtCore.Qt.AlignCenter)
                    vlay.addWidget(lbl_img, alignment=QtCore.Qt.AlignHCenter)
                    
                    # Store label reference
                    self.image_labels[image_path] = lbl_img
                    image_paths_to_load.append(image_path)
                else:
                    vlay.addWidget(QLabel("Image not found"))

            # Only show from path if it exists
            if from_path:
                lbl_from = QLabel(f"From: {os.path.basename(from_path)}")
                lbl_from.setStyleSheet("color: blue; font-weight: bold;")
                lbl_from.setWordWrap(True)
                vlay.addWidget(lbl_from)

            # Dynamic field for main column value
            h_main = QHBoxLayout()
            l_main = QLabel(f"{main_column}:")
            e_main = QLineEdit(str(rowdata.get(main_column,'')))
            e_main.textChanged.connect(
                lambda text, idx=i, col=main_column, path=csv_path: self.on_column_value_changed(text, idx, col, path)
            )
            h_main.addWidget(l_main)
            h_main.addWidget(e_main)
            vlay.addLayout(h_main)

            h_to = QHBoxLayout()
            l_to = QLabel("To:")
            e_to = QLineEdit(str(rowdata.get('to','')))
            h_to.addWidget(l_to)
            h_to.addWidget(e_to)
            vlay.addLayout(h_to)

            # Only show crossed-out field if it exists in the dataframe
            if 'co' in self.current_df.columns:
                h_co = QHBoxLayout()
                l_co = QLabel("Crossed Out:")
                e_co = QLineEdit(str(rowdata.get('co','')))
                h_co.addWidget(l_co)
                h_co.addWidget(e_co)
                vlay.addLayout(h_co)
            else:
                e_co = None  # Placeholder so the dictionary below doesn't break

            # Keep the skip checkbox if it exists
            skip_chk = None
            if 'skip' in self.current_df.columns:
                skip_chk = QCheckBox("Skip")
                skip_chk.setChecked(str(rowdata.get('skip', '')).lower() == 'x')
                skip_chk.stateChanged.connect(
                    lambda state, idx=i, path=csv_path: self.on_skip_checkbox_changed(state, idx, path)
                )
                vlay.addWidget(skip_chk)

            # Create dynamic widget references
            widget_dict = {
                'df_index': i,
                'to_line': e_to
            }
            
            # Add main column line edit
            widget_dict[f'{main_column}_line'] = e_main
            
            # Add optional fields if they exist
            if 'co' in self.current_df.columns:
                widget_dict['co_line'] = e_co
            if 'skip' in self.current_df.columns:
                widget_dict['skip_chk'] = skip_chk
                
            self.ui_widgets.append(widget_dict)

            self.grid_layout.addWidget(container, row_idx, col_idx)
            col_idx += 1
            if col_idx >= cols:
                col_idx = 0
                row_idx += 1

        self.grid_layout.setRowStretch(row_idx, 1)
        self.grid_layout.setColumnStretch(cols, 1)
        
        # Start loading images in a thread
        if image_paths_to_load:
            self.start_image_loading(image_paths_to_load)

    def collect_ui_changes_into_df(self):
        main_column = CONFIG['main_column']
        for w in self.ui_widgets:
            idx = w['df_index']
            # Handle main column
            if f'{main_column}_line' in w:
                self.current_df.at[idx, main_column] = w[f'{main_column}_line'].text().strip()
            # Handle 'to' field
            if 'to_line' in w:
                self.current_df.at[idx, 'to'] = w['to_line'].text().strip()
            # Handle skip checkbox if it exists
            if 'skip_chk' in w:
                self.current_df.at[idx, 'skip'] = 'x' if w['skip_chk'].isChecked() else ''
            # Handle crossed out field if it exists
            if 'co_line' in w:
                self.current_df.at[idx, 'co'] = w['co_line'].text().strip()

    def on_column_value_changed(self, text, idx, column, csv_path):
        if self.current_df is not None and not self.current_df.empty and idx < len(self.current_df):
            self.current_df.at[idx, column] = text
            
    def on_skip_checkbox_changed(self, state, idx, csv_path):
        if self.current_df is not None and not self.current_df.empty and idx < len(self.current_df):
            self.current_df.at[idx, 'skip'] = 'x' if state == QtCore.Qt.Checked else ''

    def populate_continue_dropdown(self):
        self.continue_dropdown.clear()
        self.continue_dropdown.addItem("Start Over")

        files = os.listdir(self.rename_files_dir)
        partials = [(f, os.path.getmtime(os.path.join(self.rename_files_dir, f))) 
                    for f in files
                    if f.startswith(CONFIG['temp_output_prefix']) and f.endswith('.csv')]
        
        partials.sort(key=lambda x: x[1], reverse=True)
        
        if partials:
            for p, _ in partials:
                self.continue_dropdown.addItem(f"Continue from {p}")
            self.continue_dropdown.setCurrentIndex(1)

    def populate_checkoutput_csv_dropdown(self):
        if not self.rename_files_dir:
            return
            
        files = os.listdir(self.rename_files_dir)
        csv_files = [f for f in files if f.endswith('.csv') and (
            f.startswith(CONFIG['temp_output_prefix']) or
            CONFIG['output_prefix'] in f or
            f.startswith("checked_"))]
        
        csv_with_times = [(csv, max(os.path.getctime(os.path.join(self.rename_files_dir, csv)),
                                  os.path.getmtime(os.path.join(self.rename_files_dir, csv))))
                         for csv in csv_files]
        
        csv_with_times.sort(key=lambda x: x[1], reverse=True)
        
        self.csv_dropdown.clear()
        self.csv_dropdown.addItem("(select CSV)")
        
        for csv, _ in csv_with_times:
            self.csv_dropdown.addItem(csv)
        
        if self.csv_dropdown.count() > 1:
            self.csv_dropdown.setCurrentIndex(1)

    def populate_preview_dropdown(self):
        self.preview_dropdown.clear()
        if not self.input_directory:
            return
            
        # Get all compressed image files
        image_files = [f for f in os.listdir(self.input_directory) 
                      if ImageExtensionHandler.is_compressed_image(os.path.splitext(f)[1])]
        
        if not image_files:
            return
            
        for img in image_files:
            self.preview_dropdown.addItem(img)
            
        if self.preview_dropdown.count() > 0:
            self.preview_dropdown.setCurrentIndex(0)
            selected_img = self.preview_dropdown.currentText()
            img_path = os.path.join(self.input_directory, selected_img)
            self.show_jpg_previews(img_path)
            
        # Update batch preview dropdown
        self.populate_batch_preview_dropdown()

    def populate_batch_preview_dropdown(self):
        # Remember current selection if any
        current_selection = None
        if self.batch_preview_dropdown.count() > 0:
            current_selection = self.batch_preview_dropdown.currentData()
        
        # Temporarily disconnect signal to avoid triggering updates
        self.batch_preview_dropdown.blockSignals(True)
        self.batch_preview_dropdown.clear()
        
        if not self.input_directory:
            self.batch_preview_dropdown.blockSignals(False)
            return
            
        # Get all compressed image files
        image_files = [f for f in os.listdir(self.input_directory) 
                      if ImageExtensionHandler.is_compressed_image(os.path.splitext(f)[1])]
        
        if not image_files:
            self.batch_preview_dropdown.blockSignals(False)
            return
            
        # Calculate total number of batches
        total_batches = math.ceil(len(image_files) / CONFIG['batch_size'])
        
        # Add batch options
        for i in range(1, total_batches + 1):
            start_idx = (i - 1) * CONFIG['batch_size']
            end_idx = min(i * CONFIG['batch_size'], len(image_files))
            batch_text = f"Batch {i} ({start_idx + 1}-{end_idx})"
            self.batch_preview_dropdown.addItem(batch_text, i)
            
        # Try to restore previous selection
        if current_selection is not None and current_selection <= total_batches:
            index = self.batch_preview_dropdown.findData(current_selection)
            if index >= 0:
                self.batch_preview_dropdown.setCurrentIndex(index)
        elif self.batch_preview_dropdown.count() > 0:
            self.batch_preview_dropdown.setCurrentIndex(0)
            
        # Re-enable signals
        self.batch_preview_dropdown.blockSignals(False)
        
        # Only trigger an update if we have items and not already updating
        # AND if we're not restoring a previous selection
        if (self.batch_preview_dropdown.count() > 0 and 
            not self.combined_preview_updating and 
            current_selection is None):
            self.update_combined_preview()

    def on_batch_preview_changed(self, index):
        """When user selects a different batch to preview"""
        if index < 0 or not self.input_directory:
            return
            
        # Simply trigger an update - the _do_update_combined_preview method will 
        # get the current batch from the dropdown
        self.update_combined_preview()

    def show_jpg_previews(self, img_path):
        try:
            img_pil = Image.open(img_path)
            
            # Show original
            pil_before = ImageProcessor.fix_orientation(img_pil)
            qimg_before = self.pil_to_qimage(pil_before, 480, 270)
            self.lbl_before_preview.setPixmap(QtGui.QPixmap.fromImage(qimg_before))
            
            # Show zoomed - apply EXIF rotation first if not PNG
            if not img_path.lower().endswith('.png'):
                img_pil = ImageProcessor.fix_orientation(img_pil)
            if self.crop_settings['zoom']:
                pil_after = ImageProcessor.crop_image(img_pil, self.crop_settings)
            else:
                pil_after = ImageProcessor.fix_orientation(img_pil)
            qimg_after = self.pil_to_qimage(pil_after, 480, 270)
            self.lbl_after_preview.setPixmap(QtGui.QPixmap.fromImage(qimg_after))
            
            # Show rotated
            rotation_degs = self.rotation_dropdown.currentData()
            pil_rotated = ImageProcessor.fix_orientation(img_pil)
            pil_rotated = pil_rotated.rotate(rotation_degs, expand=True)
            qimg_rotated = self.pil_to_qimage(pil_rotated, 480, 270)
            self.lbl_rotated_preview.setPixmap(QtGui.QPixmap.fromImage(qimg_rotated))
            
        except Exception as e:
            QMessageBox.warning(self, "Error loading image", f"{e}")

    def on_preview_dropdown_changed(self, index):
        if index < 0 or not self.input_directory:
            return
            
        selected_img = self.preview_dropdown.currentText()
        img_path = os.path.join(self.input_directory, selected_img)
        self.show_jpg_previews(img_path)

    def update_jpg_preview(self):
        if self.preview_dropdown.count() > 0:
            self.on_preview_dropdown_changed(self.preview_dropdown.currentIndex())

    def populate_model_dropdown(self):
        self.model_dropdown.clear()
        for model in self.available_models:
            self.model_dropdown.addItem(model)
            
        # Set default to latest pro exp version
        default_model = next((m for m in self.available_models if 'pro-exp' in m), None)
        if default_model:
            index = self.model_dropdown.findText(default_model)
            if index >= 0:
                self.model_dropdown.setCurrentIndex(index)
                CONFIG['model_name'] = default_model

    def on_model_changed(self, index):
        if index >= 0:
            selected_model = self.model_dropdown.currentText()
            CONFIG['model_name'] = selected_model
            if hasattr(self, 'gemini_handler'):
                self.gemini_handler.update_model(selected_model)

    def load_prompt(self):
        prompt_file = os.path.join(os.path.dirname(__file__), 'prompt.txt')
        
        if os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    loaded_prompt = f.read()
                    # Check if the prompt already contains the sample JSON
                    if '{sample_json_output}' in loaded_prompt:
                        CONFIG['prompt_text_base'] = loaded_prompt.format(sample_json_output=SAMPLE_JSON)
                    else:
                        CONFIG['prompt_text_base'] = loaded_prompt
            except Exception as e:
                print(f"Error loading prompt.txt: {e}")
                # Use the default prompt as fallback
                CONFIG['prompt_text_base'] = DEFAULT_PROMPT.format(sample_json_output=SAMPLE_JSON)
        else:
            # Use the default prompt
            CONFIG['prompt_text_base'] = DEFAULT_PROMPT.format(sample_json_output=SAMPLE_JSON)

    def save_prompt(self):
        prompt_file = os.path.join(os.path.dirname(__file__), 'prompt.txt')
        try:
            # Save the unformatted prompt (with {sample_json_output} placeholder)
            prompt_to_save = CONFIG['prompt_text_base'].replace(SAMPLE_JSON, '{sample_json_output}')
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt_to_save)
            QMessageBox.information(self, "Success", "Prompt saved successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save prompt: {e}")

    def restore_default_prompt(self):
        # Reset to the original unformatted prompt
        CONFIG['prompt_text_base'] = DEFAULT_PROMPT.format(sample_json_output=SAMPLE_JSON)
        self.save_prompt()
        self.prompt_text.setPlainText(CONFIG['prompt_text_base'])
        QMessageBox.information(self, "Success", "Default prompt restored and saved.")

    def on_main_column_changed(self):
        CONFIG['main_column'] = self.main_column_input.text().strip()
        self.update_display()

    def on_batch_size_changed(self):
        try:
            new_batch_size = int(self.batch_size_input.text())
            if new_batch_size != CONFIG['batch_size']:
                CONFIG['batch_size'] = new_batch_size
                self.populate_batch_preview_dropdown()
                self.update_combined_preview()
        except ValueError:
            # If invalid input, just don't update the batch size
            pass

    def on_merged_img_height_changed(self):
        try:
            new_height = int(self.merged_img_height_input.text())
            if new_height != CONFIG['merged_img_height']:
                CONFIG['merged_img_height'] = new_height
                self.update_combined_preview()
        except ValueError:
            # If invalid input, just don't update the total height
            pass

    def on_zoom_checkbox_changed(self, state):
        self.crop_settings['zoom'] = state == QtCore.Qt.Checked
        # Refresh the preview if one is selected
        if self.preview_dropdown.count() > 0:
            self.on_preview_dropdown_changed(self.preview_dropdown.currentIndex())
        # Update UI for crop settings
        for widget in self.crop_inputs.values():
            widget.setEnabled(self.crop_settings['zoom'])
            
        # Update the merged preview if we're on the first tab
        if self.active_tab == 0:
            self.update_combined_preview()

    def on_crop_setting_changed(self):
        try:
            for key, input_widget in self.crop_inputs.items():
                value = float(input_widget.text())
                # For bottom and right, use 1 - value to crop from that side
                if key in ['bottom', 'right']:
                    self.crop_settings[key] = 1 - value
                else:
                    self.crop_settings[key] = value
                    
            if self.preview_dropdown.count() > 0:
                self.on_preview_dropdown_changed(self.preview_dropdown.currentIndex())
        except ValueError:
            # If invalid input, just don't update the crop settings
            pass

    def on_save_prompt(self):
        CONFIG['prompt_text_base'] = self.prompt_text.toPlainText()
        self.save_prompt()

    def populate_settings_preview_dropdown(self):
        self.preview_dropdown.clear()
        if not self.input_directory:
            return
            
        # Get all compressed image files
        image_files = [f for f in os.listdir(self.input_directory) 
                      if ImageExtensionHandler.is_compressed_image(os.path.splitext(f)[1])]
        
        if not image_files:
            return
            
        for img in image_files:
            self.preview_dropdown.addItem(img)
            
        if self.preview_dropdown.count() > 0:
            self.preview_dropdown.setCurrentIndex(0)
            selected_img = self.preview_dropdown.currentText()
            img_path = os.path.join(self.input_directory, selected_img)
            self.show_settings_previews(img_path)

    def show_settings_previews(self, img_path):
        try:
            image_path = self.get_image_path(img_path)
            img_pil = Image.open(image_path)
            
            # Show original
            pil_original = ImageProcessor.fix_orientation(img_pil)
            qimg_original = self.pil_to_qimage(pil_original, 480, 270)
            self.lbl_before_preview.setPixmap(QtGui.QPixmap.fromImage(qimg_original))
            
            # Show zoomed or unzoomed based on zoom setting
            if self.crop_settings['zoom']:
                pil_preview = ImageProcessor.crop_image(img_pil, self.crop_settings)
            else:
                pil_preview = ImageProcessor.fix_orientation(img_pil)
                
            qimg_preview = self.pil_to_qimage(pil_preview, 480, 270)
            self.lbl_after_preview.setPixmap(QtGui.QPixmap.fromImage(qimg_preview))
            
        except Exception as e:
            QMessageBox.warning(self, "Error loading image", f"{e}")

    def update_combined_preview(self):
        """Schedule a combined preview update if not already in progress and we're on the right tab"""
        # Only update if we're on the Process Images tab
        if self.active_tab != 0:
            return
            
        # If an update is already in progress, don't start another one
        if self.combined_preview_updating:
            return
            
        # Mark that we're starting an update
        self.combined_preview_updating = True
        
        # Directly do the update without timer delay
        self._do_update_combined_preview()

    def _do_update_combined_preview(self):
        """Actually perform the combined preview update"""
        try:
            if not self.input_directory:
                return
                
            # Get all compressed image files
            image_files = [f for f in os.listdir(self.input_directory) 
                          if ImageExtensionHandler.is_compressed_image(os.path.splitext(f)[1])]
            
            if not image_files:
                return
                
            # Get the selected batch from dropdown if available
            batch_num = 1  # Default to first batch
            if hasattr(self, 'batch_preview_dropdown') and self.batch_preview_dropdown.currentData():
                batch_num = self.batch_preview_dropdown.currentData()
                
            # Calculate indices for the selected batch
            batch_size = CONFIG['batch_size']
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, len(image_files))
            batch_images = image_files[start_idx:end_idx]
            
            # Process and merge images
            images = []
            for i, img_file in enumerate(batch_images):
                img_path = os.path.join(self.input_directory, img_file)
                try:
                    # Use the actual row number from the DataFrame
                    row_number = start_idx + i + 1
                    img = ImageProcessor.preprocess_image(img_path, row_number, self.crop_settings)
                    images.append(img)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    
            if images:
                merged = ImageProcessor.merge_images(images, merged_img_height=CONFIG['merged_img_height'])
                if merged:
                    # Calculate the path for the merged image
                    total_batches = math.ceil(len(image_files)/CONFIG['batch_size'])
                    merged_path = os.path.join(self.rename_files_dir, f"temp_merged_b{batch_num}of{total_batches}.jpg")
                    
                    # Save the merged image
                    try:
                        merged.save(merged_path)
                        print(f"Saved merged preview to: {merged_path}")
                    except Exception as e:
                        print(f"Error saving merged preview: {e}")
                        return
                    
                    # Convert to QImage and display
                    qimg = self.pil_to_qimage(merged, 480, 270)
                    self.lbl_combined_preview.setPixmap(QtGui.QPixmap.fromImage(qimg))
                    
                    # Make the combined preview clickable
                    self.lbl_combined_preview.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                    self.lbl_combined_preview.installEventFilter(self)
                    self.lbl_combined_preview.image_path = merged_path
        finally:
            # Always mark that we're done updating, even if there was an error
            self.combined_preview_updating = False

    def on_grayscale_checkbox_changed(self, state):
        self.crop_settings['grayscale'] = state == QtCore.Qt.Checked
        # Refresh the preview if one is selected
        if self.preview_dropdown.count() > 0:
            self.on_preview_dropdown_changed(self.preview_dropdown.currentIndex())
        # Update the merged preview if we're on the first tab
        if self.active_tab == 0:
            self.update_combined_preview()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 
