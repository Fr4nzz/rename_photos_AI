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
    QScrollArea, QLineEdit, QGridLayout, QMessageBox, QProgressBar
)
import google.generativeai as genai
from googleapiclient.errors import HttpError

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
                merged = ImageProcessor.merge_images(images, res_fact=8)
                
                if merged and not self.should_stop:
                    temp_file = f"temp_merged_b{bnum}of{self.total_batches}.jpg"
                    temp_path = os.path.join(self.rename_files_dir, temp_file)
                    merged.save(temp_path)
                    
                    response, _ = gemini_handler.send_request(temp_path)
                    resp_text = getattr(response, 'text', str(response))
                    
                    match = re.search(r'```json\s*\n([\s\S]*?)\n```', resp_text)
                    if match and not self.should_stop:
                        jdata = json.loads(match.group(1))
                        
                        for idx in batch_df.index:
                            photo_id = str(batch_df.at[idx, 'photo_ID'])
                            if photo_id in jdata:
                                self.df.at[idx, 'CAM'] = jdata[photo_id].get('CAM', 'MISSING_CAM')
                                self.df.at[idx, 'note'] = jdata[photo_id].get('n', '')
                                self.df.at[idx, 'skip'] = jdata[photo_id].get('skip', '')
                                self.df.at[idx, 'co'] = jdata[photo_id].get('co', '')
                            else:
                                self.df.at[idx, 'CAM'] = "MISSING_CAM"
                                self.df.at[idx, 'note'] = "(No JSON key found)"
                                self.df.at[idx, 'skip'] = ''
                                self.df.at[idx, 'co'] = ''
                                
                        partial_csv = f"{self.config['temp_output_prefix']}_b{bnum}of{self.total_batches}.csv"
                        self.df.to_csv(os.path.join(self.rename_files_dir, partial_csv), index=False)
                        
                        # Emit progress and batch completion
                        progress = int((bnum / self.total_batches) * 100)
                        self.progress.emit(progress)
                        self.batch_completed.emit(self.df.copy(), bnum, self.total_batches)
            
            if not self.should_stop:
                self.finished.emit()
                
        except Exception as e:
            self.error.emit(str(e))

class ImageLoadWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    image_loaded = QtCore.pyqtSignal(str, QtGui.QImage)
    error = QtCore.pyqtSignal(str, str)

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def process(self):
        for path in self.image_paths:
            if self.should_stop:
                break
                
            try:
                pil_img = Image.open(path)
                pil_img = ImageProcessor.crop_image(pil_img)
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

# Configuration
CONFIG = {
    'api_keys': [
        "AIzaSyCOxnlCgNWi0nnqXHvJqMdogtvy87EVXfw",
        "AIzaSyDY9C_bLMeicBNDPAMFt2BKMYliUigw-KQ",
        "AIzaSyCiINFgVZvx969F2Y6GjlkmazM6_Sqru_M",
        "AIzaSyAjxH4ISWKsYoEjKlJDke_KDnc6x5PZZ2w"
    ],
    'directory_path': '',
    'file_ext': [".JPG", ".CR2"],
    'batch_size': 9,
    'prompt_text_base': """Extract CAM (CAM07xxxx) and notes (n) from the image.
- 2 wing photos (dorsal and ventral) per individual (CAM) are arranged in a grid left to right, top to bottom.
- If no CAMID is visible or image should be skipped, set skip: 'x', else skip: ''
- If CAMID is crossed out, set 'co' to the crossed out CAMID and put the new CAMID in 'CAM'
- CAMIDs have no spaces, remember CAM format (CAM07xxxx)
- Use notes (n) to indicate anything unusual (e.g., repeated, rotated 90°, etc).
- Put skipped reason in notes 'n'
- Double-check numbers are correctly OCRed; consecutive photos might not have consecutive CAMs
- Return JSON as shown in example; always give all keys even if empty. Example:
{sample_json_output}
""",
    'temp_output_prefix': 'temp_output',
    'output_prefix': 'output',
}

SAMPLE_JSON = """{
  "1": {"CAM": "CAM074806", "co": "", "n": "", "skip": ""},
  "2": {"CAM": "CAM074806", "co": "", "n": "", "skip": ""},
  "3": {"CAM": "Empty", "co": "", "n": "CAM missing", "skip": "x"},
  "4": {"CAM": "CAM070555", "co": "CAM072554", "n": "", "skip": ""},
  "5": {"CAM": "CAM070545", "co": "CAM072554", "n": "", "skip": ""},
  "6": {"CAM": "CAM072190", "co": "", "n": "", "skip": ""},
  "7": {"CAM": "CAM072190", "co": "", "n": "", "skip": ""},
  "8": {"CAM": "CAM074749", "co": "", "n": "", "skip": ""},
  "9": {"CAM": "CAM074749", "co": "", "n": "", "skip": ""},
  "10": {"CAM": "CAM074749", "co": "", "n": "repeated", "skip": ""},
  "11": {"CAM": "CAM074541", "co": "", "n": "", "skip": ""},
  "12": {"CAM": "CAM074541", "co": "", "n": "", "skip": ""}
}"""

CONFIG['prompt_text_base'] = CONFIG['prompt_text_base'].format(sample_json_output=SAMPLE_JSON)

class ImageProcessor:
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
    def crop_image(img):
        img = ImageProcessor.fix_orientation(img)
        w, h = img.size
        crop_upper = int(h * 0.1)
        crop_lower = int(h * 0.5)
        return img.crop((0, crop_upper, w//2, crop_lower))

    @staticmethod
    def preprocess_image(image_path, label):
        img = Image.open(image_path)
        img = ImageProcessor.crop_image(img)
        img = ImageOps.grayscale(img)
        img = ImageOps.expand(img, border=10, fill='black')
        
        draw = ImageDraw.Draw(img)
        font_size = int(0.12 * img.height)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
            
        text = str(label)
        bbox = draw.textbbox((0, 0), text, font=font)
        x = (img.width - (bbox[2] - bbox[0])) // 2
        y = 10
        
        draw.text((x, y), text, fill='black', font=font)
        return img

    @staticmethod
    def merge_images(images, res_fact=8):
        if not images:
            return None
            
        n = len(images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n/cols)
        w, h = images[0].size
        
        blank = Image.new('RGB', (w, h), 'white')
        needed = rows * cols - n
        images_padded = images + [blank] * needed
        grid = Image.new('RGB', (cols * w, rows * h), 'white')
        
        for idx, img in enumerate(images_padded):
            r, c = divmod(idx, cols)
            grid.paste(img, (c * w, r * h))
            
        new_size = (grid.width // res_fact, grid.height // res_fact)
        return grid.resize(new_size, Image.LANCZOS)

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
                        self._switch_api_key()
                        time.sleep(min(2 ** attempt + random.random(), 10))  # Cap at 10 seconds
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
                            self._switch_api_key()
                            time.sleep(min(2 ** attempt + random.random(), 10))
                            continue
                    raise

            except HttpError as e:
                last_error = e
                print(f"\nHTTP Error: {str(e)}")
                if e.resp.status in [429, 500, 503] or "quota" in str(e).lower():
                    if self.retry_count < self.max_retries:
                        self._switch_api_key()
                        time.sleep(min(2 ** attempt + random.random(), 10))
                        continue
                raise

            except Exception as e:
                last_error = e
                print(f"\nUnexpected error: {str(e)}")
                if self.retry_count < self.max_retries:
                    self._switch_api_key()
                    time.sleep(min(2 ** attempt + random.random(), 10))
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
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if any(f.upper().endswith(ext.upper()) for ext in valid_exts)]

class NameCalculator:
    @staticmethod
    def calculate_suffixes_for_cam(cam_df, suffix_mode):
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
    def recalc_final_names(df, suffix_mode, ext_list):
        if df.empty:
            return df
            
        df = df.copy()
        df['suffix'] = ''
        non_skipped_mask = df['skip'] != 'x'
        
        for cam_name, group in df[non_skipped_mask].groupby('CAM', group_keys=False):
            if not cam_name:
                continue
            suffixes = NameCalculator.calculate_suffixes_for_cam(group, suffix_mode)
            for idx, suffix in zip(group.index, suffixes):
                df.at[idx, 'suffix'] = suffix
        
        main_ext = ext_list[0] if ext_list else ".JPG"
        df.loc[non_skipped_mask, 'to'] = (
            df.loc[non_skipped_mask, 'CAM'].astype(str) + 
            df.loc[non_skipped_mask, 'suffix'].astype(str)
        )
        df.loc[~non_skipped_mask, 'to'] = ''
        
        counts = {}
        new_names = []
        for val in df['to']:
            if not val:
                new_names.append('')
                continue
            counts[val] = counts.get(val, 0) + 1
            new_names.append(val + (f"_{counts[val]}" if counts[val] > 1 else ""))
        
        df['to'] = [nm + main_ext if nm else '' for nm in new_names]
        return df

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wing Photos - CAM ID Processor")
        self.resize(1400, 800)
        
        self.input_directory, self.parent_directory = FileManager.load_last_dir()
        self.rename_files_dir = ""
        if self.input_directory and os.path.isdir(self.input_directory):
            self.rename_files_dir = FileManager.ensure_rename_files_dir(self.input_directory)
            
        self.current_df = pd.DataFrame()
        self.suffix_mode = 'wings'
        self.ext_list = [".JPG", ".CR2"]
        
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
        
        if self.input_directory and os.path.isdir(self.input_directory):
            self.lab_dir_path.setText(self.input_directory)
            self.refresh_ui()

    def setup_ui(self):
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        self.build_ask_gemini_tab()
        self.build_check_output_tab()
        
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

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
        self.gemini_worker.progress.connect(self.gemini_progress.setValue)
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
        worker = ImageLoadWorker(image_paths)
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
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.lab_dir_path = QLabel("(none)")
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.on_browse_directory)
        dir_layout.addWidget(QLabel("Input Folder:"))
        dir_layout.addWidget(self.lab_dir_path)
        dir_layout.addWidget(btn_browse)
        layout.addLayout(dir_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        self.model_dropdown = QComboBox()
        self.populate_model_dropdown()
        self.model_dropdown.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(QLabel("Gemini Model:"))
        model_layout.addWidget(self.model_dropdown)
        layout.addLayout(model_layout)
        
        # Rotation settings
        rotation_layout = QHBoxLayout()
        self.rotation_dropdown = QComboBox()
        for deg, text in [(0, "0° (No rotation)"), (90, "90° anticlockwise"),
                         (180, "180° anticlockwise"), (270, "270° anticlockwise")]:
            self.rotation_dropdown.addItem(text, deg)
        self.rotation_dropdown.setCurrentIndex(2)
        self.rotation_dropdown.currentIndexChanged.connect(self.update_jpg_preview)
        rotation_layout.addWidget(QLabel("Rotate JPG:"))
        rotation_layout.addWidget(self.rotation_dropdown)
        layout.addLayout(rotation_layout)
        
        # Preview selection
        preview_layout = QHBoxLayout()
        self.preview_dropdown = QComboBox()
        self.preview_dropdown.currentIndexChanged.connect(self.on_preview_dropdown_changed)
        preview_layout.addWidget(QLabel("Preview Image:"))
        preview_layout.addWidget(self.preview_dropdown)
        layout.addLayout(preview_layout)
        
        # Preview images
        preview_images = QHBoxLayout()
        self.lbl_before_preview = self.create_preview_label()
        self.lbl_after_preview = self.create_preview_label()
        preview_images.addWidget(self.lbl_before_preview)
        preview_images.addWidget(self.lbl_after_preview)
        layout.addLayout(preview_images)
        
        # Progress bar
        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 100)
        layout.addWidget(self.export_progress)
        
        # Rotate button
        btn_rotate = QPushButton("Rotate JPGs")
        btn_rotate.clicked.connect(self.on_rotate_jpgs)
        layout.addWidget(btn_rotate)
        
        # Continue dropdown
        self.continue_dropdown = QComboBox()
        self.continue_dropdown.addItem("Start Over")
        layout.addWidget(self.continue_dropdown)
        
        # Gemini progress
        self.gemini_progress = QProgressBar()
        self.gemini_progress.setRange(0, 100)
        layout.addWidget(self.gemini_progress)
        
        # Ask Gemini button
        ask_btn = QPushButton("Ask Gemini (Start Processing)")
        ask_btn.clicked.connect(self.on_ask_gemini)
        layout.addWidget(ask_btn)
        
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Ask Gemini")

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
        
        self.ext_line_edit = QLineEdit()
        self.ext_line_edit.setPlaceholderText(".JPG, .CR2")
        controls.addWidget(self.ext_line_edit)
        
        self.show_dupes_chk = QCheckBox("Show duplicates only")
        self.show_dupes_chk.stateChanged.connect(self.update_display)
        controls.addWidget(self.show_dupes_chk)
        
        for btn_text, handler in [
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
        self.tab_widget.addTab(tab, "Check Output")

    def create_preview_label(self):
        label = QLabel()
        label.setFixedSize(200, 200)
        label.setStyleSheet("background-color: #aaa;")
        return label

    def create_clickable_label(self, image_path):
        label = QLabel()
        label.image_path = image_path
        label.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        label.installEventFilter(self)
        return label

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            if hasattr(obj, 'image_path'):
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(obj.image_path))
                return True
        return super().eventFilter(obj, event)

    def pil_to_qimage(self, pil_img, w_target, h_target):
        pil_img = pil_img.copy()
        pil_img.thumbnail((w_target, h_target), Image.LANCZOS)
        data = pil_img.tobytes("raw", "RGB")
        return QtGui.QImage(data, pil_img.width, pil_img.height, QtGui.QImage.Format_RGB888)

    def refresh_ui(self):
        self.populate_continue_dropdown()
        self.populate_checkoutput_csv_dropdown()
        self.populate_preview_dropdown()
        self.auto_select_latest_csv()

    def on_tab_changed(self, index):
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
            
        jpg_files = [f for f in os.listdir(self.input_directory) if f.lower().endswith('.jpg')]
        if not jpg_files:
            QMessageBox.information(self, "No JPG", "No .JPG files found in this folder.")
            return
            
        self.export_progress.setValue(0)
        total_files = len(jpg_files)
        rotation_degs = self.rotation_dropdown.currentData()
        
        deg_to_orientation = {0: 1, 90: 6, 180: 3, 270: 8}
        target_orientation = deg_to_orientation.get(rotation_degs, 1)
        
        for i, filename in enumerate(jpg_files):
            jpg_path = os.path.join(self.input_directory, filename)
            
            try:
                with Image.open(jpg_path) as img:
                    if 'exif' in img.info:
                        exif_dict = piexif.load(img.info['exif'])
                        exif_dict['0th'][piexif.ImageIFD.Orientation] = target_orientation
                        piexif.insert(piexif.dump(exif_dict), jpg_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
            self.export_progress.setValue(int((i + 1) / total_files * 100))
            QApplication.processEvents()
            
        QMessageBox.information(self, "Done", "JPG rotation completed.")

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
            all_files = FileManager.get_all_files(self.input_directory, [".JPG", ".JPEG"])
            if not all_files:
                print("No JPG files found in the input folder.")
                QMessageBox.warning(self, "No files", "No JPG files found in the input folder.")
                return
                
            print(f"Found {len(all_files)} JPG files to process")
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
                
                # Fill NaN values with empty strings for text columns
                text_columns = ['from', 'CAM', 'to', 'skip', 'note', 'co']
                for col in text_columns:
                    if col in df.columns:
                        df[col] = df[col].fillna('')
                
                # Ensure skip column exists and is properly formatted
                if 'skip' not in df.columns:
                    df['skip'] = ''
                else:
                    df['skip'] = df['skip'].apply(lambda x: 'x' if str(x).lower() == 'x' else '')
                    
                # Ensure required columns exist
                for col in ['from', 'CAM', 'to']:
                    if col not in df.columns:
                        df[col] = ''
                        
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
        
        exts_txt = self.ext_line_edit.text().strip()
        if exts_txt:
            self.ext_list = [x.strip() for x in exts_txt.split(',') if x.strip()]
            
        self.current_df = NameCalculator.recalc_final_names(
            self.current_df, self.suffix_mode, self.ext_list)
        self.update_display()
        QMessageBox.information(self, "Done", "Recalculated final names.")

    def on_rename_files(self):
        if self.current_df.empty:
            print("No CSV loaded. Cannot rename files.")
            QMessageBox.warning(self, "No data", "No CSV loaded.")
            return
            
        print("\n=== Starting File Rename ===")
        self.collect_ui_changes_into_df()
        non_skipped_df = self.current_df[self.current_df['skip'] != 'x']
        
        cts = non_skipped_df['to'].value_counts()
        if not cts[cts > 1].empty:
            print("Cannot rename due to duplicate target names")
            QMessageBox.warning(self, "Duplicates", "Cannot rename because of duplicates")
            return
            
        log_entries = []
        renamed_count = 0
        
        print(f"Found {len(non_skipped_df)} files to rename")
        
        for _, row in non_skipped_df.iterrows():
            if not row['to']:
                continue
                
            src_jpg = row['from']
            if os.path.exists(src_jpg):
                base_name = os.path.splitext(row['to'])[0]
                dst_jpg = os.path.join(os.path.dirname(src_jpg), base_name + ".JPG")
                
                try:
                    print(f"Renaming: {os.path.basename(src_jpg)} -> {os.path.basename(dst_jpg)}")
                    os.rename(src_jpg, dst_jpg)
                    log_entries.append({'original': src_jpg, 'renamed': dst_jpg})
                    renamed_count += 1
                    
                    src_cr2 = os.path.splitext(src_jpg)[0] + ".CR2"
                    if os.path.exists(src_cr2):
                        dst_cr2 = os.path.join(os.path.dirname(src_cr2), base_name + ".CR2")
                        print(f"Renaming CR2: {os.path.basename(src_cr2)} -> {os.path.basename(dst_cr2)}")
                        os.rename(src_cr2, dst_cr2)
                        log_entries.append({'original': src_cr2, 'renamed': dst_cr2})
                        renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {src_jpg}: {e}")
                    
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

        df_disp = self.current_df.copy()
        if self.show_dupes_chk.isChecked():
            cam_counts = df_disp['CAM'].value_counts()
            non_pair_cams = cam_counts[cam_counts != 2].index
            df_disp = df_disp[df_disp['CAM'].isin(non_pair_cams)]
            
            if df_disp.empty:
                msg_label = QLabel("All CAMs appear exactly twice (normal dorsal/ventral pairs)")
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

            cam_value = str(rowdata.get('CAM', ''))
            if cam_value:
                cam_count = len(df_disp[df_disp['CAM'] == cam_value])
                total_count = len(self.current_df[self.current_df['CAM'] == cam_value])
                if total_count != 2:
                    count_label = QLabel(f"Warning: CAM appears {total_count} times")
                    count_label.setStyleSheet("color: red; font-weight: bold;")
                    vlay.addWidget(count_label)
                
                from_path = rowdata['from']
                if os.path.exists(from_path):
                    # Create placeholder label
                    lbl_img = self.create_clickable_label(from_path)
                    lbl_img.setText("Loading...")
                    lbl_img.setAlignment(QtCore.Qt.AlignCenter)
                    vlay.addWidget(lbl_img, alignment=QtCore.Qt.AlignHCenter)
                    
                    # Store label reference
                    self.image_labels[from_path] = lbl_img
                    image_paths_to_load.append(from_path)
                else:
                    vlay.addWidget(QLabel("Image not found"))

            lbl_from = QLabel(f"From: {os.path.basename(from_path)}")
            lbl_from.setStyleSheet("color: blue; font-weight: bold;")
            lbl_from.setWordWrap(True)
            vlay.addWidget(lbl_from)

            h_cam = QHBoxLayout()
            l_cam = QLabel("CAM:")
            e_cam = QLineEdit(str(rowdata.get('CAM','')))
            e_cam.textChanged.connect(
                lambda text, idx=i, path=csv_path: self.on_cam_value_changed(text, idx, path)
            )
            h_cam.addWidget(l_cam)
            h_cam.addWidget(e_cam)
            vlay.addLayout(h_cam)

            h_to = QHBoxLayout()
            l_to = QLabel("To:")
            e_to = QLineEdit(str(rowdata.get('to','')))
            h_to.addWidget(l_to)
            h_to.addWidget(e_to)
            vlay.addLayout(h_to)

            h_co = QHBoxLayout()
            l_co = QLabel("Crossed Out:")
            e_co = QLineEdit(str(rowdata.get('co','')))
            h_co.addWidget(l_co)
            h_co.addWidget(e_co)
            vlay.addLayout(h_co)

            skip_chk = QCheckBox("Skip")
            skip_chk.setChecked(str(rowdata.get('skip', '')).lower() == 'x')
            skip_chk.stateChanged.connect(
                lambda state, idx=i, path=csv_path: self.on_skip_checkbox_changed(state, idx, path)
            )
            vlay.addWidget(skip_chk)

            self.ui_widgets.append({
                'df_index': i,
                'cam_line': e_cam,
                'to_line': e_to,
                'skip_chk': skip_chk,
                'co_line': e_co
            })

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
        for w in self.ui_widgets:
            idx = w['df_index']
            self.current_df.at[idx, 'CAM'] = w['cam_line'].text().strip()
            self.current_df.at[idx, 'to'] = w['to_line'].text().strip()
            self.current_df.at[idx, 'skip'] = 'x' if w['skip_chk'].isChecked() else ''
            self.current_df.at[idx, 'co'] = w['co_line'].text().strip()

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
            
        jpg_files = [f for f in os.listdir(self.input_directory) if f.lower().endswith('.jpg')]
        if not jpg_files:
            return
            
        for jpg in jpg_files:
            self.preview_dropdown.addItem(jpg)
            
        if self.preview_dropdown.count() > 0:
            self.preview_dropdown.setCurrentIndex(0)
            selected_jpg = self.preview_dropdown.currentText()
            jpg_path = os.path.join(self.input_directory, selected_jpg)
            self.show_jpg_previews(jpg_path)

    def show_jpg_previews(self, jpg_path):
        try:
            img_pil = Image.open(jpg_path)
            
            pil_before = ImageProcessor.fix_orientation(img_pil)
            pil_before = pil_before.convert('RGB')
            qimg_before = self.pil_to_qimage(pil_before, 200, 200)
            self.lbl_before_preview.setPixmap(QtGui.QPixmap.fromImage(qimg_before))
            
            rotation_degs = self.rotation_dropdown.currentData()
            pil_after = ImageProcessor.fix_orientation(img_pil)
            pil_after = pil_after.rotate(rotation_degs, expand=True)
            pil_after = pil_after.convert('RGB')
            qimg_after = self.pil_to_qimage(pil_after, 200, 200)
            self.lbl_after_preview.setPixmap(QtGui.QPixmap.fromImage(qimg_after))
            
        except Exception as e:
            QMessageBox.warning(self, "Error loading JPG", f"{e}")

    def on_preview_dropdown_changed(self, index):
        if index < 0 or not self.input_directory:
            return
            
        selected_jpg = self.preview_dropdown.currentText()
        jpg_path = os.path.join(self.input_directory, selected_jpg)
        self.show_jpg_previews(jpg_path)

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

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 
