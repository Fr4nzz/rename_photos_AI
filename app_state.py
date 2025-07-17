# ai-photo-processor/app_state.py

import os
import shutil
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

# Attempt to register HEIC/HEIF support once at application startup.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    print("[INFO ] HEIC/HEIF image support enabled.")
except ImportError:
    print("[WARN ] 'pillow-heif' not found. To process .heic/.heif files, run: pip install pillow-heif")

from utils.file_management import ensure_directory_exists
from utils.logger import SimpleLogger

CONFIG_DIR_NAME = "user_config"
SETTINGS_FILE = "settings.json"
API_KEYS_FILE = "api_keys.txt"
LAST_DIR_FILE = "last_dir.txt"

DEFAULT_PROMPT = """Extract CAM (CAM07xxxx) and notes (n) from the image.
- 2 wing photos (dorsal and ventral) per individual (CAM) are arranged in a grid left to right, top to bottom.
- If no CAMID is visible or image should be skipped, set skip: 'x', else skip: ''
- If CAMID is crossed out, set 'co' to the crossed out CAMID and put the new CAMID in 'CAM'
- CAMIDs have no spaces, remember CAM format (CAM07xxxx)
- Use notes (n) to indicate anything unusual (e.g., repeated, rotated 90°, etc).
- Put skipped reason in notes 'n'
- Double-check numbers are correctly OCRed; consecutive photos might not have consecutive CAMs
- Return JSON as shown in example; always give all keys even if empty. Example:
{
  "1": {"CAM": "CAM074806", "co": "", "n": "", "skip": ""},
  "2": {"CAM": "CAM074806", "co": "", "n": "", "skip": ""},
  "3": {"CAM": "Empty", "co": "", "n": "CAM missing", "skip": "x"},
  "4": {"CAM": "CAM070555", "co": "CAM072554", "n": "", "skip": ""}
}
"""

class AppState:
    """Manages the application's entire state, settings, and configuration persistence."""

    def __init__(self, logger: SimpleLogger):
        self.logger = logger
        self.logger.info("Initializing application state...")

        self.current_df: pd.DataFrame = pd.DataFrame()
        self.input_directory: str = ""
        self.rename_files_dir: str = ""
        self.api_keys: List[str] = []
        self.available_models: List[str] = []

        self.settings: Dict[str, Any] = {
            'batch_size': 9, 'merged_img_height': 1080, 'main_column': 'CAM',
            'model_name': '', 'prompt_text': DEFAULT_PROMPT,
            'exiftool_path': self._find_exiftool(), 'rotation_angle': 180,
            'use_exif': True, 'preview_raw': False, 'review_crop_enabled': True,
            'review_items_per_page': 50,
            'review_thumb_height': '720p',
            'suffix_mode': 'Standard',  # New setting
            'custom_suffixes': 'd,v',  # New setting
            'crop_settings': {
                'top': 0.1, 'bottom': 0.0, 'left': 0.0, 'right': 0.5,
                'zoom': True, 'grayscale': True
            }
        }

        script_dir = Path(__file__).resolve().parent
        self._config_path = script_dir / CONFIG_DIR_NAME
        ensure_directory_exists(self._config_path)

        self._load_configuration_from_disk()
        self.logger.info("Application state loaded.")

    def _load_configuration_from_disk(self):
        settings_path = self._config_path / SETTINGS_FILE
        try:
            saved_settings = json.loads(settings_path.read_text('utf-8'))
            def recursive_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        d[k] = recursive_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            self.settings = recursive_update(self.settings, saved_settings)
            self.logger.info(f"Loaded settings from {SETTINGS_FILE}")
        except FileNotFoundError:
            self.logger.info(f"{SETTINGS_FILE} not found. Using default settings.")
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.warn(f"Could not parse {SETTINGS_FILE}. Using default settings. Error: {e}")

        keys_path = self._config_path / API_KEYS_FILE
        if keys_path.exists():
            self.api_keys = [line.strip() for line in keys_path.read_text('utf-8').splitlines() if line.strip()]
            self.logger.info(f"Loaded {len(self.api_keys)} API key(s).")

        last_dir_path = self._config_path / LAST_DIR_FILE
        if last_dir_path.exists():
            last_dir = last_dir_path.read_text('utf-8').strip()
            if os.path.isdir(last_dir):
                self.set_input_directory(last_dir)

    def save_settings(self):
        settings_path = self._config_path / SETTINGS_FILE
        try:
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4)
            self.logger.info(f"Settings saved to {settings_path.name}")
        except Exception as e:
            self.logger.error(f"Failed to save settings to {settings_path.name}", exception=e)

    def save_api_keys(self):
        path = self._config_path / API_KEYS_FILE
        with open(path, 'w', encoding='utf-8') as f: f.write("\n".join(self.api_keys))
        self.logger.info(f"API keys saved to {path.name}")

    def set_input_directory(self, path: str):
        self.input_directory = path
        self.rename_files_dir = os.path.join(path, "rename_files")
        ensure_directory_exists(self.rename_files_dir)
        self._save_last_directory(path)

    def _save_last_directory(self, directory: str):
        path = self._config_path / LAST_DIR_FILE
        with open(path, 'w', encoding='utf-8') as f: f.write(directory)

    def _find_exiftool(self) -> Optional[str]:
        path = shutil.which('exiftool')
        if not path:
             self.logger.warn("'exiftool' not found in system PATH. RAW file rotation may be disabled.")
        return path