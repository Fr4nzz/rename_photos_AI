# ai-photo-processor/utils/image_processing.py

import math
import re  # <-- Import 're' for parsing exiftool output
import piexif
import rawpy
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageOps, ImageDraw, ImageFont

from .file_management import get_image_files
from .logger import SimpleLogger

# Constants
ORIENTATION_TO_ANGLE = { 1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: 270, 7: 270, 8: 90 }
ANGLE_TO_ORIENTATION = {0: 1, 90: 8, 180: 3, 270: 6}

def fix_orientation(img: Image.Image) -> Image.Image:
    """Rotates a PIL Image to respect its EXIF orientation tag for viewing."""
    try:
        exif = img.getexif()
        orientation = exif.get(0x0112, 1)
    except (AttributeError, KeyError):
        return img

    orientation_map = {
        2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE, 6: Image.ROTATE_270, 7: Image.TRANSVERSE, 8: Image.ROTATE_90,
    }
    return img.transpose(orientation_map[orientation]) if orientation in orientation_map else img

def decode_raw_image(image_path: Path, use_exif: bool) -> Optional[Image.Image]:
    """
    Decodes a RAW file into a viewable PIL Image, respecting the 'use_exif' flag.
    """
    try:
        with rawpy.imread(str(image_path)) as raw:
            # When use_exif is False, user_flip=0 tells rawpy to ignore the orientation tag.
            flip_override = 0 if not use_exif else None
            rgb_array = raw.postprocess(use_camera_wb=True, no_auto_bright=True, user_flip=flip_override)
            return Image.fromarray(rgb_array)
    except rawpy.LibRawError as e:
        print(f"CRITICAL: Failed to decode RAW file {image_path.name}: {e}")
        return None

# --- MODIFIED FUNCTION ---
def get_angle_from_exif(img_path: Path, file_type: str, exiftool_path: Optional[str] = None) -> int:
    """
    Reads the orientation from a file and returns its rotation angle in degrees.
    Uses exiftool via subprocess for RAW files and piexif for compressed files.
    """
    orientation_tag = 1
    try:
        if file_type == 'raw':
            if not exiftool_path:
                return 0 # Cannot proceed without exiftool for RAWs
            # Use a direct subprocess call to read the orientation tag numerically.
            command = [exiftool_path, "-n", "-Orientation", str(img_path)]
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
            if result.returncode == 0 and result.stdout:
                # Example output: "Orientation                     : 6"
                match = re.search(r':\s*(\d+)', result.stdout)
                if match:
                    orientation_tag = int(match.group(1))
        else: # 'compressed'
            # piexif is faster for JPGs and doesn't require a subprocess.
            exif_dict = piexif.load(str(img_path))
            orientation_tag = exif_dict.get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
    except (subprocess.TimeoutExpired, ValueError, TypeError, piexif.InvalidImageDataError):
        # On any error (timeout, parsing, bad image), default to 1 (unrotated).
        pass
    except Exception:
        # Catch-all for other potential issues.
        pass
    return ORIENTATION_TO_ANGLE.get(orientation_tag, 0)


# --- MODIFIED FUNCTION ---
def process_single_file_for_rotation(img_path: Path, config: Dict, logger: SimpleLogger):
    """
    Calculates the final orientation and writes it to the file using exiftool or piexif.
    """
    try:
        # Determine current angle based on 'use_exif' setting
        if config['use_exif']:
            exiftool_path = config.get('exiftool_path') if config['file_type'] == 'raw' else None
            current_angle = get_angle_from_exif(img_path, config['file_type'], exiftool_path)
        else:
            current_angle = 0

        final_angle = (current_angle + config['rotation_angle']) % 360
        final_orientation_tag = ANGLE_TO_ORIENTATION.get(final_angle, 1)
        mode = "using EXIF" if config['use_exif'] else "ignoring EXIF"
        logger.info(f"Rotating {img_path.name}: angle={current_angle}°+{config['rotation_angle']}°, new_tag={final_orientation_tag} ({mode})")

        if config['file_type'] == 'raw':
            # For RAW files, always use exiftool (via subprocess) to write the tag.
            exiftool_path = config.get('exiftool_path')
            command = [
                exiftool_path,
                f"-Orientation={final_orientation_tag}",
                "-overwrite_original",
                "-n",  # Crucial: treat Orientation value as a number
                str(img_path)
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
            if result.returncode != 0:
                logger.error(f"ExifTool failed for {img_path.name}. Stderr: {result.stderr.strip()}")
        else:
            # For compressed files, use piexif.
            exif_dict = piexif.load(str(img_path))
            exif_dict['0th'][piexif.ImageIFD.Orientation] = final_orientation_tag
            piexif.insert(piexif.dump(exif_dict), str(img_path))

    except subprocess.TimeoutExpired:
        logger.error(f"ExifTool timed out processing {img_path.name}. The file may be corrupt or locked.")
    except Exception as e:
        logger.error(f"Failed to write orientation for {img_path.name}", exception=e)

# --- MODIFIED FUNCTION ---
def apply_rotation_to_folder(config: Dict, logger: SimpleLogger):
    """
    Orchestrates the rotation process for all files in a folder.
    This no longer creates a persistent exiftool instance.
    """
    file_type, folder_path = config['file_type'], config['folder_path']
    image_files = get_image_files(folder_path, file_type)

    if not image_files:
        logger.warn(f"No '{file_type}' files found in '{folder_path}' to rotate.")
        if 'progress_callback' in config:
             config['progress_callback'](100, "No files found")
        return

    total_files = len(image_files)
    if file_type == 'raw':
        exiftool_path = config.get('exiftool_path')
        if not (exiftool_path and Path(exiftool_path).exists()):
            raise RuntimeError(f"ExifTool path is not configured or is invalid: '{exiftool_path}'")
        logger.info(f"Processing {total_files} RAW files using direct subprocess calls...")
    else:
        logger.info(f"Processing {total_files} compressed files using piexif...")

    for i, img_path in enumerate(image_files):
        # The call is now simpler, without the exiftool instance argument
        process_single_file_for_rotation(img_path, config, logger)
        if 'progress_callback' in config:
            progress_text = "Rotating RAWs... %p%" if file_type == 'raw' else "Rotating JPGs... %p%"
            config['progress_callback'](int((i + 1) / total_files * 100), progress_text)

    logger.info("Rotation process finished.")


# --- NO CHANGES BELOW THIS LINE ---

def crop_image(img: Image.Image, crop_settings: Dict) -> Image.Image:
    if not crop_settings.get('zoom', False): return img
    w, h = img.size
    left_coord = int(w * crop_settings.get('left', 0.0))
    top_coord = int(h * crop_settings.get('top', 0.0))
    right_coord = int(w * (1 - crop_settings.get('right', 0.0)))
    bottom_coord = int(h * (1 - crop_settings.get('bottom', 0.0)))
    if left_coord >= right_coord or top_coord >= bottom_coord:
        print(f"Warning: Invalid crop settings. Skipping crop.")
        return img
    return img.crop((left_coord, top_coord, right_coord, bottom_coord))

def get_system_font(size=50) -> ImageFont.FreeTypeFont:
    for font_name in ["Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc", "Verdana.ttf"]:
        try: return ImageFont.truetype(font_name, size)
        except IOError: continue
    return ImageFont.load_default()

def preprocess_image(image: Image.Image, label_text: str, crop_settings: Dict) -> Image.Image:
    img = crop_image(image.copy(), crop_settings)
    if crop_settings.get('grayscale', False): img = ImageOps.grayscale(img)
    img, draw = img.convert('RGBA'), ImageDraw.Draw(img)
    font = get_system_font(size=max(20, int(img.height / 20)))
    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    padding = text_h // 2
    bg_x1, bg_y1 = (img.width - text_w) / 2 - padding, padding / 2
    bg_x2, bg_y2 = (img.width + text_w) / 2 + padding, text_h + padding * 1.5
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 255, 255, 128))
    draw.text(((img.width - text_w) / 2, padding), label_text, fill='black', font=font)
    return img.convert('RGB')

def merge_images(images: List[Image.Image], merged_img_height: int) -> Optional[Image.Image]:
    if not images: return None
    n = len(images)
    if n == 0: return None
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / float(cols)))
    if rows == 0: return None
    if not hasattr(images[0], 'size') or not all(images[0].size): return None
    ref_w, ref_h = images[0].size
    if ref_h == 0: return None
    cell_h = merged_img_height // rows
    cell_w = int(cell_h * (ref_w / ref_h))
    if cell_w == 0 or cell_h == 0: return None
    grid_img = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')
    for i, img in enumerate(images):
        resized_img = img.resize((cell_w, cell_h), Image.LANCZOS)
        row, col = divmod(i, cols)
        grid_img.paste(resized_img, (col * cell_w, row * cell_h))
    return grid_img