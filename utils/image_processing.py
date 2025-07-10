# ai-photo-processor/utils/image_processing.py

import math
import piexif
import exiftool
import rawpy
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageOps, ImageDraw, ImageFont

from .file_management import get_image_files
from .logger import SimpleLogger

# Constants remain the same
ORIENTATION_TO_ANGLE = { 1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: 270, 7: 270, 8: 90 }
ANGLE_TO_ORIENTATION = {0: 1, 90: 8, 180: 3, 270: 6}

# --- START OF REVERTED AND SIMPLIFIED LOGIC ---

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
            # This is the elegant fix. Let rawpy do the work.
            # If use_exif is False, user_flip=0 overrides the default auto-rotation.
            # If use_exif is True, we pass None, letting rawpy use the file's orientation.
            flip_override = 0 if not use_exif else None
            rgb_array = raw.postprocess(use_camera_wb=True, no_auto_bright=True, user_flip=flip_override)
            return Image.fromarray(rgb_array)
    except rawpy.LibRawError as e:
        print(f"CRITICAL: Failed to decode RAW file {image_path.name}: {e}")
        return None

# The get_angle_from_exif and rotation folder logic are still needed for the "Apply Rotation" button,
# so they remain unchanged. Their logic is correct for writing new metadata.
def get_angle_from_exif(img_path: Path, et_instance: Optional[exiftool.ExifTool]) -> int:
    orientation_tag = 1
    try:
        if et_instance:
            metadata = et_instance.get_metadata(str(img_path))
            orientation_tag = int(metadata.get('EXIF:Orientation', metadata.get('Orientation', 1)))
        else:
            exif_dict = piexif.load(str(img_path))
            orientation_tag = exif_dict.get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
    except Exception: pass
    return ORIENTATION_TO_ANGLE.get(orientation_tag, 0)

def process_single_file_for_rotation(img_path: Path, config: Dict, logger: SimpleLogger, et_instance: Optional[exiftool.ExifTool]):
    current_angle = get_angle_from_exif(img_path, et_instance) if config['use_exif'] else 0
    final_angle = (current_angle + config['rotation_angle']) % 360
    final_orientation_tag = ANGLE_TO_ORIENTATION.get(final_angle, 1)
    mode = "using EXIF" if config['use_exif'] else "ignoring EXIF"
    logger.info(f"Rotating {img_path.name}: base_angle={current_angle}°, rot_angle={config['rotation_angle']}°, final_tag={final_orientation_tag} ({mode})")
    try:
        if et_instance:
            command = f"-Orientation={final_orientation_tag}".encode('utf-8')
            et_instance.execute(command, b"-overwrite_original", str(img_path).encode('utf-8'))
        else:
            exif_dict = piexif.load(str(img_path))
            exif_dict['0th'][piexif.ImageIFD.Orientation] = final_orientation_tag
            piexif.insert(piexif.dump(exif_dict), str(img_path))
    except Exception as e:
        logger.error(f"Failed to write orientation for {img_path.name}", exception=e)

def apply_rotation_to_folder(config: Dict, logger: SimpleLogger):
    file_type, folder_path = config['file_type'], config['folder_path']
    image_files = get_image_files(folder_path, file_type)
    if not image_files:
        logger.warn(f"No '{file_type}' files found in '{folder_path}' to rotate.")
        config['progress_callback'](100, "No files found - 100%")
        return
    if file_type == 'raw':
        exiftool_path = config.get('exiftool_path')
        if not exiftool_path: raise RuntimeError("ExifTool path not configured.")
        logger.info(f"Initializing ExifTool for {len(image_files)} RAW files...")
        with exiftool.ExifTool(executable=exiftool_path) as et:
            for i, img_path in enumerate(image_files):
                process_single_file_for_rotation(img_path, config, logger, et)
                config['progress_callback'](int((i + 1) / len(image_files) * 100), "Rotating RAWs... %p%")
    else:
        logger.info(f"Processing {len(image_files)} compressed files...")
        for i, img_path in enumerate(image_files):
            process_single_file_for_rotation(img_path, config, logger, None)
            config['progress_callback'](int((i + 1) / len(image_files) * 100), "Rotating JPGs... %p%")
    logger.info("Rotation process finished.")

# --- Image Pre-processing for Previews and API (These are fine) ---
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