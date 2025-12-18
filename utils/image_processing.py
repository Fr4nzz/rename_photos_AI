# ai-photo-processor/utils/image_processing.py

import math
import re
import piexif
import rawpy
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageOps, ImageDraw, ImageFont

from .file_management import get_image_files, SUPPORTED_RAW_EXTENSIONS
from .logger import SimpleLogger

# Constants
ORIENTATION_TO_ANGLE = { 1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: 270, 7: 270, 8: 90 }
ANGLE_TO_ORIENTATION = {0: 1, 90: 8, 180: 3, 270: 6}

# File extension sets for EXIF handling routing
HEIC_EXTENSIONS = {'.heic', '.heif'}
JPEG_EXTENSIONS = {'.jpg', '.jpeg'}


def _get_heic_orientation(img_path: Path) -> int:
    """Read EXIF orientation from HEIC files via pillow-heif."""
    try:
        import pillow_heif
        heif = pillow_heif.open_heif(str(img_path))
        for meta in heif.info.get('metadata', []):
            if meta.get('type') == 'Exif':
                data = meta.get('data', b'')
                if data.startswith(b'Exif\x00\x00'):
                    data = data[6:]
                if data:
                    return piexif.load(data).get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
        if 'exif' in heif.info:
            return piexif.load(heif.info['exif']).get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
    except Exception:
        pass
    return 1


def _get_orientation_via_exiftool(img_path: Path, exiftool_path: str) -> int:
    """Read EXIF orientation via exiftool (works for RAW, HEIC, etc.)."""
    try:
        result = subprocess.run(
            [exiftool_path, "-n", "-Orientation", str(img_path)],
            capture_output=True, text=True, check=False, timeout=10
        )
        if result.returncode == 0 and (match := re.search(r':\s*(\d+)', result.stdout)):
            return int(match.group(1))
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return 1


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
    """Decode RAW file to PIL Image. If use_exif=False, ignores camera orientation."""
    try:
        with rawpy.imread(str(image_path))as raw:
            flip_override = 0 if not use_exif else None
            rgb_array = raw.postprocess(use_camera_wb=True, no_auto_bright=True, user_flip=flip_override)
            return Image.fromarray(rgb_array)
    except rawpy.LibRawError as e:
        print(f"CRITICAL: Failed to decode RAW file {image_path.name}: {e}")
        return None

def get_angle_from_exif(img_path: Path, file_type: str, exiftool_path: Optional[str] = None) -> int:
    """Get rotation angle (0/90/180/270) from EXIF. Routes by extension: HEIC→pillow-heif, JPEG→piexif, RAW→exiftool."""
    orientation_tag = 1
    suffix = img_path.suffix.lower()

    try:
        if suffix in HEIC_EXTENSIONS:
            orientation_tag = _get_heic_orientation(img_path)
            if orientation_tag == 1 and exiftool_path:
                orientation_tag = _get_orientation_via_exiftool(img_path, exiftool_path)
        elif suffix in JPEG_EXTENSIONS:
            exif_dict = piexif.load(str(img_path))
            orientation_tag = exif_dict.get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
        elif suffix in SUPPORTED_RAW_EXTENSIONS or file_type == 'raw':
            if exiftool_path:
                orientation_tag = _get_orientation_via_exiftool(img_path, exiftool_path)
        else:
            try:
                exif_dict = piexif.load(str(img_path))
                orientation_tag = exif_dict.get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
            except (piexif.InvalidImageDataError, ValueError):
                if exiftool_path:
                    orientation_tag = _get_orientation_via_exiftool(img_path, exiftool_path)
    except Exception:
        pass

    return ORIENTATION_TO_ANGLE.get(orientation_tag, 0)


def process_single_file_for_rotation(img_path: Path, config: Dict, logger: SimpleLogger):
    """Write EXIF orientation tag. Uses exiftool for RAW/HEIC, piexif for JPEG."""
    try:
        current_angle = get_angle_from_exif(img_path, config['file_type'], config.get('exiftool_path')) if config['use_exif'] else 0
        final_angle = (current_angle + config['rotation_angle']) % 360
        final_orientation_tag = ANGLE_TO_ORIENTATION.get(final_angle, 1)
        mode = "using EXIF" if config['use_exif'] else "ignoring EXIF"
        logger.info(f"Rotating {img_path.name}: angle={current_angle}°+{config['rotation_angle']}°, new_tag={final_orientation_tag} ({mode})")

        suffix = img_path.suffix.lower()
        exiftool_path = config.get('exiftool_path')

        # Determine which method to use for writing EXIF
        use_exiftool = (
            config['file_type'] == 'raw' or
            suffix in SUPPORTED_RAW_EXTENSIONS or
            suffix in HEIC_EXTENSIONS  # HEIC requires exiftool - piexif can't write HEIC
        )

        if use_exiftool:
            if not exiftool_path:
                logger.error(f"ExifTool required for {img_path.name} but path not configured.")
                return

            command = [exiftool_path, f"-Orientation={final_orientation_tag}", "-overwrite_original", "-n", str(img_path)]
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
            if result.returncode != 0:
                logger.error(f"ExifTool failed for {img_path.name}. Stderr: {result.stderr.strip()}")
        else:
            # JPEG and other formats that piexif supports
            exif_dict = piexif.load(str(img_path))
            exif_dict['0th'][piexif.ImageIFD.Orientation] = final_orientation_tag
            piexif.insert(piexif.dump(exif_dict), str(img_path))

    except subprocess.TimeoutExpired:
        logger.error(f"ExifTool timed out processing {img_path.name}.")
    except Exception as e:
        logger.error(f"Failed to write orientation for {img_path.name}", exception=e)

def apply_rotation_to_folder(config: Dict, logger: SimpleLogger):
    file_type, folder_path = config['file_type'], config['folder_path']
    image_files = get_image_files(folder_path, file_type)

    if not image_files:
        logger.warn(f"No '{file_type}' files found in '{folder_path}' to rotate.")
        if 'progress_callback' in config: config['progress_callback'](100, "No files found")
        return

    total_files = len(image_files)
    if file_type == 'raw':
        exiftool_path = config.get('exiftool_path')
        if not (exiftool_path and Path(exiftool_path).exists()):
            raise RuntimeError(f"ExifTool path is not configured or is invalid: '{exiftool_path}'")

    for i, img_path in enumerate(image_files):
        process_single_file_for_rotation(img_path, config, logger)
        if 'progress_callback' in config:
            progress_text = f"Rotating {file_type}... %p%"
            config['progress_callback'](int((i + 1) / total_files * 100), progress_text)

    logger.info("Rotation process finished.")


def crop_image(img: Image.Image, crop_settings: Dict) -> Image.Image:
    if not crop_settings.get('zoom', False): return img
    w, h = img.size
    left = int(w * crop_settings.get('left', 0.0))
    top = int(h * crop_settings.get('top', 0.0))
    right = int(w * (1 - crop_settings.get('right', 0.0)))
    bottom = int(h * (1 - crop_settings.get('bottom', 0.0)))
    if left >= right or top >= bottom:
        print(f"Warning: Invalid crop settings {left, top, right, bottom}. Skipping crop.")
        return img
    return img.crop((left, top, right, bottom))

def get_system_font(size=50) -> ImageFont.FreeTypeFont:
    for font_name in ["arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"]:
        try: return ImageFont.truetype(font_name, size)
        except IOError: continue
    return ImageFont.load_default()

def preprocess_image(image: Image.Image, label_text: str, crop_settings: Dict) -> Image.Image:
    """Apply crop/grayscale, add border, and overlay label text."""
    img_cropped = crop_image(image.copy(), crop_settings)
    img_processed = ImageOps.grayscale(img_cropped) if crop_settings.get('grayscale', False) else img_cropped
    img_bordered = ImageOps.expand(img_processed, border=10, fill='black')

    img_final = img_bordered.convert('RGBA')
    draw = ImageDraw.Draw(img_final)

    font_size = max(40, int(img_cropped.height / 12))
    font = get_system_font(size=font_size)

    # Asymmetrical padding: minimal top, generous bottom
    side_padding, top_padding, bottom_padding = int(font_size * 0.25), int(font_size * 0.10), int(font_size * 0.25)

    try:
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label_text, font=font)

    block_top_y = int(img_final.height * 0.04)
    text_x = (img_final.width - text_w) // 2
    text_y = block_top_y + top_padding

    bg_x1, bg_y1 = text_x - side_padding, block_top_y
    bg_x2, bg_y2 = text_x + text_w + side_padding, block_top_y + top_padding + text_h + bottom_padding

    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 255, 255, 128))
    draw.text((text_x, text_y), label_text, fill='black', font=font)

    return img_final.convert('RGB')


def merge_images(images: List[Image.Image], merged_img_height: int) -> Optional[Image.Image]:
    if not images: return None
    n = len(images)
    cols = int(math.ceil(math.sqrt(n)))
    if n == 0 or cols == 0: return None
    rows = int(math.ceil(n / float(cols)))
    if rows == 0: return None
    
    ref_w, ref_h = images[0].size
    if ref_h == 0: return None
    
    cell_h = merged_img_height // rows
    cell_w = int(cell_h * (ref_w / ref_h))
    if cell_w == 0 or cell_h == 0: return None
    
    grid_img = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')
    for i, img in enumerate(images):
        resized_img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        row, col = divmod(i, cols)
        grid_img.paste(resized_img, (col * cell_w, row * cell_h))
        
    return grid_img