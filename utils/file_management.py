# ai-photo-processor/utils/file_management.py

import os
from pathlib import Path
from typing import Literal, List

# Define supported image extensions for clarity and easy modification.
SUPPORTED_COMPRESSED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
SUPPORTED_RAW_EXTENSIONS = {'.cr2', '.orf', '.tif', '.tiff'}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_COMPRESSED_EXTENSIONS.union(SUPPORTED_RAW_EXTENSIONS)

def get_image_files(
    directory: str,
    file_type: Literal['all', 'compressed', 'raw'] = 'all'
) -> List[Path]:
    """
    Scans a directory and returns a sorted list of image file paths.

    Args:
        directory: The path to the directory to scan.
        file_type: The type of images to return. Can be 'all', 'compressed', or 'raw'.

    Returns:
        A sorted list of pathlib.Path objects for the found image files.
    """
    if not directory or not os.path.isdir(directory):
        return []

    if file_type == 'compressed':
        extensions_to_check = SUPPORTED_COMPRESSED_EXTENSIONS
    elif file_type == 'raw':
        extensions_to_check = SUPPORTED_RAW_EXTENSIONS
    else:  # 'all'
        extensions_to_check = ALL_SUPPORTED_EXTENSIONS

    found_files = []
    # Using Path.iterdir() for a more robust iteration
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions_to_check:
            found_files.append(file_path)

    # Sort the files by name for consistent ordering
    return sorted(found_files)


def ensure_directory_exists(path: str) -> None:
    """Creates a directory if it doesn't already exist."""
    # Using pathlib's mkdir for a cleaner implementation
    Path(path).mkdir(parents=True, exist_ok=True)