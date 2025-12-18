# ai-photo-processor/utils/helpers.py

"""Common utility functions to reduce redundant code throughout the project."""

from pathlib import Path
from typing import Optional, Union


def safe_int(value: Union[str, int, float, None], default: int = 0) -> int:
    """
    Safely convert a value to int with a default fallback.

    Args:
        value: The value to convert (string, int, float, or None).
        default: The default value if conversion fails.

    Returns:
        The converted integer or the default value.

    Examples:
        >>> safe_int("42")
        42
        >>> safe_int("", 10)
        10
        >>> safe_int(None, 5)
        5
    """
    if value is None:
        return default
    try:
        # Handle string input - strip whitespace
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Union[str, int, float, None], default: float = 0.0) -> float:
    """
    Safely convert a value to float with a default fallback.

    Args:
        value: The value to convert (string, int, float, or None).
        default: The default value if conversion fails.

    Returns:
        The converted float or the default value.

    Examples:
        >>> safe_float("3.14")
        3.14
        >>> safe_float("", 0.5)
        0.5
        >>> safe_float(None, 1.0)
        1.0
    """
    if value is None:
        return default
    try:
        # Handle string input - strip whitespace
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return float(value)
    except (ValueError, TypeError):
        return default


def ensure_path(path: Union[str, Path, None]) -> Optional[Path]:
    """
    Convert to Path if valid file exists.

    Args:
        path: A string or Path object, or None.

    Returns:
        A Path object if the file exists, otherwise None.
    """
    if not path:
        return None
    p = Path(path) if isinstance(path, str) else path
    return p if p.is_file() else None


def ensure_dir(path: Union[str, Path, None]) -> Optional[Path]:
    """
    Return Path if valid directory exists.

    Args:
        path: A string or Path object, or None.

    Returns:
        A Path object if the directory exists, otherwise None.
    """
    if not path:
        return None
    p = Path(path) if isinstance(path, str) else path
    return p if p.is_dir() else None
