# ai-photo-processor/utils/helpers.py

from pathlib import Path
from typing import Optional, Union


def safe_int(value: Union[str, int, float, None], default: int = 0) -> int:
    """Convert to int, returning default on failure or empty string."""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Union[str, int, float, None], default: float = 0.0) -> float:
    """Convert to float, returning default on failure or empty string."""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return float(value)
    except (ValueError, TypeError):
        return default


def ensure_path(path: Union[str, Path, None]) -> Optional[Path]:
    """Return Path if file exists, else None."""
    if not path:
        return None
    p = Path(path) if isinstance(path, str) else path
    return p if p.is_file() else None


def ensure_dir(path: Union[str, Path, None]) -> Optional[Path]:
    """Return Path if directory exists, else None."""
    if not path:
        return None
    p = Path(path) if isinstance(path, str) else path
    return p if p.is_dir() else None
