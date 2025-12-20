# /build.spec
# -*- mode: python ; coding: utf-8 -*-

import sys
import subprocess
import shutil
import tempfile
from pathlib import Path

block_cipher = None

# --- Icon paths ---
ICONS_DIR = Path('icons')
ICO_PATH = ICONS_DIR / 'app.ico'
ICNS_PATH = ICONS_DIR / 'app.icns'
SVG_PATH = ICONS_DIR / 'app.svg'


def convert_svg_to_icns(svg_path: Path, output_path: Path) -> bool:
    """Convert SVG to ICNS for macOS using cairosvg and iconutil.

    Requires: pip install cairosvg
    macOS built-in: iconutil

    Returns True if successful, False otherwise.
    """
    try:
        import cairosvg
    except ImportError:
        print("[build.spec] ERROR: cairosvg not installed. Run: pip install cairosvg")
        return False

    # macOS icon sizes (normal and @2x retina)
    sizes = [16, 32, 64, 128, 256, 512]

    # Create temporary iconset directory
    with tempfile.TemporaryDirectory() as tmpdir:
        iconset_path = Path(tmpdir) / 'app.iconset'
        iconset_path.mkdir()

        print(f"[build.spec] Converting {svg_path} to ICNS...")

        # Generate PNGs at each required size
        for size in sizes:
            # Normal resolution
            png_name = f'icon_{size}x{size}.png'
            png_path = iconset_path / png_name
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=str(png_path),
                output_width=size,
                output_height=size
            )
            print(f"  - Generated {png_name}")

            # Retina (@2x) resolution
            retina_size = size * 2
            if retina_size <= 1024:  # Max supported size
                png_name_2x = f'icon_{size}x{size}@2x.png'
                png_path_2x = iconset_path / png_name_2x
                cairosvg.svg2png(
                    url=str(svg_path),
                    write_to=str(png_path_2x),
                    output_width=retina_size,
                    output_height=retina_size
                )
                print(f"  - Generated {png_name_2x}")

        # Use iconutil to create the .icns file
        try:
            result = subprocess.run(
                ['iconutil', '-c', 'icns', str(iconset_path), '-o', str(output_path)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"[build.spec] ERROR: iconutil failed: {result.stderr}")
                return False
            print(f"[build.spec] Successfully created {output_path}")
            return True
        except FileNotFoundError:
            print("[build.spec] ERROR: iconutil not found. Are you running on macOS?")
            return False


# --- Detect/generate icon file ---
icon_file = None

if sys.platform == 'win32':
    # Windows: use .ico
    if ICO_PATH.exists():
        icon_file = str(ICO_PATH)
        print(f"[build.spec] Using icon: {icon_file}")
    else:
        print(f"[build.spec] WARNING: {ICO_PATH} not found")

elif sys.platform == 'darwin':
    # macOS: prefer .icns, generate from .svg if needed
    if ICNS_PATH.exists():
        icon_file = str(ICNS_PATH)
        print(f"[build.spec] Using existing icon: {icon_file}")
    elif SVG_PATH.exists():
        # Try to convert SVG to ICNS
        if convert_svg_to_icns(SVG_PATH, ICNS_PATH):
            icon_file = str(ICNS_PATH)
        else:
            print("[build.spec] WARNING: SVG to ICNS conversion failed")
            # Fall back to .ico if available
            if ICO_PATH.exists():
                icon_file = str(ICO_PATH)
                print(f"[build.spec] Falling back to: {icon_file}")
    elif ICO_PATH.exists():
        icon_file = str(ICO_PATH)
        print(f"[build.spec] Using fallback icon: {icon_file}")
    else:
        print(f"[build.spec] WARNING: No icon found in {ICONS_DIR}/")

else:
    # Linux: icons typically not embedded in executable
    print("[build.spec] Linux build - icon embedding not applicable")


# --- Bundle data files ---
bundled_datas = []

# Bundle icons folder (for taskbar icon)
if ICONS_DIR.exists():
    bundled_datas.append((str(ICONS_DIR), 'icons'))
    print(f"[build.spec] Found {ICONS_DIR}/ - will bundle for taskbar icon")

# Detect and bundle exiftool files (Windows only)
if sys.platform == 'win32':
    # Check for exiftool.exe in current directory
    if Path('exiftool.exe').exists():
        bundled_datas.append(('exiftool.exe', '.'))
        print("[build.spec] Found exiftool.exe - will bundle")

    # Check for exiftool_files directory (required by exiftool on Windows)
    if Path('exiftool_files').exists():
        bundled_datas.append(('exiftool_files', 'exiftool_files'))
        print("[build.spec] Found exiftool_files/ - will bundle")

    if not any('exiftool' in str(d) for d in bundled_datas):
        print("[build.spec] WARNING: exiftool.exe not found. Place it next to build.spec to bundle.")

a = Analysis(
    ['main.py'],  # Entry point
    pathex=[],
    binaries=[],
    datas=bundled_datas,  # Include icons, exiftool, etc.
    hiddenimports=[
        'pandas',
        'numpy',
        'google.genai',
        'google.genai.types',
        'google.genai.errors',
        'google.genai.models',
        'google.genai.chats',
        'httpx',
        'pillow_heif',
        'rawpy',
        'pyexiftool',
        'piexif',
        'PyQt5.QtMultimedia',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AIPhotoProcessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for logging; can set to False for cleaner UX
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

# --- macOS Specific Bundle Configuration ---
app = BUNDLE(
    exe,
    name='AI Photo Processor.app',
    icon=icon_file,
    bundle_identifier='com.ai.photoprocessor',
)
