# /build.spec
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# --- Detect and bundle exiftool files (Windows) ---
exiftool_datas = []
if sys.platform == 'win32':
    # Check for exiftool.exe in current directory
    if Path('exiftool.exe').exists():
        exiftool_datas.append(('exiftool.exe', '.'))
        print("[build.spec] Found exiftool.exe - will bundle")

    # Check for exiftool_files directory (required by exiftool on Windows)
    if Path('exiftool_files').exists():
        exiftool_datas.append(('exiftool_files', 'exiftool_files'))
        print("[build.spec] Found exiftool_files/ - will bundle")

    if not exiftool_datas:
        print("[build.spec] WARNING: exiftool.exe not found. Place it next to build.spec to bundle.")

a = Analysis(
    ['main.py'],  # Entry point
    pathex=[],
    binaries=[],
    datas=exiftool_datas,  # Include exiftool if found
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
    icon='screenshots/app_icon.jpeg',
)

# --- macOS Specific Bundle Configuration ---
app = BUNDLE(
    exe,
    name='AI Photo Processor.app',
    icon='screenshots/app_icon.jpeg',
    bundle_identifier='com.ai.photoprocessor',
)
