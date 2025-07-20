# /build.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],  # <-- UPDATED: Entry point is now main.py
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'pandas',
        'numpy',
        'google.generativeai',
        'google.api_core.exceptions',
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
    name='AIPhotoProcessor', # Executable name
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='screenshots/app_icon.jpeg', # <-- UPDATED: Path to your icon
)

# --- macOS Specific Bundle Configuration ---
app = BUNDLE(
    exe,
    name='AI Photo Processor.app', # <-- UPDATED: App bundle name
    icon='screenshots/app_icon.jpeg', # <-- UPDATED: Path to your icon
    bundle_identifier='com.ai.photoprocessor', # You can customize this
)