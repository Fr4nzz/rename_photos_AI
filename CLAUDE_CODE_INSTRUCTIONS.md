# AI Photo Processor - Optimization & Migration Plan

## Overview

This document contains instructions for Claude Code to optimize, fix bugs, and modernize the AI Photo Processor codebase. Execute these tasks in the order specified.

---

## Project Context

This is a PyQt5 desktop application that:
- Uses Google Gemini Vision API to extract text (specimen IDs) from batched images
- Processes images (rotation, cropping, grayscale) before sending to AI
- Allows users to review AI results and bulk rename files
- Supports JPEG, PNG, HEIC/HEIF, and RAW formats (.CR2, .ORF, .TIF)

**Key files:**
- `main.py` - Entry point
- `main_window.py` - Main application window
- `app_state.py` - Application state and settings management
- `controllers/process_tab_handler.py` - Image processing tab logic
- `controllers/review_tab_handler.py` - Review/rename tab logic
- `utils/gemini_handler.py` - Gemini API wrapper
- `utils/image_processing.py` - Image manipulation functions
- `workers.py` - Background thread workers
- `build.spec` - PyInstaller build configuration

---

## Phase 1: Critical - Migrate Google SDK (HIGHEST PRIORITY)

**Official Documentation:** https://googleapis.github.io/python-genai/
**Migration Guide:** https://ai.google.dev/gemini-api/docs/migrate
**Codegen Instructions (for LLMs):** https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md

### Task 1.1: Update requirements.txt

**File:** `requirements.txt`

**Changes:**
1. Remove `google-generativeai` 
2. Add `google-genai>=1.0.0`

The old `google.generativeai` package is deprecated and shows this warning:
```
All support for the `google.generativeai` package has ended. Please switch to the `google.genai` package.
```

---

### Task 1.2: Rewrite utils/gemini_handler.py

**File:** `utils/gemini_handler.py`

**CRITICAL: The new SDK architecture is fundamentally different. Study these patterns carefully.**

**OLD SDK (deprecated) - Current code uses this:**
```python
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)
chat = model.start_chat(history=[])
response = chat.send_message([prompt_text, img])
return response.text
```

**NEW SDK - Required changes:**
```python
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Create a client (must be done for each API key when rotating)
client = genai.Client(api_key=api_key)

# For simple generation (no chat history needed):
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[prompt_text, img],  # Can pass PIL Image directly!
)
return response.text

# For chat-based conversations (if needed):
chat = client.chats.create(model='gemini-2.5-flash')
response = chat.send_message('message here')
```

**Key migration points for GeminiHandler class:**
1. Replace `genai.configure(api_key=...)` with `client = genai.Client(api_key=...)`
2. Replace `genai.GenerativeModel(model_name)` - not needed, pass model to generate_content
3. Replace `model.start_chat()` with `client.chats.create(model=...)` if chat is needed
4. Replace `chat.send_message([prompt, image])` with `client.models.generate_content(model=..., contents=[prompt, image])`
5. Replace exception imports:
   - OLD: `from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted`
   - NEW: `from google.genai.errors import APIError`
6. PIL Images can be passed directly in contents - no need to call `Image.open()` separately
7. Response format: `response.text` still works the same way

**API Key rotation logic:**
- When switching keys, create a new `genai.Client(api_key=new_key)`
- The client object should be recreated for each key

**Error handling pattern:**
```python
from google.genai.errors import APIError

try:
    response = client.models.generate_content(...)
except APIError as e:
    print(e.code)  # HTTP status code like 404, 429
    print(e.message)  # Error message
    # Handle rate limiting (429) by switching API key
```

**Passing images to the API:**
```python
from PIL import Image

# The new SDK accepts PIL Images directly in the contents list!
img = Image.open(image_path)
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[prompt_text, img]  # PIL Image passed directly
)
```

**Alternative: Using types.Part for more control:**
```python
from google.genai import types

# From bytes
with open('image.jpg', 'rb') as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        prompt_text,
        types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
    ]
)
```

---

### Task 1.3: Update model listing in process_tab_handler.py

**File:** `controllers/process_tab_handler.py`

**Function to rewrite:** `update_models_dropdown()`

**Current problems:**
1. Shows too many models (TTS, image generators, audio models, etc.)
2. Version regex `r'(\d+\.\d+)'` doesn't match models like `gemini-3-flash` (no decimal)
3. Uses deprecated `genai.list_models()` API

**NEW SDK pattern for listing models:**
```python
from google import genai

client = genai.Client(api_key=api_keys[0])

for model in client.models.list():
    model_name = model.name  # e.g., "models/gemini-2.5-flash" or just "gemini-2.5-flash"
    # Extract just the model ID if it has "models/" prefix
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
```

**Filter OUT models containing these patterns (case-insensitive):**
- `-tts` (text-to-speech)
- `-image` (image generation models like gemini-2.5-flash-image)
- `-audio` (audio models)
- `-video` (video models)
- `embedding` (embedding models)
- `aqa` (attributed QA)
- `bisheng` (internal)
- `learnlm` (learning models)
- `imagen` (image generation)
- `veo` (video generation)

**Fix version regex to handle both formats:**
- `gemini-2.5-flash` → major=2, minor=5
- `gemini-3-flash` → major=3, minor=0

New regex pattern: `r'gemini[- ]?(\d+)(?:\.(\d+))?'`

**Sort models by:**
1. Version (highest first)
2. Flash BEFORE Pro (free-tier friendly)
3. Stable before preview

**Expected result:** Dropdown shows only ~5-10 useful multimodal text models, with Flash models prioritized for free-tier compatibility:
1. gemini-3-flash-preview (or gemini-3-flash when out of preview) ← **DEFAULT SELECTION**
2. gemini-3-pro-preview (or gemini-3-pro)
3. gemini-2.5-flash
4. gemini-2.5-pro
5. etc.

**Example implementation pattern:**
```python
def update_models_dropdown(self):
    self.ui.model_dropdown.clear()
    
    if not self.app_state.api_keys:
        self.ui.model_dropdown.addItem("No API Key Set")
        return
    
    try:
        from google import genai
        import re
        
        client = genai.Client(api_key=self.app_state.api_keys[0])
        
        EXCLUDED_PATTERNS = [
            '-tts', '-image', '-audio', '-video', 'embedding',
            'aqa', 'bisheng', 'learnlm', 'imagen', 'veo'
        ]
        
        usable_models = []
        
        for model in client.models.list():
            # Get model name, strip "models/" prefix if present
            model_name = model.name
            if '/' in model_name:
                model_name = model_name.split('/')[-1]
            
            model_lower = model_name.lower()
            
            # Skip non-Gemini models
            if 'gemini' not in model_lower:
                continue
            
            # Skip excluded patterns
            if any(pattern in model_lower for pattern in EXCLUDED_PATTERNS):
                continue
            
            # Parse version - handles "gemini-2.5-flash" and "gemini-3-flash"
            version_match = re.search(r'gemini-(\d+)(?:\.(\d+))?', model_lower)
            if not version_match:
                continue
            
            major = int(version_match.group(1))
            minor = int(version_match.group(2)) if version_match.group(2) else 0
            
            # Filter: version >= 2.5
            if major < 2 or (major == 2 and minor < 5):
                continue
            
            usable_models.append({
                'name': model_name,
                'major': major,
                'minor': minor,
                'is_preview': 'preview' in model_lower,
                'is_flash': 'flash' in model_lower,
                'is_pro': 'pro' in model_lower,
            })
        
        # Sort priority (highest first):
        # 1. Higher version (3.x before 2.x)
        # 2. Flash BEFORE Pro (free tier friendly - Pro often not available on free tier)
        # 3. Stable before preview (but both should work)
        usable_models.sort(
            key=lambda m: (
                m['major'],           # Higher major version first
                m['minor'],           # Higher minor version first  
                m['is_flash'],        # Flash BEFORE Pro (True sorts after False, so flash=True goes first with reverse)
                not m['is_preview'],  # Stable before preview
            ),
            reverse=True
        )
        
        # Populate dropdown
        if usable_models:
            model_names = [m['name'] for m in usable_models]
            self.app_state.available_models = model_names
            self.ui.model_dropdown.addItems(model_names)
            
            # Try to restore saved model, otherwise use first (best) option
            saved_model = self.app_state.settings.get('model_name')
            if saved_model and saved_model in model_names:
                self.ui.model_dropdown.setCurrentText(saved_model)
            else:
                # Default to first model (should be gemini-3-flash or gemini-3-flash-preview)
                self.ui.model_dropdown.setCurrentIndex(0)
                self.app_state.settings['model_name'] = model_names[0]
        else:
            self.ui.model_dropdown.addItem("No compatible models found")
            
    except Exception as e:
        self.logger.error("Failed to fetch models", exception=e)
        self.ui.model_dropdown.addItem("Error fetching models")
```

**Expected dropdown order (top to bottom):**
1. gemini-3-flash (or gemini-3-flash-preview while in preview)
2. gemini-3-pro (or gemini-3-pro-preview)
3. gemini-2.5-flash
4. gemini-2.5-pro
5. etc.

**Note:** The sorting handles future model naming gracefully:
- When "preview" is removed from model names, they'll still sort correctly
- Flash models always appear before Pro models at each version level
- This is intentional because Pro models often aren't available on the free API tier

---

### Task 1.4: Update build.spec hidden imports

**File:** `build.spec`

**Changes:**
1. Remove `'google.generativeai'` from hiddenimports
2. Remove `'google.api_core.exceptions'` from hiddenimports
3. Add these new imports:
   - `'google.genai'`
   - `'google.genai.types'`
   - `'google.genai.errors'`
   - `'google.genai.models'`
   - `'google.genai.chats'`
   - `'httpx'` (the new SDK uses httpx for HTTP requests)

---

## Phase 2: Critical - Fix HEIC EXIF Orientation Support

### Task 2.1: Update get_angle_from_exif function

**File:** `utils/image_processing.py`

**Current bug:** The function routes HEIC files to `piexif.load()` which does NOT support HEIC format. This causes HEIC orientation to always return 1 (no rotation).

**Fix required:**

Create a routing system based on file extension:

1. **HEIC/HEIF files** (`.heic`, `.heif`):
   - First try: Use `pillow_heif.open_heif()` to read EXIF metadata
   - The EXIF data is in `heif_file.info.get('metadata', [])` where type='Exif'
   - May need to strip EXIF header bytes (`Exif\x00\x00`) before passing to piexif
   - Fallback: Use exiftool via subprocess if pillow-heif fails

2. **JPEG files** (`.jpg`, `.jpeg`):
   - Keep using `piexif.load()` - it's fast and works well

3. **RAW files** (`.cr2`, `.orf`, `.tif`, `.tiff`):
   - Keep using exiftool via subprocess

4. **Other formats**:
   - Fallback to exiftool

**Also update the write/rotation functions:**
- `process_single_file_for_rotation()` needs similar routing
- HEIC write operations require exiftool (piexif can't write HEIC)

**Add HEIC_EXTENSIONS constant:** `{'.heic', '.heif'}`

---

### Task 2.2: Update file_management.py if needed

**File:** `utils/file_management.py`

Verify that HEIC/HEIF extensions are included in `SUPPORTED_COMPRESSED_EXTENSIONS`. They should already be there but confirm:
```python
SUPPORTED_COMPRESSED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
```

---

## Phase 3: Code Optimization

### Task 3.1: Create utils/helpers.py with common utilities

**New file:** `utils/helpers.py`

Create helper functions to reduce redundant code throughout the project:

1. `safe_int(value, default=0)` - Safely convert to int with default fallback
2. `safe_float(value, default=0.0)` - Safely convert to float with default
3. `ensure_path(path) -> Optional[Path]` - Convert to Path if valid file exists
4. `ensure_dir(path) -> Optional[Path]` - Return Path if valid directory

Then update files that have redundant try/except blocks for int/float conversion:
- `controllers/process_tab_handler.py` - `_sync_settings_from_ui()`
- `controllers/review_tab_handler.py` - `_sync_settings_from_ui()`

---

### Task 3.2: Simplify redundant checks throughout codebase

Search for and simplify patterns like:

```python
# BEFORE
if self.app_state.input_directory and self.app_state.input_directory != "":
# AFTER  
if self.app_state.input_directory:

# BEFORE
if path and os.path.exists(path) and os.path.isdir(path):
# AFTER
if path and Path(path).is_dir():
```

---

### Task 3.3: Fix the "Invalid crop value entered" warning on startup

**File:** `controllers/process_tab_handler.py`

**Bug:** On startup, `_sync_settings_from_ui()` is called before UI fields are properly populated, causing float conversion to fail.

**Fix:** Add validation that UI fields have valid text before trying to convert:
```python
if ui.crop_top_input.text().strip():
    cs['top'] = float(ui.crop_top_input.text())
```

Or ensure `populate_initial_ui()` completes before any sync calls.

---

### Task 3.4: Consolidate image processing code

**File:** `utils/image_processing.py`

**Issue:** `fix_orientation()` is called multiple times on the same image in different places.

**Improvement:** Consider creating an `ImageProcessor` class or ensuring images are only orientation-fixed once when loaded, then passed around in their corrected state.

---

## Phase 4: Build Configuration - Single Executable

### Task 4.1: Update build.spec to bundle exiftool

**File:** `build.spec`

**Goal:** Create a single .exe that includes exiftool files internally.

**Changes:**

1. Add logic to detect and include exiftool files:
```python
exiftool_datas = []
if sys.platform == 'win32':
    if Path('exiftool.exe').exists():
        exiftool_datas.append(('exiftool.exe', '.'))
    if Path('exiftool_files').exists():
        exiftool_datas.append(('exiftool_files', 'exiftool_files'))
```

2. Add `exiftool_datas` to the `datas` parameter in `Analysis()`

3. Consider changing `console=True` to `console=False` for cleaner user experience (but keep a way to see logs, perhaps a log file)

---

### Task 4.2: Update exiftool path resolution in app_state.py

**File:** `app_state.py`

**Function:** `_find_exiftool()`

**Update to check these locations in order:**

1. **Bundled location** (PyInstaller): `Path(sys._MEIPASS) / 'exiftool.exe'`
2. **Next to executable** (legacy): `Path(sys.executable).parent / 'exiftool.exe'`
3. **System PATH**: `shutil.which('exiftool')`
4. **Common Windows locations**: 
   - `%LOCALAPPDATA%/exiftool/exiftool.exe`
   - `C:/Program Files/exiftool/exiftool.exe`

Log which location was found for debugging.

---

## Phase 5: Minor Improvements

### Task 5.1: Add type hints to key functions

Add type hints to improve code readability and IDE support in these files:
- `utils/gemini_handler.py`
- `utils/image_processing.py`  
- `utils/name_calculator.py`
- `workers.py`

Use `from typing import Optional, List, Dict, Any, Union` and `from pathlib import Path`.

---

### Task 5.2: Improve worker cleanup

**Files:** `controllers/process_tab_handler.py`, `controllers/review_tab_handler.py`

Ensure workers are properly stopped and threads are cleaned up:
- Add timeout to `thread.wait()` calls
- Consider using `thread.terminate()` as last resort if `wait()` times out
- Clear references to worker and thread after cleanup

---

### Task 5.3: Update README.md

**File:** `README.md`

After completing the migration:
1. Update any references to the old Google SDK
2. Add note about HEIC support
3. Update the model selection description
4. Note that exiftool is now bundled in the Windows build

---

## Phase 6: Testing Checklist

After implementing changes, verify:

- [ ] App starts without deprecation warnings
- [ ] Model dropdown shows only useful models (gemini-2.5-*, gemini-3-*)
- [ ] No TTS, image-gen, audio, or embedding models in dropdown
- [ ] **Flash models appear BEFORE Pro models** (e.g., gemini-3-flash before gemini-3-pro)
- [ ] **Default selected model is the latest Flash version** (e.g., gemini-3-flash-preview)
- [ ] Gemini API calls work with new SDK
- [ ] HEIC files rotate correctly (test with a HEIC that has non-standard orientation)
- [ ] JPEG files still rotate correctly
- [ ] RAW files (.CR2, .ORF) still rotate correctly with exiftool
- [ ] No "Invalid crop value entered" warning on startup
- [ ] Build produces working single executable (Windows)
- [ ] Bundled exiftool is found and works in built exe

---

## Execution Order Summary

```
Session 1: SDK Migration (CRITICAL)
├── 1.1 Update requirements.txt
├── 1.2 Rewrite gemini_handler.py
├── 1.3 Update model dropdown logic
└── 1.4 Update build.spec imports

Session 2: HEIC Fix (CRITICAL)  
├── 2.1 Fix get_angle_from_exif for HEIC
└── 2.2 Verify file extensions

Session 3: Code Optimization
├── 3.1 Create helpers.py
├── 3.2 Simplify redundant checks
├── 3.3 Fix startup warning
└── 3.4 Consolidate image processing

Session 4: Build Configuration
├── 4.1 Update build.spec for bundling
└── 4.2 Update exiftool discovery

Session 5: Polish
├── 5.1 Add type hints
├── 5.2 Improve worker cleanup
└── 5.3 Update README

Session 6: Testing
└── Run through testing checklist
```

---

## Notes for Claude Code

**IMPORTANT: Before starting Phase 1, read the official Google codegen instructions:**
- URL: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md
- This file is specifically designed to help LLMs generate correct code for the new SDK
- Key patterns to follow are documented there

**Testing guidelines:**
- Always run `python main.py` after making changes to verify the app still works
- The app requires at least one valid Gemini API key in the API Keys tab to test model listing
- Test HEIC orientation with a photo taken on iPhone (they often have orientation metadata)
- When updating the Google SDK, carefully handle the response format differences
- Keep backward compatibility with existing settings.json files

**Key SDK migration reminders:**
- OLD: `import google.generativeai as genai` → NEW: `from google import genai`
- OLD: `genai.configure(api_key=...)` → NEW: `client = genai.Client(api_key=...)`
- OLD: `genai.GenerativeModel(model)` → NEW: Pass model to `client.models.generate_content(model=...)`
- OLD: `model.generate_content(contents)` → NEW: `client.models.generate_content(model=..., contents=...)`
- OLD: `genai.list_models()` → NEW: `client.models.list()`
- OLD: `from google.api_core.exceptions import ...` → NEW: `from google.genai.errors import APIError`
- PIL Images can be passed directly in contents list - the SDK handles them automatically

**Official documentation links:**
- SDK Docs: https://googleapis.github.io/python-genai/
- Migration Guide: https://ai.google.dev/gemini-api/docs/migrate
- PyPI: https://pypi.org/project/google-genai/

**Free tier consideration:**
- Most users will be using free-tier API keys
- Pro models (gemini-3-pro, gemini-2.5-pro) are often NOT available on free tier
- Flash models should always be listed FIRST and selected by DEFAULT
- The model dropdown sorting prioritizes: Flash > Pro at each version level
