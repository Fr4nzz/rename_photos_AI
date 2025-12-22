# AI Photo Processor

The AI Photo Processor is a desktop application designed to streamline the process of cataloging large batches of photos, particularly for scientific specimen collections. It leverages **Google's Gemini API** to automatically extract handwritten data (like specimen IDs, notes, and labels) from images using OCR and provides a powerful interface to review, correct, and rename the corresponding files.

![AI Photo Processor Screenshot](screenshots/2025-07-17.png)

## Key Features

### AI-Powered Data Extraction
- **Gemini OCR**: Extract specimen IDs, handwritten notes, and other data from photos using Google's Gemini API (`google-genai` SDK) with fully customizable prompts
- **Smart Model Selection**: Automatically filters to Gemini 2.5+ models, prioritizing Flash models for free-tier compatibility
- **Batch Processing**: Process entire folders with real-time progress tracking and multi-API key rotation for rate limit handling
- **Auto-Retry**: Automatically retries failed API calls with empty responses

### Image Processing
- **Live Previews**: Preview rotation, cropping, and message grids before sending to the API
- **EXIF-Safe Rotation**: Modify orientation tags (not pixels) for JPEG (piexif), HEIC/HEIF (pillow-heif + exiftool), and RAW formats (exiftool)
- **Pre-processing Options**: Crop to focus area, grayscale filter, and pre-rotation for improved OCR accuracy

### Review & Rename
- **Smart Review Grid**: Images grouped by ID with pair verification and mismatch highlighting
- **In-Line Editing**: Edit data directly with autosave; paginated view with adjustable thumbnail quality
- **Safe Renaming**: Generate filenames from extracted IDs, rename with RAW file pairing, full undo via logged restore

---

## Download Standalone App

Ready-to-use standalone applications are available with no installation required. Just download, unzip, and run!

[**Download Latest Release (v3.0)**](https://github.com/Fr4nzz/rename_photos_AI/releases/latest)

- **Windows**: `AIPhotoProcessor-windows-v3.0.zip`

The Windows zip includes ExifTool v13.32 (from the official [ExifTool website](https://exiftool.org/)) so you can process RAW and HEIC images right away.

**Windows zip contents:**
```
AIPhotoProcessor/
├── AIPhotoProcessor.exe
├── exiftool.exe
└── exiftool_files/
    └── ... (Perl libraries required by exiftool)
```

---

## Prerequisites

- **Google Gemini API Key**: Required for AI features. Get your free key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- **Python 3.9+** (source code users only): Required if running from source.
- **ExifTool** (source code users only): Required for rotating RAW and HEIC files. Download from [exiftool.org](https://exiftool.org/). Bundled in the Windows standalone version.

---

## Installation (From Source)

### 1. Clone the Repository
```bash
git clone https://github.com/Fr4nzz/rename_photos_AI.git
cd rename_photos_AI
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python main.py
```

---

## User Interface Guide

### API Keys Tab
1. Paste your Google Gemini API key(s), one per line
2. Click **Save API Keys**
3. Multiple keys enable automatic rotation when rate limits are hit

---

### Process Images Tab

#### Input Folder
- **Browse**: Select the folder containing your images (JPEG, HEIC, RAW formats supported)
- **ExifTool Path**: Specify path to exiftool executable (required for RAW/HEIC rotation)

#### Preview Selection
- **Image**: Select an individual image to preview processing settings
- **Message**: Select which message (grid of images) to preview before sending to Gemini

#### Image Rotation
| Option | Description |
|--------|-------------|
| **Rotation** | Manual rotation angle: 0°, 90° CCW, 180°, or 90° CW |
| **Use EXIF rotation** | Apply the camera's orientation tag when displaying images |
| **Apply Rotation to Files** | Permanently modify EXIF orientation tags on disk (non-destructive to pixels) |

#### Cropping & Filters
| Option | Description |
|--------|-------------|
| **Enable Cropping** | Crop images before sending to Gemini (useful to focus on label area) |
| **Convert to Grayscale** | Convert to grayscale for potentially better OCR |
| **Pre-rotate images for Gemini** | Apply the selected rotation angle to images before merging into the grid. When enabled, respects the "Use EXIF rotation" setting. When disabled, EXIF rotation is always applied. |
| **Top/Bottom/Left/Right %** | Percentage of image to crop from each edge (0.0 = no crop, 0.5 = crop 50%) |

#### API Settings
| Option | Description |
|--------|-------------|
| **Model** | Select Gemini model (auto-populated with 2.5+ models, Flash models prioritized) |
| **Images per Prompt** | Number of merged grids to send in a single API call (default: 5) |
| **Grid Size** | Rows × Columns for the image grid (default: 3×3 = 9 images per grid) |
| **Merged Height** | Target height in pixels for merged grid images (default: 1080) |
| **Main Column** | Column name for the primary extracted data (default: CAM) |

#### Prompt
Customize the instructions sent to Gemini. The default prompt is designed for extracting specimen IDs (CAM numbers) from wing photos, but you can modify it for any OCR task.

#### Run Mode
| Mode | Description |
|------|-------------|
| **Start Over** | Begin fresh processing of all images in the folder |
| **Continue from [CSV]** | Resume from a previous session's progress |
| **Retry specific messages** | Re-process specific message numbers (e.g., "1,3,5-7") using an existing CSV |

#### Progress
- **Ask Gemini (Start)**: Begin processing
- **Stop**: Cancel processing (progress is saved)
- Progress bar shows current message number and percentage

---

### Review Results Tab

#### Loading Data
- **CSV Dropdown**: Select which output CSV to review
- **Load Selected CSV**: Load the chosen file for review

#### Filtering & Sorting
| Filter | Description |
|--------|-------------|
| **Show All** | Display all images |
| **Show Empty [Column]** | Show only images missing data in the main column |
| **Show Filled [Column]** | Show only images with data in the main column |

| Sort Option | Description |
|-------------|-------------|
| **File Name (A-Z / Z-A)** | Sort alphabetically by filename |
| **Capture Date (New-Old / Old-New)** | Sort by EXIF capture date |
| **CAM ID (A-Z / Z-A)** | Sort by extracted specimen ID |
| **Message (1-N / N-1)** | Sort by the message number from processing |

#### Pagination
- **Items per page**: Adjust how many images to show (fewer = faster loading)
- **Navigation**: Use Previous/Next or type a page number directly

#### Review Grid
Each image card displays:
- **Thumbnail**: Click to open full image in default viewer
- **Message N**: Blue label showing which API message this image was in
- **Editable fields**: CAM (specimen ID), co (correction), n (notes), skip (x to skip)
- **Status indicators**: Warnings for missing pairs, duplicates, etc.

#### Actions
| Button | Description |
|--------|-------------|
| **Recalculate Final Names** | Generate 'to' filenames based on extracted IDs. Creates a "_checked" CSV version |
| **Rename Files** | Apply the generated filenames to actual files on disk |
| **Undo Last Rename** | Restore files to their original names using the rename log |

---

## Workflow Example

1. **Setup**: Add your Gemini API key in the API Keys tab
2. **Select Folder**: Choose a folder with specimen photos
3. **Configure**: Adjust rotation/cropping to focus on the label area; preview with Message selector
4. **Process**: Click "Ask Gemini (Start)" and wait for completion
5. **Review**: Switch to Review Results tab, correct any OCR errors
6. **Rename**: Click "Recalculate Final Names" then "Rename Files"

---

## Tips

- **Grid Size**: Larger grids (e.g., 4×4) process more images per API call but may reduce OCR accuracy for small text
- **Cropping**: If labels are in a consistent position, crop to just that area for better results
- **Multiple API Keys**: Add several keys to avoid rate limiting during large batch processing
- **Pre-rotate**: Use this if your photos need rotation AND you want control over EXIF behavior
- **Message numbers**: If some messages fail, note their numbers and use "Retry specific messages" mode

---

## License

This project uses:
- [ExifTool](https://exiftool.org/) by Phil Harvey (Perl Artistic License / GPL)
- [Google Generative AI SDK](https://github.com/google/generative-ai-python) (Apache 2.0)
