import type { AppSettings, CropSettings } from '@/types'

export const SUPPORTED_COMPRESSED_EXTENSIONS = new Set([
  '.jpg', '.jpeg', '.png', '.heic', '.heif',
])

export const BROWSER_ROTATABLE_EXTENSIONS = new Set([
  '.jpg', '.jpeg', '.png',
])

export const SUPPORTED_RAW_EXTENSIONS = new Set([
  '.cr2', '.orf', '.tif', '.tiff', '.nef', '.arw', '.dng', '.raf',
])

export const ALL_SUPPORTED_EXTENSIONS = new Set([
  ...SUPPORTED_COMPRESSED_EXTENSIONS,
  ...SUPPORTED_RAW_EXTENSIONS,
])

export const ORIENTATION_TO_ANGLE: Record<number, number> = {
  1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: 270, 7: 270, 8: 90,
}

export const ANGLE_TO_ORIENTATION: Record<number, number> = {
  0: 1, 90: 8, 180: 3, 270: 6,
}

export const QUALITY_TO_HEIGHT: Record<string, number | null> = {
  '480p': 480,
  '540p': 540,
  '720p': 720,
  '900p': 900,
  '1080p': 1080,
  'Original': null,
}

export const DEFAULT_CROP_SETTINGS: CropSettings = {
  top: 0.1,
  bottom: 0.0,
  left: 0.0,
  right: 0.5,
  zoom: true,
  grayscale: false,
  prerotate: false,
}

export const DEFAULT_PROMPT = `Extract CAM (CAM07xxxx) and notes (n) from the image.
- 2 wing photos (dorsal and ventral) per individual (CAM) are arranged in a grid left to right, top to bottom.
- If no CAMID is visible or image should be skipped, set skip: 'x', else skip: ''
- If CAMID is crossed out, set 'co' to the crossed out CAMID and put the new CAMID in 'CAM'
- CAMIDs have no spaces, remember CAM format (CAM07xxxx)
- Use notes (n) to indicate anything unusual (e.g., repeated, rotated 90°, etc).
- Put skipped reason in notes 'n'
- Double-check numbers are correctly OCRed; consecutive photos might not have consecutive CAMs
- Return JSON as shown in example; always give all keys even if empty. Example:
{
  "1": {"CAM": "CAM074806", "co": "", "n": "", "skip": ""},
  "2": {"CAM": "CAM074806", "co": "", "n": "", "skip": ""},
  "3": {"CAM": "Empty", "co": "", "n": "CAM missing", "skip": "x"},
  "4": {"CAM": "CAM070555", "co": "CAM072554", "n": "", "skip": ""}
}`

export const DEFAULT_SETTINGS: AppSettings = {
  imagesPerPrompt: 1,
  gridRows: 2,
  gridCols: 2,
  mergedImgHeight: 1600,
  parallelRequests: 5,
  mainColumn: 'CAM',
  modelName: 'gemini-3.1-flash-lite-preview',
  promptText: DEFAULT_PROMPT,
  rotationAngle: 180,
  useExif: true,
  previewRaw: false,
  cropSettings: { ...DEFAULT_CROP_SETTINGS },
  reviewCropEnabled: true,
  reviewItemsPerPage: 50,
  reviewThumbHeight: '720p',
  reviewThumbSize: 180,
  previewTileHeight: 240,
  suffixMode: 'Standard',
  customSuffixes: 'd,v',
}

export const BACKEND_URL = 'http://localhost:3847'
export const BACKEND_DOWNLOAD_URL_WINDOWS =
  'https://github.com/Fr4nzz/rename_photos_AI/releases/download/backend-latest/AIPhotoProcessor-Backend.exe'
export const BACKEND_RELEASES_URL =
  'https://github.com/Fr4nzz/rename_photos_AI/releases/tag/backend-latest'
