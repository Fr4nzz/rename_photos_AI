export interface CropSettings {
  top: number
  bottom: number
  left: number
  right: number
  zoom: boolean
  grayscale: boolean
  prerotate: boolean
}

export interface AppSettings {
  imagesPerPrompt: number
  gridRows: number
  gridCols: number
  mergedImgHeight: number
  parallelRequests: number
  mainColumn: string
  modelName: string
  promptText: string
  rotationAngle: number
  useExif: boolean
  previewRaw: boolean
  cropSettings: CropSettings
  reviewCropEnabled: boolean
  reviewItemsPerPage: number
  reviewThumbHeight: string
  reviewThumbSize: number
  previewTileHeight: number
  suffixMode: SuffixMode
  customSuffixes: string
  /** 'local': the built-in CAMID reader (no API key); 'gemini': the Gemini prompt */
  engine: 'local' | 'gemini'
  /** local reader: turn each photo so its envelope text is upright right after reading */
  autoRotate: boolean
}

export interface PhotoRow {
  from: string
  currentPath: string
  photoId: number
  mainValue: string
  co: string
  n: string
  skip: string
  to: string
  suffix: string
  batchNumber: number
  captureDate: string | null
  status: 'Original' | 'Renamed' | 'New' | 'Missing'
  /** local OCR: why this photo needs a person ('' = confirmed or renamed automatically) */
  review: string
  /** local OCR: suggested CAMIDs, best first, space-separated */
  suggest: string
  /** clockwise rotation that makes the photo upright, from the CAMID reading ('' = unknown) */
  rotSuggested: string
  /** clockwise rotation to use for this photo ('' = none) */
  rotChosen: string
  /** where rotChosen came from: the reader, a manual change, or learned from manual changes */
  rotSource: '' | 'ocr' | 'manual' | 'learned'
  /** clockwise rotation already written to the file(s) by this app ('' = none) */
  rotApplied: string
}

export interface RotationLogEntry {
  original: string
  /** 'tag': only the EXIF Orientation value changed (undo = write `before` back);
   *  're-encode': pixels rewritten, original kept in rotation_backups/`backup` */
  method?: 'tag' | 're-encode'
  backup?: string
  before?: number
  after?: number
  angle: number
  timestamp: string
}

export type SuffixMode = 'Standard' | 'Wing Clips' | 'Custom'

export type RunMode = 'start_over' | 'continue' | 'retry_specific'

export interface FileEntry {
  name: string
  path: string
  file: File
  extension: string
}

export type DownloadTier = 'directory' | 'zip-picker' | 'zip-fallback'
