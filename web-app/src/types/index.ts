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
  /** the review's photos: whole, or zoomed to the wings (the envelope beside) */
  reviewZoom: 'photo' | 'wings'
  /** the review's cards: one per specimen (its photos side by side), or one per photo */
  reviewLayout: 'specimens' | 'photos'
  previewTileHeight: number
  suffixMode: SuffixMode
  customSuffixes: string
  /** 'local': the built-in CAMID reader (no API key); 'gemini': the Gemini prompt */
  engine: 'local' | 'gemini'
  /** local reader: turn each photo so its envelope text is upright right after reading */
  autoRotate: boolean
  /** local reader: photos read at the same time, one per worker (0: as many as suit the computer) */
  photoWorkers: number
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
  /** clockwise rotation that makes the photo upright, from its own reading (wings or envelope) or its camera's other photos ('' = unknown) */
  rotSuggested: string
  /** clockwise rotation to use for this photo ('' = none) */
  rotChosen: string
  /** where rotChosen came from: the photo's own reading, a manual change, the camera's other photos, or learned from manual changes */
  rotSource: '' | 'ocr' | 'manual' | 'learned' | 'neighbours'
  /** clockwise rotation already written to the file(s) by this app ('' = none) */
  rotApplied: string
  /** where `suffix` came from: '' the order taken, 'wings' the wings (against that order), 'manual' typed (kept when names are worked out again) */
  viewBy?: '' | 'wings' | 'manual'
  /** P(dorsal) of the photo's wings, from the wing check ('' = not checked) */
  pDorsal?: string
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
