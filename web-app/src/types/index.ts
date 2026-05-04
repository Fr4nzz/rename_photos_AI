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
}

export type SuffixMode = 'Standard' | 'Wing Clips' | 'Custom'

export type RunMode = 'start_over' | 'continue' | 'retry_specific'

export interface BackendStatus {
  connected: boolean
  url: string
}

export interface FileEntry {
  name: string
  path: string
  file: File
  extension: string
}

export type DownloadTier = 'directory' | 'zip-picker' | 'zip-fallback'
