import exifr from 'exifr'
import { SUPPORTED_RAW_EXTENSIONS } from './constants'
import type { CropSettings } from '@/types'

/**
 * Read EXIF orientation and return rotation info.
 * Uses exifr which supports JPEG, TIFF, HEIC, and RAW files.
 */
export async function getOrientationAngle(file: File): Promise<number> {
  try {
    const orientation = await exifr.orientation(file)
    if (!orientation) return 0
    const map: Record<number, number> = {
      1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: 270, 7: 270, 8: 90,
    }
    return map[orientation] ?? 0
  } catch {
    return 0
  }
}

/**
 * Load a File into an HTMLImageElement as a blob URL.
 * For RAW files, extracts the embedded JPEG thumbnail via exifr.
 */
export async function loadImage(file: File): Promise<HTMLImageElement> {
  const ext = '.' + file.name.split('.').pop()!.toLowerCase()
  let url: string

  if (SUPPORTED_RAW_EXTENSIONS.has(ext)) {
    const thumbUrl = await exifr.thumbnailUrl(file)
    if (!thumbUrl) throw new Error(`No thumbnail in RAW file: ${file.name}`)
    url = thumbUrl
  } else {
    url = URL.createObjectURL(file)
  }

  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load: ${file.name}`))
    img.src = url
  })
}

/**
 * Apply EXIF orientation correction by drawing to canvas.
 */
export async function fixOrientation(
  img: HTMLImageElement,
  file: File
): Promise<HTMLCanvasElement> {
  const rotation = await exifr.rotation(file).catch(() => null)
  const canvas = document.createElement('canvas')

  if (!rotation || rotation.deg === 0) {
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    return canvas
  }

  const { deg, dimensionSwapped } = rotation
  if (dimensionSwapped) {
    canvas.width = img.naturalHeight
    canvas.height = img.naturalWidth
  } else {
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
  }

  const ctx = canvas.getContext('2d')!
  ctx.translate(canvas.width / 2, canvas.height / 2)
  ctx.rotate((deg * Math.PI) / 180)
  ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2)
  return canvas
}

/**
 * Rotate a canvas by a given angle (0, 90, 180, 270).
 */
export function rotateCanvas(
  source: HTMLCanvasElement,
  angle: number
): HTMLCanvasElement {
  if (angle === 0) return source

  const canvas = document.createElement('canvas')
  const swap = angle === 90 || angle === 270
  canvas.width = swap ? source.height : source.width
  canvas.height = swap ? source.width : source.height

  const ctx = canvas.getContext('2d')!
  ctx.translate(canvas.width / 2, canvas.height / 2)
  ctx.rotate((angle * Math.PI) / 180)
  ctx.drawImage(source, -source.width / 2, -source.height / 2)
  return canvas
}

/**
 * Crop a canvas based on percentage settings.
 * Port of Python crop_image().
 */
export function cropCanvas(
  source: HTMLCanvasElement,
  settings: CropSettings
): HTMLCanvasElement {
  if (!settings.zoom) return source

  const w = source.width
  const h = source.height
  const left = Math.floor(w * settings.left)
  const top = Math.floor(h * settings.top)
  const right = Math.floor(w * (1 - settings.right))
  const bottom = Math.floor(h * (1 - settings.bottom))

  if (left >= right || top >= bottom) return source

  const cropW = right - left
  const cropH = bottom - top
  const canvas = document.createElement('canvas')
  canvas.width = cropW
  canvas.height = cropH

  const ctx = canvas.getContext('2d')!
  ctx.drawImage(source, left, top, cropW, cropH, 0, 0, cropW, cropH)
  return canvas
}

/**
 * Convert canvas to grayscale.
 */
export function grayscaleCanvas(source: HTMLCanvasElement): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = source.width
  canvas.height = source.height
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(source, 0, 0)

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const d = imageData.data
  for (let i = 0; i < d.length; i += 4) {
    const gray = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
    d[i] = d[i + 1] = d[i + 2] = gray
  }
  ctx.putImageData(imageData, 0, 0)
  return canvas
}

/**
 * Preprocess image: crop, grayscale, add border and label overlay.
 * Port of Python preprocess_image().
 */
export function preprocessImage(
  source: HTMLCanvasElement,
  labelText: string,
  cropSettings: CropSettings
): HTMLCanvasElement {
  let processed = cropCanvas(source, cropSettings)
  if (cropSettings.grayscale) {
    processed = grayscaleCanvas(processed)
  }

  // Add 10px black border
  const borderSize = 10
  const bordered = document.createElement('canvas')
  bordered.width = processed.width + borderSize * 2
  bordered.height = processed.height + borderSize * 2
  const bCtx = bordered.getContext('2d')!
  bCtx.fillStyle = 'black'
  bCtx.fillRect(0, 0, bordered.width, bordered.height)
  bCtx.drawImage(processed, borderSize, borderSize)

  // Label overlay
  const fontSize = Math.max(40, Math.floor(processed.height / 12))
  const ctx = bordered.getContext('2d')!
  ctx.font = `${fontSize}px Arial, sans-serif`

  const sidePad = Math.floor(fontSize * 0.25)
  const topPad = Math.floor(fontSize * 0.1)
  const bottomPad = Math.floor(fontSize * 0.25)

  const metrics = ctx.measureText(labelText)
  const textW = metrics.width
  const textH = fontSize

  const blockTopY = Math.floor(bordered.height * 0.04)
  const textX = (bordered.width - textW) / 2
  const textY = blockTopY + topPad

  // Semi-transparent white background
  ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'
  ctx.fillRect(
    textX - sidePad,
    blockTopY,
    textW + sidePad * 2,
    topPad + textH + bottomPad
  )

  // Label text
  ctx.fillStyle = 'black'
  ctx.textBaseline = 'top'
  ctx.fillText(labelText, textX, textY)

  return bordered
}

/**
 * Merge multiple canvases into a grid.
 * Port of Python merge_images().
 */
export function mergeImages(
  canvases: HTMLCanvasElement[],
  mergedHeight: number,
  gridRows: number,
  gridCols: number
): HTMLCanvasElement | null {
  if (canvases.length === 0) return null

  const rows = gridRows
  const cols = gridCols
  if (rows === 0 || cols === 0) return null

  const refW = canvases[0].width
  const refH = canvases[0].height
  if (refH === 0) return null

  const cellH = Math.floor(mergedHeight / rows)
  const cellW = Math.floor(cellH * (refW / refH))
  if (cellW === 0 || cellH === 0) return null

  const grid = document.createElement('canvas')
  grid.width = cols * cellW
  grid.height = rows * cellH
  const ctx = grid.getContext('2d')!
  ctx.fillStyle = 'white'
  ctx.fillRect(0, 0, grid.width, grid.height)

  for (let i = 0; i < canvases.length; i++) {
    const row = Math.floor(i / cols)
    const col = i % cols
    ctx.drawImage(canvases[i], col * cellW, row * cellH, cellW, cellH)
  }

  return grid
}

/**
 * Convert canvas to base64 data (without data: prefix).
 */
export function canvasToBase64(
  canvas: HTMLCanvasElement,
  mimeType = 'image/png'
): string {
  const dataUrl = canvas.toDataURL(mimeType)
  return dataUrl.split(',')[1]
}

/**
 * Convert canvas to Blob.
 */
export function canvasToBlob(
  canvas: HTMLCanvasElement,
  mimeType = 'image/jpeg',
  quality = 0.9
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('toBlob failed'))),
      mimeType,
      quality
    )
  })
}

/**
 * Convert canvas to a blob URL for display in <img>.
 */
export function canvasToBlobUrl(canvas: HTMLCanvasElement): Promise<string> {
  return canvasToBlob(canvas).then((blob) => URL.createObjectURL(blob))
}
