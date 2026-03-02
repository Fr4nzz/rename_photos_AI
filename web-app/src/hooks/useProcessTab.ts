import { useState, useCallback, useRef, useEffect } from 'react'
import { toast } from 'sonner'
import type { FileEntry, PhotoRow } from '@/types'
import { useSettingsStore } from '@/stores/settingsStore'
import { useApiKeysStore } from '@/stores/apiKeysStore'
import { useProcessingStore } from '@/stores/processingStore'
import { useBackendStore } from '@/stores/backendStore'
import {
  supportsDirectoryPicker,
  openDirectoryPicker,
  getImageFilesFromHandle,
  getImageFilesFromInput,
} from '@/lib/fileAccess'
import {
  loadImage,
  fixOrientation,
  rotateCanvas,
  preprocessImage,
  mergeImages,
  canvasToBase64,
  canvasToBlobUrl,
} from '@/lib/imageProcessing'
import { AIClient, parseJsonResponse, responseHasData } from '@/lib/aiClient'
import {
  saveCsvToStorage,
  loadCsvFromStorage,
  parseCsv,
  toCsvString,
  saveLastCsvName,
  saveDirHandle,
  getSavedDirHandle,
} from '@/lib/csvHandler'
import { logger } from '@/lib/logger'

export interface Previews {
  original: string | null
  rotated: string | null
  processed: string | null
  grid: string | null
}

export function useProcessTab() {
  const settings = useSettingsStore()
  const { apiKeys } = useApiKeysStore()
  const processing = useProcessingStore()
  const { status: backendStatus } = useBackendStore()

  const [imageFiles, setImageFiles] = useState<FileEntry[]>([])
  const [selectedImageIndex, setSelectedImageIndex] = useState(0)
  const [selectedGridIndex, setSelectedGridIndex] = useState(0)
  const [previews, setPreviews] = useState<Previews>({
    original: null, rotated: null, processed: null, grid: null,
  })
  const [dirHandle, setDirHandle] = useState<FileSystemDirectoryHandle | null>(null)
  const [folderName, setFolderName] = useState('')
  const abortRef = useRef<AbortController | null>(null)

  const fileType = settings.previewRaw ? 'all' : 'compressed'

  // Load saved directory handle on mount
  useEffect(() => {
    if (!supportsDirectoryPicker()) return
    getSavedDirHandle().then(async (handle) => {
      if (!handle) return
      try {
        const perm = await (handle as any).queryPermission({ mode: 'read' })
        if (perm === 'granted') {
          setDirHandle(handle)
          setFolderName(handle.name)
          const files = await getImageFilesFromHandle(handle, fileType)
          setImageFiles(files)
          // Also populate fileMap for Review tab thumbnails
          const fMap = new Map<string, File>()
          for (const entry of files) fMap.set(entry.name, entry.file)
          processing.setFileMap(fMap)
          logger.info(`Restored folder: ${handle.name} (${files.length} images)`)
        }
      } catch {
        // Permission denied or handle invalid — ignore
      }
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Select directory
  const selectDirectory = useCallback(async () => {
    try {
      if (supportsDirectoryPicker()) {
        const handle = await openDirectoryPicker()
        setDirHandle(handle)
        setFolderName(handle.name)
        const files = await getImageFilesFromHandle(handle, fileType)
        setImageFiles(files)
        setSelectedImageIndex(0)
        setSelectedGridIndex(0)
        saveDirHandle(handle).catch(() => {})
        logger.info(`Opened folder: ${handle.name} (${files.length} images)`)
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        toast.error(`Failed to open folder: ${e.message}`)
      }
    }
  }, [fileType])

  // Handle fallback file input
  const handleFileInput = useCallback(
    (fileList: FileList) => {
      const files = getImageFilesFromInput(fileList, fileType)
      setImageFiles(files)
      setSelectedImageIndex(0)
      setSelectedGridIndex(0)
      if (files.length > 0) {
        const path = (files[0].file as any).webkitRelativePath || ''
        const folder = path.split('/')[0] || 'Selected folder'
        setFolderName(folder)
      }
      logger.info(`Selected ${files.length} images via file input`)
    },
    [fileType]
  )

  // Update previews when selection or settings change
  const updatePreviews = useCallback(async () => {
    if (imageFiles.length === 0 || selectedImageIndex >= imageFiles.length) {
      setPreviews({ original: null, rotated: null, processed: null, grid: null })
      return
    }

    try {
      const entry = imageFiles[selectedImageIndex]
      const img = await loadImage(entry.file)
      const oriented = await fixOrientation(img, entry.file)

      const originalUrl = await canvasToBlobUrl(oriented)
      const rotated = rotateCanvas(oriented, settings.rotationAngle)
      const rotatedUrl = await canvasToBlobUrl(rotated)

      // Processed: apply prerotate logic then preprocess
      let forGemini: HTMLCanvasElement
      if (settings.cropSettings.prerotate) {
        forGemini = rotateCanvas(
          settings.useExif ? oriented : await toPlainCanvas(img),
          settings.rotationAngle
        )
      } else {
        forGemini = oriented
      }
      const processed = preprocessImage(forGemini, '1', settings.cropSettings)
      const processedUrl = await canvasToBlobUrl(processed)

      setPreviews((prev) => ({
        ...prev,
        original: originalUrl,
        rotated: rotatedUrl,
        processed: processedUrl,
      }))
    } catch (e: any) {
      logger.error(`Preview error: ${e.message}`)
    }
  }, [imageFiles, selectedImageIndex, settings.rotationAngle, settings.useExif, settings.cropSettings])

  // Update grid preview
  const updateGridPreview = useCallback(async () => {
    if (imageFiles.length === 0) return

    const gridSize = settings.gridRows * settings.gridCols
    const startIdx = selectedGridIndex * gridSize

    try {
      const canvases: HTMLCanvasElement[] = []
      for (let i = startIdx; i < Math.min(startIdx + gridSize, imageFiles.length); i++) {
        const entry = imageFiles[i]
        const img = await loadImage(entry.file)
        const oriented = await fixOrientation(img, entry.file)

        let forGemini: HTMLCanvasElement
        if (settings.cropSettings.prerotate) {
          forGemini = rotateCanvas(oriented, settings.rotationAngle)
        } else {
          forGemini = oriented
        }
        canvases.push(
          preprocessImage(forGemini, String(i + 1), settings.cropSettings)
        )
      }

      const merged = mergeImages(canvases, settings.mergedImgHeight, settings.gridRows, settings.gridCols)
      if (merged) {
        const url = await canvasToBlobUrl(merged)
        setPreviews((prev) => ({ ...prev, grid: url }))
      }
    } catch (e: any) {
      logger.error(`Grid preview error: ${e.message}`)
    }
  }, [imageFiles, selectedGridIndex, settings.gridRows, settings.gridCols, settings.mergedImgHeight, settings.rotationAngle, settings.cropSettings])

  // Trigger preview updates
  useEffect(() => {
    updatePreviews()
  }, [updatePreviews])

  useEffect(() => {
    updateGridPreview()
  }, [updateGridPreview])

  // Parse retry message ranges like "1,3,5-7" → Set<number>
  function parseRetryBatches(input: string): Set<number> {
    const result = new Set<number>()
    for (const part of input.split(',')) {
      const trimmed = part.trim()
      if (trimmed.includes('-')) {
        const [start, end] = trimmed.split('-').map(Number)
        for (let i = start; i <= end; i++) result.add(i)
      } else {
        const n = Number(trimmed)
        if (!isNaN(n)) result.add(n)
      }
    }
    return result
  }

  // Start processing
  const startProcessing = useCallback(async () => {
    if (apiKeys.length === 0) {
      toast.error('No API keys configured. Go to the API Keys tab.')
      return
    }
    if (imageFiles.length === 0) {
      toast.error('No images loaded. Select a folder first.')
      return
    }
    if (!settings.modelName) {
      toast.error('No model selected.')
      return
    }

    const { runMode, retryMessages, continueCsvName } = useProcessingStore.getState()

    processing.setProcessing(true)
    processing.setProgress(0, 'Starting...')
    processing.setFailedBatches([])
    abortRef.current = new AbortController()

    try {
      const client = new AIClient(apiKeys, settings.modelName)
      const gridSize = settings.gridRows * settings.gridCols
      const photosPerApiCall = settings.imagesPerPrompt * gridSize
      const totalPhotos = imageFiles.length

      // Build initial photo rows — keep local ref to avoid stale closure reads
      let rows: PhotoRow[] = imageFiles.map((entry, i) => ({
        from: entry.name,
        currentPath: entry.name,
        photoId: i + 1,
        mainValue: '',
        co: '',
        n: '',
        skip: '',
        to: '',
        suffix: '',
        batchNumber: 0,
        captureDate: null,
        status: 'Original',
      }))

      // Continue mode: merge existing CSV data into rows
      if (runMode === 'continue' && continueCsvName) {
        const csvText = await loadCsvFromStorage(continueCsvName)
        if (csvText) {
          const existingRows = parseCsv(csvText, settings.mainColumn)
          const existingMap = new Map<string, PhotoRow>()
          for (const r of existingRows) existingMap.set(r.from, r)

          rows = rows.map((r) => {
            const existing = existingMap.get(r.from)
            if (existing && existing.mainValue?.trim()) {
              return { ...r, ...existing, photoId: r.photoId }
            }
            return r
          })
          logger.info(`Continue mode: loaded ${existingRows.length} rows from ${continueCsvName}`)
        }
      }

      processing.setPhotoRows(rows)

      // Store file references for Review tab thumbnails
      const fMap = new Map<string, File>()
      for (const entry of imageFiles) {
        fMap.set(entry.name, entry.file)
      }
      processing.setFileMap(fMap)

      const totalBatches = Math.ceil(totalPhotos / photosPerApiCall)
      const failedBatches: number[] = []

      // Determine which batches to process
      const retrySet = runMode === 'retry_specific' ? parseRetryBatches(retryMessages) : null

      for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
        if (abortRef.current.signal.aborted) break

        const batchNum = batchIdx + 1

        // Skip batches based on mode
        if (retrySet && !retrySet.has(batchNum)) continue
        if (runMode === 'continue') {
          // Skip batches where all rows already have data
          const batchStart = batchIdx * photosPerApiCall
          const batchEnd = Math.min(batchStart + photosPerApiCall, totalPhotos)
          const allHaveData = rows.slice(batchStart, batchEnd).every((r) => r.mainValue?.trim())
          if (allHaveData) {
            processing.setProgress(
              Math.floor(((batchIdx + 1) / totalBatches) * 100),
              `Skipping message ${batchNum}/${totalBatches} (already processed)`
            )
            continue
          }
        }

        const batchStart = batchIdx * photosPerApiCall
        const batchEnd = Math.min(batchStart + photosPerApiCall, totalPhotos)

        const mergedImages: string[] = []
        const labelRanges: [number, number][] = []

        for (let imgIdx = 0; imgIdx < settings.imagesPerPrompt; imgIdx++) {
          if (abortRef.current.signal.aborted) break

          const segStart = batchStart + imgIdx * gridSize
          const segEnd = Math.min(segStart + gridSize, batchEnd)
          if (segStart >= batchEnd) break

          const canvases: HTMLCanvasElement[] = []
          for (let i = segStart; i < segEnd; i++) {
            const entry = imageFiles[i]
            const img = await loadImage(entry.file)
            const oriented = await fixOrientation(img, entry.file)

            let forGemini: HTMLCanvasElement
            if (settings.cropSettings.prerotate) {
              forGemini = rotateCanvas(oriented, settings.rotationAngle)
            } else {
              forGemini = oriented
            }
            canvases.push(
              preprocessImage(forGemini, String(i + 1), settings.cropSettings)
            )
          }

          const merged = mergeImages(
            canvases,
            settings.mergedImgHeight,
            settings.gridRows,
            settings.gridCols
          )
          if (merged) {
            mergedImages.push(canvasToBase64(merged))
            labelRanges.push([segStart + 1, segEnd])
          }

          const step = batchIdx * settings.imagesPerPrompt + imgIdx + 1
          const totalSteps = totalBatches * (settings.imagesPerPrompt + 1)
          processing.setProgress(
            Math.floor((step / totalSteps) * 100),
            `Merging image ${imgIdx + 1}/${settings.imagesPerPrompt} for message ${batchNum}`
          )
        }

        if (abortRef.current.signal.aborted || mergedImages.length === 0) continue

        processing.setProgress(
          useProcessingStore.getState().progress.percent,
          `Sending ${mergedImages.length} images to AI...`
        )

        const labelDesc = labelRanges
          .map(([s, e], i) => `Image ${i + 1}: labels ${s}-${e}`)
          .join(', ')
        const prompt = `${settings.promptText}\n\nAnalyze ${mergedImages.length} images. ${labelDesc}.`

        const { text: responseText, success } = await client.sendRequest(
          prompt,
          mergedImages
        )

        if (!success) {
          toast.error(`Message ${batchNum} failed: ${responseText}`)
          break
        }

        if (!responseHasData(responseText, settings.mainColumn)) {
          failedBatches.push(batchNum)
          logger.warn(`Message ${batchNum} returned empty response`)
        }

        // Parse and update rows using local `rows` to avoid stale closure
        const data = parseJsonResponse(responseText, settings.mainColumn)
        if (data) {
          rows = rows.map((r) => {
            const itemData = data[String(r.photoId)]
            if (itemData) {
              return {
                ...r,
                mainValue: itemData[settings.mainColumn] ?? '',
                co: itemData.co ?? '',
                n: itemData.n ?? '',
                skip: itemData.skip ?? '',
                batchNumber: batchNum,
              }
            }
            return r
          })
          processing.setPhotoRows(rows)
        }

        processing.setProgress(
          Math.floor(((batchIdx + 1) / totalBatches) * 100),
          `Message ${batchNum}/${totalBatches} complete`
        )

        // Auto-save intermediate results after each batch
        if (rows.length > 0) {
          const csvName = `${folderName || 'results'}_${new Date().toISOString().slice(0, 10)}.csv`
          const csvText = toCsvString(rows, settings.mainColumn)
          await saveCsvToStorage(csvName, csvText)
          await saveLastCsvName(csvName)
          processing.setCurrentCsvName(csvName)
        }
      }

      processing.setFailedBatches(failedBatches)

      // Final save
      if (rows.length > 0 && !abortRef.current.signal.aborted) {
        const csvName = `${folderName || 'results'}_${new Date().toISOString().slice(0, 10)}.csv`
        const csvText = toCsvString(rows, settings.mainColumn)
        await saveCsvToStorage(csvName, csvText)
        await saveLastCsvName(csvName)
        processing.setCurrentCsvName(csvName)
        logger.info(`Saved results as: ${csvName}`)
      }

      if (failedBatches.length > 0) {
        toast.warning(
          `Processing complete. Messages with empty responses: ${failedBatches.join(', ')}`
        )
      } else if (!abortRef.current.signal.aborted) {
        toast.success('Processing complete! Results saved.')
      }
    } catch (e: any) {
      toast.error(`Processing failed: ${e.message}`)
      logger.error(`Processing error: ${e.message}`)
    } finally {
      processing.setProcessing(false)
    }
  }, [apiKeys, imageFiles, settings, processing, folderName])

  const stopProcessing = useCallback(() => {
    abortRef.current?.abort()
    processing.setProcessing(false)
    toast.info('Processing stopped')
  }, [processing])

  const gridCount = Math.ceil(
    imageFiles.length / (settings.gridRows * settings.gridCols)
  )

  return {
    imageFiles,
    selectedImageIndex,
    setSelectedImageIndex,
    selectedGridIndex,
    setSelectedGridIndex,
    previews,
    folderName,
    dirHandle,
    gridCount,
    backendConnected: backendStatus.connected,
    selectDirectory,
    handleFileInput,
    startProcessing,
    stopProcessing,
  }
}

async function toPlainCanvas(img: HTMLImageElement): Promise<HTMLCanvasElement> {
  const c = document.createElement('canvas')
  c.width = img.naturalWidth
  c.height = img.naturalHeight
  c.getContext('2d')!.drawImage(img, 0, 0)
  return c
}
