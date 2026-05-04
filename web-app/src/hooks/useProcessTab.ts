import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { toast } from 'sonner'
import type { PhotoRow } from '@/types'
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
  loadImagePreview,
  clearPreviewCache,
  rotateCanvas,
  rotateBrowserImageFile,
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
  saveCsvToFolder,
  listCsvsFromFolder,
  loadCsvFromFolder,
} from '@/lib/csvHandler'
import { BROWSER_ROTATABLE_EXTENSIONS } from '@/lib/constants'
import { getErrorMessage, getErrorName } from '@/lib/errors'
import { getSelectedImageFiles } from '@/lib/selection'
import { logger } from '@/lib/logger'

export interface Previews {
  original: string | null
  rotated: string | null
  processed: string | null
  grid: string | null
}

function getMaxParallelRequests(modelName: string, configuredParallelRequests: number): number {
  const normalized = modelName.toLowerCase()
  const configured = Math.max(1, configuredParallelRequests)
  if (/gemini-3(?:\.\d+)?-.*flash/.test(normalized)) return configured
  return 2
}

function getRequestsPerMinuteLimit(modelName: string): number {
  const normalized = modelName.toLowerCase()
  if (normalized.includes('gemini-3.1') && normalized.includes('flash-lite')) return 15
  if (/gemini-3(?:\.\d+)?-.*flash/.test(normalized)) return 5
  return 2
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.resolve()
  return new Promise((resolve) => {
    const timeout = window.setTimeout(resolve, ms)
    signal.addEventListener(
      'abort',
      () => {
        window.clearTimeout(timeout)
        resolve()
      },
      { once: true }
    )
  })
}

export function useProcessTab() {
  const settings = useSettingsStore()
  const { apiKeys } = useApiKeysStore()
  const processing = useProcessingStore()
  const { status: backendStatus } = useBackendStore()

  const imageFiles = useMemo(
    () => getSelectedImageFiles(processing.imageFiles, processing.selectedImageNames),
    [processing.imageFiles, processing.selectedImageNames]
  )
  const [selectedImageIndex, setSelectedImageIndex] = useState(0)
  const [selectedGridIndex, setSelectedGridIndex] = useState(0)
  const [previews, setPreviews] = useState<Previews>({
    original: null, rotated: null, processed: null, grid: null,
  })
  const [dirHandle, setDirHandle] = useState<FileSystemDirectoryHandle | null>(null)
  const [folderName, setFolderName] = useState('')
  const abortRef = useRef<AbortController | null>(null)

  const fileType = settings.previewRaw ? 'all' : 'compressed'

  // Import folder CSVs into IndexedDB + auto-load latest
  const importFolderCsvs = useCallback(async (handle: FileSystemDirectoryHandle) => {
    logger.time('Import folder CSVs')
    const folderCsvs = await listCsvsFromFolder(handle)
    if (folderCsvs.length === 0) { logger.timeEnd('Import folder CSVs'); return }

    // Import each CSV from folder into IndexedDB
    for (const name of folderCsvs) {
      const text = await loadCsvFromFolder(handle, name)
      if (text) await saveCsvToStorage(name, text)
    }

    // Auto-load the most recent CSV (last alphabetically, since names include dates)
    const latest = folderCsvs[folderCsvs.length - 1]
    const text = await loadCsvFromFolder(handle, latest)
    if (text) {
      const { mainColumn } = useSettingsStore.getState()
      const rows = parseCsv(text, mainColumn)
      if (rows.length > 0) {
        processing.setPhotoRows(rows)
        processing.setCurrentCsvName(latest)
        await saveLastCsvName(latest)
        toast.success(`Loaded ${rows.length} rows from ${latest}`)
        logger.info(`Auto-loaded CSV from rename_files/: ${latest} (${rows.length} rows)`)
      }
    }
    logger.timeEnd('Import folder CSVs')
  }, [processing])

  // Load saved directory handle on mount
  useEffect(() => {
    if (!supportsDirectoryPicker()) return
    logger.time('Startup: restore folder')
    getSavedDirHandle().then(async (handle) => {
      if (!handle) { logger.timeEnd('Startup: restore folder'); return }
      try {
        const perm = await handle.queryPermission?.({ mode: 'read' })
        if (perm === 'granted') {
          setDirHandle(handle)
          setFolderName(handle.name)
          processing.setDirHandle(handle)

          logger.time('Startup: list image files')
          const files = await getImageFilesFromHandle(handle, fileType)
          logger.timeEnd('Startup: list image files')

          processing.setImageFiles(files, true)
          // Also populate fileMap for Review tab thumbnails
          logger.time('Startup: build fileMap')
          const fMap = new Map<string, File>()
          for (const entry of files) fMap.set(entry.name, entry.file)
          processing.setFileMap(fMap)
          logger.timeEnd('Startup: build fileMap')

          logger.info(`Restored folder: ${handle.name} (${files.length} images)`)

          // Import CSVs from rename_files/ if no data loaded yet
          if (useProcessingStore.getState().photoRows.length === 0) {
            await importFolderCsvs(handle)
          }
        }
      } catch {
        // Permission denied or handle invalid — ignore
      }
      logger.timeEnd('Startup: restore folder')
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Select directory
  const selectDirectory = useCallback(async () => {
    try {
      if (supportsDirectoryPicker()) {
        const handle = await openDirectoryPicker()
        clearPreviewCache()
        setDirHandle(handle)
        setFolderName(handle.name)
        processing.setDirHandle(handle)

        logger.time('Open folder: list images')
        const files = await getImageFilesFromHandle(handle, fileType)
        logger.timeEnd('Open folder: list images')

        processing.setImageFiles(files, true)
        setSelectedImageIndex(0)
        setSelectedGridIndex(0)
        // Build fileMap for Review tab thumbnails
        const fMap = new Map<string, File>()
        for (const entry of files) fMap.set(entry.name, entry.file)
        processing.setFileMap(fMap)
        saveDirHandle(handle).catch(() => {})
        logger.info(`Opened folder: ${handle.name} (${files.length} images)`)

        // Import CSVs from rename_files/ subfolder
        await importFolderCsvs(handle)
      }
    } catch (e: unknown) {
      if (getErrorName(e) !== 'AbortError') {
        toast.error(`Failed to open folder: ${getErrorMessage(e)}`)
      }
    }
  }, [fileType, importFolderCsvs, processing])

  // Handle fallback file input
  const handleFileInput = useCallback(
    (fileList: FileList) => {
      const files = getImageFilesFromInput(fileList, fileType)
      processing.setImageFiles(files, true)
      setSelectedImageIndex(0)
      setSelectedGridIndex(0)
      if (files.length > 0) {
        const path = (files[0].file as FileWithRelativePath).webkitRelativePath || ''
        const folder = path.split('/')[0] || 'Selected folder'
        setFolderName(folder)
      }
      logger.info(`Selected ${files.length} images via file input`)
    },
    [fileType, processing]
  )

  const applyRotationToFiles = useCallback(async () => {
    if (!dirHandle) {
      toast.error('Open a folder first so the app can write rotated files.')
      return
    }
    if (settings.rotationAngle === 0) {
      toast.info('Rotation is set to 0°, so no files need changes.')
      return
    }

    const rotatableFiles = imageFiles.filter((entry) =>
      BROWSER_ROTATABLE_EXTENSIONS.has(entry.extension)
    )
    const skippedCount = imageFiles.length - rotatableFiles.length

    if (rotatableFiles.length === 0) {
      toast.error('No JPEG or PNG files are loaded for browser rotation.')
      return
    }

    const confirmed = window.confirm(
      `Rotate ${rotatableFiles.length} JPEG/PNG file(s) in place? ` +
      'This rewrites image pixels and may remove some metadata. Keep your originals backed up.'
    )
    if (!confirmed) return

    processing.setProcessing(true)
    processing.setProgress(0, 'Applying rotation to files...')

    let rotatedCount = 0
    const failed: string[] = []

    for (let i = 0; i < rotatableFiles.length; i++) {
      const entry = rotatableFiles[i]
      try {
        const fileHandle = await dirHandle.getFileHandle(entry.path)
        const latestFile = await fileHandle.getFile()
        const rotatedBlob = await rotateBrowserImageFile(
          latestFile,
          settings.rotationAngle,
          settings.useExif
        )
        if (!rotatedBlob) continue

        const writable = await fileHandle.createWritable()
        await writable.write(rotatedBlob)
        await writable.close()
        rotatedCount++
      } catch (error: unknown) {
        failed.push(`${entry.name}: ${getErrorMessage(error)}`)
        logger.warn(`Could not rotate ${entry.name}: ${getErrorMessage(error)}`)
      }

      processing.setProgress(
        Math.round(((i + 1) / rotatableFiles.length) * 100),
        `Rotated ${i + 1}/${rotatableFiles.length} browser-supported files`
      )
    }

    clearPreviewCache()
    const refreshedFiles = await getImageFilesFromHandle(dirHandle, fileType)
    processing.setImageFiles(refreshedFiles, false)
    const fMap = new Map<string, File>()
    for (const entry of refreshedFiles) fMap.set(entry.name, entry.file)
    processing.setFileMap(fMap)

    processing.setProcessing(false)
    processing.setProgress(100, 'Rotation complete')

    if (failed.length > 0) {
      toast.warning(`Rotated ${rotatedCount} file(s); ${failed.length} failed. Check console logs.`)
    } else if (skippedCount > 0) {
      toast.success(`Rotated ${rotatedCount} JPEG/PNG file(s). Skipped ${skippedCount} unsupported file(s).`)
    } else {
      toast.success(`Rotated ${rotatedCount} JPEG/PNG file(s).`)
    }
  }, [dirHandle, fileType, imageFiles, processing, settings.rotationAngle, settings.useExif])

  // Update previews when selection or settings change
  const updatePreviews = useCallback(async () => {
    if (imageFiles.length === 0 || selectedImageIndex >= imageFiles.length) {
      setPreviews({ original: null, rotated: null, processed: null, grid: null })
      return
    }

    try {
      logger.time('Preview: generate')
      const entry = imageFiles[selectedImageIndex]
      const oriented = await loadImagePreview(entry.file, 800)

      const originalUrl = await canvasToBlobUrl(oriented)
      const rotated = rotateCanvas(oriented, settings.rotationAngle)
      const rotatedUrl = await canvasToBlobUrl(rotated)

      // Processed: apply prerotate logic then preprocess
      let forGemini: HTMLCanvasElement
      if (settings.cropSettings.prerotate) {
        const source = settings.useExif
          ? oriented
          : await loadImagePreview(entry.file, 800, false)
        forGemini = rotateCanvas(source, settings.rotationAngle)
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
      logger.timeEnd('Preview: generate')
    } catch (e: unknown) {
      logger.error(`Preview error: ${getErrorMessage(e)}`)
    }
  }, [imageFiles, selectedImageIndex, settings.rotationAngle, settings.useExif, settings.cropSettings])

  // Update grid preview
  const updateGridPreview = useCallback(async () => {
    if (imageFiles.length === 0) return

    const gridSize = settings.gridRows * settings.gridCols
    const startIdx = selectedGridIndex * gridSize

    try {
      logger.time('Grid preview: generate')
      const endIdx = Math.min(startIdx + gridSize, imageFiles.length)
      const canvases = await Promise.all(
        imageFiles.slice(startIdx, endIdx).map(async (entry, k) => {
          const oriented = await loadImagePreview(entry.file, 800)
          const forGemini = settings.cropSettings.prerotate
            ? rotateCanvas(oriented, settings.rotationAngle)
            : oriented
          return preprocessImage(forGemini, String(startIdx + k + 1), settings.cropSettings)
        })
      )

      const merged = mergeImages(canvases, settings.mergedImgHeight, settings.gridRows, settings.gridCols)
      if (merged) {
        const url = await canvasToBlobUrl(merged)
        setPreviews((prev) => ({ ...prev, grid: url }))
      }
      logger.timeEnd('Grid preview: generate')
    } catch (e: unknown) {
      logger.error(`Grid preview error: ${getErrorMessage(e)}`)
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
    // Re-read directory for fresh File objects (handles renamed/restored files)
    let files = imageFiles
    const currentDirHandle = useProcessingStore.getState().dirHandle
    if (currentDirHandle) {
      const refreshed = await getImageFilesFromHandle(currentDirHandle, fileType)
      const selectedNames = useProcessingStore.getState().selectedImageNames
      processing.setImageFiles(refreshed, false)
      processing.setSelectedImageNames(selectedNames)
      files = getSelectedImageFiles(refreshed, selectedNames)
    }

    if (files.length === 0) {
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
    const abortController = new AbortController()
    abortRef.current = abortController
    logger.time('Total processing')

    try {
      const client = new AIClient(apiKeys, settings.modelName)
      const gridSize = settings.gridRows * settings.gridCols
      const photosPerApiCall = settings.imagesPerPrompt * gridSize
      const totalPhotos = files.length

      // Build initial photo rows — keep local ref to avoid stale closure reads
      let rows: PhotoRow[] = files.map((entry, i) => ({
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
      for (const entry of files) {
        fMap.set(entry.name, entry.file)
      }
      processing.setFileMap(fMap)

      const totalBatches = Math.ceil(totalPhotos / photosPerApiCall)
      const failedBatches: number[] = []

      // Determine which batches to process
      const retrySet = runMode === 'retry_specific' ? parseRetryBatches(retryMessages) : null
      const activeBatches: number[] = []

      for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
        if (retrySet && !retrySet.has(batchIdx + 1)) continue
        if (runMode === 'continue') {
          const bStart = batchIdx * photosPerApiCall
          const bEnd = Math.min(bStart + photosPerApiCall, totalPhotos)
          if (rows.slice(bStart, bEnd).every((r) => r.mainValue?.trim())) {
            processing.setProgress(
              Math.floor(((batchIdx + 1) / totalBatches) * 100),
              `Skipping message ${batchIdx + 1}/${totalBatches} (already processed)`
            )
            continue
          }
        }
        activeBatches.push(batchIdx)
      }

      // Sliding worker queue: starts the next message as soon as one finishes.
      const requestsPerMinuteLimit = getRequestsPerMinuteLimit(settings.modelName)
      const maxParallelRequests = Math.min(
        getMaxParallelRequests(settings.modelName, settings.parallelRequests ?? 5),
        requestsPerMinuteLimit
      )
      const requestStartTimes: number[] = []
      let hadFailure = false
      let nextActiveBatchIndex = 0
      let completedBatches = 0

      async function waitForRequestSlot(batchNum: number) {
        while (!abortController.signal.aborted && !hadFailure) {
          const now = Date.now()
          while (requestStartTimes.length > 0 && now - requestStartTimes[0] >= 60_000) {
            requestStartTimes.shift()
          }

          if (requestStartTimes.length < requestsPerMinuteLimit) {
            requestStartTimes.push(now)
            return
          }

          const waitMs = Math.max(250, 60_000 - (now - requestStartTimes[0]))
          const waitSeconds = Math.ceil(waitMs / 1000)
          processing.setProgress(
            useProcessingStore.getState().progress.percent,
            `Waiting ${waitSeconds}s for Gemini rate limit before message ${batchNum}/${totalBatches}...`
          )
          await sleep(waitMs, abortController.signal)
        }
      }

      async function processBatch(batchIdx: number) {
        if (abortController.signal.aborted || hadFailure) return
        const batchNum = batchIdx + 1
        const batchStart = batchIdx * photosPerApiCall
        const batchEnd = Math.min(batchStart + photosPerApiCall, totalPhotos)

        // Build + process all grids in parallel
        const gridTasks: { idx: number; segStart: number; segEnd: number }[] = []
        for (let imgIdx = 0; imgIdx < settings.imagesPerPrompt; imgIdx++) {
          const segStart = batchStart + imgIdx * gridSize
          const segEnd = Math.min(segStart + gridSize, batchEnd)
          if (segStart >= batchEnd) break
          gridTasks.push({ idx: imgIdx, segStart, segEnd })
        }

        logger.time(`Batch ${batchNum}: image processing`)
        const gridResults = await Promise.all(
          gridTasks.map(async ({ idx, segStart, segEnd }) => {
            if (abortController.signal.aborted) return null
            const gridLabel = `Batch ${batchNum} grid ${idx + 1}`
            logger.time(`${gridLabel}: load+preprocess`)
            const results = await Promise.all(
              files.slice(segStart, segEnd).map(async (entry, k) => {
                try {
                  const oriented = await loadImagePreview(entry.file, settings.mergedImgHeight)
                  const forGemini = settings.cropSettings.prerotate
                    ? rotateCanvas(oriented, settings.rotationAngle)
                    : oriented
                  return preprocessImage(forGemini, String(segStart + k + 1), settings.cropSettings)
                } catch (e: unknown) {
                  logger.warn(`Skipping ${entry.name}: ${getErrorMessage(e)}`)
                  return null
                }
              })
            )
            const canvases = results.filter((c): c is HTMLCanvasElement => c !== null)
            logger.timeEnd(`${gridLabel}: load+preprocess`)

            logger.time(`${gridLabel}: merge+encode`)
            const merged = canvases.length > 0
              ? mergeImages(canvases, settings.mergedImgHeight, settings.gridRows, settings.gridCols)
              : null
            const base64 = merged ? canvasToBase64(merged, 'image/jpeg') : null
            logger.timeEnd(`${gridLabel}: merge+encode`)

            return base64 ? { base64, range: [segStart + 1, segEnd] as [number, number] } : null
          })
        )
        logger.timeEnd(`Batch ${batchNum}: image processing`)

        const mergedImages: string[] = []
        const labelRanges: [number, number][] = []
        for (const r of gridResults) {
          if (r) { mergedImages.push(r.base64); labelRanges.push(r.range) }
        }

        if (abortController.signal.aborted || mergedImages.length === 0) return

        processing.setProgress(
          useProcessingStore.getState().progress.percent,
          `Sending message ${batchNum}/${totalBatches} (${maxParallelRequests} parallel)...`
        )

        const labelDesc = labelRanges
          .map(([s, e], i) => `Image ${i + 1}: labels ${s}-${e}`)
          .join(', ')
        const prompt = `${settings.promptText}\n\nAnalyze ${mergedImages.length} images. ${labelDesc}.`

        await waitForRequestSlot(batchNum)
        if (abortController.signal.aborted || hadFailure) return

        logger.time(`Batch ${batchNum}: API call`)
        const { text: responseText, success } = await client.sendRequest(prompt, mergedImages)
        logger.timeEnd(`Batch ${batchNum}: API call`)

        if (!success) {
          toast.error(`Message ${batchNum} failed: ${responseText}`)
          hadFailure = true
          return
        }

        if (!responseHasData(responseText, settings.mainColumn)) {
          failedBatches.push(batchNum)
          logger.warn(`Message ${batchNum} returned empty response`)
        }

        // Each batch updates different photoId ranges — no conflicts
        const data = parseJsonResponse(responseText)
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

        completedBatches += 1
        processing.setProgress(
          Math.floor((completedBatches / Math.max(activeBatches.length, 1)) * 100),
          `Message ${batchNum}/${totalBatches} complete`
        )

        // Auto-save after each batch completes
        if (rows.length > 0) {
          const csvName = `${folderName || 'results'}_${new Date().toISOString().slice(0, 10)}.csv`
          const csvText = toCsvString(rows, settings.mainColumn)
          await saveCsvToStorage(csvName, csvText)
          await saveLastCsvName(csvName)
          processing.setCurrentCsvName(csvName)
          const dh = useProcessingStore.getState().dirHandle
          if (dh) await saveCsvToFolder(dh, csvName, csvText)
        }
      }

      async function batchWorker() {
        while (!abortController.signal.aborted && !hadFailure) {
          const batchIdx = activeBatches[nextActiveBatchIndex]
          nextActiveBatchIndex += 1
          if (batchIdx === undefined) return
          await processBatch(batchIdx)
        }
      }

      await Promise.all(
        Array.from(
          { length: Math.min(maxParallelRequests, activeBatches.length) },
          () => batchWorker()
        )
      )

      processing.setFailedBatches(failedBatches)

      // Final save (IndexedDB + folder)
      if (rows.length > 0 && !abortController.signal.aborted) {
        const csvName = `${folderName || 'results'}_${new Date().toISOString().slice(0, 10)}.csv`
        const csvText = toCsvString(rows, settings.mainColumn)
        await saveCsvToStorage(csvName, csvText)
        await saveLastCsvName(csvName)
        processing.setCurrentCsvName(csvName)
        const dh = useProcessingStore.getState().dirHandle
        if (dh) await saveCsvToFolder(dh, csvName, csvText)
        logger.info(`Saved results as: ${csvName}`)
      }

      if (failedBatches.length > 0) {
        toast.warning(
          `Processing complete. Messages with empty responses: ${failedBatches.join(', ')}`
        )
      } else if (!abortController.signal.aborted) {
        toast.success('Processing complete! Results saved.')
      }
    } catch (e: unknown) {
      toast.error(`Processing failed: ${getErrorMessage(e)}`)
      logger.error(`Processing error: ${getErrorMessage(e)}`)
    } finally {
      logger.timeEnd('Total processing')
      processing.setProcessing(false)
    }
  }, [apiKeys, imageFiles, settings, processing, folderName, fileType])

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
    canApplyBrowserRotation: !!dirHandle && settings.rotationAngle !== 0 && imageFiles.some((entry) =>
      BROWSER_ROTATABLE_EXTENSIONS.has(entry.extension)
    ),
    selectDirectory,
    handleFileInput,
    applyRotationToFiles,
    startProcessing,
    stopProcessing,
  }
}
