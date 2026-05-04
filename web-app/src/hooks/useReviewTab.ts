import { useState, useCallback, useMemo, useEffect, useRef } from 'react'
import { toast } from 'sonner'
import type { PhotoRow } from '@/types'
import { useSettingsStore } from '@/stores/settingsStore'
import { useProcessingStore } from '@/stores/processingStore'
import { calculateFinalNames } from '@/lib/nameCalculator'
import {
  listStoredCsvs,
  loadCsvFromStorage,
  saveCsvToStorage,
  parseCsv,
  toCsvString,
  downloadCsv,
  getLastCsvName,
  saveLastCsvName,
  saveCsvToFolder,
} from '@/lib/csvHandler'
import { logger } from '@/lib/logger'

export type FilterType = 'all' | 'crossedOut' | 'hasNotes' | 'skipped' | 'mismatches'
export type SortOption =
  | 'name-asc' | 'name-desc'
  | 'batch-asc' | 'batch-desc'
  | 'cam-asc' | 'cam-desc'
  | 'date-asc' | 'date-desc'

export function useReviewTab() {
  const { mainColumn, suffixMode, customSuffixes, reviewItemsPerPage } = useSettingsStore()
  const { photoRows, setPhotoRows } = useProcessingStore()

  const [csvFiles, setCsvFiles] = useState<string[]>([])
  const [selectedCsv, setSelectedCsv] = useState('')
  const [filter, setFilter] = useState<FilterType>('all')
  const [sortOption, setSortOption] = useState<SortOption>('name-asc')
  const [currentPage, setCurrentPage] = useState(1)

  // Debounced autosave
  const autosaveTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const prevRowsRef = useRef<PhotoRow[]>(photoRows)

  const refreshCsvList = useCallback(async () => {
    const csvs = await listStoredCsvs()
    setCsvFiles(csvs)
  }, [])

  // Load CSV list on mount + auto-load last CSV if no data in store
  useEffect(() => {
    listStoredCsvs().then(setCsvFiles)
    if (photoRows.length === 0) {
      logger.time('Startup: auto-load CSV')
      getLastCsvName().then(async (name) => {
        if (!name) { logger.timeEnd('Startup: auto-load CSV'); return }
        const text = await loadCsvFromStorage(name)
        if (!text) { logger.timeEnd('Startup: auto-load CSV'); return }
        const rows = parseCsv(text, mainColumn)
        if (rows.length > 0) {
          setPhotoRows(rows)
          setSelectedCsv(name)
          logger.info(`Auto-loaded last CSV: ${name} (${rows.length} rows)`)
        }
        logger.timeEnd('Startup: auto-load CSV')
      })
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Autosave when photoRows change (debounced 2s)
  useEffect(() => {
    if (photoRows.length === 0 || photoRows === prevRowsRef.current) return
    prevRowsRef.current = photoRows

    clearTimeout(autosaveTimer.current)
    autosaveTimer.current = setTimeout(async () => {
      const { currentCsvName, dirHandle } = useProcessingStore.getState()
      const csvName = selectedCsv || currentCsvName
      if (!csvName) return
      const text = toCsvString(photoRows, mainColumn)
      await saveCsvToStorage(csvName, text)
      // Also persist to rename_files/ on disk
      if (dirHandle) await saveCsvToFolder(dirHandle, csvName, text)
      logger.info(`Auto-saved: ${csvName}`)
    }, 2000)

    return () => clearTimeout(autosaveTimer.current)
  }, [photoRows, selectedCsv, mainColumn])

  const loadCsv = useCallback(
    async (name: string) => {
      const text = await loadCsvFromStorage(name)
      if (!text) {
        toast.error(`CSV "${name}" not found`)
        return
      }
      const rows = parseCsv(text, mainColumn)
      setPhotoRows(rows)
      setSelectedCsv(name)
      setCurrentPage(1)
      logger.info(`Loaded CSV: ${name} (${rows.length} rows)`)
    },
    [mainColumn, setPhotoRows]
  )

  // Compute duplicate CAM+suffix pairs for warning display
  const duplicatePairs = useMemo(() => {
    const counts = new Map<string, number>()
    for (const r of photoRows) {
      if (r.mainValue?.trim() && r.suffix?.trim()) {
        const key = `${r.mainValue.trim()}__${r.suffix.trim()}`
        counts.set(key, (counts.get(key) ?? 0) + 1)
      }
    }
    const dupes = new Set<string>()
    for (const [key, count] of counts) {
      if (count > 1) dupes.add(key)
    }
    return dupes
  }, [photoRows])

  // Filter rows
  const filteredRows = useMemo(() => {
    let rows = photoRows

    switch (filter) {
      case 'crossedOut':
        rows = rows.filter((r) => r.co?.trim())
        break
      case 'hasNotes':
        rows = rows.filter((r) => r.n?.trim())
        break
      case 'skipped':
        rows = rows.filter((r) => r.skip === 'x')
        break
      case 'mismatches': {
        const idCounts = new Map<string, number>()
        const suffixKeys = new Set<string>()
        const duplicateSuffix = new Set<number>()

        rows.forEach((r) => {
          if (r.mainValue?.trim()) {
            idCounts.set(r.mainValue, (idCounts.get(r.mainValue) ?? 0) + 1)
          }
          if (r.mainValue && r.suffix) {
            const key = `${r.mainValue}__${r.suffix}`
            if (suffixKeys.has(key)) {
              duplicateSuffix.add(r.photoId)
            }
            suffixKeys.add(key)
          }
        })

        rows = rows.filter(
          (r) =>
            (r.mainValue?.trim() && idCounts.get(r.mainValue) !== 2) ||
            duplicateSuffix.has(r.photoId)
        )
        break
      }
    }

    // Sort
    const sorted = [...rows]
    sorted.sort((a, b) => {
      switch (sortOption) {
        case 'name-asc': return a.from.localeCompare(b.from)
        case 'name-desc': return b.from.localeCompare(a.from)
        case 'batch-asc': return a.batchNumber - b.batchNumber
        case 'batch-desc': return b.batchNumber - a.batchNumber
        case 'cam-asc': return (a.mainValue ?? '').localeCompare(b.mainValue ?? '')
        case 'cam-desc': return (b.mainValue ?? '').localeCompare(a.mainValue ?? '')
        case 'date-asc': return (a.captureDate ?? '').localeCompare(b.captureDate ?? '')
        case 'date-desc': return (b.captureDate ?? '').localeCompare(a.captureDate ?? '')
        default: return 0
      }
    })

    return sorted
  }, [photoRows, filter, sortOption])

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / reviewItemsPerPage))
  const pagedRows = filteredRows.slice(
    (currentPage - 1) * reviewItemsPerPage,
    currentPage * reviewItemsPerPage
  )

  const updateRow = useCallback(
    (photoId: number, updates: Partial<PhotoRow>) => {
      const rows = photoRows.map((r) =>
        r.photoId === photoId ? { ...r, ...updates } : r
      )
      setPhotoRows(rows)
    },
    [photoRows, setPhotoRows]
  )

  const recalculateNames = useCallback(() => {
    const updated = calculateFinalNames(photoRows, mainColumn, suffixMode, customSuffixes)
    setPhotoRows(updated)
    toast.success('Names recalculated')
  }, [photoRows, mainColumn, suffixMode, customSuffixes, setPhotoRows])

  const saveChanges = useCallback(async () => {
    const csvName = selectedCsv || `results_${Date.now()}.csv`
    const text = toCsvString(photoRows, mainColumn)
    await saveCsvToStorage(csvName, text)
    await saveLastCsvName(csvName)
    // Also persist to rename_files/ on disk
    const dh = useProcessingStore.getState().dirHandle
    if (dh) await saveCsvToFolder(dh, csvName, text)
    setSelectedCsv(csvName)
    await refreshCsvList()
    toast.success(`Saved: ${csvName}`)
  }, [photoRows, mainColumn, selectedCsv, refreshCsvList])

  const exportCsv = useCallback(() => {
    const text = toCsvString(photoRows, mainColumn)
    const name = selectedCsv || `results_${Date.now()}.csv`
    downloadCsv(text, name)
  }, [photoRows, mainColumn, selectedCsv])

  return {
    csvFiles,
    selectedCsv,
    setSelectedCsv,
    filter,
    setFilter,
    sortOption,
    setSortOption,
    currentPage,
    setCurrentPage,
    totalPages,
    filteredRows,
    pagedRows,
    photoRows,
    duplicatePairs,
    refreshCsvList,
    loadCsv,
    updateRow,
    recalculateNames,
    saveChanges,
    exportCsv,
  }
}
