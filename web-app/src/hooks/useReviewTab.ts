import { useState, useCallback, useMemo, useEffect } from 'react'
import { toast } from 'sonner'
import type { PhotoRow, SuffixMode } from '@/types'
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
} from '@/lib/csvHandler'
import { logger } from '@/lib/logger'

export type FilterType = 'all' | 'crossedOut' | 'hasNotes' | 'skipped' | 'mismatches'
export type SortOption =
  | 'name-asc' | 'name-desc'
  | 'batch-asc' | 'batch-desc'
  | 'cam-asc' | 'cam-desc'

export function useReviewTab() {
  const { mainColumn, suffixMode, customSuffixes, reviewItemsPerPage } = useSettingsStore()
  const { photoRows, setPhotoRows } = useProcessingStore()

  const [csvFiles, setCsvFiles] = useState<string[]>([])
  const [selectedCsv, setSelectedCsv] = useState('')
  const [filter, setFilter] = useState<FilterType>('all')
  const [sortOption, setSortOption] = useState<SortOption>('name-asc')
  const [currentPage, setCurrentPage] = useState(1)

  // Load CSV list on mount + auto-load last CSV if no data in store
  useEffect(() => {
    refreshCsvList()
    if (photoRows.length === 0) {
      getLastCsvName().then(async (name) => {
        if (!name) return
        const text = await loadCsvFromStorage(name)
        if (!text) return
        const rows = parseCsv(text, mainColumn)
        if (rows.length > 0) {
          setPhotoRows(rows)
          setSelectedCsv(name)
          logger.info(`Auto-loaded last CSV: ${name} (${rows.length} rows)`)
        }
      })
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const refreshCsvList = useCallback(async () => {
    const csvs = await listStoredCsvs()
    setCsvFiles(csvs)
  }, [])

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
        // IDs that don't appear exactly 2 times, or duplicate CAM+suffix combos
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
    refreshCsvList,
    loadCsv,
    updateRow,
    recalculateNames,
    saveChanges,
    exportCsv,
  }
}
