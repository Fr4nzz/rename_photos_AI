import { create } from 'zustand'
import type { FileEntry, PhotoRow, RotationLogEntry, RunMode } from '@/types'
import type { PhotoReading } from '@/lib/ocr/pipeline'
import { selectAllImageNames } from '@/lib/selection'
import type { HostChange } from '@/lib/embed/protocol'
import type { WingCheck } from '@/lib/wings/checks'

interface ProcessingState {
  photoRows: PhotoRow[]
  imageFiles: FileEntry[]
  selectedImageNames: Set<string>
  reviewSelectedOnly: boolean
  renameCompanions: boolean
  rotationLog: RotationLogEntry[]
  /** Map of filename → File object for thumbnail display in Review tab */
  fileMap: Map<string, File>
  /** Directory handle for filesystem CSV persistence */
  dirHandle: FileSystemDirectoryHandle | null
  isProcessing: boolean
  progress: { percent: number; message: string }
  runMode: RunMode
  retryMessages: string
  continueCsvName: string
  currentCsvName: string
  failedBatches: number[]
  /** local OCR readings of this session (line boxes for the review zoom), by file name */
  ocrReadings: Map<string, PhotoReading>
  setOcrReadings: (m: Map<string, PhotoReading>) => void
  /** the wing check's results, by file name, and its progress while it runs */
  wingChecks: Map<string, WingCheck>
  setWingChecks: (m: Map<string, WingCheck>) => void
  wingRun: { done: number; total: number } | null
  setWingRun: (run: { done: number; total: number } | null) => void
  /** embedded mode: the corrections made by the page hosting the app, in order (to take them back)… */
  hostLog: HostChange[]
  /** …and, by file, what was changed in each row (shown on its card) and who made them */
  hostChanges: Map<string, HostChange>
  hostName: string
  /** the controls few people need (results files, the Gemini reader, crop, ZIP) are shown; hidden at first inside another app */
  advanced: boolean
  setAdvanced: (value: boolean) => void
  addHostChanges: (changes: HostChange[], by: string) => void
  clearHostChanges: () => void

  setPhotoRows: (rows: PhotoRow[]) => void
  setImageFiles: (files: FileEntry[], selectAll?: boolean) => void
  setSelectedImageNames: (names: Set<string>) => void
  toggleSelectedImage: (name: string) => void
  setReviewSelectedOnly: (value: boolean) => void
  setRenameCompanions: (value: boolean) => void
  setRotationLog: (entries: RotationLogEntry[]) => void
  setFileMap: (map: Map<string, File>) => void
  setDirHandle: (handle: FileSystemDirectoryHandle | null) => void
  updateRow: (index: number, updates: Partial<PhotoRow>) => void
  setProcessing: (value: boolean) => void
  setProgress: (percent: number, message: string) => void
  setRunMode: (mode: RunMode) => void
  setRetryMessages: (messages: string) => void
  setContinueCsvName: (name: string) => void
  setCurrentCsvName: (name: string) => void
  setFailedBatches: (batches: number[]) => void
  reset: () => void
}

export const useProcessingStore = create<ProcessingState>()((set) => ({
  photoRows: [],
  imageFiles: [],
  selectedImageNames: new Set(),
  reviewSelectedOnly: false,
  renameCompanions: true,
  rotationLog: [],
  fileMap: new Map(),
  dirHandle: null,
  isProcessing: false,
  progress: { percent: 0, message: '' },
  runMode: 'start_over',
  retryMessages: '',
  continueCsvName: '',
  currentCsvName: '',
  failedBatches: [],
  ocrReadings: new Map(),
  setOcrReadings: (m) => set({ ocrReadings: m }),
  wingChecks: new Map(),
  setWingChecks: (m) => set({ wingChecks: m }),
  wingRun: null,
  setWingRun: (run) => set({ wingRun: run }),
  hostLog: [],
  hostChanges: new Map(),
  hostName: '',
  advanced: !(typeof window !== 'undefined' && window.parent !== window && new URLSearchParams(window.location.search).has('embed')),
  setAdvanced: (value) => set({ advanced: value }),
  addHostChanges: (changes, by) =>
    set((state) => {
      const byFile = new Map(state.hostChanges)
      for (const c of changes) {
        const earlier = byFile.get(c.file)
        // an ID changed twice: from what the reader had to what the host set last
        const id = c.id && earlier?.id ? { from: earlier.id.from, to: c.id.to } : (c.id ?? earlier?.id)
        byFile.set(c.file, { file: c.file, what: [...(earlier?.what ?? []), ...c.what], before: { ...c.before, ...(earlier?.before ?? {}) }, ...(id ? { id } : {}) })
      }
      return { hostLog: [...state.hostLog, ...changes], hostChanges: byFile, hostName: by }
    }),
  clearHostChanges: () => set({ hostLog: [], hostChanges: new Map() }),

  setPhotoRows: (rows) => set({ photoRows: rows }),
  setImageFiles: (files, selectAll = true) =>
    set({
      imageFiles: files,
      selectedImageNames: selectAll ? selectAllImageNames(files) : new Set(),
    }),
  setSelectedImageNames: (names) => set({ selectedImageNames: new Set(names) }),
  toggleSelectedImage: (name) =>
    set((state) => {
      const selected = new Set(state.selectedImageNames)
      if (selected.has(name)) selected.delete(name)
      else selected.add(name)
      return { selectedImageNames: selected }
    }),
  setReviewSelectedOnly: (value) => set({ reviewSelectedOnly: value }),
  setRenameCompanions: (value) => set({ renameCompanions: value }),
  setRotationLog: (entries) => set({ rotationLog: entries }),
  setFileMap: (map) => set({ fileMap: map }),
  setDirHandle: (handle) => set({ dirHandle: handle }),
  updateRow: (index, updates) =>
    set((state) => {
      const rows = [...state.photoRows]
      rows[index] = { ...rows[index], ...updates }
      return { photoRows: rows }
    }),
  setProcessing: (value) => set({ isProcessing: value }),
  setProgress: (percent, message) =>
    set((state) => ({
      progress: {
        // parallel batches may report out of order; 0 starts a new run
        percent: percent === 0 ? 0 : Math.max(state.progress.percent, percent),
        message,
      },
    })),
  setRunMode: (mode) => set({ runMode: mode }),
  setRetryMessages: (messages) => set({ retryMessages: messages }),
  setContinueCsvName: (name) => set({ continueCsvName: name }),
  setCurrentCsvName: (name) => set({ currentCsvName: name }),
  setFailedBatches: (batches) => set({ failedBatches: batches }),
  reset: () =>
    set({
      photoRows: [],
      imageFiles: [],
      selectedImageNames: new Set(),
      isProcessing: false,
      progress: { percent: 0, message: '' },
      failedBatches: [],
      hostLog: [],
      hostChanges: new Map(),
      wingChecks: new Map(),
      wingRun: null,
    }),
}))
