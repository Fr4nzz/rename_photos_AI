import { create } from 'zustand'
import type { FileEntry, PhotoRow, RotationLogEntry, RunMode } from '@/types'
import { selectAllImageNames } from '@/lib/selection'

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
        percent: Math.max(state.progress.percent, percent),
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
    }),
}))
