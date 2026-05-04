import { create } from 'zustand'
import type { PhotoRow, RunMode } from '@/types'

interface ProcessingState {
  photoRows: PhotoRow[]
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
      isProcessing: false,
      progress: { percent: 0, message: '' },
      failedBatches: [],
    }),
}))
