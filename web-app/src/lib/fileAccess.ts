import type { FileEntry } from '@/types'
import { ALL_SUPPORTED_EXTENSIONS, SUPPORTED_COMPRESSED_EXTENSIONS, SUPPORTED_RAW_EXTENSIONS } from './constants'

export function supportsDirectoryPicker(): boolean {
  return 'showDirectoryPicker' in window
}

export function supportsSaveFilePicker(): boolean {
  return 'showSaveFilePicker' in window
}

const extensionSets = {
  all: ALL_SUPPORTED_EXTENSIONS,
  compressed: SUPPORTED_COMPRESSED_EXTENSIONS,
  raw: SUPPORTED_RAW_EXTENSIONS,
} as const

function getExtension(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot >= 0 ? name.slice(dot).toLowerCase() : ''
}

export async function openDirectoryPicker(): Promise<FileSystemDirectoryHandle> {
  return await (window as any).showDirectoryPicker({ mode: 'readwrite' })
}

export async function getImageFilesFromHandle(
  handle: FileSystemDirectoryHandle,
  type: 'all' | 'compressed' | 'raw' = 'all'
): Promise<FileEntry[]> {
  const allowed = extensionSets[type]
  const entries: FileEntry[] = []

  for await (const [name, fileHandle] of (handle as any).entries()) {
    if (fileHandle.kind !== 'file') continue
    const ext = getExtension(name)
    if (!allowed.has(ext)) continue
    const file = await fileHandle.getFile()
    entries.push({ name, path: name, file, extension: ext })
  }

  return entries.sort((a, b) => a.name.localeCompare(b.name))
}

export function getImageFilesFromInput(
  fileList: FileList,
  type: 'all' | 'compressed' | 'raw' = 'all'
): FileEntry[] {
  const allowed = extensionSets[type]
  const entries: FileEntry[] = []

  for (let i = 0; i < fileList.length; i++) {
    const file = fileList[i]
    const ext = getExtension(file.name)
    if (!allowed.has(ext)) continue
    entries.push({
      name: file.name,
      path: (file as any).webkitRelativePath || file.name,
      file,
      extension: ext,
    })
  }

  return entries.sort((a, b) => a.name.localeCompare(b.name))
}
