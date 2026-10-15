import type { FileEntry, PhotoRow } from '@/types'
import { SUPPORTED_RAW_EXTENSIONS } from './constants'

export type ImageSortOption = 'name-asc' | 'name-desc' | 'date-asc' | 'date-desc' | 'type-asc'

export function getSelectedImageFiles(
  files: FileEntry[],
  selectedNames: Set<string>
): FileEntry[] {
  return files.filter((entry) => selectedNames.has(entry.name))
}

export function selectAllImageNames(files: FileEntry[]): Set<string> {
  return new Set(files.map((entry) => entry.name))
}

export function invertImageSelection(files: FileEntry[], selectedNames: Set<string>): Set<string> {
  return new Set(files.filter((entry) => !selectedNames.has(entry.name)).map((entry) => entry.name))
}

export function matchImageNames(files: FileEntry[], query: string): Set<string> {
  const normalized = query.trim().toLowerCase()
  if (!normalized) return new Set()
  return new Set(
    files
      .filter((entry) => entry.name.toLowerCase().includes(normalized))
      .map((entry) => entry.name)
  )
}

export function sortImageFiles(files: FileEntry[], sortOption: ImageSortOption): FileEntry[] {
  const sorted = [...files]
  sorted.sort((a, b) => {
    switch (sortOption) {
      case 'name-desc':
        return b.name.localeCompare(a.name)
      case 'date-asc':
        return a.file.lastModified - b.file.lastModified
      case 'date-desc':
        return b.file.lastModified - a.file.lastModified
      case 'type-asc': {
        const ext = a.extension.localeCompare(b.extension)
        return ext !== 0 ? ext : a.name.localeCompare(b.name)
      }
      case 'name-asc':
      default:
        return a.name.localeCompare(b.name)
    }
  })
  return sorted
}

const stemOf = (name: string) => (name.lastIndexOf('.') > 0 ? name.slice(0, name.lastIndexOf('.')) : name).toLowerCase()
const isRaw = (entry: FileEntry) => SUPPORTED_RAW_EXTENSIONS.has(entry.extension.toLowerCase())

/**
 * The photos a reading takes: a RAW file shot with a JPG (same name) is the same photo and follows
 * the JPG's renaming and rotation, so only RAWs without one are read.
 */
export function photosToRead(files: FileEntry[]): FileEntry[] {
  const withJpg = new Set(files.filter((entry) => !isRaw(entry)).map((entry) => stemOf(entry.name)))
  return files.filter((entry) => !isRaw(entry) || !withJpg.has(stemOf(entry.name)))
}

/** Rows by file name, a RAW file shown with its JPG's reading (and its new name, in its own extension). */
export function rowsWithRaws(rows: PhotoRow[], files: FileEntry[]): Map<string, PhotoRow> {
  const byName = new Map(rows.map((r) => [r.from, r]))
  const byStem = new Map(rows.map((r) => [stemOf(r.from), r]))
  for (const entry of files) {
    if (byName.has(entry.name) || !isRaw(entry)) continue
    const row = byStem.get(stemOf(entry.name))
    if (!row) continue
    const ext = entry.name.slice(entry.name.lastIndexOf('.'))
    const renamed = row.currentPath.lastIndexOf('.') > 0 ? row.currentPath.slice(0, row.currentPath.lastIndexOf('.')) + ext : row.currentPath
    byName.set(entry.name, { ...row, currentPath: renamed })
  }
  return byName
}
