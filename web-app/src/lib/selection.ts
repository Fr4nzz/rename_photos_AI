import type { FileEntry } from '@/types'

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
