/**
 * A batch's results against its folder now: photos of the results no longer in the folder (moved,
 * deleted, renamed elsewhere), and photos in the folder the results do not have (added after the
 * reading). A RAW file goes with its photo.
 */
import type { FileEntry, PhotoRow } from '@/types'
import { photosToRead } from './selection'

const lower = (s: string) => s.toLowerCase()
const stemOf = (name: string) => lower(name.lastIndexOf('.') > 0 ? name.slice(0, name.lastIndexOf('.')) : name)

export function matchFolder(rows: PhotoRow[], files: FileEntry[]): { missing: PhotoRow[]; unread: FileEntry[] } {
  if (!rows.length || !files.length) return { missing: [], unread: [] }
  const names = new Set(files.map((f) => lower(f.name)))
  const known = new Set(rows.flatMap((r) => [lower(r.from), lower(r.currentPath)]))
  const stems = new Set(rows.flatMap((r) => [stemOf(r.from), stemOf(r.currentPath)]))
  return {
    missing: rows.filter((r) => !names.has(lower(r.currentPath)) && !names.has(lower(r.from))),
    unread: photosToRead(files).filter((f) => !known.has(lower(f.name)) && !stems.has(stemOf(f.name))),
  }
}
