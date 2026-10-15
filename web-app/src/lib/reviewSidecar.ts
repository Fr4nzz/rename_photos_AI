/**
 * What a batch's results CSV does not hold, kept beside it in rename_files/ (<csv>.review.json), so a
 * folder opened again resumes as it was left: the reader's results (the envelope and ID line it found,
 * how sure it was, the other IDs it read), when each photo was taken, and the assistant's changes
 * (the strip of changes and, on each card, the ID read beside the one set), and the wing check's results.
 * The IDs, turns, skips and notes themselves are in the CSV.
 */
import { loadCsvFromFolder, saveCsvToFolder } from './csvHandler'
import type { HostChange } from './embed/protocol'
import type { PhotoReading } from './ocr/pipeline'
import type { WingCheck } from './wings/checks'
import { useProcessingStore } from '@/stores/processingStore'
import type { PhotoRow } from '@/types'

interface Sidecar {
  version: 1
  readings: Record<string, PhotoReading>
  taken: Record<string, string>
  host: { log: HostChange[]; name: string }
  wings?: Record<string, WingCheck>
}
const nameOf = (csv: string) => `${csv}.review.json`

/** Saves what the store holds beside the CSV of that name. */
export async function saveSidecar(dir: FileSystemDirectoryHandle, csvName: string): Promise<void> {
  const s = useProcessingStore.getState()
  const data: Sidecar = {
    version: 1,
    readings: Object.fromEntries(s.ocrReadings),
    taken: Object.fromEntries(s.photoRows.flatMap((r) => (r.captureDate ? [[r.from, r.captureDate]] : []))),
    host: { log: s.hostLog, name: s.hostName },
    wings: Object.fromEntries(s.wingChecks),
  }
  await saveCsvToFolder(dir, nameOf(csvName), JSON.stringify(data))
}

/** Puts back into the store what was kept beside the CSV, and the photos' times into `rows` (answered). */
export async function restoreSidecar(dir: FileSystemDirectoryHandle, csvName: string, rows: PhotoRow[]): Promise<PhotoRow[]> {
  const text = await loadCsvFromFolder(dir, nameOf(csvName))
  const s = useProcessingStore.getState()
  s.clearHostChanges()
  s.setWingChecks(new Map())
  if (!text) return rows
  let data: Sidecar
  try {
    data = JSON.parse(text)
  } catch {
    return rows
  }
  s.setOcrReadings(new Map(Object.entries(data.readings ?? {})))
  s.setWingChecks(new Map(Object.entries(data.wings ?? {})))
  if (data.host?.log?.length) s.addHostChanges(data.host.log, data.host.name || 'Claude')
  return rows.map((r) => (data.taken?.[r.from] && !r.captureDate ? { ...r, captureDate: data.taken[r.from] } : r))
}
