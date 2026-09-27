/**
 * Local CAMID run: read every photo in the OCR workers, collect EXIF capture times, then apply the
 * decision rules (database, batch check) and fill the rows used by the Review tab.
 */
import exifr from 'exifr'
import type { FileEntry, PhotoRow } from '@/types'
import { loadCamidSet, type CamidSet } from './database'
import { decide, type Decision, type PhotoInput } from './decide'
import { readPhotos } from './engine'
import type { PhotoReading } from './pipeline'

export const REASON_LABEL: Record<string, string> = {
  'no-reading': 'No CAMID read',
  'not-in-database': 'Not in database',
  'out-of-sequence': 'Out of sequence',
  'repeated-id': 'Same ID on another photo',
  'unreadable-photo': 'Photo not readable',
}

async function captureTime(file: File): Promise<number | null> {
  try {
    const t = await exifr.parse(file, ['DateTimeOriginal', 'CreateDate'])
    const d: Date | undefined = t?.DateTimeOriginal ?? t?.CreateDate
    return d instanceof Date && !isNaN(d.getTime()) ? d.getTime() : null
  } catch {
    return null
  }
}

export interface LocalRunResult {
  rows: PhotoRow[]
  readings: Map<string, PhotoReading>
  decisions: Decision[]
  database: CamidSet
}

export async function runLocalOcr(
  files: FileEntry[],
  baseRows: PhotoRow[],
  onProgress: (done: number, total: number, partial: Map<string, PhotoReading>) => void,
  signal: AbortSignal,
): Promise<LocalRunResult> {
  const databasePromise = loadCamidSet()
  const timesPromise = Promise.all(files.map((f) => captureTime(f.file)))
  const readings = new Map<string, PhotoReading>()
  const errors = new Map<string, string>()
  let done = 0
  await readPhotos(files.map((f) => f.file), (r) => {
    if (r.reading) readings.set(r.name, r.reading)
    else errors.set(r.name, r.error ?? 'unreadable')
    done++
    onProgress(done, files.length, readings)
  }, signal)
  const [database, times] = await Promise.all([databasePromise, timesPromise])

  const inputs: PhotoInput[] = files.map((f, i) => ({
    name: f.name, reading: readings.get(f.name), error: errors.get(f.name), capturedAt: times[i],
  }))
  const decisions = decide(inputs, database.ids)
  const byName = new Map(decisions.map((d) => [d.name, d]))
  const rows = baseRows.map((row, i) => {
    const d = byName.get(row.from)
    if (!d) return row
    const reading = readings.get(row.from)
    const others = reading?.candidates.map((c) => c.id).filter((id) => id !== d.camid) ?? []
    return {
      ...row,
      mainValue: d.camid ?? (d.prefill.length === 9 && !d.prefill.includes('?') ? d.prefill : ''),
      co: [...new Set(others)].join(' '),
      n: reading?.source === 'fallback' ? 'no envelope found; read from the whole photo' : row.n,
      review: d.auto ? '' : d.reasons.join(','),
      suggest: [d.prefill, ...d.candidates].filter((s, k, a) => s.length === 9 && a.indexOf(s) === k).join(' '),
      captureDate: times[i] ? new Date(times[i]!).toISOString() : row.captureDate,
      batchNumber: 0,
    }
  })
  return { rows, readings, decisions, database }
}
