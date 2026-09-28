/**
 * Orientation suggestions from the CAMID reading, manual corrections, and learning from them.
 *
 * Angles are clockwise degrees (0, 90, 180, 270) relative to how the photo currently displays
 * (its EXIF orientation applied). "Envelope text upright" is taken to mean "photo upright".
 */
import type { PhotoReading } from './ocr/pipeline'
import type { PhotoRow } from '@/types'

export type RotSource = '' | 'ocr' | 'manual' | 'learned'

const norm = (a: number) => ((a % 360) + 360) % 360

/**
 * Clockwise rotation that turns the photo upright, from the way its CAMID line was read:
 * a line read after turning it 90° counter-clockwise means the photo needs 270° clockwise, a line
 * read upside down means 180°. Null when no CAMID was read.
 */
export function uprightRotation(reading: PhotoReading | undefined, camid: string | null): number | null {
  if (!reading || !camid) return null
  const line = reading.lines.find((l) => l.text.replace(/\s+/g, '').toUpperCase() === camid)
  if (!line) return null
  return norm(-(reading.turned + (line.turn ?? 0)))
}

/** Pending rotation still to be written to the file (chosen minus already applied). */
export function pendingRotation(row: Pick<PhotoRow, 'rotChosen' | 'rotApplied'>): number {
  return norm(Number(row.rotChosen || 0) - Number(row.rotApplied || 0))
}

const SESSION_GAP_MS = 30 * 60 * 1000

/** Group rows into shooting sessions: consecutive capture times less than 30 minutes apart. */
export function sessionsOf(rows: PhotoRow[]): number[][] {
  const order = rows.map((r, i) => ({ i, t: r.captureDate ? Date.parse(r.captureDate) : NaN }))
    .sort((a, b) => (a.t || Infinity) - (b.t || Infinity) || a.i - b.i)
  const sessions: number[][] = []
  let last = NaN
  for (const { i, t } of order) {
    if (!sessions.length || isNaN(t) || isNaN(last) || t - last > SESSION_GAP_MS) sessions.push([])
    sessions[sessions.length - 1].push(i)
    last = t
  }
  return sessions
}

/**
 * When at least two manual corrections in a session turn their photos by the same offset from the
 * reader's suggestion (e.g. the envelope always lies sideways next to the wings), apply that offset
 * to the session's other photos that still follow the reader ('ocr') or an earlier learned offset.
 * A single correction stays an exception. Photos already written to disk keep their state.
 */
export function applyLearnedOffsets(rows: PhotoRow[]): PhotoRow[] {
  const out = rows.map((r) => ({ ...r }))
  for (const session of sessionsOf(rows)) {
    const offsets = new Map<number, number>()
    for (const i of session) {
      const r = rows[i]
      if (r.rotSource === 'manual' && r.rotSuggested !== '') {
        const off = norm(Number(r.rotChosen || 0) - Number(r.rotSuggested))
        offsets.set(off, (offsets.get(off) ?? 0) + 1)
      }
    }
    const ranked = [...offsets].sort((a, b) => b[1] - a[1])
    const learned = ranked.length && ranked[0][1] >= 2 && (ranked.length === 1 || ranked[0][1] > ranked[1][1]) ? ranked[0][0] : null
    for (const i of session) {
      const r = out[i]
      if (r.rotSource !== 'ocr' && r.rotSource !== 'learned') continue
      if (r.rotSuggested === '') continue
      if (learned === null || learned === 0) {
        // no (or a zero) learned offset: back to the reader's suggestion
        if (r.rotSource === 'learned') { r.rotChosen = r.rotSuggested; r.rotSource = 'ocr' }
        continue
      }
      r.rotChosen = String(norm(Number(r.rotSuggested) + learned))
      r.rotSource = 'learned'
    }
  }
  return out
}
