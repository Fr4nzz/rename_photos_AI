/**
 * Orientation suggestions from the CAMID reading, manual corrections, and learning from them.
 *
 * Angles are clockwise degrees (0, 90, 180, 270) relative to how the photo currently displays
 * (its EXIF orientation applied). "Envelope text upright" is taken to mean "photo upright".
 */
import type { PhotoReading } from './ocr/pipeline'
import type { PhotoRow } from '@/types'

export type RotSource = '' | 'ocr' | 'manual' | 'learned' | 'neighbours'

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

/**
 * Photos without a trusted reading take their rotation from the read photos around them in the
 * same session (capture order). The comparison is made in the camera's own frame, ignoring the
 * orientation tag: on a copy stand the camera points at the floor, so its gyro-based tag flips
 * between shots while the sensor frame stays the same. `tagCw` gives, per photo, the clockwise
 * turn its current orientation tag applies (null when the tag is mirrored: no suggestion).
 * The majority of the (up to) two nearest read photos on each side wins; a tie goes to the nearest.
 *
 * A read photo whose envelope points another way than the nearest read photos on both sides, while
 * those two agree (e.g. an envelope laid down turned), takes their rotation instead and is flagged
 * 'rotation-outlier' for review.
 */
export function fillFromNeighbours(rows: PhotoRow[], tagCw: Map<string, number | null>): PhotoRow[] {
  const out = rows.map((r) => ({ ...r }))
  const time = (r: PhotoRow) => (r.captureDate ? Date.parse(r.captureDate) : NaN)
  // upright orientation of each read photo, in its sensor frame
  const sensor = (r: PhotoRow): number | null => {
    const t = tagCw.get(r.from)
    return r.rotSuggested === '' || t === null || t === undefined ? null : norm(t + Number(r.rotSuggested))
  }
  for (const session of sessionsOf(rows)) {
    const ordered = [...session].sort((a, b) => time(rows[a]) - time(rows[b]))
    // outliers among the read photos
    for (let k = 0; k < ordered.length; k++) {
      const own = sensor(rows[ordered[k]])
      const t = tagCw.get(rows[ordered[k]].from)
      if (own === null || t === null || t === undefined || rows[ordered[k]].rotSource === 'manual') continue
      const side = (dir: number) => {
        for (let j = k + dir; j >= 0 && j < ordered.length; j += dir) {
          const a = sensor(rows[ordered[j]])
          if (a !== null) return a
        }
        return null
      }
      const before = side(-1), after = side(1)
      if (before === null || before !== after || before === own) continue
      const r = out[ordered[k]]
      const turn = String(norm(before - t))
      Object.assign(r, { rotSuggested: turn, rotChosen: turn, rotSource: 'neighbours' as const,
        review: [...r.review.split(',').filter(Boolean), 'rotation-outlier'].join(',') })
    }
    for (let k = 0; k < ordered.length; k++) {
      const r = out[ordered[k]]
      const t = tagCw.get(r.from)
      if (rows[ordered[k]].rotSuggested !== '' || r.rotSource === 'manual' || t === null || t === undefined) continue
      const near: { a: number; dt: number }[] = []
      for (const dir of [-1, 1]) {
        let found = 0
        for (let j = k + dir; j >= 0 && j < ordered.length && found < 2; j += dir) {
          const a = sensor(rows[ordered[j]])
          if (a === null) continue
          near.push({ a, dt: Math.abs(time(rows[ordered[j]]) - time(rows[ordered[k]])) })
          found++
        }
      }
      if (!near.length) continue
      const votes = new Map<number, number>()
      for (const n of near) votes.set(n.a, (votes.get(n.a) ?? 0) + 1)
      const top = Math.max(...votes.values())
      const tied = new Set([...votes].filter(([, v]) => v === top).map(([a]) => a))
      const pick = near.filter((n) => tied.has(n.a)).sort((x, y) => x.dt - y.dt)[0].a
      const turn = String(norm(pick - t))
      Object.assign(r, { rotSuggested: turn, rotChosen: turn, rotSource: 'neighbours' as const })
    }
  }
  return out
}
