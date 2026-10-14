/**
 * Orientation suggestions from the CAMID reading, manual corrections, and learning from them.
 *
 * Angles are clockwise degrees (0, 90, 180, 270) relative to how the photo currently displays
 * (its EXIF orientation applied). An envelope read upright or upside down votes for the batch's one
 * turn (oneTurn); the photos stay landscape, so an envelope laid sideways does not count.
 */
import type { PhotoReading } from './ocr/pipeline'
import type { PhotoRow } from '@/types'

/** 'learned' only appears in results files saved by earlier versions */
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

/**
 * One turn for the whole batch. On a copy stand the camera stays put, so in its own frame (its
 * orientation tag ignored: the tag flips between shots) every photo needs the same turn, and the
 * photos are landscape (the wings side by side, forewings above hindwings): in the camera's frame the
 * turn is none or a half for a camera whose frame is landscape. The envelopes read upright or upside
 * down vote between those two; one laid down sideways says nothing. With a clear majority (at least
 * TURN_MIN votes, TURN_AGREES of them alike) every photo takes it; otherwise they take the one most
 * voted (none without votes) and are flagged 'turn-unsure': which way up needs a look at the wings.
 * `landscape`: whether each photo shows landscape now (its tag applied; null: unknown); a manual
 * turn is left as it is.
 */
const TURN_MIN = 3
const TURN_AGREES = 0.8
export function oneTurn(rows: PhotoRow[], landscape: Map<string, boolean | null>, tagCw: Map<string, number | null>): PhotoRow[] {
  const out = rows.map((r) => ({ ...r }))
  /** Whether the camera's own frame is landscape, from how the photo shows and its tag. */
  const frameWide = (r: PhotoRow): boolean | null => {
    const wide = landscape.get(r.from)
    const t = tagCw.get(r.from)
    return wide === null || wide === undefined || t === null || t === undefined ? null : (norm(t) % 180 === 0) === wide
  }
  const fits = (sensorTurn: number, wide: boolean) => (norm(sensorTurn) % 180 === 0) === wide
  const votes = new Map<number, number>()
  for (const r of rows) {
    const t = tagCw.get(r.from)
    const wide = frameWide(r)
    if (r.rotSource !== 'ocr' || r.rotSuggested === '' || t === null || t === undefined || wide === null) continue
    const turn = norm(t + Number(r.rotSuggested))
    if (fits(turn, wide)) votes.set(turn, (votes.get(turn) ?? 0) + 1)
  }
  const total = [...votes.values()].reduce((a, b) => a + b, 0)
  const [best, most] = [...votes].sort((a, b) => b[1] - a[1])[0] ?? [0, 0]
  const sure = total >= TURN_MIN && most / total >= TURN_AGREES
  for (const r of out) {
    const t = tagCw.get(r.from)
    const wide = frameWide(r)
    if (r.rotSource === 'manual' || t === null || t === undefined || wide === null) continue
    // A camera whose frame stands (not this setup's, usually): the quarter that lays it down.
    const turn = String(norm((fits(best, wide) ? best : best + 90) - t))
    const review = r.review.split(',').filter(Boolean).filter((x) => x !== 'rotation-outlier' && x !== 'turn-unsure')
    Object.assign(r, {
      rotSuggested: turn,
      rotChosen: turn,
      rotSource: r.rotSource === 'ocr' && r.rotSuggested === turn ? 'ocr' : ('neighbours' as const),
      review: [...review, ...(sure ? [] : ['turn-unsure'])].join(','),
    })
  }
  return out
}
