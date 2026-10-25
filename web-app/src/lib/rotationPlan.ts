/**
 * Orientation suggestions from the photo's own readings (its envelope, its wings), manual corrections,
 * and the camera's other photos for a photo that says nothing itself.
 *
 * Angles are clockwise degrees (0, 90, 180, 270) relative to how the photo currently displays
 * (its EXIF orientation applied).
 */
import { AUTO_SURE, findCamid, type PhotoReading } from './ocr/pipeline'
import { TURN_CLOSE, TURN_SURE } from './ocr/envelopeText'
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

/** An envelope's text turn is trusted with at least this many characters read in it. */
export const TEXT_MIN_CHARS = 8

/**
 * Clockwise turn that makes the photo's envelope upright, when the reading is sure of it (null
 * otherwise), in any of the four directions: the turn its CAM ID line was read in when that line
 * read AUTO_SURE (`camid`: the ID taken for the photo, null when it is not one to trust); else
 * the turn its whole text reads best in (PhotoReading.text), when at least TEXT_MIN_CHARS were
 * read and the other turn compared read TURN_CLOSE worse per character (or, not compared, the
 * text read TURN_SURE).
 */
export function envelopeTurn(reading: PhotoReading | undefined, camid: string | null): number | null {
  if (!reading) return null
  const line = camid ? reading.lines.find((l) => l.conf >= AUTO_SURE && findCamid(l.text)?.id === camid) : undefined
  if (line) return norm(-(reading.turned + (line.turn ?? 0)))
  const t = reading.text
  if (!t || reading.source !== 'envelope') return null
  const chars = t.lines.filter((l) => l.turn === t.turn).reduce((n, l) => n + l.text.replace(/\s+/g, '').length, 0)
  const best = t.scores[t.turn], other = t.scores[norm(t.turn + 180) as 0 | 90 | 180 | 270]
  if (chars < TEXT_MIN_CHARS || best === undefined) return null
  if (other === undefined ? best < TURN_SURE : best - other < TURN_CLOSE) return null
  return norm(-t.turn)
}

/** The reasons planTurns gives (dropped before it decides again); 'rotation-outlier' only in older files. */
export const TURN_FLAGS = ['turn-unsure', 'turn-disagree', 'turn-camera', 'rotation-outlier']

/** The camera's turn is taken as sure with at least TURN_MIN photos saying it, TURN_AGREES of them alike. */
const TURN_MIN = 3
const TURN_AGREES = 0.8

/**
 * Each photo's turn. A photo takes its own sure reading, in any direction: its wings' turn when they
 * are sure (`wingCw`: the clockwise turn from how it shows that puts its forewings on top, null when
 * the wings are not sure), else its envelope's (the rows' rotSuggested with rotSource 'ocr', from
 * envelopeTurn). When both are sure and disagree, the wings win: they show the specimen itself, while
 * an envelope may lie any way beside it (10 of 60 photos of a copy-stand test, 25 Oct 2026, had their
 * envelope sideways or upside down next to upright wings; the sure wings were right on every one of
 * its 112 photos where they were sure). Such a photo is flagged 'turn-disagree', unless its turn is
 * also its camera's sure usual turn (below): an envelope laid another way on a copy stand. A photo
 * turned by its envelope alone against its camera's sure usual turn is flagged 'turn-camera': without
 * the wings, an envelope laid another way cannot be told from a photo turned another way.
 *
 * A photo with neither takes its camera's usual turn, in the camera's own frame (its orientation
 * tag ignored: on a copy stand the camera stays put and the tag flips between shots), from the
 * photos that have their own and stay landscape with it; that one keeps the photo landscape
 * (`landscape`: whether each photo shows landscape now, its tag applied; null: unknown, left as it
 * is). Unless TURN_MIN photos say it, TURN_AGREES of them alike, it is flagged 'turn-unsure'.
 * A manual turn is left as it is. Input rows carry the envelope's own turn (as runLocalOcr fills
 * them), not a turn this function chose.
 */
export function planTurns(rows: PhotoRow[], landscape: Map<string, boolean | null>, tagCw: Map<string, number | null>, wingCw = new Map<string, number | null>()): PhotoRow[] {
  const own = new Map<string, { cw: number; by: 'wings' | 'envelope'; disagree: boolean }>()
  for (const r of rows) {
    if (r.rotSource === 'manual') continue
    const envelope = r.rotSource === 'ocr' && r.rotSuggested !== '' ? norm(Number(r.rotSuggested)) : null
    const wings = wingCw.get(r.from) ?? null
    if (wings !== null) own.set(r.from, { cw: norm(wings), by: 'wings', disagree: envelope !== null && envelope !== norm(wings) })
    else if (envelope !== null) own.set(r.from, { cw: envelope, by: 'envelope', disagree: false })
  }
  /** Whether the camera's own frame is landscape, from how the photo shows and its tag. */
  const frameWide = (r: PhotoRow): boolean | null => {
    const wide = landscape.get(r.from)
    const t = tagCw.get(r.from)
    return wide === null || wide === undefined || t === null || t === undefined ? null : (norm(t) % 180 === 0) === wide
  }
  const fits = (sensorTurn: number, wide: boolean) => (norm(sensorTurn) % 180 === 0) === wide
  const votes = new Map<number, number>()
  for (const r of rows) {
    const o = own.get(r.from), t = tagCw.get(r.from), wide = frameWide(r)
    if (!o || t === null || t === undefined || wide === null) continue
    const turn = norm(t + o.cw)
    if (fits(turn, wide)) votes.set(turn, (votes.get(turn) ?? 0) + 1)
  }
  const total = [...votes.values()].reduce((a, b) => a + b, 0)
  const [best, most] = [...votes].sort((a, b) => b[1] - a[1])[0] ?? [0, 0]
  const sure = total >= TURN_MIN && most / total >= TURN_AGREES
  /** The camera's usual turn from how this photo shows (null: its tag or shape unknown). */
  const usual = (r: PhotoRow): number | null => {
    const t = tagCw.get(r.from)
    const wide = frameWide(r)
    // A camera whose frame stands (not this setup's, usually): the quarter that lays it down.
    return t === null || t === undefined || wide === null ? null : norm((fits(best, wide) ? best : best + 90) - t)
  }
  return rows.map((row) => {
    const r = { ...row }
    if (r.rotSource === 'manual') return r
    const review = r.review.split(',').filter(Boolean).filter((x) => !TURN_FLAGS.includes(x))
    const o = own.get(r.from)
    if (o) {
      const turn = String(o.cw)
      const u = sure ? usual(r) : null
      const look = o.disagree && u !== o.cw ? ['turn-disagree'] : o.by === 'envelope' && u !== null && u !== o.cw ? ['turn-camera'] : []
      return Object.assign(r, { rotSuggested: turn, rotChosen: turn, rotSource: 'ocr' as const, review: [...review, ...look].join(',') })
    }
    const u = usual(r)
    if (u === null) return r
    const turn = String(u)
    return Object.assign(r, { rotSuggested: turn, rotChosen: turn, rotSource: 'neighbours' as const, review: [...review, ...(sure ? [] : ['turn-unsure'])].join(',') })
  })
}

/**
 * planTurns for each camera of the batch: a folder may hold the photos of two set-ups (another camera,
 * held another way), and the photos of one say nothing of the other's usual turn. `camera`: each photo's
 * make and model ('' unknown: those go with each other).
 */
export function planTurnsByCamera(
  rows: PhotoRow[], landscape: Map<string, boolean | null>, tagCw: Map<string, number | null>, camera: Map<string, string>, wingCw?: Map<string, number | null>,
): PhotoRow[] {
  const groups = new Map<string, PhotoRow[]>()
  for (const r of rows) {
    const key = camera.get(r.from) ?? ''
    groups.set(key, [...(groups.get(key) ?? []), r])
  }
  const turned = new Map<number, PhotoRow>()
  for (const group of groups.values()) for (const r of planTurns(group, landscape, tagCw, wingCw)) turned.set(r.photoId, r)
  return rows.map((r) => turned.get(r.photoId) ?? r)
}
