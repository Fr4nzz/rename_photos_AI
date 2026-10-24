/**
 * The envelope's whole text, read in one turn per envelope (not line by line).
 *
 * Every line of an envelope is read in the turn (0, 90, 180 or 270 degrees counter-clockwise) that
 * makes the envelope's text upright, chosen by how confidently its lines read together. Reading each
 * sideways line in whichever turn the recogniser liked best mixed both turns on one envelope, and lines
 * read upside down invented ♂/♀ (archive scan, Oct 2026).
 *
 * The rule, per envelope:
 * - Lines are grouped by shape: wide (along the photo's x axis), tall (along y) and square-ish
 *   (neither side 1.5 times the other: a lone ♂/♀, a short number). The envelope's turn is the one of
 *   the group with more text (summed long sides): 0/180 for wide lines, 90/270 for tall ones; square
 *   lines follow it. Lines of the other shape (a CAM ID written across the rest) get that group's own
 *   turn, chosen the same way, so they are not read crosswise.
 * - Within a group the two turns are compared by the mean log-probability of the characters read on
 *   all its lines together (length-weighted). The CAM ID line's turn, when the CAM ID reader is sure
 *   of it, settles a close call; the photo as shown (already turned by its camera tag or the batch's
 *   turn) favours 0 for wide lines, so 180 is only tried when 0 reads poorly.
 */
import type { RGBAImage } from './image'
import { quadCrop, rotate } from './image'
import type { Pt } from './lineDet'

export type Turn = 0 | 90 | 180 | 270

/** One recognition of a line crop turned by some angle. */
export interface LineRead {
  text: string
  /** mean probability of its characters */
  conf: number
  /** sum of the log-probabilities of its characters */
  logp: number
}

export interface TextLine {
  /** quadrilateral in the working image (the photo as shown, long side <= 1600), not turned */
  box: Pt[]
  text: string
  conf: number
  logp: number
  /** counter-clockwise degrees the line crop was turned before it was read */
  turn: Turn
  /** the crop as the recogniser read it (only with `keepCrops`) */
  image?: RGBAImage
  /** every turn read for this line (only with `allTurns`, for evaluation) */
  reads?: Partial<Record<Turn, LineRead>>
}

export interface EnvelopeText {
  /** counter-clockwise degrees that make the envelope's text upright */
  turn: Turn
  /** the turn of lines written across the rest (other shape), when there are any */
  crossTurn?: Turn
  /** mean log-probability per character over the lines read in `turn` (closer to 0: surer) */
  score: number
  /** lines in reading order (top to bottom, then left to right, in the turned envelope) */
  lines: TextLine[]
  /** the score of each turn that was compared (a turn not tried is absent) */
  scores: Partial<Record<Turn, number>>
  /** how many line crops were read for this text beyond the CAM ID reading */
  extraReads: number
}

export type Shape = 'wide' | 'tall' | 'square'

/** A crop's shape: tall as the CAM ID reader treats it (height > 1.5 x width), wide the other way. */
export function shapeOf(width: number, height: number): Shape {
  return height > 1.5 * width ? 'tall' : width > 1.5 * height ? 'wide' : 'square'
}

/** Mean log-probability per character of a set of readings (length-weighted); -Infinity when empty. */
export function meanLogp(reads: LineRead[]): number {
  let lp = 0, n = 0
  for (const r of reads) { lp += r.logp; n += [...r.text].length }
  return n ? lp / n : -Infinity
}

/** A group's mean character probability at or above this needs no second look (exp(-0.105) = 0.90). */
export const TURN_SURE = Math.log(0.9)
/** within this many nats per character, the sure CAM ID line's turn (or the photo as shown) decides */
export const TURN_CLOSE = 0.1

/**
 * A crop of the other shape whose long side is at most this many times the main lines' thickness
 * (median short side) is a lone sign, not a line written across: a ♀ is taller than wide on an
 * upright envelope, and read turned 270 it is a ♂ (dev sample, 24 Oct: 37 of 37 such crops on
 * upright envelopes were signs).
 */
export const GLYPH = 1.6

export interface OrientInput {
  /** width and height of each line's crop as cut from the envelope (not turned) */
  sizes: [number, number][]
  /** reading of line i turned by `turn` (cached by the caller) */
  read: (i: number, turn: Turn) => Promise<LineRead>
  /** the turn the CAM ID was read in, when the CAM ID reader was sure, and which line it was */
  camid?: { line: number; turn: Turn } | null
  /** the turn expected from the photo as shown (0 unless the caller knows better) */
  prior?: Turn
}

export interface OrientResult {
  turn: Turn
  crossTurn?: Turn
  /** the turn each line is read in */
  lineTurns: Turn[]
  scores: Partial<Record<Turn, number>>
}

const norm = (t: number) => (((t % 360) + 360) % 360) as Turn

/**
 * Choose the envelope's turn and each line's (see the module's comment). Reads only what the
 * decision needs: both turns of tall lines (the CAM ID reader has read them already), 180 for
 * wide lines only when 0 reads poorly or the CAM ID says 180.
 */
export async function chooseTurn(input: OrientInput): Promise<OrientResult> {
  const { sizes, read } = input
  const prior = norm(input.prior ?? 0)
  const shapes = sizes.map(([w, h]) => shapeOf(w, h))
  const lengths = sizes.map(([w, h]) => Math.max(w, h))
  const weight = { wide: 0, tall: 0 }
  shapes.forEach((s, i) => { if (s !== 'square') weight[s] += lengths[i] })
  // the group with more text sets the envelope's turn; without wide or tall lines, the photo as shown
  const main: 'wide' | 'tall' = weight.tall > weight.wide || (weight.tall === weight.wide && prior % 180 === 90) ? 'tall' : 'wide'
  // lone signs of the other shape are read with the main lines
  const thick = sizes.filter((_, i) => shapes[i] === main).map(([w, h]) => Math.min(w, h)).sort((a, b) => a - b)
  const median = thick.length ? thick[thick.length >> 1] : 0
  shapes.forEach((s, i) => { if (s !== main && s !== 'square' && lengths[i] <= GLYPH * median) shapes[i] = 'square' })
  const scores: Partial<Record<Turn, number>> = {}
  const pick = async (group: 'wide' | 'tall'): Promise<Turn | undefined> => {
    const members = shapes.flatMap((s, i) => (s === group || (s === 'square' && group === main) ? [i] : []))
    if (!members.length) return undefined
    const pair: [Turn, Turn] = group === 'wide' ? [0, 180] : [90, 270]
    const cam = input.camid && members.includes(input.camid.line) && pair.includes(input.camid.turn) ? input.camid.turn : undefined
    // the turn tried first: the CAM ID's, else the photo's own (or its nearer one for tall lines)
    const first: Turn = cam ?? (pair.includes(prior) ? prior : pair[0])
    const other: Turn = first === pair[0] ? pair[1] : pair[0]
    const score = async (t: Turn) => (scores[t] ??= meanLogp(await Promise.all(members.map((i) => read(i, t)))))
    const s1 = await score(first)
    // wide lines: the photo is shown upright, so the turned-over view is read only on a poor reading
    // (a sure CAM ID line already settled it); tall lines have both turns read by the CAM ID reader
    if (group === 'wide' && (cam !== undefined || s1 >= TURN_SURE)) return first
    const s2 = await score(other)
    return s2 > s1 + TURN_CLOSE || (s2 > s1 && cam === undefined && first !== prior) ? other : first
  }
  const mainTurn = (await pick(main)) ?? (main === 'wide' ? 0 : 90)
  const cross = main === 'wide' ? 'tall' : 'wide'
  const crossTurn = await pick(cross)
  const lineTurns = shapes.map((s) => (s === cross && crossTurn !== undefined ? crossTurn : mainTurn))
  return { turn: mainTurn, ...(crossTurn !== undefined ? { crossTurn } : {}), lineTurns, scores }
}

/** A line's crop from the envelope, turned as it is read. */
export function lineCrop(envelope: RGBAImage, box: Pt[], turn: Turn): RGBAImage {
  const c = quadCrop(envelope, box)
  return turn ? rotate(c, turn) : c
}

/** Reading order in the envelope turned by `turn`: top to bottom, then left to right (by line centre). */
export function readingOrder(boxes: Pt[][], turn: Turn, width: number, height: number): number[] {
  const centre = (b: Pt[]) => [b.reduce((s, p) => s + p[0], 0) / b.length, b.reduce((s, p) => s + p[1], 0) / b.length]
  // the centre in the envelope turned counter-clockwise by `turn`
  const turned = (x: number, y: number) =>
    turn === 90 ? [y, width - x] : turn === 180 ? [width - x, height - y] : turn === 270 ? [height - y, x] : [x, y]
  const keys = boxes.map((b, i) => {
    const [x, y] = turned(...(centre(b) as [number, number]))
    const ys = b.map((p) => turned(p[0], p[1])[1])
    return { i, x, y, h: Math.max(...ys) - Math.min(...ys) }
  })
  keys.sort((a, b) => a.y - b.y)
  // lines whose centres are within half a line height are one row, read left to right
  const out: typeof keys = []
  for (const k of keys) {
    let j = out.length
    while (j > 0 && Math.abs(out[j - 1].y - k.y) < 0.5 * Math.min(out[j - 1].h, k.h) && out[j - 1].x > k.x) j--
    out.splice(j, 0, k)
  }
  return out.map((k) => k.i)
}
