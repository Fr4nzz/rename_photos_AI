/**
 * Parts of a photo for the review: its wings (the box around the wings the wing check found) and its
 * envelope (the box the reader found), cut from the photo as it was read and turned as it will be.
 * Boxes are fractions of the photo (0–1) in the frame it was read in: the reading's, and the wing
 * check's once turned back by what had been written to the file when it ran.
 */
import type { PhotoRow } from '@/types'
import type { PhotoReading } from './ocr/pipeline'
import type { WingCheck } from './wings/checks'

export type Box = [number, number, number, number]

const norm = (a: number) => ((a % 360) + 360) % 360

/** The same box once the photo is turned `cw` degrees clockwise (0, 90, 180, 270). */
export function turnBox(box: Box, cw: number): Box {
  let [x0, y0, x1, y1] = box
  for (let k = 0; k < norm(cw) / 90; k++) [x0, y0, x1, y1] = [1 - y1, x0, 1 - y0, x1]
  return [x0, y0, x1, y1]
}

const grow = ([x0, y0, x1, y1]: Box, by: number): Box => [Math.max(0, x0 - by), Math.max(0, y0 - by), Math.min(1, x1 + by), Math.min(1, y1 + by)]

/** Around the wings found, in the frame the photo was read in (null: no wings, or no check). */
export function wingsBox(check: WingCheck | undefined): Box | null {
  if (!check?.wings.length) return null
  const [W, H] = check.size
  const b: Box = [
    Math.min(...check.wings.map((w) => w.box[0])) / W, Math.min(...check.wings.map((w) => w.box[1])) / H,
    Math.max(...check.wings.map((w) => w.box[2])) / W, Math.max(...check.wings.map((w) => w.box[3])) / H,
  ]
  // the check ran on the file as written then, quarter-turned photos checked again turned upright
  const frame = check.applied + (check.checkedUpright ? 90 * check.turnK : 0)
  return grow(turnBox(b, 360 - norm(frame)), 0.04)
}

/** Around the envelope the reader found, in the frame the photo was read in. */
export function envelopeBox(reading: PhotoReading | undefined): Box | null {
  if (!reading?.envelope) return null
  const [W, H] = reading.size
  const [x0, y0, x1, y1] = reading.envelope
  return grow([x0 / W, y0 / H, x1 / W, y1 / H], 0.01)
}

/** `box` of the photo (null: all of it), as the card shows it: the turn chosen applied. */
export function cutView(preview: HTMLCanvasElement, row: Pick<PhotoRow, 'rotChosen' | 'rotApplied'>, box: Box | null): HTMLCanvasElement {
  const applied = norm(Number(row.rotApplied || 0))
  // the preview shows the file now (with what was written): back to the photo as read
  const read = applied ? turn(preview, 360 - applied) : preview
  let out = read
  if (box) {
    const [x0, y0, x1, y1] = box
    const x = Math.round(x0 * read.width), y = Math.round(y0 * read.height)
    const w = Math.round((x1 - x0) * read.width), h = Math.round((y1 - y0) * read.height)
    if (w > 4 && h > 4) {
      out = document.createElement('canvas')
      out.width = w
      out.height = h
      out.getContext('2d')!.drawImage(read, x, y, w, h, 0, 0, w, h)
    }
  }
  return turn(out, norm(Number(row.rotChosen || 0)))
}

function turn(source: HTMLCanvasElement, cw: number): HTMLCanvasElement {
  if (!norm(cw)) return source
  const swap = norm(cw) % 180 !== 0
  const out = document.createElement('canvas')
  out.width = swap ? source.height : source.width
  out.height = swap ? source.width : source.height
  const ctx = out.getContext('2d')!
  ctx.translate(out.width / 2, out.height / 2)
  ctx.rotate((norm(cw) * Math.PI) / 180)
  ctx.drawImage(source, -source.width / 2, -source.height / 2)
  return out
}
