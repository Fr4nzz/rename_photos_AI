/**
 * Read the CAMID on a specimen photo: envelope detector -> text-line detector -> CAMID recognizer,
 * with the whole-photo fallback (upright and turned 180 degrees) when no envelope is found.
 * Port of the Python pipeline evaluated in ocr-next/camid-bigtest and ocr-next/fresh-test.
 */
import type { RGBAImage } from './image'
import { crop, quadCrop, resize, rotate } from './image'
import { decodeBoxes, envelopeCropBounds, letterbox, type Box } from './envelope'
import { detBoxes, detInput, type Pt } from './lineDet'
import { ctcDecode, recInput } from './rec'

export type ModelName = 'envelope' | 'line' | 'rec'

export interface Backend {
  /** Run a model on one input tensor and return its first output. */
  run(model: ModelName, data: Float32Array, dims: number[]): Promise<{ data: Float32Array; dims: number[] }>
}

export interface LineReading {
  box: Pt[]
  text: string
  conf: number
}

export interface PhotoReading {
  /** best whole-line CAMID reading (CAM + 6 digits), or null (see chooseCamid for the database rule) */
  camid: string | null
  conf: number
  /** every whole-line CAMID reading on the photo, most confident first */
  candidates: { id: string; conf: number }[]
  source: 'envelope' | 'fallback' | 'none'
  envelope: [number, number, number, number] | null
  /** 0 or 180: orientation at which the fallback found the ID */
  turned: number
  lines: LineReading[]
  /** size of the working image (long side <= 1600) the boxes refer to */
  size: [number, number]
}

export const CAMID_RE = /^CAM[0-9]{6}$/
const WORK_SIDE = 1600 // the models were trained and tested on 1600 px renditions

export const normalizeReading = (text: string) => text.replace(/\s+/g, '').toUpperCase()

async function readLine(backend: Backend, img: RGBAImage, chars: string[]): Promise<{ text: string; conf: number }> {
  const views = img.height > 1.5 * img.width ? [rotate(img, 90), rotate(img, 270)] : [img]
  let best = { text: '', conf: -1 }
  for (const view of views) {
    const input = recInput(view)
    const out = await backend.run('rec', input.data, input.dims)
    const [, steps, classes] = out.dims
    const r = ctcDecode(out.data, steps, classes, chars)
    if (r.conf > best.conf) best = r
  }
  return best
}

async function readRegion(backend: Backend, img: RGBAImage, chars: string[], limit: number, maxSide: number, skipTall: boolean): Promise<LineReading[]> {
  const det = detInput(img, limit, maxSide)
  const out = await backend.run('line', det.data, det.dims)
  const [ph, pw] = out.dims.slice(-2)
  const lines: LineReading[] = []
  for (const box of detBoxes(out.data, pw, ph, det.shape)) {
    const lineImg = quadCrop(img, box)
    if (skipTall && lineImg.height > 1.5 * lineImg.width) continue
    lines.push({ box, ...(await readLine(backend, lineImg, chars)) })
  }
  return lines
}

function bestCamid(lines: LineReading[]): { camid: string | null; conf: number; candidates: { id: string; conf: number }[] } {
  const found = lines.map((l) => ({ id: normalizeReading(l.text), conf: l.conf })).filter((f) => CAMID_RE.test(f.id))
  found.sort((a, b) => b.conf - a.conf)
  return { camid: found[0]?.id ?? null, conf: found[0]?.conf ?? 0, candidates: found }
}

/** Database rule: the most confident CAMID reading that exists in the database, or null. */
export function chooseCamid(reading: PhotoReading, known: Set<string> | null): { id: string; conf: number } | null {
  const pool = known ? reading.candidates.filter((c) => known.has(c.id)) : reading.candidates
  return pool[0] ?? null
}

export async function readPhoto(backend: Backend, photo: RGBAImage, chars: string[]): Promise<PhotoReading> {
  const scale = Math.min(1, WORK_SIDE / Math.max(photo.width, photo.height))
  const img = scale < 1 ? resize(photo, Math.round(photo.width * scale), Math.round(photo.height * scale)) : photo
  const size: [number, number] = [img.width, img.height]

  const lb = letterbox(img)
  const det = await backend.run('envelope', lb.data, lb.dims)
  const boxes = decodeBoxes(det.data, det.dims[2], lb, img)
  if (boxes.length) {
    const largest = boxes.reduce((a, b) => ((b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) > (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) ? b : a))
    const bounds = envelopeCropBounds(largest.box as Box, img)
    const envelope = crop(img, ...bounds)
    const lines = await readRegion(backend, envelope, chars, 64, 4000, false)
    // report line boxes in working-image coordinates
    for (const l of lines) l.box = l.box.map(([x, y]) => [x + bounds[0], y + bounds[1]] as Pt)
    return { ...bestCamid(lines), source: 'envelope', envelope: bounds, turned: 0, lines, size }
  }

  // No envelope found (e.g. a newer photo setup): read the whole photo, upright and turned over
  let best: PhotoReading = { camid: null, conf: 0, candidates: [], source: 'none', envelope: null, turned: 0, lines: [], size }
  for (const turned of [0, 180] as const) {
    const view = turned ? rotate(img, 180) : img
    const lines = await readRegion(backend, view, chars, 736, 1600, true)
    const b = bestCamid(lines)
    if (b.camid && (!best.camid || b.conf > best.conf)) best = { ...b, source: 'fallback', envelope: null, turned, lines, size }
    else if (!best.camid && lines.length > best.lines.length) best = { ...best, lines }
  }
  return best
}
