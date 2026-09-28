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
  /** a tall (vertical) line was read turned by this many degrees counter-clockwise */
  turn?: 90 | 270
  /** made by joining a CAM prefix with the 6 digits read as a separate piece on the same row */
  joined?: boolean
}

/**
 * A crossed-out or widely spaced ID line is often detected as pieces ("CAM7" + "077332"). When a
 * piece of exactly 6 digits has, on the same row and just to its left, a piece starting with CAM,
 * offer CAM + those digits as a reading (the database and sequence checks still apply).
 */
export function joinPieces(lines: LineReading[]): LineReading[] {
  const flat = lines.filter((l) => !l.turn)
  const geo = (l: LineReading) => {
    const xs = l.box.map((p) => p[0]), ys = l.box.map((p) => p[1])
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) }
  }
  const out: LineReading[] = []
  for (const d of flat) {
    const digits = normalizeReading(d.text)
    if (!/^[0-9]{6}$/.test(digits)) continue
    const g = geo(d), h = g.y1 - g.y0, cy = (g.y0 + g.y1) / 2
    for (const c of flat) {
      if (c === d || !/^CAM/.test(normalizeReading(c.text)) || CAMID_RE.test(normalizeReading(c.text))) continue
      const k = geo(c), kcy = (k.y0 + k.y1) / 2
      if (Math.abs(kcy - cy) > 0.6 * h || k.x1 > g.x0 + 0.3 * h || g.x0 - k.x1 > 4 * h) continue
      out.push({ box: [[k.x0, Math.min(k.y0, g.y0)], [g.x1, Math.min(k.y0, g.y0)], [g.x1, Math.max(k.y1, g.y1)], [k.x0, Math.max(k.y1, g.y1)]],
        text: `CAM${digits}`, conf: Math.min(c.conf, d.conf), joined: true })
    }
  }
  return out
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

/**
 * Full-resolution pixels of a region of the working image, when the caller has the original photo:
 * `scale` = full-resolution pixels per working-image pixel.
 */
export type FullCrop = (bounds: [number, number, number, number]) => Promise<{ image: RGBAImage; scale: number } | null>

/** How text lines are read when full resolution is available (evaluation switch). */
export const LINE_MODE = { value: 'auto' as 'working' | 'hires' | 'both' | 'hidet' | 'union' | 'auto' }
/** evaluation counter: photos that needed the full-resolution pass */
export const HIRES_PASSES = { count: 0 }

export const CAMID_RE = /^CAM[0-9]{6}$/
/** below this confidence the envelope is read again at full resolution (sealed test: 7% of photos) */
const AUTO_SURE = 0.95
const WORK_SIDE = 1600 // the models were trained and tested on 1600 px renditions

export const normalizeReading = (text: string) => text.replace(/\s+/g, '').toUpperCase()

async function readLine(backend: Backend, img: RGBAImage, chars: string[]): Promise<{ text: string; conf: number; turn?: 90 | 270 }> {
  const views: [RGBAImage, 90 | 270 | undefined][] = img.height > 1.5 * img.width
    ? [[rotate(img, 90), 90], [rotate(img, 270), 270]] : [[img, undefined]]
  let best: { text: string; conf: number; turn?: 90 | 270 } = { text: '', conf: -1 }
  for (const [view, turn] of views) {
    const input = recInput(view)
    const out = await backend.run('rec', input.data, input.dims)
    const [, steps, classes] = out.dims
    const r = ctcDecode(out.data, steps, classes, chars)
    if (r.conf > best.conf) best = turn ? { ...r, turn } : r
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

/**
 * Read every text line of an envelope upright and turned 180 degrees (the line detector runs once;
 * each line crop is re-read turned over). Returns [upright, flipped]; flipped boxes are in the
 * coordinates of the envelope turned 180 degrees. Tall (sideways) lines are already read both
 * ways and are left out of the flipped list.
 */
async function readEnvelopeBothWays(
  backend: Backend, env: RGBAImage, chars: string[], hi: { image: RGBAImage; scale: number } | null,
  mode: typeof LINE_MODE.value = LINE_MODE.value,
): Promise<[LineReading[], LineReading[]]> {
  const upright: LineReading[] = [], flipped: LineReading[] = []
  if (hi && (mode === 'hidet' || mode === 'union')) {
    // find and read the lines on the full-resolution envelope; boxes back in working coordinates
    const det = detInput(hi.image, 64, 4000)
    const out = await backend.run('line', det.data, det.dims)
    const [ph, pw] = out.dims.slice(-2)
    for (const fbox of detBoxes(out.data, pw, ph, det.shape)) {
      const lineImg = quadCrop(hi.image, fbox)
      const box = fbox.map(([x, y]) => [x / hi.scale, y / hi.scale] as Pt)
      upright.push({ box, ...(await readLine(backend, lineImg, chars)) })
      if (lineImg.height <= 1.5 * lineImg.width) {
        const r = await readLine(backend, rotate(lineImg, 180), chars)
        flipped.push({ box: [2, 3, 0, 1].map((k) => [env.width - box[k][0], env.height - box[k][1]] as Pt), ...r })
      }
    }
    if (mode === 'hidet') return [upright, flipped]
  }
  const det = detInput(env, 64, 4000)
  const out = await backend.run('line', det.data, det.dims)
  const [ph, pw] = out.dims.slice(-2)
  // lines are found on the working image (as trained) and, with the original at hand, read from
  // its full-resolution pixels: small print (e.g. under a barcode) keeps its detail
  const views = (box: Pt[]) => {
    const work = quadCrop(env, box)
    if (!hi || mode === 'working' || mode === 'union') return [work]
    const full = quadCrop(hi.image, box.map(([x, y]) => [x * hi.scale, y * hi.scale] as Pt))
    return mode === 'both' ? [work, full] : [full]
  }
  const best = async (imgs: RGBAImage[]) => {
    let top = await readLine(backend, imgs[0], chars)
    for (const im of imgs.slice(1)) { const r = await readLine(backend, im, chars); if (r.conf > top.conf) top = r }
    return top
  }
  for (const box of detBoxes(out.data, pw, ph, det.shape)) {
    const lineImgs = views(box)
    const lineImg = lineImgs[0]
    upright.push({ box, ...(await best(lineImgs)) })
    if (lineImg.height <= 1.5 * lineImg.width) {
      const r = await best(lineImgs.map((im) => rotate(im, 180)))
      // the same quadrilateral in the envelope turned 180: (x, y) -> (w - x, h - y), corners re-ordered
      const turnedBox = [2, 3, 0, 1].map((k) => [env.width - box[k][0], env.height - box[k][1]] as Pt)
      flipped.push({ box: turnedBox, ...r })
    }
  }
  return [upright, flipped]
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

export async function readPhoto(backend: Backend, photo: RGBAImage, chars: string[], fullCrop?: FullCrop): Promise<PhotoReading> {
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
    let upright: LineReading[], flipped: LineReading[]
    if (LINE_MODE.value === 'auto') {
      // normal reading first; the full-resolution pass only when it found no confident CAMID
      ;[upright, flipped] = await readEnvelopeBothWays(backend, envelope, chars, null, 'working')
      const sure = Math.max(bestCamid(upright).conf, bestCamid(flipped).conf) >= AUTO_SURE
      const hi = !sure && fullCrop ? await fullCrop(bounds).catch(() => null) : null
      if (hi) {
        HIRES_PASSES.count++
        const [u2, f2] = await readEnvelopeBothWays(backend, envelope, chars, hi, 'hidet')
        upright.push(...u2)
        flipped.push(...f2)
      }
    } else {
      const hi = fullCrop ? await fullCrop(bounds).catch(() => null) : null
      ;[upright, flipped] = await readEnvelopeBothWays(backend, envelope, chars, hi)
    }
    // sealed test: +1 correct, no new wrong readings
    upright.push(...joinPieces(upright))
    flipped.push(...joinPieces(flipped))
    // Upside-down photos: every line is also read turned over. The orientation that gives the most
    // confident whole-line CAMID wins (a correct reading is near-certain; an upside-down misread
    // rarely is); without any CAMID reading, the photo is taken as upright.
    const bestUp = bestCamid(upright), bestFlip = bestCamid(flipped)
    const useFlipped = !!bestFlip.camid && (!bestUp.camid || bestFlip.conf > bestUp.conf)
    const [ew, eh] = [bounds[2] - bounds[0], bounds[3] - bounds[1]]
    const lines = useFlipped ? flipped : upright
    for (const l of lines) {
      l.box = useFlipped
        // flipped boxes are given in the whole working image turned 180 (as for the fallback)
        ? l.box.map(([x, y]) => [img.width - (bounds[0] + ew - x), img.height - (bounds[1] + eh - y)] as Pt)
        : l.box.map(([x, y]) => [x + bounds[0], y + bounds[1]] as Pt)
    }
    return { ...bestCamid(lines), source: 'envelope', envelope: bounds, turned: useFlipped ? 180 : 0, lines, size }
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
