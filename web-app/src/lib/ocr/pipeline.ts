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
import { lexiconPosteriors, type IdView } from './lexicon'
import { chooseTurn, lineCrop, meanLogp, readingOrder, type EnvelopeText, type LineRead, type TextLine, type Turn } from './envelopeText'

export type ModelName = 'envelope' | 'line' | 'rec'

export interface Backend {
  /** Run a model on one input tensor and return its first output. */
  run(model: ModelName, data: Float32Array, dims: number[]): Promise<{ data: Float32Array; dims: number[] }>
  /** The side of a model's square input when the file fixes it (the envelope detector's letterbox). */
  inputSide?(model: ModelName): Promise<number | undefined>
}

export interface LineReading {
  box: Pt[]
  text: string
  conf: number
  /** a tall (vertical) line was read turned by this many degrees counter-clockwise */
  turn?: 90 | 270
  /** made by joining a CAM prefix with the 6 digits read as a separate piece on the same row */
  joined?: boolean
  /** this line's views that read CAM + 6 digits (moved to PhotoReading.idViews) */
  idViews?: IdView[]
}

/**
 * A crossed-out or widely spaced ID line is often detected as pieces ("CAM7" + "077332", "CAM078" + "715").
 * When a piece of digits has, on the same row and just to its left, a piece starting with CAM, offer
 * CAM + 6 digits as a reading (the database and sequence checks still apply): a piece of exactly 6 digits
 * follows any CAM piece (its digits are taken as crossed out); a shorter piece only completes a piece that
 * is exactly CAM + the missing number of digits.
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
    if (!/^[0-9]{1,6}$/.test(digits)) continue
    const g = geo(d), h = g.y1 - g.y0, cy = (g.y0 + g.y1) / 2
    for (const c of flat) {
      const head = normalizeReading(c.text)
      if (c === d || !/^CAM/.test(head) || CAMID_RE.test(head)) continue
      const id = digits.length === 6 ? `CAM${digits}` : head.length + digits.length === 9 && /^CAM[0-9]*$/.test(head) ? head + digits : null
      if (!id) continue
      const k = geo(c), kcy = (k.y0 + k.y1) / 2
      if (Math.abs(kcy - cy) > 0.6 * h || k.x1 > g.x0 + 0.3 * h || g.x0 - k.x1 > 4 * h) continue
      out.push({ box: [[k.x0, Math.min(k.y0, g.y0)], [g.x1, Math.min(k.y0, g.y0)], [g.x1, Math.max(k.y1, g.y1)], [k.x0, Math.max(k.y1, g.y1)]],
        text: id, conf: Math.min(c.conf, d.conf), joined: true })
    }
  }
  return out
}

/**
 * How a CAM ID was found in a line: the whole line; a token ending a longer line (CAM + 6 digits followed
 * by nothing but a sex sign or punctuation: "CAM075192♀", "Ela-meme-1-CAM079134"); or with its first letter
 * cut off or blotted, or its first two at the edge of a photo of the envelope's top strip ("AM074594",
"CM077250", "M077392"; at the start of the line or after a separator).
 */
export type IdFind = 'line' | 'token' | 'cut'

/** Without the database, an ID read with letters of CAM missing is taken only this sure. */
export const CUT_SURE = 0.99

const TAIL = '[♀♂.,;:)\\]|/_-]*'
const TOKEN_RE = new RegExp(`CAM([0-9]{6})${TAIL}$`)
const CUT_RE = new RegExp(`(?:^|[^A-Z0-9])(?:AM|CM|M)([0-9]{6})${TAIL}$`)

/** The CAM ID in a line's text, or null. Strict: exactly 6 digits, nothing but signs after them. */
export function findCamid(text: string): { id: string; how: IdFind } | null {
  const t = normalizeReading(text)
  if (CAMID_RE.test(t)) return { id: t, how: 'line' }
  let m = TOKEN_RE.exec(t)
  if (m) return { id: `CAM${m[1]}`, how: 'token' }
  m = CUT_RE.exec(t)
  if (m) return { id: `CAM${m[1]}`, how: 'cut' }
  return null
}

export interface PhotoReading {
  /** best whole-line CAMID reading (CAM + 6 digits), or null (see chooseCamid for the database rule) */
  camid: string | null
  conf: number
  /** every CAMID reading on the photo: whole lines first, then IDs inside longer lines (`how`), each most confident first */
  candidates: IdCandidate[]
  source: 'envelope' | 'fallback' | 'none'
  envelope: [number, number, number, number] | null
  /** 0 or 180: orientation at which the fallback found the ID */
  turned: number
  lines: LineReading[]
  /** size of the working image (long side <= 1600) the boxes refer to */
  size: [number, number]
  /** every line view, in either orientation, that read CAM + 6 digits, for the lexicon margin
   * (decide.ts); dropped once the batch is decided */
  idViews?: IdView[]
  /** the envelope's whole text, read in one turn (only when asked for: ReadOptions.text) */
  text?: EnvelopeText
}

export interface ReadOptions {
  /** also read every line of the envelope in its one turn (PhotoReading.text) */
  text?: boolean
  /** the turn the envelope is expected in, counter-clockwise (default 0: the photo as shown is upright) */
  prior?: Turn
  /** keep each text line's crop as it was read (TextLine.image), for cutting training crops */
  keepCrops?: boolean
  /** evaluation: read every text line in all four turns (TextLine.reads) */
  allTurns?: boolean
}

/** The working-image lines of an envelope and what was read of them, shared by the CAM ID and text readings. */
interface LineCache {
  boxes: Pt[][]
  /** line i's crop from the working-image envelope, not turned */
  crop: (i: number) => RGBAImage
  /** `${line}:${turn}` -> reading */
  reads: Map<string, LineRead>
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
export const AUTO_SURE = 0.95
/**
 * An upright line that reads CAM + 6 digits this sure ends the envelope's reading: the lines below it and
 * the turned-over views are not read (app speed campaign, 17 Oct 2026: on its locked set of 2,014 photos
 * no decision changed, with or without the margin rule, and the recogniser ran 64 % less).
 */
const STOP_SURE = 0.99
const WORK_SIDE = 1600 // the models were trained and tested on 1600 px renditions

export const normalizeReading = (text: string) => text.replace(/\s+/g, '').toUpperCase()

type Read = { text: string; conf: number; turn?: 90 | 270; idViews?: IdView[] }

/** Recognise one crop as it is. */
async function recognise(backend: Backend, img: RGBAImage, chars: string[]) {
  const input = recInput(img)
  const out = await backend.run('rec', input.data, input.dims)
  const [, steps, classes] = out.dims
  return { ...ctcDecode(out.data, steps, classes, chars), out: out.data, steps, classes }
}

/** `seen`: each view read, with its turn relative to `img` (for the envelope's text). */
async function readLine(backend: Backend, img: RGBAImage, chars: string[], seen?: (turn: Turn, r: LineRead) => void): Promise<Read> {
  const views: [RGBAImage, 90 | 270 | undefined][] = img.height > 1.5 * img.width
    ? [[rotate(img, 90), 90], [rotate(img, 270), 270]] : [[img, undefined]]
  let best: Read = { text: '', conf: -1 }
  const idViews: IdView[] = []
  for (const [view, turn] of views) {
    const { text, conf, logp, out, steps, classes } = await recognise(backend, view, chars)
    seen?.(turn ?? 0, { text, conf, logp })
    const r = { text, conf }
    // both turns of a tall line count for the margin, not only the one kept
    const id = normalizeReading(r.text)
    if (CAMID_RE.test(id)) idViews.push({ id, steps, probs: lexiconPosteriors(out, steps, classes, chars) })
    if (r.conf > best.conf) best = turn ? { ...r, turn } : r
  }
  return idViews.length ? { ...best, idViews } : best
}

/** The CAMID views of these lines, taken off them (the lines stay small for the review). */
function takeIdViews(lines: LineReading[]): IdView[] {
  const out: IdView[] = []
  for (const l of lines) if (l.idViews) { out.push(...l.idViews); delete l.idViews }
  return out
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
 * ways and are left out of the flipped list. On the working image the lines are read upright first,
 * in the detector's order, up to the first sure ID (STOP_SURE); turned over only when none was.
 */
async function readEnvelopeBothWays(
  backend: Backend, env: RGBAImage, chars: string[], hi: { image: RGBAImage; scale: number } | null,
  mode: typeof LINE_MODE.value = LINE_MODE.value, cache?: LineCache,
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
  const boxes = detBoxes(out.data, pw, ph, det.shape)
  // working-image crops, cut when first needed; kept with their readings for the envelope's text
  const work: RGBAImage[] = []
  const workCrop = (i: number) => (work[i] ??= quadCrop(env, boxes[i]))
  if (cache) { cache.boxes = boxes; cache.crop = workCrop }
  // lines are found on the working image (as trained) and, with the original at hand, read from
  // its full-resolution pixels: small print (e.g. under a barcode) keeps its detail
  const views = (i: number) => {
    if (!hi || mode === 'working' || mode === 'union') return [workCrop(i)]
    const full = quadCrop(hi.image, boxes[i].map(([x, y]) => [x * hi.scale, y * hi.scale] as Pt))
    return mode === 'both' ? [workCrop(i), full] : [full]
  }
  // `keep`: given when the line's working-image crop is read alone; its views go to the cache
  const best = async (imgs: RGBAImage[], keep?: (t: Turn, r: LineRead) => void) => {
    let top = await readLine(backend, imgs[0], chars, keep)
    const idViews = [...(top.idViews ?? [])]
    for (const im of imgs.slice(1)) {
      const r = await readLine(backend, im, chars)
      idViews.push(...(r.idViews ?? []))
      if (r.conf > top.conf) top = r
    }
    return { ...top, idViews }
  }
  const keeper = (i: number, imgs: RGBAImage[], base: Turn) => cache && imgs.length === 1 && imgs[0] === work[i]
    ? (t: Turn, r: LineRead) => { cache.reads.set(`${i}:${(t + base) % 360}`, r) } : undefined
  const viewsOf: RGBAImage[][] = []
  for (const [i, box] of boxes.entries()) {
    viewsOf.push(views(i))
    const line = { box, ...(await best(viewsOf[i], keeper(i, viewsOf[i], 0))) }
    upright.push(line)
    // the line keeps the posteriors of its views (both turns of a tall line) for the margin
    if (!hi && CAMID_RE.test(normalizeReading(line.text)) && line.conf >= STOP_SURE) return [upright, flipped]
  }
  for (const [i, box] of boxes.entries()) {
    const lineImgs = viewsOf[i]
    if (lineImgs[0].height > 1.5 * lineImgs[0].width) continue
    const r = await best(lineImgs.map((im) => rotate(im, 180)), keeper(i, lineImgs, 180))
    // the same quadrilateral in the envelope turned 180: (x, y) -> (w - x, h - y), corners re-ordered
    const turnedBox = [2, 3, 0, 1].map((k) => [env.width - box[k][0], env.height - box[k][1]] as Pt)
    flipped.push({ box: turnedBox, ...r })
  }
  return [upright, flipped]
}

export interface IdCandidate { id: string; conf: number; how?: IdFind }

const takeable = (known: Set<string> | null) => (c: IdCandidate) =>
  known ? known.has(c.id) : c.how !== 'cut' || c.conf >= CUT_SURE

/**
 * The CAM ID readings of these lines: whole lines (and joined pieces), most confident first; with
 * `inLines`, then the IDs found inside longer lines or with letters of CAM missing (findCamid). `camid` is the first
 * that can be taken without the database. The turn of the envelope is chosen on whole lines only.
 */
function bestCamid(lines: LineReading[], inLines = false): { camid: string | null; conf: number; candidates: IdCandidate[] } {
  const whole: IdCandidate[] = [], inside: IdCandidate[] = []
  for (const l of lines) {
    const f = findCamid(l.text)
    if (f?.how === 'line') whole.push({ id: f.id, conf: l.conf })
    else if (f && inLines) inside.push({ id: f.id, conf: l.conf, how: f.how })
  }
  whole.sort((a, b) => b.conf - a.conf)
  inside.sort((a, b) => b.conf - a.conf)
  const candidates = [...whole, ...inside]
  const first = candidates.find(takeable(null))
  return { camid: first?.id ?? null, conf: first?.conf ?? 0, candidates }
}

/**
 * Database rule: the first CAMID reading (whole lines first, then IDs inside lines, each most confident
 * first) that exists in the database, or null. Without the database, an ID read with letters of CAM
 * missing is taken only when read CUT_SURE or more.
 */
export function chooseCamid(reading: PhotoReading, known: Set<string> | null): { id: string; conf: number } | null {
  const c = reading.candidates.find(takeable(known))
  return c ? { id: c.id, conf: c.conf } : null
}

/** The envelope's lines found on the working image, nothing read yet. */
async function workingLines(backend: Backend, env: RGBAImage): Promise<LineCache> {
  const det = detInput(env, 64, 4000)
  const out = await backend.run('line', det.data, det.dims)
  const [ph, pw] = out.dims.slice(-2)
  const boxes = detBoxes(out.data, pw, ph, det.shape)
  const crops: RGBAImage[] = []
  return { boxes, crop: (i) => (crops[i] ??= quadCrop(env, boxes[i])), reads: new Map() }
}

/**
 * Every line of the envelope, read in the envelope's one turn (envelopeText.ts). Reuses what the CAM ID
 * reading read on the same crops; `camid`: that reading's best whole-line CAM ID, whose turn is a prior
 * when the reader was sure of it.
 */
async function readText(
  backend: Backend, env: RGBAImage, chars: string[], cache: LineCache, bounds: [number, number, number, number],
  camid: { camid: string | null; conf: number }, opts: ReadOptions,
): Promise<EnvelopeText> {
  let extraReads = 0
  const read = async (i: number, turn: Turn): Promise<LineRead> => {
    const key = `${i}:${turn}`
    let r = cache.reads.get(key)
    if (!r) {
      const { text, conf, logp } = await recognise(backend, lineCrop(env, cache.boxes[i], turn), chars)
      cache.reads.set(key, (r = { text, conf, logp }))
      extraReads++
    }
    return r
  }
  const n = cache.boxes.length
  const dims = Array.from({ length: n }, (_, i) => cache.crop(i))
  let camidLine: { line: number; turn: Turn } | null = null
  if (camid.camid && camid.conf >= AUTO_SURE) {
    let top = -1
    for (const [key, r] of cache.reads) {
      if (normalizeReading(r.text) !== camid.camid || r.conf < AUTO_SURE || r.conf <= top) continue
      const [line, turn] = key.split(':').map(Number)
      camidLine = { line, turn: turn as Turn }
      top = r.conf
    }
  }
  const choice = await chooseTurn({
    sizes: dims.map((c) => [c.width, c.height] as [number, number]),
    read, camid: camidLine, prior: opts.prior,
  })
  const lines: TextLine[] = []
  for (const i of readingOrder(cache.boxes, choice.turn, env.width, env.height)) {
    const turn = choice.lineTurns[i]
    const r = await read(i, turn)
    const line: TextLine = { box: cache.boxes[i].map(([x, y]) => [x + bounds[0], y + bounds[1]] as Pt), ...r, turn }
    if (opts.keepCrops) line.image = lineCrop(env, cache.boxes[i], turn)
    if (opts.allTurns) {
      line.reads = {}
      for (const t of [0, 90, 180, 270] as const) line.reads[t] = await read(i, t)
    }
    lines.push(line)
  }
  const main = lines.filter((l) => l.turn === choice.turn)
  return {
    turn: choice.turn, ...(choice.crossTurn !== undefined ? { crossTurn: choice.crossTurn } : {}),
    score: main.length ? meanLogp(main) : 0, lines, scores: choice.scores, extraReads,
  }
}

export async function readPhoto(backend: Backend, photo: RGBAImage, chars: string[], fullCrop?: FullCrop, opts: ReadOptions = {}): Promise<PhotoReading> {
  const scale = Math.min(1, WORK_SIDE / Math.max(photo.width, photo.height))
  const img = scale < 1 ? resize(photo, Math.round(photo.width * scale), Math.round(photo.height * scale)) : photo
  const size: [number, number] = [img.width, img.height]

  const lb = letterbox(img, await backend.inputSide?.('envelope'))
  const det = await backend.run('envelope', lb.data, lb.dims)
  const boxes = decodeBoxes(det.data, det.dims[2], lb, img)
  if (boxes.length) {
    const largest = boxes.reduce((a, b) => ((b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) > (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) ? b : a))
    const bounds = envelopeCropBounds(largest.box as Box, img)
    const envelope = crop(img, ...bounds)
    let upright: LineReading[], flipped: LineReading[]
    let cache: LineCache | undefined = opts.text ? { boxes: [], crop: () => envelope, reads: new Map() } : undefined
    if (LINE_MODE.value === 'auto') {
      // normal reading first; the full-resolution pass only when it found no confident CAMID
      ;[upright, flipped] = await readEnvelopeBothWays(backend, envelope, chars, null, 'working', cache)
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
      // the text is read on the working image's lines whatever the evaluation mode
      if (cache) cache = await workingLines(backend, envelope)
    }
    // before the CAM ID decision below changes the lines' boxes
    const text = cache ? await readText(backend, envelope, chars, cache, bounds, bestCamid([...upright, ...flipped]), opts) : undefined
    const idViews = takeIdViews([...upright, ...flipped])
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
    return { ...bestCamid(lines, true), source: 'envelope', envelope: bounds, turned: useFlipped ? 180 : 0, lines, size, idViews, ...(text ? { text } : {}) }
  }

  // No envelope found (e.g. a newer photo setup): read the whole photo, upright and turned over
  let best: PhotoReading = { camid: null, conf: 0, candidates: [], source: 'none', envelope: null, turned: 0, lines: [], size }
  const idViews: IdView[] = []
  for (const turned of [0, 180] as const) {
    const view = turned ? rotate(img, 180) : img
    const lines = await readRegion(backend, view, chars, 736, 1600, true)
    idViews.push(...takeIdViews(lines))
    const b = bestCamid(lines, true)
    if (b.camid && (!best.camid || b.conf > best.conf)) best = { ...b, source: 'fallback', envelope: null, turned, lines, size }
    else if (!best.camid && lines.length > best.lines.length) best = { ...best, lines }
  }
  return { ...best, idViews }
}
