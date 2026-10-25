/// <reference lib="webworker" />
/**
 * Photo worker: reads a photo's CAMID and checks its wings from one decoding of the photo. A camera's
 * 18 MP JPEG takes ~150 ms to decode and ~50 ms to shrink; the reader and the wing check used to do
 * both in workers of their own. The photo is decoded as it shows (JPEG/PNG via createImageBitmap with
 * its EXIF orientation; RAW via its largest embedded JPEG and the RAW's orientation tag; HEIC with
 * libheif) and shrunk to 1600 px; the reader gets that shrunk photo re-saved as a JPEG (see
 * asRendition), the wings get it as it is. Each model loads on first use, on the device the pool
 * chose for it (see runtime.ts).
 */
import { readOrientation } from './orientation'
import { extractLargestJpeg } from './rawPreview'
import { decodeHeic, HEIC_EXTENSIONS } from './heic'
import { rotate, type RGBAImage } from './ocr/image'
import { readPhoto, type Backend, type FullCrop, type PhotoReading } from './ocr/pipeline'
import { analysePhoto, type WingBackend, type WingConfig, type WingResult } from './wings/model'
import { openRuntime, probeDevices, type ProbeTimes, type Runtime, type RuntimePlan } from './runtime'

export interface PhotoRequest { id: number; file: File; raw: boolean; read: boolean; wings: boolean }
export type PhotoResponse =
  | { id: number; ok: true; reading?: PhotoReading; readError?: string; wings?: WingResult; wingError?: string; ms: number }
  | { id: number; ok: false; error: string }
/**
 * `bigLines`: where this worker sends its large line detections; `serves`: the ports of the workers
 * whose large line detections this one runs (see BIG_LINE_INPUT).
 */
export type PhotoInit = { init: string; plan: RuntimePlan; bigLines?: MessagePort; serves?: MessagePort[] }
export type PhotoReady = { id: -1; ready: true } | { id: -1; ready: false; error: string }
/** times the models on this computer's devices (devicePlan.ts) */
export type PhotoProbe = { probe: string }
export type ProbeResult = { id: -2; times: ProbeTimes } | { id: -2; error: string }

let runtime: Runtime | null = null
let base = ''
let chars: Promise<string[]> | null = null
let wingCfg: Promise<WingConfig> | null = null

/**
 * A line detection on more pixels than this (the full-resolution pass on an envelope: up to 4,000 px a
 * side) grows the processor's memory of the worker running it by up to 1 GB, which it keeps until the
 * worker ends. Only some workers run them (photoPool.ts), for the others too, so that the readers' memory
 * stays as it was with two reading workers. The whole-photo pass (1,600 x 1,088) stays where it is.
 */
const BIG_LINE_INPUT = 2_000_000
let bigLines: MessagePort | null = null
const bigPending = new Map<number, (r: { data: Float32Array; dims: number[] } | { error: string }) => void>()
let bigId = 1

function runBigLine(data: Float32Array, dims: number[]): Promise<{ data: Float32Array; dims: number[] }> {
  const id = bigId++
  return new Promise((resolve, reject) => {
    bigPending.set(id, (r) => ('error' in r ? reject(new Error(r.error)) : resolve(r)))
    bigLines!.postMessage({ id, data, dims }, [data.buffer])
  })
}

/** Runs the large line detections of other workers. */
function serve(port: MessagePort) {
  port.onmessage = async (e: MessageEvent<{ id: number; data: Float32Array; dims: number[] }>) => {
    const { id, data, dims } = e.data
    try {
      await ready
      const out = await runtime!.run('line', data, dims)
      port.postMessage({ id, data: out.data, dims: out.dims }, [out.data.buffer])
    } catch (err) {
      port.postMessage({ id, error: err instanceof Error ? err.message : String(err) })
    }
  }
}

const readerBackend: Backend = {
  async run(model, data, dims) {
    if (model === 'line' && bigLines && dims[2] * dims[3] > BIG_LINE_INPUT && runtime!.deviceOf('line') === 'cpu') return runBigLine(data, dims)
    return runtime!.run(model, data, dims)
  },
  // a model file cached from before a change of its input size still gets the input it was made for
  async inputSide(model) {
    const side = (await runtime!.inputShape(model))?.[3]
    return typeof side === 'number' ? side : undefined
  },
}

const wingBackend: WingBackend = {
  async segment(x, S) {
    const [det, proto] = await runtime!.runAll('wingseg', x, [1, 3, S, S])
    return { det: det.data, detDims: det.dims, proto: proto.data, protoDims: proto.dims }
  },
  async classify(x, n, size) {
    return (await runtime!.run('wingcls', x, [n, 3, size, size])).data
  },
}

const WORK_SIDE = 1600

/**
 * The reader was trained on 1600 px JPEG renditions: re-saving the shrunk photo as a quality 85
 * JPEG makes browser input match them (sealed test: 626/651 right instead of 593 without it).
 */
async function asRendition(canvas: OffscreenCanvas): Promise<RGBAImage> {
  const jpeg = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.85 })
  const bmp = await createImageBitmap(jpeg)
  const out = new OffscreenCanvas(bmp.width, bmp.height)
  const ctx = out.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bmp, 0, 0)
  bmp.close()
  return { width: out.width, height: out.height, data: ctx.getImageData(0, 0, out.width, out.height).data }
}

type Source = CanvasImageSource & { width: number; height: number; close?: () => void }

/**
 * Region [x0, y0, x1, y1] of a working image `w` px wide, cut from the full-size photo. The photo is
 * decoded again for it: holding every photo in flight at full size (~72 MB for 18 MP) for the few
 * read at full resolution cost more memory than the second decoding costs time.
 */
function cropper(load: () => Promise<Source>, w: number): FullCrop {
  return async ([x0, y0, x1, y1]) => {
    const source = await load()
    try {
      const scale = source.width / w
      const sx = Math.round(x0 * scale), sy = Math.round(y0 * scale)
      const sw = Math.max(1, Math.round((x1 - x0) * scale)), sh = Math.max(1, Math.round((y1 - y0) * scale))
      const c = new OffscreenCanvas(sw, sh)
      const ctx = c.getContext('2d', { willReadFrequently: true })!
      ctx.drawImage(source, sx, sy, sw, sh, 0, 0, sw, sh)
      return { image: { width: sw, height: sh, data: ctx.getImageData(0, 0, sw, sh).data }, scale }
    } finally {
      source.close?.()
    }
  }
}

interface Decoded {
  /** what the reader reads (the rendition), when asked */
  reading?: RGBAImage
  /** full-resolution pixels of a region of `reading` (text lines are read from these) */
  full?: FullCrop
  /** what the wing check looks at (the photo shrunk, as shown), when asked */
  wings?: RGBAImage
}

/** The photo at 1600 px on its canvas: the wings' pixels and the reader's rendition both come from it. */
function shrink(source: Source): OffscreenCanvas {
  const scale = Math.min(1, WORK_SIDE / Math.max(source.width, source.height))
  const w = Math.round(source.width * scale), h = Math.round(source.height * scale)
  const canvas = new OffscreenCanvas(w, h)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(source, 0, 0, w, h)
  return canvas
}

const pixelsOf = (canvas: OffscreenCanvas): RGBAImage =>
  ({ width: canvas.width, height: canvas.height, data: canvas.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, canvas.width, canvas.height).data })

async function decode(file: File, raw: boolean, read: boolean, wings: boolean): Promise<Decoded> {
  if (HEIC_EXTENSIONS.has(file.name.slice(file.name.lastIndexOf('.')).toLowerCase())) {
    const full = await decodeHeic(file)
    const src = new OffscreenCanvas(full.width, full.height)
    src.getContext('2d')!.putImageData(new ImageData(full.data, full.width, full.height), 0, 0)
    const small = shrink(src)
    return {
      wings: wings ? pixelsOf(small) : undefined,
      ...(read ? { reading: await asRendition(small), full: cropper(async () => src, small.width) } : {}),
    }
  }
  let source: Blob = file
  let turnCcw = 0
  if (raw) {
    const preview = await extractLargestJpeg(file)
    if (!preview) throw new Error('no embedded preview in this RAW file')
    source = preview
    const slot = await readOrientation(file)
    turnCcw = slot ? ({ 3: 180, 6: 270, 8: 90 } as Record<number, number>)[slot.value] ?? 0 : 0
  }
  const bmp = await createImageBitmap(source, { imageOrientation: 'from-image' })
  const small = shrink(bmp)
  const shown = wings ? pixelsOf(small) : undefined
  const rendition = read ? await asRendition(small) : undefined
  bmp.close()
  if (turnCcw) {
    // RAW previews turned by the RAW's tag: read at working resolution only
    const turn = turnCcw as 90 | 180 | 270
    return { reading: rendition && rotate(rendition, turn), wings: shown && rotate(shown, turn) }
  }
  const full = read ? cropper(() => createImageBitmap(source, { imageOrientation: 'from-image' }), small.width) : undefined
  return { reading: rendition, wings: shown, full }
}

let ready: Promise<void> | null = null

async function init(msg: PhotoInit) {
  base = msg.init
  bigLines = msg.bigLines ?? null
  if (bigLines) bigLines.onmessage = (e) => { bigPending.get(e.data.id)?.(e.data); bigPending.delete(e.data.id) }
  for (const port of msg.serves ?? []) serve(port)
  runtime = await openRuntime(msg.plan, base)
}

self.onmessage = async (event: MessageEvent<PhotoRequest | PhotoInit | PhotoProbe>) => {
  const msg = event.data
  if ('probe' in msg) {
    try {
      self.postMessage({ id: -2, times: await probeDevices(msg.probe) } satisfies ProbeResult)
    } catch (e) {
      self.postMessage({ id: -2, error: String(e) } satisfies ProbeResult)
    }
    return
  }
  if ('init' in msg) {
    ready = init(msg)
    try {
      await ready
      self.postMessage({ id: -1, ready: true } satisfies PhotoReady)
    } catch (e) {
      self.postMessage({ id: -1, ready: false, error: String(e) } satisfies PhotoReady)
    }
    return
  }
  try {
    await ready
    const t0 = performance.now()
    chars ??= fetch(`${base}models/envelope_reader_rec_chars.json`).then((r) => r.json() as Promise<string[]>)
    wingCfg ??= fetch(`${base}models/wingside.json`).then((r) => r.json() as Promise<WingConfig>)
    const decoded = await decode(msg.file, msg.raw, msg.read, msg.wings)
    const out: PhotoResponse & { ok: true } = { id: msg.id, ok: true, ms: 0 }
    if (decoded.reading) {
      try {
        out.reading = await readPhoto(readerBackend, decoded.reading, await chars, decoded.full)
      } catch (e) {
        out.readError = e instanceof Error ? e.message : String(e)
      }
    }
    if (decoded.wings) {
      try {
        out.wings = await analysePhoto(wingBackend, await wingCfg, decoded.wings)
      } catch (e) {
        out.wingError = e instanceof Error ? e.message : String(e)
      }
    }
    out.ms = performance.now() - t0
    self.postMessage(out)
  } catch (e) {
    self.postMessage({ id: msg.id, ok: false, error: e instanceof Error ? e.message : String(e) } satisfies PhotoResponse)
  }
}
