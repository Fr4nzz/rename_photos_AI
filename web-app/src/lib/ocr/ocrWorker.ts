/// <reference lib="webworker" />
/**
 * OCR worker: loads the three ONNX models once (onnxruntime-web, WebAssembly), decodes each photo
 * (JPEG/PNG via createImageBitmap with EXIF orientation; RAW via its largest embedded JPEG and the
 * RAW's orientation tag), shrinks it to 1600 px and runs the CAMID pipeline.
 */
import * as ort from 'onnxruntime-web/wasm'
import { readOrientation } from '../orientation'
import { extractLargestJpeg } from '../rawPreview'
import { decodeHeic, HEIC_EXTENSIONS } from '../heic'
import { rotate, type RGBAImage } from './image'
import { readPhoto, type Backend, type FullCrop, type ModelName, type PhotoReading } from './pipeline'

export interface OcrRequest { id: number; file: File; raw: boolean }
export type OcrResponse =
  | { id: number; ok: true; reading: PhotoReading; ms: number }
  | { id: number; ok: false; error: string }
  | { id: -1; ready: true } | { id: -1; ready: false; error: string }

let sessions: Record<ModelName, ort.InferenceSession> | null = null
let chars: string[] = []

async function init(modelBase: string) {
  ort.env.logLevel = 'error' // the converted models log harmless 'unused initializer' warnings
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1
  const opts: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3 }
  const load = async (name: string) => ort.InferenceSession.create(await (await fetch(`${modelBase}models/${name}`)).arrayBuffer(), opts)
  const [envelope, line, rec, charList] = await Promise.all([
    load('envelope_det.onnx'), load('line_det.onnx'), load('camid_rec.onnx'),
    fetch(`${modelBase}models/camid_rec_chars.json`).then((r) => r.json() as Promise<string[]>),
  ])
  sessions = { envelope, line, rec }
  chars = charList
}

const backend: Backend = {
  async run(model, data, dims) {
    const s = sessions![model]
    const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', data, dims) })
    const t = out[s.outputNames[0]]
    return { data: t.data as Float32Array, dims: t.dims as number[] }
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

interface Decoded {
  img: RGBAImage
  /** full-resolution pixels of a region of `img` (text lines are read from these) */
  full?: FullCrop
  release: () => void
}

/** Region [x0, y0, x1, y1] of a working image `w` px wide, cut from a full-size source. */
function cropper(source: CanvasImageSource & { width: number; height: number }, w: number): FullCrop {
  return async ([x0, y0, x1, y1]) => {
    const scale = source.width / w
    const sx = Math.round(x0 * scale), sy = Math.round(y0 * scale)
    const sw = Math.max(1, Math.round((x1 - x0) * scale)), sh = Math.max(1, Math.round((y1 - y0) * scale))
    const c = new OffscreenCanvas(sw, sh)
    const ctx = c.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(source, sx, sy, sw, sh, 0, 0, sw, sh)
    return { image: { width: sw, height: sh, data: ctx.getImageData(0, 0, sw, sh).data }, scale }
  }
}

async function decode(file: File, raw: boolean): Promise<Decoded> {
  if (HEIC_EXTENSIONS.has(file.name.slice(file.name.lastIndexOf('.')).toLowerCase())) {
    const full = await decodeHeic(file)
    const scale = Math.min(1, WORK_SIDE / Math.max(full.width, full.height))
    const w = Math.round(full.width * scale), h = Math.round(full.height * scale)
    const src = new OffscreenCanvas(full.width, full.height)
    src.getContext('2d')!.putImageData(new ImageData(full.data, full.width, full.height), 0, 0)
    const dst = new OffscreenCanvas(w, h)
    const ctx = dst.getContext('2d', { willReadFrequently: true })!
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(src, 0, 0, w, h)
    return { img: await asRendition(dst), full: cropper(src, w), release: () => {} }
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
  const probe = await createImageBitmap(source, { imageOrientation: 'from-image' })
  const scale = Math.min(1, WORK_SIDE / Math.max(probe.width, probe.height))
  const w = Math.round(probe.width * scale), h = Math.round(probe.height * scale)
  const canvas = new OffscreenCanvas(w, h)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(probe, 0, 0, w, h)
  const img = await asRendition(canvas)
  if (turnCcw) {
    // RAW previews turned by the RAW's tag: read at working resolution only
    probe.close()
    return { img: rotate(img, turnCcw as 90 | 180 | 270), release: () => {} }
  }
  return { img, full: cropper(probe, w), release: () => probe.close() }
}

let ready: Promise<void> | null = null

self.onmessage = async (event: MessageEvent<OcrRequest | { init: string }>) => {
  const msg = event.data
  if ('init' in msg) {
    ready = init(msg.init)
    try {
      await ready
      self.postMessage({ id: -1, ready: true } satisfies OcrResponse)
    } catch (e) {
      self.postMessage({ id: -1, ready: false, error: String(e) } satisfies OcrResponse)
    }
    return
  }
  try {
    await ready
    const t0 = performance.now()
    const decoded = await decode(msg.file, msg.raw)
    let reading: PhotoReading
    try {
      reading = await readPhoto(backend, decoded.img, chars, decoded.full)
    } finally {
      decoded.release()
    }
    self.postMessage({ id: msg.id, ok: true, reading, ms: performance.now() - t0 } satisfies OcrResponse)
  } catch (e) {
    self.postMessage({ id: msg.id, ok: false, error: e instanceof Error ? e.message : String(e) } satisfies OcrResponse)
  }
}
