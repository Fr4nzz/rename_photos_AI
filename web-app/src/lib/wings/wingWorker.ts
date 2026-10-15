/// <reference lib="webworker" />
/**
 * Wing worker: loads the two wing models once (onnxruntime-web, WebAssembly), decodes each photo as
 * it shows (its orientation tag applied; a RAW by its camera JPEG and the RAW's tag), shrinks it to
 * 1600 px and runs the wing chain.
 */
import * as ort from 'onnxruntime-web/wasm'
import { readOrientation } from '../orientation'
import { extractLargestJpeg } from '../rawPreview'
import { decodeHeic, HEIC_EXTENSIONS } from '../heic'
import { rotate, type RGBAImage } from '../ocr/image'
import { analysePhoto, type WingBackend, type WingConfig, type WingResult } from './model'

export interface WingRequest { id: number; file: File; raw: boolean }
export type WingResponse =
  | { id: number; ok: true; result: WingResult; ms: number }
  | { id: number; ok: false; error: string }
  | { id: -1; ready: true } | { id: -1; ready: false; error: string }

let seg: ort.InferenceSession | null = null
let cls: ort.InferenceSession | null = null
let cfg: WingConfig | null = null

async function init(modelBase: string) {
  ort.env.logLevel = 'error'
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1
  const opts: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3 }
  const load = async (name: string) => ort.InferenceSession.create(await (await fetch(`${modelBase}models/${name}`)).arrayBuffer(), opts)
  cfg = await (await fetch(`${modelBase}models/wingside.json`)).json()
  ;[seg, cls] = await Promise.all([load(cfg!.segmentation.file), load(cfg!.classifier.file)])
}

const backend: WingBackend = {
  async segment(x, S) {
    const o = await seg!.run({ [seg!.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, S, S]) })
    const [det, proto] = seg!.outputNames.map((n) => o[n])
    return { det: det.data as Float32Array, detDims: det.dims as number[], proto: proto.data as Float32Array, protoDims: proto.dims as number[] }
  },
  async classify(x, n, size) {
    const o = await cls!.run({ [cls!.inputNames[0]]: new ort.Tensor('float32', x, [n, 3, size, size]) })
    return o[cls!.outputNames[0]].data as Float32Array
  },
}

const SIDE = 1600

function pixels(source: CanvasImageSource & { width: number; height: number }): RGBAImage {
  const scale = Math.min(1, SIDE / Math.max(source.width, source.height))
  const w = Math.round(source.width * scale), h = Math.round(source.height * scale)
  const canvas = new OffscreenCanvas(w, h)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(source, 0, 0, w, h)
  return { width: w, height: h, data: ctx.getImageData(0, 0, w, h).data }
}

async function decode(file: File, raw: boolean): Promise<RGBAImage> {
  if (HEIC_EXTENSIONS.has(file.name.slice(file.name.lastIndexOf('.')).toLowerCase())) {
    const full = await decodeHeic(file)
    const src = new OffscreenCanvas(full.width, full.height)
    src.getContext('2d')!.putImageData(new ImageData(full.data, full.width, full.height), 0, 0)
    return pixels(src)
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
  try {
    const img = pixels(bmp)
    return turnCcw ? rotate(img, turnCcw as 90 | 180 | 270) : img
  } finally {
    bmp.close()
  }
}

let ready: Promise<void> | null = null

self.onmessage = async (event: MessageEvent<WingRequest | { init: string }>) => {
  const msg = event.data
  if ('init' in msg) {
    ready = init(msg.init)
    try {
      await ready
      self.postMessage({ id: -1, ready: true } satisfies WingResponse)
    } catch (e) {
      self.postMessage({ id: -1, ready: false, error: String(e) } satisfies WingResponse)
    }
    return
  }
  try {
    await ready
    const t0 = performance.now()
    const result = await analysePhoto(backend, cfg!, await decode(msg.file, msg.raw))
    self.postMessage({ id: msg.id, ok: true, result, ms: performance.now() - t0 } satisfies WingResponse)
  } catch (e) {
    self.postMessage({ id: msg.id, ok: false, error: e instanceof Error ? e.message : String(e) } satisfies WingResponse)
  }
}
