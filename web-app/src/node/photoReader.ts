/**
 * The CAMID reader for a server: the same models and pipeline as the browser's (src/lib/ocr), for photos
 * already on the server's disk. The Ithomiini app runs it to read the photos someone uploaded to Drive
 * without renaming them (its server/drive-photos.mjs downloads them one by one and deletes them after).
 *
 *   node photo-reader.mjs --models <dir> [--known <camids.json>] [--threads 2]
 *
 * stdin, one JSON line per photo: { id, path, preview?, crop? } (preview, crop: where to write a small
 * upright JPEG of the photo and of its envelope). stdout, one JSON line each, in order:
 * { id, camid, conf, candidates, source, turned, rotation, orientation, width, height, envelope, ms } or
 * { id, error }. `rotation`: clockwise degrees that turn the photo as shown (its orientation tag applied)
 * upright; `orientation`: its EXIF orientation tag (1 without one); width/height as shown.
 * A line {"ready": true} comes first, once the models are loaded.
 */
import { readFileSync } from 'node:fs'
import { createInterface } from 'node:readline'
import * as ort from 'onnxruntime-node'
import sharp from 'sharp'
import type { RGBAImage } from '../lib/ocr/image'
import { chooseCamid, readPhoto, type Backend, type FullCrop, type ModelName } from '../lib/ocr/pipeline'
import { uprightRotation } from '../lib/rotationPlan'

const arg = (name: string, fallback?: string): string | undefined => {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 ? process.argv[i + 1] : fallback
}
const models = arg('models')
if (!models) throw new Error('--models <dir> is required')
const threads = Number(arg('threads', '2'))
const known = arg('known') ? new Set<string>(JSON.parse(readFileSync(arg('known')!, 'utf8'))) : null
const chars: string[] = JSON.parse(readFileSync(`${models}/camid_rec_chars.json`, 'utf8'))
/** Long side of the working image (as the browser's worker). */
const WORK_SIDE = 1600
const PREVIEW_SIDE = 1024
const CROP_SIDE = 640

// Errors only: the models' own export notes are not news.
ort.env.logLevel = 'error'
const options = { intraOpNumThreads: threads, interOpNumThreads: 1, logSeverityLevel: 3 as const }
const sessions: Record<ModelName, ort.InferenceSession> = {
  envelope: await ort.InferenceSession.create(`${models}/envelope_det.onnx`, options),
  line: await ort.InferenceSession.create(`${models}/line_det.onnx`, options),
  rec: await ort.InferenceSession.create(`${models}/camid_rec.onnx`, options),
}
const backend: Backend = {
  async run(model, data, dims) {
    const s = sessions[model]
    const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', data, dims) })
    const t = out[s.outputNames[0]]
    return { data: t.data as Float32Array, dims: t.dims as number[] }
  },
}

const rgba = async (image: sharp.Sharp): Promise<RGBAImage> => {
  const { data, info } = await image.ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  return { width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) }
}

interface Job { id: string; path: string; preview?: string; crop?: string }

async function read(job: Job) {
  const started = Date.now()
  const meta = await sharp(job.path).metadata()
  const orientation = meta.orientation ?? 1
  // As shown: width and height swap for the tags that turn the photo a quarter.
  const sideways = orientation >= 5
  const width = (sideways ? meta.height : meta.width) ?? 0
  const height = (sideways ? meta.width : meta.height) ?? 0
  const scale = Math.min(1, WORK_SIDE / Math.max(width, height))
  const work = await rgba(sharp(job.path).rotate().resize(Math.round(width * scale), Math.round(height * scale), { fit: 'fill' }))
  // Full-resolution pixels of a region of the working image, for the envelopes read again closer.
  const full: FullCrop = async ([x0, y0, x1, y1]) => {
    const k = width / work.width
    const left = Math.max(0, Math.round(x0 * k))
    const top = Math.max(0, Math.round(y0 * k))
    const w = Math.min(width - left, Math.max(1, Math.round((x1 - x0) * k)))
    const h = Math.min(height - top, Math.max(1, Math.round((y1 - y0) * k)))
    return { image: await rgba(sharp(job.path).rotate().extract({ left, top, width: w, height: h })), scale: k }
  }
  const reading = await readPhoto(backend, work, chars, full)
  const chosen = chooseCamid(reading, known)
  const camid = chosen?.id ?? reading.camid
  if (job.preview)
    await sharp(job.path).rotate().resize(PREVIEW_SIDE, PREVIEW_SIDE, { fit: 'inside' }).jpeg({ quality: 78 }).toFile(job.preview)
  if (job.crop && reading.envelope) {
    const k = width / reading.size[0]
    const [x0, y0, x1, y1] = reading.envelope.map((v) => Math.round(v * k))
    const left = Math.max(0, x0)
    const top = Math.max(0, y0)
    await sharp(job.path)
      .rotate()
      .extract({ left, top, width: Math.max(1, Math.min(width, x1) - left), height: Math.max(1, Math.min(height, y1) - top) })
      .resize(CROP_SIDE, CROP_SIDE, { fit: 'inside' })
      .jpeg({ quality: 82 })
      .toFile(job.crop)
  }
  return {
    id: job.id,
    camid,
    conf: chosen?.conf ?? reading.conf,
    candidates: reading.candidates.slice(0, 4),
    inWorkbook: known ? !!chosen : null,
    source: reading.source,
    turned: reading.turned,
    rotation: uprightRotation(reading, camid),
    orientation,
    width,
    height,
    envelope: reading.envelope ? reading.envelope.map((v) => Math.round((v * width) / reading.size[0])) : null,
    ms: Date.now() - started,
  }
}

process.stdout.write(`${JSON.stringify({ ready: true })}\n`)
const lines = createInterface({ input: process.stdin })
for await (const line of lines) {
  if (!line.trim()) continue
  let job: Job
  try {
    job = JSON.parse(line)
  } catch {
    continue
  }
  try {
    process.stdout.write(`${JSON.stringify(await read(job))}\n`)
  } catch (e) {
    process.stdout.write(`${JSON.stringify({ id: job.id, error: e instanceof Error ? e.message : String(e) })}\n`)
  }
}
