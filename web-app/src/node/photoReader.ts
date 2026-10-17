/**
 * The CAMID reader for a server: the same models and pipeline as the browser's (src/lib/ocr), for photos
 * already on the server's disk. The Ithomiini app runs it to read the photos someone uploaded to Drive
 * without renaming them (its server/drive-photos.mjs downloads them one by one and deletes them after).
 *
 *   node photo-reader.mjs --models <dir> [--known <camids.json>] [--threads 2]
 *
 * stdin, one JSON line per photo: { id, path, preview?, crop? } (preview, crop: where to write a small
 * upright JPEG of the photo and of its envelope), or { id, op: "cut", path, box, out, side } (a part
 * of the photo as shown, `box` [x0, y0, x1, y1] in fractions of it, written upright to `out` with its
 * long side at most `side`: answered { id, out, width, height }). stdout, one JSON line each, in order:
 * { id, camid, conf, margin, rival, candidates, source, turned, rotation, orientation, width, height, envelope, ms }
 * or { id, error }. `margin`: nats by which the recogniser prefers `camid` to `rival`, the closest other
 * database ID (lib/ocr/lexicon.ts; below 1 a person should look; null without --known or for a joined ID). `rotation`: clockwise degrees that turn the photo as shown (its orientation tag applied)
 * upright; `orientation`: its EXIF orientation tag (1 without one); width/height as shown.
 * A line {"ready": true} comes first; the models are loaded with the first photo to read.
 */
import { readFileSync } from 'node:fs'
import { createInterface } from 'node:readline'
import * as ort from 'onnxruntime-node'
import sharp from 'sharp'
import type { RGBAImage } from '../lib/ocr/image'
import { chooseCamid, readPhoto, type Backend, type FullCrop, type ModelName } from '../lib/ocr/pipeline'
import { lexiconMargin } from '../lib/ocr/lexicon'
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
let loaded: Promise<Record<ModelName, ort.InferenceSession>> | null = null
const sessions = () =>
  (loaded ??= (async () => ({
    envelope: await ort.InferenceSession.create(`${models}/envelope_det.onnx`, options),
    line: await ort.InferenceSession.create(`${models}/line_det.onnx`, options),
    rec: await ort.InferenceSession.create(`${models}/camid_rec.onnx`, options),
  }))())
const backend: Backend = {
  async run(model, data, dims) {
    const s = (await sessions())[model]
    const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', data, dims) })
    const t = out[s.outputNames[0]]
    return { data: t.data as Float32Array, dims: t.dims as number[] }
  },
}

const rgba = async (image: sharp.Sharp): Promise<RGBAImage> => {
  const { data, info } = await image.ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  return { width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) }
}

interface Job { id: string; path: string; preview?: string; crop?: string; op?: 'cut'; box?: number[]; out?: string; side?: number }

/** A part of the photo as shown, upright, written as a JPEG. */
async function cut(job: Job) {
  const meta = await sharp(job.path).metadata()
  const sideways = (meta.orientation ?? 1) >= 5
  const width = (sideways ? meta.height : meta.width) ?? 0
  const height = (sideways ? meta.width : meta.height) ?? 0
  const [x0, y0, x1, y1] = (job.box ?? [0, 0, 1, 1]).map((v) => Math.min(1, Math.max(0, Number(v) || 0)))
  const left = Math.round(Math.min(x0, x1) * width)
  const top = Math.round(Math.min(y0, y1) * height)
  const w = Math.max(1, Math.min(width - left, Math.round(Math.abs(x1 - x0) * width)))
  const h = Math.max(1, Math.min(height - top, Math.round(Math.abs(y1 - y0) * height)))
  const side = Math.max(64, Math.min(2400, Number(job.side) || 1600))
  const info = await sharp(job.path).rotate().extract({ left, top, width: w, height: h }).resize(side, side, { fit: 'inside', withoutEnlargement: true }).jpeg({ quality: 85 }).toFile(job.out!)
  return { id: job.id, out: job.out, width: info.width, height: info.height, from: [w, h] }
}

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
  const lex = known && chosen ? lexiconMargin(reading.idViews ?? [], chosen.id, known) : null
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
    margin: lex ? Math.round(lex.margin * 100) / 100 : null,
    rival: lex?.rival ?? null,
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
    process.stdout.write(`${JSON.stringify(job.op === 'cut' ? await cut(job) : await read(job))}\n`)
  } catch (e) {
    process.stdout.write(`${JSON.stringify({ id: job.id, error: e instanceof Error ? e.message : String(e) })}\n`)
  }
}
