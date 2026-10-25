/**
 * Every text line of each photo's envelope, cut and read the way the app reads it: one turn per envelope
 * (lib/ocr/envelopeText.ts), each line turned before reading. For cutting recogniser training crops and
 * for test-set predictions, so both see the lines exactly as the app does.
 *
 *   npm run envelope-lines -- --out <dir> [--models <dir>] [--rec <onnx>] [--chars <json>]
 *     [--known <camids.json>] [--no-crops] [--all-turns] [--threads 2] <photo | list.txt>...
 *
 * Photos: paths, or text files listing one path per line (relative paths: from the list's folder).
 * --models: the app's models (default public/models); --rec/--chars: another recogniser and its alphabet
 * (json: index 0 the CTC blank, the dictionary, then the space; the recogniser's output width).
 * Writes <out>/lines.jsonl, one line per photo, in the photos' order:
 *   { photo, camid, conf, inDatabase, source, size, envelope, turn, crossTurn, score, scores,
 *     lines: [{ crop, text, conf, logp, turn, box }], ms }
 * or { photo, error }. `turn`: counter-clockwise degrees that make the envelope's text upright;
 * each line's `turn` is the one its crop was turned by (the envelope's, or `crossTurn` for lines written
 * across the rest); `box`: the line in the photo as shown (its orientation tag applied), pixels;
 * `crop`: the PNG of the line as it was read, under <out>/crops/ (none with --no-crops); `logp`: sum of
 * its characters' log-probabilities; `score`: mean log-probability per character over the envelope's
 * lines; `camid`: the CAM ID reading (the database rule with --known). A photo already in lines.jsonl
 * is skipped, so a run can be resumed. Without an envelope found, `lines` is empty.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs'
import { basename, dirname, extname, resolve } from 'node:path'
import * as ort from 'onnxruntime-node'
import sharp from 'sharp'
import type { RGBAImage } from '../lib/ocr/image'
import { chooseCamid, readPhoto, type Backend, type ModelName } from '../lib/ocr/pipeline'

const argv = process.argv.slice(2)
const flags = new Set(['no-crops', 'all-turns'])
const opts: Record<string, string | true> = {}
const inputs: string[] = []
for (let i = 0; i < argv.length; i++) {
  const a = argv[i]
  if (!a.startsWith('--')) { inputs.push(a); continue }
  const name = a.slice(2)
  opts[name] = flags.has(name) ? true : argv[++i]
}
const str = (k: string) => (typeof opts[k] === 'string' ? (opts[k] as string) : undefined)
const out = str('out')
if (!out || !inputs.length) {
  console.error('usage: envelope-lines --out <dir> [--models <dir>] [--rec <onnx>] [--chars <json>] [--known <camids.json>] [--no-crops] [--all-turns] [--threads 2] <photo | list.txt>...')
  process.exit(2)
}
const models = str('models') ?? resolve(dirname(new URL(import.meta.url).pathname), '../../public/models')
const recPath = str('rec') ?? `${models}/envelope_reader_rec.onnx`
const chars: string[] = JSON.parse(readFileSync(str('chars') ?? `${models}/envelope_reader_rec_chars.json`, 'utf8'))
const known = str('known') ? new Set<string>(JSON.parse(readFileSync(str('known')!, 'utf8'))) : null
const threads = Number(str('threads') ?? 2)
const crops = !opts['no-crops']
const WORK_SIDE = 1600

ort.env.logLevel = 'error'
const so = { intraOpNumThreads: threads, interOpNumThreads: 1, logSeverityLevel: 3 as const }
const sessions: Record<ModelName, ort.InferenceSession> = {
  envelope: await ort.InferenceSession.create(`${models}/envelope_det.onnx`, so),
  line: await ort.InferenceSession.create(`${models}/line_det.onnx`, so),
  rec: await ort.InferenceSession.create(recPath, so),
}
let checked = false
const backend: Backend = {
  async run(model, data, dims) {
    const s = sessions[model]
    const res = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', data, dims) })
    const t = res[s.outputNames[0]]
    if (model === 'rec' && !checked) {
      checked = true
      if (t.dims[2] !== chars.length) {
        console.error(`The recogniser has ${t.dims[2]} classes and the alphabet ${chars.length}: give its alphabet with --chars.`)
        process.exit(2)
      }
    }
    return { data: t.data as Float32Array, dims: t.dims as number[] }
  },
  async inputSide(model) {
    const m = sessions[model].inputMetadata[0]
    const side = m?.isTensor ? m.shape[3] : undefined
    return typeof side === 'number' ? side : undefined
  },
}

/** The photos: arguments, and the paths listed in text files. */
const photos = inputs.flatMap((p) => {
  if (/\.(txt|lst|list)$/i.test(p) && statSync(p).isFile())
    return readFileSync(p, 'utf8').split('\n').map((l) => l.trim()).filter(Boolean).map((l) => resolve(dirname(p), l))
  return [resolve(p)]
})

mkdirSync(`${out}/crops`, { recursive: true })
const jsonl = `${out}/lines.jsonl`
const done = new Set(existsSync(jsonl)
  ? readFileSync(jsonl, 'utf8').split('\n').filter(Boolean).flatMap((l) => { try { return [JSON.parse(l).photo as string] } catch { return [] } })
  : [])
const used = new Set<string>()
/** a folder name for a photo's crops, unique within the run (its parent folder when names repeat) */
const cropDir = (photo: string) => {
  const stem = basename(photo, extname(photo))
  let name = used.has(stem) ? `${basename(dirname(photo))}__${stem}` : stem
  for (let k = 2; used.has(name); k++) name = `${stem}_${k}`
  used.add(name)
  return name
}

const rgba = async (image: sharp.Sharp): Promise<RGBAImage> => {
  const { data, info } = await image.ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  return { width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) }
}
const r3 = (x: number) => Math.round(x * 1000) / 1000

let n = 0
const started = Date.now()
for (const photo of photos) {
  const dir = cropDir(photo)
  if (done.has(photo)) continue
  const t0 = Date.now()
  try {
    // as photoReader.ts: the photo as shown (orientation tag applied), long side at most 1600
    const meta = await sharp(photo).metadata()
    const sideways = (meta.orientation ?? 1) >= 5
    const width = (sideways ? meta.height : meta.width) ?? 0
    const height = (sideways ? meta.width : meta.height) ?? 0
    const scale = Math.min(1, WORK_SIDE / Math.max(width, height))
    const work = await rgba(sharp(photo).rotate().resize(Math.round(width * scale), Math.round(height * scale), { fit: 'fill' }))
    const reading = await readPhoto(backend, work, chars, undefined, { text: true, keepCrops: crops, allTurns: !!opts['all-turns'] })
    const chosen = chooseCamid(reading, known)
    const text = reading.text
    const lines = []
    for (const [k, l] of (text?.lines ?? []).entries()) {
      let crop: string | undefined
      if (l.image) {
        crop = `crops/${dir}/${String(k).padStart(2, '0')}.png`
        mkdirSync(`${out}/crops/${dir}`, { recursive: true })
        await sharp(Buffer.from(l.image.data.buffer, l.image.data.byteOffset, l.image.data.length), { raw: { width: l.image.width, height: l.image.height, channels: 4 } })
          .removeAlpha().png().toFile(`${out}/${crop}`)
      }
      const k1 = width / work.width
      lines.push({
        ...(crop ? { crop } : {}), text: l.text, conf: r3(l.conf), logp: r3(l.logp), turn: l.turn,
        box: l.box.map(([x, y]) => [Math.round(x * k1), Math.round(y * k1)]),
        ...(l.reads ? { reads: Object.fromEntries(Object.entries(l.reads).map(([t, r]) => [t, { text: r.text, conf: r3(r.conf), logp: r3(r.logp) }])) } : {}),
      })
    }
    appendFileSync(jsonl, JSON.stringify({
      photo, camid: chosen?.id ?? reading.camid, conf: r3(chosen?.conf ?? reading.conf), inDatabase: known ? !!chosen : null,
      source: reading.source, size: [width, height],
      envelope: reading.envelope ? reading.envelope.map((v) => Math.round((v * width) / reading.size[0])) : null,
      ...(text ? { turn: text.turn, ...(text.crossTurn !== undefined ? { crossTurn: text.crossTurn } : {}), score: r3(text.score), scores: Object.fromEntries(Object.entries(text.scores).map(([t, s]) => [t, r3(s!)])) } : {}),
      lines, ms: Date.now() - t0,
    }) + '\n')
  } catch (e) {
    appendFileSync(jsonl, JSON.stringify({ photo, error: e instanceof Error ? e.message : String(e) }) + '\n')
  }
  if (++n % 50 === 0) console.error(`${n} photos, ${Math.round((Date.now() - started) / n)} ms each`)
}
console.error(`${n} photos read into ${jsonl}`)
