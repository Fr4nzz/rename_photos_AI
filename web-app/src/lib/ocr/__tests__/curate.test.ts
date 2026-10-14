/**
 * Curation scan (not part of the app): read every photo of the Sanger archive with the CAMID
 * pipeline and save the readings, resumable. Runs only with CURATE=1:
 *   CURATE=1 npx vitest run src/lib/ocr/__tests__/curate.test.ts
 */
import { describe, it } from 'vitest'
import { appendFileSync, readdirSync, readFileSync } from 'node:fs'
import jpeg from 'jpeg-js'
import * as ort from 'onnxruntime-node'
import { chooseCamid, readPhoto, type Backend, type ModelName } from '../pipeline'

const IMAGES = `${process.env.HOME}/Documents/CodeProjs/ithomiini_maps/data/specimen_classifier/sanger_images_1600`
const DIR = `${process.env.HOME}/.local/share/sanger-envelope-sam3-20260924/ocr-next/curation-20260928`
// SHARD=i/n splits the archive between n processes; each writes its own readings-i.jsonl
const [SI, SN] = (process.env.SHARD ?? '0/1').split('/').map(Number)
const OUT = `${DIR}/readings-${SI}.jsonl`
const MODELS = new URL('../../../../public/models/', import.meta.url).pathname

describe.skipIf(process.env.CURATE !== '1')('curation scan', () => {
  it('reads the archive', async () => {
    const opt = { intraOpNumThreads: Number(process.env.THREADS ?? 4) }
    const s: Record<ModelName, ort.InferenceSession> = {
      envelope: await ort.InferenceSession.create(`${MODELS}envelope_det.onnx`, opt),
      line: await ort.InferenceSession.create(`${MODELS}line_det.onnx`, opt),
      rec: await ort.InferenceSession.create(`${MODELS}camid_rec.onnx`, opt),
    }
    const backend: Backend = { async run(m, data, dims) { const x = s[m]; const o = await x.run({ [x.inputNames[0]]: new ort.Tensor('float32', data, dims) }); const t = o[x.outputNames[0]]; return { data: t.data as Float32Array, dims: t.dims as number[] } } }
    const chars: string[] = JSON.parse(readFileSync(`${MODELS}camid_rec_chars.json`, 'utf8'))
    const known = new Set<string>(JSON.parse(readFileSync(`${MODELS}../db/camids.json`, 'utf8')))
    const done = new Set<string>(readdirSync(DIR).filter((f) => /^readings.*\.jsonl$/.test(f))
      .flatMap((f) => readFileSync(`${DIR}/${f}`, 'utf8').split('\n').filter(Boolean).flatMap((l) => { try { return [JSON.parse(l).file] } catch { return [] } })))
    const files = readdirSync(IMAGES, { withFileTypes: true }).filter((d) => d.isDirectory())
      .flatMap((d) => readdirSync(`${IMAGES}/${d.name}`).filter((f) => f.endsWith('.jpg')).map((f) => `${d.name}/${f}`)).sort()
      .filter((_, i) => i % SN === SI)
    for (const file of files) {
      if (done.has(file)) continue
      try {
        const raw = jpeg.decode(readFileSync(`${IMAGES}/${file}`), { useTArray: true, maxMemoryUsageInMB: 1024 })
        const r = await readPhoto(backend, { width: raw.width, height: raw.height, data: new Uint8ClampedArray(raw.data.buffer) }, chars)
        appendFileSync(OUT, JSON.stringify({
          file, folder: file.split('/')[0], name: file.split('/')[1].split('__')[0],
          camid: chooseCamid(r, known)?.id ?? null, conf: chooseCamid(r, known)?.conf ?? 0,
          candidates: r.candidates, source: r.source, envelope: r.envelope, turned: r.turned, size: r.size,
          lines: r.lines.map((l) => ({ t: l.text, c: Math.round(l.conf * 1000) / 1000, b: l.box.map(([x, y]) => [Math.round(x), Math.round(y)]), ...(l.turn ? { turn: l.turn } : {}), ...(l.joined ? { joined: true } : {}) })),
        }) + '\n')
      } catch (e) {
        appendFileSync(OUT, JSON.stringify({ file, error: String(e) }) + '\n')
      }
    }
  }, 24 * 3600_000)
})
