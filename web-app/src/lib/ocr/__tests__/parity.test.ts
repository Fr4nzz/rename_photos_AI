/**
 * Parity of the TypeScript pipeline with the evaluated Python pipeline, on the sealed round-2 test
 * half of the fresh collections (Panama, Brasil, Peru, Guyana, insectary). Runs only with
 * PARITY=1 (needs the local evaluation data and onnxruntime-node):
 *   PARITY=1 PARITY_LIMIT=651 npx vitest run src/lib/ocr/__tests__/parity.test.ts
 */
import { describe, expect, it } from 'vitest'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import jpeg from 'jpeg-js'
import * as ort from 'onnxruntime-node'
import { chooseCamid, readPhoto, type Backend, type ModelName } from '../pipeline'

const FRESH = `${process.env.HOME}/.local/share/sanger-envelope-sam3-20260924/ocr-next/fresh-test`
const MODELS = new URL('../../../../public/models/', import.meta.url).pathname
const RUN = process.env.PARITY === '1' && existsSync(FRESH)

describe.skipIf(!RUN)('parity with the Python pipeline', () => {
  it('reads the sealed test half like the Python pipeline', async () => {
    const sessions: Record<ModelName, ort.InferenceSession> = {
      envelope: await ort.InferenceSession.create(`${MODELS}envelope_det.onnx`),
      line: await ort.InferenceSession.create(`${MODELS}line_det.onnx`),
      rec: await ort.InferenceSession.create(`${MODELS}camid_rec.onnx`),
    }
    const backend: Backend = {
      async run(model, data, dims) {
        const s = sessions[model]
        const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', data, dims) })
        const t = out[s.outputNames[0]]
        return { data: t.data as Float32Array, dims: t.dims as number[] }
      },
    }
    const chars: string[] = JSON.parse(readFileSync(`${MODELS}camid_rec_chars.json`, 'utf8'))
    const known = new Set<string>(JSON.parse(readFileSync(`${MODELS}../db/camids.json`, 'utf8')))
    const split: Record<string, string> = JSON.parse(readFileSync(`${FRESH}/round2-split.json`, 'utf8'))
    const checks: Record<string, { envelope_camid?: string; no_visible_id?: boolean }> =
      JSON.parse(readFileSync(`${FRESH}/visual-checks.json`, 'utf8'))
    // Python reference: p3 on the v1 envelope lines + whole-photo fallback, database rule
    const pyReads = new Map<string, { lines: { p3: [string, number] }[] }>(
      readFileSync(`${FRESH}/reads_p3.jsonl`, 'utf8').trim().split('\n').map((l) => { const r = JSON.parse(l); return [r.camid, r] }))
    const pyFallback: Record<string, string | null> = JSON.parse(readFileSync(`${FRESH}/fallback_p3.json`, 'utf8'))
    const python = (c: string) => {
      const r = pyReads.get(c)
      if (!r) return pyFallback[c] && known.has(pyFallback[c]!) ? pyFallback[c] : null
      const found = r.lines.map((l) => ({ id: l.p3[0].replace(/\s+/g, '').toUpperCase(), conf: l.p3[1] }))
        .filter((f) => /^CAM[0-9]{6}$/.test(f.id) && known.has(f.id)).sort((a, b) => b.conf - a.conf)
      return found[0]?.id ?? null
    }

    const ids = Object.keys(split).filter((c) => split[c] === 'test' && !checks[c]?.no_visible_id).sort()
      .slice(0, Number(process.env.PARITY_LIMIT ?? 60))
    const tally = { ts: { correct: 0, wrong: 0, none: 0 }, py: { correct: 0, wrong: 0, none: 0 }, agree: 0 }
    const rows: unknown[] = []
    const t0 = Date.now()
    for (const c of ids) {
      const raw = jpeg.decode(readFileSync(`${FRESH}/images/${c}.jpg`), { useTArray: true, maxMemoryUsageInMB: 1024 })
      const img = { width: raw.width, height: raw.height, data: new Uint8ClampedArray(raw.data.buffer) }
      const reading = await readPhoto(backend, img, chars)
      const ts = chooseCamid(reading, known)?.id ?? null
      const py = python(c)
      const truth = checks[c]?.envelope_camid ?? c
      const k = (p: string | null) => (!p ? 'none' : p === truth ? 'correct' : 'wrong') as 'none' | 'correct' | 'wrong'
      tally.ts[k(ts)]++
      tally.py[k(py)]++
      if (ts === py) tally.agree++
      rows.push({ camid: c, truth, ts, py, source: reading.source })
    }
    const perPhoto = (Date.now() - t0) / ids.length
    writeFileSync('/tmp/parity-result.json', JSON.stringify({ tally, perPhotoMs: perPhoto, rows }, null, 1))
    console.log(JSON.stringify(tally), `${Math.round(perPhoto)} ms/photo`)
    expect(tally.agree / ids.length).toBeGreaterThan(0.9)
    expect(tally.ts.wrong).toBeLessThanOrEqual(tally.py.wrong + 2)
  }, 3_600_000)
})
