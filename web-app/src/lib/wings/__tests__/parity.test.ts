/**
 * The port against the models' reference chain (Node, sharp): the same verdicts on archive photos.
 * Runs when the reference results are there:
 *   node ref.mjs   (in /tmp/wingparity, list.txt: photo paths; writes ref.json)
 *   npx vitest run src/lib/wings/__tests__/parity.test.ts
 */
import { describe, expect, it } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import jpeg from 'jpeg-js'
import * as ort from 'onnxruntime-node'
import { analysePhoto, type WingBackend, type WingConfig } from '../model'

const REF = '/tmp/wingparity/ref.json'
const MODELS = new URL('../../../../public/models/', import.meta.url).pathname

describe.skipIf(!existsSync(REF))('wing models, port vs reference', () => {
  it('gives the reference verdicts', async () => {
    const opt = { intraOpNumThreads: 4 }
    const seg = await ort.InferenceSession.create(`${MODELS}wingseg_640.onnx`, opt)
    const cls = await ort.InferenceSession.create(`${MODELS}wing_side_cls.onnx`, opt)
    const cfg: WingConfig = JSON.parse(readFileSync(`${MODELS}wingside.json`, 'utf8'))
    const backend: WingBackend = {
      async segment(x, S) {
        const o = await seg.run({ images: new ort.Tensor('float32', x, [1, 3, S, S]) })
        return { det: o.output0.data as Float32Array, detDims: o.output0.dims as number[], proto: o.output1.data as Float32Array, protoDims: o.output1.dims as number[] }
      },
      async classify(x, n, size) {
        const o = await cls.run({ [cls.inputNames[0]]: new ort.Tensor('float32', x, [n, 3, size, size]) })
        return o[cls.outputNames[0]].data as Float32Array
      },
    }
    const ref: Record<string, { side: string; p: number | null; n: number; k: number; missing: string[] }> = JSON.parse(readFileSync(REF, 'utf8'))
    const diffs: number[] = []
    let sameSide = 0, sameCount = 0, sameTurn = 0
    for (const [file, want] of Object.entries(ref)) {
      const raw = jpeg.decode(readFileSync(file), { useTArray: true, maxMemoryUsageInMB: 1024 })
      const got = await analysePhoto(backend, cfg, { width: raw.width, height: raw.height, data: new Uint8ClampedArray(raw.data.buffer) })
      if (got.side === want.side) sameSide++
      if (got.wings.length === want.n) sameCount++
      if (got.turnK === want.k) sameTurn++
      if (got.p !== null && want.p !== null) diffs.push(Math.abs(got.p - want.p))
    }
    const n = Object.keys(ref).length
    diffs.sort((a, b) => a - b)
    console.log({ photos: n, sameSide, sameCount, sameTurn, medianDiff: diffs[diffs.length >> 1], maxDiff: diffs[diffs.length - 1] })
    expect(sameSide).toBeGreaterThanOrEqual(n - 1)
    expect(sameCount).toBeGreaterThanOrEqual(n - 1)
    expect(sameTurn).toBe(n)
    expect(diffs[diffs.length >> 1]).toBeLessThan(0.01)
  }, 600_000)
})
