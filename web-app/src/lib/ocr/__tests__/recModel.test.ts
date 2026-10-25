/** The recogniser the app ships and its alphabet belong together: one output class per entry (blank, dictionary, ♂/♀, space). */
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import * as ort from 'onnxruntime-node'
import { recInput } from '../rec'
import { makeImage } from '../image'

const MODELS = new URL('../../../../public/models/', import.meta.url).pathname

describe('envelope reader model', () => {
  it('has one output class per alphabet entry, with the sex signs and the space', async () => {
    const chars: string[] = JSON.parse(readFileSync(`${MODELS}envelope_reader_rec_chars.json`, 'utf8'))
    expect(chars.length).toBe(440)
    expect(chars.slice(-3)).toEqual(['♂', '♀', ' '])
    for (const c of '0123456789CAM') expect(chars).toContain(c)
    const s = await ort.InferenceSession.create(`${MODELS}envelope_reader_rec.onnx`, { logSeverityLevel: 3 })
    const x = recInput(makeImage(320, 48))
    const out = (await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', x.data, x.dims) }))[s.outputNames[0]]
    expect(out.dims[2]).toBe(chars.length)
  })
})
