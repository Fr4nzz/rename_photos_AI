/**
 * readPhoto with the envelope's text, on a made-up photo and models: the CAM ID reading is the same with
 * and without the text, and the text is read in one turn.
 */
import { describe, expect, it } from 'vitest'
import { makeImage } from '../image'
import { readPhoto, type Backend } from '../pipeline'

const CHARS = ['blank', ...'0123456789CAM', ' ']
const W = 640, H = 480

/** A photo whose red grows to the right and green downwards: the recogniser can tell how a crop was turned. */
function photo() {
  const img = makeImage(W, H)
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 4
    img.data[i] = (x * 255) / W
    img.data[i + 1] = (y * 255) / H
    img.data[i + 2] = 0
  }
  return img
}

/** Text regions in the envelope, as fractions of it: two lines written sideways and one across them. */
const REGIONS = [[0.2, 0.1, 0.27, 0.75], [0.4, 0.1, 0.47, 0.8], [0.55, 0.85, 0.95, 0.93]]

function backend(calls: { rec: number }): Backend {
  return {
    async run(model, data, dims) {
      if (model === 'envelope') {
        // one box in the 512 letterbox: x 56..456, y 106..406 (the photo is scaled 0.8, padded 64 above)
        const out = new Float32Array(37)
        out.set([256, 256, 400, 300, 0.9])
        return { data: out, dims: [1, 37, 1] }
      }
      if (model === 'line') {
        const [, , h, w] = dims
        const out = new Float32Array(h * w)
        for (const [x0, y0, x1, y1] of REGIONS)
          for (let y = Math.round(y0 * h); y < Math.round(y1 * h); y++) for (let x = Math.round(x0 * w); x < Math.round(x1 * w); x++) out[y * w + x] = 0.9
        return { data: out, dims: [1, 1, h, w] }
      }
      calls.rec++
      // how the crop was turned: which colour grows or falls from its left half to its right half
      const w = dims[3], plane = 48 * w
      let cw = 0
      for (let x = 0; x < w; x++) for (let y = 0; y < 48; y++) if (data[plane + y * w + x] !== 0) cw = x + 1
      const half = (c: number, a: number, b: number) => {
        let s = 0, n = 0
        for (let y = 0; y < 48; y++) for (let x = a; x < b; x++) { s += data[c * plane + y * w + x]; n++ }
        return s / n
      }
      const dR = half(2, cw >> 1, cw) - half(2, 0, cw >> 1), dG = half(1, cw >> 1, cw) - half(1, 0, cw >> 1)
      const turn = Math.abs(dR) > Math.abs(dG) ? (dR > 0 ? 0 : 180) : dG > 0 ? 90 : 270
      // the sideways lines read at 90 and the line across them at 0; anything else is a poor guess
      const tall = Math.abs(dG) > Math.abs(dR) ? turn === 90 || turn === 270 : false
      const good = turn === 90 || (turn === 0 && !tall)
      const text = good ? 'CAM012345' : 'MA'
      const steps = 40, classes = CHARS.length
      const out = new Float32Array(steps * classes)
      for (let t = 0; t < steps; t++) {
        const k = t < 2 * text.length && t % 2 === 0 ?CHARS.indexOf(text[t / 2]) : 0
        for (let c = 0; c < classes; c++) out[t * classes + c] = c === k ? (good ? 0.99 : 0.6) : (good ? 0.01 : 0.4) / (classes - 1)
      }
      return { data: out, dims: [1, steps, classes] }
    },
  }
}

describe('readPhoto with the envelope text', () => {
  it('leaves the CAM ID reading as it was and reads every line in one turn', async () => {
    const before = { rec: 0 }, after = { rec: 0 }
    const plain = await readPhoto(backend(before), photo(), CHARS)
    const withText = await readPhoto(backend(after), photo(), CHARS, undefined, { text: true, keepCrops: true })
    const { text, ...rest } = withText
    expect(rest).toEqual(plain)
    expect(plain.camid).toBe('CAM012345')
    expect(text).toBeDefined()
    expect(text!.turn).toBe(90)
    expect(text!.crossTurn).toBe(0)
    // reading order in the envelope turned 90: the line along its right edge comes first
    expect(text!.lines.map((l) => [l.text, l.turn])).toEqual([['CAM012345', 0], ['CAM012345', 90], ['CAM012345', 90]])
    // the sideways lines' readings are the CAM ID reader's; only what it did not read is read again
    expect(after.rec - before.rec).toBe(text!.extraReads)
    expect(text!.lines[0].image!.width).toBeGreaterThan(text!.lines[0].image!.height)
    // boxes in the photo, inside the envelope
    for (const l of text!.lines) for (const [x, y] of l.box) {
      expect(x).toBeGreaterThanOrEqual(plain.envelope![0])
      expect(y).toBeLessThanOrEqual(plain.envelope![3])
    }
  })
})
