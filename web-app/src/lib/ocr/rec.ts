/** CAMID recognizer (PaddleOCR PP-OCRv5 mobile, fine-tuned): input tensor and CTC decoding. */
import type { RGBAImage } from './image'
import { resize } from './image'

/** Height 48, width by aspect ratio (at least 320), BGR, scaled to [-1, 1], zero padding. */
export function recInput(img: RGBAImage): { data: Float32Array; dims: [1, 3, 48, number] } {
  const { width: w, height: h } = img
  const ratio = Math.max(320 / 48, w / h)
  const width = Math.trunc(48 * ratio)
  const rw = Math.min(width, Math.ceil((48 * w) / h))
  const r = resize(img, rw, 48)
  const data = new Float32Array(3 * 48 * width) // zeros = padding
  const plane = 48 * width
  for (let y = 0; y < 48; y++) {
    for (let x = 0; x < rw; x++) {
      const s = (y * rw + x) * 4, t = y * width + x
      for (let c = 0; c < 3; c++) data[c * plane + t] = (r.data[s + (2 - c)] / 255 - 0.5) / 0.5
    }
  }
  return { data, dims: [1, 3, 48, width] }
}

/** Greedy CTC decoding: the text and the mean probability of its characters (Paddle's score). */
export function ctcDecode(probs: Float32Array, steps: number, classes: number, chars: string[]): { text: string; conf: number } {
  let text = '', sum = 0, n = 0, prev = -1
  for (let t = 0; t < steps; t++) {
    let best = 0, bestP = -1
    const row = t * classes
    for (let k = 0; k < classes; k++) {
      const p = probs[row + k]
      if (p > bestP) { bestP = p; best = k }
    }
    if (best !== 0 && best !== prev) { text += chars[best] ?? ''; sum += bestP; n++ }
    prev = best
  }
  return { text, conf: n ? sum / n : 0 }
}
