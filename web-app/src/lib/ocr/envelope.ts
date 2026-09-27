/** Envelope detector (YOLO segmentation, ONNX export at 1024 px): letterbox input and box decoding. */
import type { RGBAImage } from './image'
import { resize } from './image'

const SIZE = 1024

export interface Letterbox {
  data: Float32Array
  dims: [1, 3, number, number]
  scale: number
  padX: number
  padY: number
}

/** Ultralytics letterbox: fit inside 1024x1024 keeping the aspect ratio, pad with grey 114, RGB 0..1. */
export function letterbox(img: RGBAImage): Letterbox {
  const scale = Math.min(SIZE / img.width, SIZE / img.height)
  const nw = Math.round(img.width * scale), nh = Math.round(img.height * scale)
  const padX = Math.round((SIZE - nw) / 2 - 0.1), padY = Math.round((SIZE - nh) / 2 - 0.1)
  const r = resize(img, nw, nh)
  const plane = SIZE * SIZE
  const data = new Float32Array(3 * plane).fill(114 / 255)
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const s = (y * nw + x) * 4, t = (y + padY) * SIZE + x + padX
      data[t] = r.data[s] / 255
      data[plane + t] = r.data[s + 1] / 255
      data[2 * plane + t] = r.data[s + 2] / 255
    }
  }
  return { data, dims: [1, 3, SIZE, SIZE], scale, padX, padY }
}

export type Box = [number, number, number, number] // x1, y1, x2, y2 in the original image

const iou = (a: Box, b: Box) => {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const inter = ix * iy
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
}

/**
 * Boxes from output0 [1, 4 + 1 + 32, N] (centre x, centre y, width, height, class score, mask
 * coefficients), confidence >= 0.25, non-maximum suppression at IoU 0.7 (Ultralytics defaults).
 */
export function decodeBoxes(out: Float32Array, n: number, lb: Letterbox, img: RGBAImage, conf = 0.25): { box: Box; score: number }[] {
  const cand: { box: Box; score: number }[] = []
  for (let i = 0; i < n; i++) {
    const score = out[4 * n + i]
    if (score < conf) continue
    const cx = out[i], cy = out[n + i], w = out[2 * n + i], h = out[3 * n + i]
    const box: Box = [
      Math.max(0, (cx - w / 2 - lb.padX) / lb.scale), Math.max(0, (cy - h / 2 - lb.padY) / lb.scale),
      Math.min(img.width, (cx + w / 2 - lb.padX) / lb.scale), Math.min(img.height, (cy + h / 2 - lb.padY) / lb.scale),
    ]
    cand.push({ box, score })
  }
  cand.sort((a, b) => b.score - a.score)
  const kept: typeof cand = []
  for (const c of cand) if (kept.every((k) => iou(k.box, c.box) < 0.7)) kept.push(c)
  return kept
}

/** The largest detected envelope, cropped with the same margin as the training data (8%, >= 12 px). */
export function envelopeCropBounds(box: Box, img: RGBAImage): [number, number, number, number] {
  const [x1, y1, x2, y2] = box
  const m = Math.max(12, Math.round(Math.max(x2 - x1, y2 - y1) * 0.08))
  return [Math.max(0, Math.floor(x1 - m)), Math.max(0, Math.floor(y1 - m)),
          Math.min(img.width, Math.ceil(x2 + m)), Math.min(img.height, Math.ceil(y2 + m))]
}
