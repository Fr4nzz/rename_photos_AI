/**
 * PaddleOCR DB text-line detector: pre-processing and box extraction (port of onnx_check.py's
 * det_input / det_boxes, which match Paddle's own boxes on all 898 test lines).
 * OpenCV's findContours + minAreaRect are replaced by connected components, a convex hull and
 * rotating calipers, which give the same rectangles for these blob-shaped text regions.
 */
import type { RGBAImage } from './image'
import { resize } from './image'

export type Pt = [number, number]

export interface DetInput {
  data: Float32Array
  dims: [1, 3, number, number]
  /** original height/width and resized height/width */
  shape: [number, number, number, number]
}

const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]

/** Resize to multiples of 32 (upscaling if the short side is below `limit`), normalise, CHW, BGR. */
export function detInput(img: RGBAImage, limit = 64, maxSide = 4000): DetInput {
  const h = img.height, w = img.width
  const ratio = Math.min(h, w) < limit ? limit / Math.min(h, w) : 1
  let rh = Math.trunc(h * ratio), rw = Math.trunc(w * ratio)
  if (Math.max(rh, rw) > maxSide) {
    const r = maxSide / Math.max(rh, rw)
    rh = Math.trunc(rh * r); rw = Math.trunc(rw * r)
  }
  rh = Math.max(Math.round(rh / 32) * 32, 32)
  rw = Math.max(Math.round(rw / 32) * 32, 32)
  const r = resize(img, rw, rh)
  const data = new Float32Array(3 * rh * rw)
  const plane = rh * rw
  for (let i = 0; i < plane; i++) {
    // channel order B, G, R (the models were trained on OpenCV BGR images)
    for (let c = 0; c < 3; c++) {
      data[c * plane + i] = (r.data[i * 4 + (2 - c)] / 255 - MEAN[c]) / STD[c]
    }
  }
  return { data, dims: [1, 3, rh, rw], shape: [h, w, rh, rw] }
}

function convexHull(points: Pt[]): Pt[] {
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1])
  if (pts.length < 3) return pts
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const lower: Pt[] = [], upper: Pt[] = []
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop()
    lower.push(p)
  }
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop()
    upper.push(p)
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1))
}

/** Minimum-area rectangle of a convex polygon: centre, side lengths and edge direction. */
function minAreaRect(hull: Pt[]): { corners: Pt[]; short: number; area: number; perimeter: number } {
  if (hull.length === 1) hull = [hull[0], [hull[0][0] + 1e-6, hull[0][1]]]
  let best = { area: Infinity, corners: [] as Pt[], w: 0, h: 0 }
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length]
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1e-9
    const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity
    for (const p of hull) {
      const u = p[0] * ux + p[1] * uy, v = -p[0] * uy + p[1] * ux
      minU = Math.min(minU, u); maxU = Math.max(maxU, u); minV = Math.min(minV, v); maxV = Math.max(maxV, v)
    }
    const area = (maxU - minU) * (maxV - minV)
    if (area < best.area) {
      const corner = (u: number, v: number): Pt => [u * ux - v * uy, u * uy + v * ux]
      best = { area, corners: [corner(minU, minV), corner(maxU, minV), corner(maxU, maxV), corner(minU, maxV)], w: maxU - minU, h: maxV - minV }
    }
  }
  return { corners: best.corners, short: Math.min(best.w, best.h), area: best.area, perimeter: 2 * (best.w + best.h) }
}

/** Order 4 corners as top-left, top-right, bottom-right, bottom-left (as get_mini_boxes). */
function orderCorners(c: Pt[]): Pt[] {
  const byX = c.slice().sort((a, b) => a[0] - b[0])
  const [l0, l1] = byX.slice(0, 2).sort((a, b) => a[1] - b[1])
  const [r0, r1] = byX.slice(2).sort((a, b) => a[1] - b[1])
  return [l0, r0, r1, l1]
}

/** Mean of `pred` inside the polygon (pixel centres), over its integer bounding box. */
function boxScore(pred: Float32Array, W: number, H: number, box: Pt[]): number {
  const xs = box.map((p) => p[0]), ys = box.map((p) => p[1])
  const x0 = Math.max(0, Math.floor(Math.min(...xs))), x1 = Math.min(W - 1, Math.ceil(Math.max(...xs)))
  const y0 = Math.max(0, Math.floor(Math.min(...ys))), y1 = Math.min(H - 1, Math.ceil(Math.max(...ys)))
  let sum = 0, n = 0
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      let inside = false
      for (let i = 0, j = 3; i < 4; j = i++) {
        const [xi, yi] = box[i], [xj, yj] = box[j]
        if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
      }
      if (inside) { sum += pred[y * W + x]; n++ }
    }
  }
  return n ? sum / n : 0
}

/** Text-line quadrilaterals in original-image coordinates, top to bottom. */
export function detBoxes(pred: Float32Array, predW: number, predH: number, shape: DetInput['shape'],
                         thresh = 0.3, boxThresh = 0.6, unclipRatio = 1.5): Pt[][] {
  const [h, w, rh, rw] = shape
  const label = new Int32Array(predW * predH).fill(-1)
  const boxes: Pt[][] = []
  const stack: number[] = []
  let components = 0
  for (let start = 0; start < pred.length && components < 1000; start++) {
    if (pred[start] <= thresh || label[start] !== -1) continue
    components++
    // flood-fill one 8-connected component, collecting boundary pixel centres
    const boundary: Pt[] = []
    stack.push(start); label[start] = components
    while (stack.length) {
      const i = stack.pop()!
      const x = i % predW, y = (i - x) / predW
      let edge = false
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue
          const nx = x + dx, ny = y + dy
          if (nx < 0 || ny < 0 || nx >= predW || ny >= predH) { edge = true; continue }
          const j = ny * predW + nx
          if (pred[j] <= thresh) { edge = true; continue }
          if (label[j] === -1) { label[j] = components; stack.push(j) }
        }
      }
      if (edge) boundary.push([x, y])
    }
    const rect = minAreaRect(convexHull(boundary))
    if (rect.short < 3) continue
    const box = orderCorners(rect.corners)
    if (boxScore(pred, predW, predH, box) < boxThresh) continue
    // unclip: offset the rectangle by d = area * ratio / perimeter on every side
    const d = (rect.area * unclipRatio) / rect.perimeter
    const cx = box.reduce((s, p) => s + p[0], 0) / 4, cy = box.reduce((s, p) => s + p[1], 0) / 4
    const e1: Pt = [box[1][0] - box[0][0], box[1][1] - box[0][1]], e2: Pt = [box[3][0] - box[0][0], box[3][1] - box[0][1]]
    const l1 = Math.hypot(...e1) || 1, l2 = Math.hypot(...e2) || 1
    const u1: Pt = [e1[0] / l1, e1[1] / l1], u2: Pt = [e2[0] / l2, e2[1] / l2]
    const half1 = l1 / 2 + d, half2 = l2 / 2 + d
    if (Math.min(half1, half2) * 2 < 5) continue
    const grown: Pt[] = [
      [cx - u1[0] * half1 - u2[0] * half2, cy - u1[1] * half1 - u2[1] * half2],
      [cx + u1[0] * half1 - u2[0] * half2, cy + u1[1] * half1 - u2[1] * half2],
      [cx + u1[0] * half1 + u2[0] * half2, cy + u1[1] * half1 + u2[1] * half2],
      [cx - u1[0] * half1 + u2[0] * half2, cy - u1[1] * half1 + u2[1] * half2],
    ]
    boxes.push(orderCorners(grown).map(([x, y]) => [
      Math.min(w, Math.max(0, Math.round((x / rw) * w))),
      Math.min(h, Math.max(0, Math.round((y / rh) * h))),
    ]))
  }
  return boxes.sort((a, b) => Math.min(...a.map((p) => p[1])) - Math.min(...b.map((p) => p[1]))
    || Math.min(...a.map((p) => p[0])) - Math.min(...b.map((p) => p[0])))
}
