/**
 * The wing models on one photo, without a browser: the wings found (segmentation), each wing's side
 * (dorsal or ventral), kind (forewing or hindwing) and how far it is turned (classifier), and the
 * photo's verdict. A port of the models' reference chain (ithomiini_maps, models/dorsal_ventral/node/
 * wingside.mjs, branch dorsal-ventral-wings) with its own resizing, so the browser worker and the Node
 * tests run the same code; the shape measurements are left out.
 */
import { rotate, type RGBAImage } from '../ocr/image'

export interface WingConfig {
  segmentation: { file: string; inputSize: number; conf: number; nmsIou: number; maxWings: number; letterboxFill: number }
  classifier: { file: string; inputSize: number; cropMargin: number; cropMaskLogit: number; grey: number; outputs: string[] }
  decision: { wingMin: number; lo: number; hi: number; photoLogitScale: number }
}

/** Runs the two models: segmentation on a letterboxed photo, the classifier on a batch of wing crops. */
export interface WingBackend {
  segment(x: Float32Array, S: number): Promise<{ det: Float32Array; detDims: number[]; proto: Float32Array; protoDims: number[] }>
  classify(x: Float32Array, n: number, size: number): Promise<Float32Array>
}

export type Side = 'dorsal' | 'ventral' | 'unsure' | 'mixed' | 'absent'
export type Position = 'forewing_left' | 'forewing_right' | 'hindwing_left' | 'hindwing_right'
export const POSITIONS: Position[] = ['forewing_left', 'forewing_right', 'hindwing_left', 'hindwing_right']

export interface Wing {
  box: [number, number, number, number]
  pDorsal: number
  pForewing: number
  kind: 'forewing' | 'hindwing' | 'extra'
  half: 'left' | 'right' | 'unknown'
}

export interface WingResult {
  side: Side
  /** P(dorsal) of all wings, of the forewings, of the hindwings (null: none) */
  p: number | null
  fore: { side: Side; p: number | null }
  hind: { side: Side; p: number | null }
  wings: Wing[]
  /** positions without a wing */
  missing: Position[]
  /** quarter turns the photo shows turned counter-clockwise (0 upright), and how sure */
  turnK: number
  pTurn: number
  /** the photo's width and height as checked (boxes are in its pixels) */
  size: [number, number]
  /** checked again turned upright (a quarter-turned photo): `size` and the boxes are of that turn */
  checkedUpright?: true
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
const logSoftmax = (v: number[]) => {
  const m = Math.max(...v)
  const s = Math.log(v.reduce((a, x) => a + Math.exp(x - m), 0)) + m
  return v.map((x) => x - s)
}

/**
 * RGB resize as Pillow's BILINEAR does it (a triangle filter widened when shrinking, horizontal pass
 * then vertical, rounded to bytes after each): the reference letterboxes with it.
 */
export function resizeLikePillow(img: RGBAImage, nw: number, nh: number): Uint8Array {
  const pass = (src: Uint8Array, sw: number, sh: number, horizontal: boolean, out: number): Uint8Array => {
    const inSize = horizontal ? sw : sh
    const scale = inSize / out, ss = 1 / Math.max(scale, 1), support = Math.max(scale, 1)
    const W = horizontal ? out : sw, H = horizontal ? sh : out
    const dst = new Uint8Array(W * H * 3)
    for (let o = 0; o < out; o++) {
      const center = (o + 0.5) * scale
      const xmin = Math.max(Math.trunc(center - support + 0.5), 0)
      const xmax = Math.min(Math.trunc(center + support + 0.5), inSize)
      const ws: number[] = []
      let total = 0
      for (let i = xmin; i < xmax; i++) { const w = Math.max(0, 1 - Math.abs((i - center + 0.5) * ss)); ws.push(w); total += w }
      for (let k = 0; k < ws.length; k++) ws[k] /= total || 1
      const lines = horizontal ? sh : sw
      for (let l = 0; l < lines; l++) {
        let r = 0, g = 0, b = 0
        for (let k = 0; k < ws.length; k++) {
          const s = horizontal ? (l * sw + xmin + k) * 3 : ((xmin + k) * sw + l) * 3
          r += src[s] * ws[k]; g += src[s + 1] * ws[k]; b += src[s + 2] * ws[k]
        }
        const t = horizontal ? (l * W + o) * 3 : (o * W + l) * 3
        dst[t] = Math.min(255, Math.max(0, Math.round(r)))
        dst[t + 1] = Math.min(255, Math.max(0, Math.round(g)))
        dst[t + 2] = Math.min(255, Math.max(0, Math.round(b)))
      }
    }
    return dst
  }
  const rgb = new Uint8Array(img.width * img.height * 3)
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) { rgb[i] = img.data[j]; rgb[i + 1] = img.data[j + 1]; rgb[i + 2] = img.data[j + 2] }
  let cur: Uint8Array = rgb, w = img.width, h = img.height
  if (nw !== w) { cur = pass(cur, w, h, true, nw); w = nw }
  if (nh !== h) { cur = pass(cur, w, h, false, nh); h = nh }
  return cur
}

interface Letterbox { x: Float32Array; r: number; px: number; py: number; S: number }

function letterbox(img: RGBAImage, S: number, fill: number): Letterbox {
  const r = Math.min(S / img.width, S / img.height)
  const nw = Math.round(img.width * r), nh = Math.round(img.height * r)
  const px = Math.round((S - nw) / 2 - 0.1), py = Math.round((S - nh) / 2 - 0.1)
  const small = resizeLikePillow(img, nw, nh)
  const plane = S * S
  const x = new Float32Array(3 * plane).fill(fill / 255)
  for (let y = 0; y < nh; y++) {
    for (let xx = 0; xx < nw; xx++) {
      const s = (y * nw + xx) * 3, t = (y + py) * S + xx + px
      x[t] = small[s] / 255; x[plane + t] = small[s + 1] / 255; x[2 * plane + t] = small[s + 2] / 255
    }
  }
  return { x, r, px, py, S }
}

const iou = (a: number[], b: number[]) => {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const i = ix * iy
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
}

interface Found { box: [number, number, number, number]; score: number; coef: Float32Array }
interface Protos extends Letterbox { proto: Float32Array; P: number }

async function segment(backend: WingBackend, cfg: WingConfig, img: RGBAImage): Promise<{ wings: Found[]; info: Protos }> {
  const { conf, nmsIou, maxWings, inputSize: S, letterboxFill } = cfg.segmentation
  const lb = letterbox(img, S, letterboxFill)
  const out = await backend.segment(lb.x, S)
  const det = out.det, N = out.detDims[1], D = out.detDims[2]
  const cand: number[] = []
  for (let i = 0; i < N; i++) if (det[i * D + 4] >= conf) cand.push(i)
  cand.sort((a, b) => det[b * D + 4] - det[a * D + 4])
  const kept: { i: number; lbox: number[] }[] = []
  for (const i of cand) {
    const box = [det[i * D], det[i * D + 1], det[i * D + 2], det[i * D + 3]]
    if (kept.every((k) => iou(k.lbox, box) < nmsIou)) kept.push({ i, lbox: box })
    if (kept.length === maxWings) break
  }
  const wings = kept.map(({ i, lbox }): Found => ({
    box: [Math.max(0, (lbox[0] - lb.px) / lb.r), Math.max(0, (lbox[1] - lb.py) / lb.r),
      Math.min(img.width, (lbox[2] - lb.px) / lb.r), Math.min(img.height, (lbox[3] - lb.py) / lb.r)],
    score: det[i * D + 4],
    coef: det.slice(i * D + 6, i * D + 38),
  }))
  return { wings, info: { ...lb, proto: out.proto, P: out.protoDims[2] } }
}

/** P x P mask logits of one wing (coefficients . prototypes). */
function protoMask(wing: Found, info: Protos): Float32Array {
  const { proto, P } = info, n = P * P
  const m = new Float32Array(n)
  for (let c = 0; c < 32; c++) {
    const w = wing.coef[c], off = c * n
    for (let j = 0; j < n; j++) m[j] += w * proto[off + j]
  }
  return m
}

/** Mask logits at photo points (xs: columns, ys: rows); -Infinity outside the wing's box. */
function maskLogits(wing: Found, info: Protos, pm: Float32Array, xs: number[], ys: number[]): Float32Array {
  const { P, S, r, px, py } = info, k = P / S
  const [bx1, by1, bx2, by2] = wing.box
  const out = new Float32Array(xs.length * ys.length)
  const gx = xs.map((x) => Math.min(P - 1, Math.max(0, (x * r + px) * k - 0.5)))
  for (let yi = 0; yi < ys.length; yi++) {
    const y = ys[yi], gy = Math.min(P - 1, Math.max(0, (y * r + py) * k - 0.5))
    const y0 = Math.floor(gy), y1 = Math.min(y0 + 1, P - 1), wy = gy - y0
    for (let xi = 0; xi < xs.length; xi++) {
      const x = xs[xi], t = yi * xs.length + xi
      if (x < bx1 || x > bx2 || y < by1 || y > by2) { out[t] = -Infinity; continue }
      const g = gx[xi], x0 = Math.floor(g), x1 = Math.min(x0 + 1, P - 1), wx = g - x0
      out[t] = (pm[y0 * P + x0] * (1 - wx) + pm[y0 * P + x1] * wx) * (1 - wy) + (pm[y1 * P + x0] * (1 - wx) + pm[y1 * P + x1] * wx) * wy
    }
  }
  return out
}

/** The largest 4-connected blob of a w x h grid. */
function largestBlob(mask: Uint8Array, w: number, h: number): Uint8Array {
  const lab = new Int32Array(w * h), stack = new Int32Array(w * h)
  let best = 0, bestSize = 0, cur = 0
  for (let s = 0; s < w * h; s++) {
    if (!mask[s] || lab[s]) continue
    cur++
    let size = 0, sp = 0
    stack[sp++] = s; lab[s] = cur
    while (sp) {
      const p = stack[--sp]; size++
      const x = p % w, y = (p / w) | 0
      if (x > 0 && mask[p - 1] && !lab[p - 1]) { lab[p - 1] = cur; stack[sp++] = p - 1 }
      if (x < w - 1 && mask[p + 1] && !lab[p + 1]) { lab[p + 1] = cur; stack[sp++] = p + 1 }
      if (y > 0 && mask[p - w] && !lab[p - w]) { lab[p - w] = cur; stack[sp++] = p - w }
      if (y < h - 1 && mask[p + w] && !lab[p + w]) { lab[p + w] = cur; stack[sp++] = p + w }
    }
    if (size > bestSize) { bestSize = size; best = cur }
  }
  const out = new Uint8Array(w * h)
  for (let s = 0; s < w * h; s++) out[s] = lab[s] === best && best ? 1 : 0
  return out
}

function sampleBilinear(img: RGBAImage, x: number, y: number, grey: number, rgb: number[]) {
  const { width: W, height: H, data } = img
  const fx = x - 0.5, fy = y - 0.5, x0 = Math.floor(fx), y0 = Math.floor(fy), wx = fx - x0, wy = fy - y0
  rgb[0] = rgb[1] = rgb[2] = 0
  for (let dy = 0; dy < 2; dy++) {
    const yi = y0 + dy, ww = dy ? wy : 1 - wy
    for (let dx = 0; dx < 2; dx++) {
      const xi = x0 + dx, w = ww * (dx ? wx : 1 - wx)
      if (xi < 0 || xi >= W || yi < 0 || yi >= H) { rgb[0] += grey * w; rgb[1] += grey * w; rgb[2] += grey * w; continue }
      const s = (yi * W + xi) * 4
      rgb[0] += data[s] * w; rgb[1] += data[s + 1] * w; rgb[2] += data[s + 2] * w
    }
  }
}

/** Masked square crop of one wing (CHW, 0–1) into `dst` at `offset`; returns the mask's centroid. */
function wingCrop(img: RGBAImage, wing: Found, info: Protos, pm: Float32Array, cfg: WingConfig['classifier'], dst: Float32Array, offset: number): [number, number] {
  const { inputSize: size, cropMargin: margin, cropMaskLogit: maskLogit, grey } = cfg
  const [x1, y1, x2, y2] = wing.box
  const side = margin * Math.max(x2 - x1, y2 - y1, 1)
  const ox = (x1 + x2) / 2 - side / 2, oy = (y1 + y2) / 2 - side / 2, step = side / size
  const cs = Array.from({ length: size }, (_, i) => ox + (i + 0.5) * step)
  const rs = Array.from({ length: size }, (_, i) => oy + (i + 0.5) * step)
  const logits = maskLogits(wing, info, pm, cs, rs)
  const raw = new Uint8Array(size * size)
  for (let i = 0; i < raw.length; i++) raw[i] = logits[i] > maskLogit ? 1 : 0
  const mask = largestBlob(raw, size, size)
  let sx = 0, sy = 0, n = 0
  for (let i = 0; i < mask.length; i++) if (mask[i]) { sx += i % size; sy += (i / size) | 0; n++ }
  const plane = size * size, rgb = [0, 0, 0], acc = [0, 0, 0]
  for (let yi = 0; yi < size; yi++) {
    for (let xi = 0; xi < size; xi++) {
      const t = yi * size + xi
      if (!mask[t]) { dst[offset + t] = dst[offset + plane + t] = dst[offset + 2 * plane + t] = grey / 255; continue }
      acc[0] = acc[1] = acc[2] = 0
      for (const oy2 of [-0.25, 0.25]) for (const ox2 of [-0.25, 0.25]) {
        sampleBilinear(img, cs[xi] + ox2 * step, rs[yi] + oy2 * step, grey, rgb)
        acc[0] += rgb[0]; acc[1] += rgb[1]; acc[2] += rgb[2]
      }
      // the reference rounds the 2x2 average to a byte before the classifier
      dst[offset + t] = Math.min(255, Math.floor(acc[0] / 4 + 0.5)) / 255
      dst[offset + plane + t] = Math.min(255, Math.floor(acc[1] / 4 + 0.5)) / 255
      dst[offset + 2 * plane + t] = Math.min(255, Math.floor(acc[2] / 4 + 0.5)) / 255
    }
  }
  return n ? [ox + (sx / n + 0.5) * step, oy + (sy / n + 0.5) * step] : [(x1 + x2) / 2, (y1 + y2) / 2]
}

interface Detection {
  box: [number, number, number, number]
  pDorsal: number; pForewing: number; pWing: number; logitDorsal: number; rot: number[]
  cx: number; cy: number
  // filled by aggregate
  ux?: number; ubox?: number[]; kind?: Wing['kind']; half?: Wing['half']
}

/** A point of a photo turned k quarter turns counter-clockwise, in the upright frame. */
const toUpright = (x: number, y: number, k: number, W: number, H: number): [number, number] =>
  ([[x, y], [H - y, x], [W - x, H - y], [y, W - x]] as [number, number][])[k]

/** Forewing/hindwing from the classifier; left/right in the upright frame (the reference's rules). */
function assignPositions(wings: Detection[]) {
  for (const w of wings) w.kind = w.pForewing >= 0.5 ? 'forewing' : 'hindwing'
  const fw = wings.filter((w) => w.kind === 'forewing').sort((a, b) => a.ux! - b.ux!)
  const hw = wings.filter((w) => w.kind === 'hindwing').sort((a, b) => a.ux! - b.ux!)
  for (const g of [fw, hw]) for (const w of g.slice(1, -1)) { w.kind = 'extra'; w.half = 'unknown' }
  let mid: number | null = null
  if (fw.length >= 2) {
    fw[0].half = 'left'; fw[fw.length - 1].half = 'right'
    mid = (fw[0].ubox![2] + fw[fw.length - 1].ubox![0]) / 2
  } else if (fw.length === 1) {
    const f = fw[0]
    f.half = f.ux! < (f.ubox![0] + f.ubox![2]) / 2 ? 'left' : 'right'
    mid = f.half === 'left' ? f.ubox![2] : f.ubox![0]
  }
  if (hw.length >= 2) { hw[0].half = 'left'; hw[hw.length - 1].half = 'right' }
  else if (hw.length === 1) hw[0].half = mid === null ? 'unknown' : hw[0].ux! < mid ? 'left' : 'right'
}

function aggregate(detections: Detection[], d: WingConfig['decision'], W: number, H: number): WingResult {
  const wings = detections.filter((w) => w.pWing >= d.wingMin)
  const rs = [0, 0, 0, 0]
  for (const w of wings) logSoftmax(w.rot).forEach((v, k) => { rs[k] += v })
  const k = wings.length ? rs.indexOf(Math.max(...rs)) : 0
  const pr = wings.length ? logSoftmax(rs).map(Math.exp) : [1, 0, 0, 0]
  for (const w of wings) {
    w.ux = toUpright(w.cx, w.cy, k, W, H)[0]
    const c = [[w.box[0], w.box[1]], [w.box[2], w.box[1]], [w.box[0], w.box[3]], [w.box[2], w.box[3]]].map(([x, y]) => toUpright(x, y, k, W, H))
    w.ubox = [Math.min(...c.map((p) => p[0])), Math.min(...c.map((p) => p[1])), Math.max(...c.map((p) => p[0])), Math.max(...c.map((p) => p[1]))]
  }
  assignPositions(wings)
  const verdictOf = (ws: Detection[]): { side: Side; p: number | null } => {
    if (!ws.length) return { side: 'absent', p: null }
    const p = sigmoid((d.photoLogitScale ?? 1) * ws.reduce((s, w) => s + w.logitDorsal, 0) / ws.length)
    return { side: p >= d.hi ? 'dorsal' : p <= d.lo ? 'ventral' : 'unsure', p }
  }
  const fore = verdictOf(wings.filter((w) => w.kind === 'forewing'))
  const hind = verdictOf(wings.filter((w) => w.kind === 'hindwing'))
  const all = verdictOf(wings)
  const decided = (s: Side) => s === 'dorsal' || s === 'ventral'
  const side: Side = decided(fore.side) && decided(hind.side) && fore.side !== hind.side ? 'mixed' : all.side
  const placed = new Set(wings.filter((w) => w.kind !== 'extra' && w.half !== 'unknown').map((w) => `${w.kind}_${w.half}`))
  return {
    side, p: all.p, fore, hind,
    wings: wings.map((w) => ({
      box: w.box.map((v) => Math.round(v)) as Wing['box'], pDorsal: w.pDorsal, pForewing: w.pForewing, kind: w.kind!, half: w.half ?? 'unknown',
    })),
    missing: POSITIONS.filter((p) => !placed.has(p)),
    turnK: k, pTurn: pr[k], size: [W, H],
  }
}

async function analyseOnce(backend: WingBackend, cfg: WingConfig, img: RGBAImage): Promise<WingResult> {
  const { wings, info } = await segment(backend, cfg, img)
  const c = cfg.classifier
  const size = c.inputSize, per = 3 * size * size, nOut = c.outputs.length
  const pms = wings.map((w) => protoMask(w, info))
  const x = new Float32Array(Math.max(1, wings.length) * per)
  const cents = wings.map((w, k) => wingCrop(img, w, info, pms[k], c, x, k * per))
  const logits = wings.length ? await backend.classify(x, wings.length, size) : new Float32Array(0)
  const detections: Detection[] = wings.map((w, k) => ({
    box: w.box, pDorsal: sigmoid(logits[k * nOut]), pForewing: sigmoid(logits[k * nOut + 1]), pWing: sigmoid(logits[k * nOut + 2]),
    logitDorsal: logits[k * nOut], rot: Array.from(logits.slice(k * nOut + 3, k * nOut + 7)),
    cx: cents[k][0], cy: cents[k][1],
  }))
  return aggregate(detections, cfg.decision, img.width, img.height)
}

/**
 * The whole chain on a photo (RGBA, its orientation tag applied, longest side 1600 px at most). The
 * segmentation was trained on upright photos and misses wings on quarter-turned ones, so a photo
 * found quarter-turned is turned upright and checked again (keeping the turn found first).
 */
export async function analysePhoto(backend: WingBackend, cfg: WingConfig, img: RGBAImage): Promise<WingResult> {
  const a = await analyseOnce(backend, cfg, img)
  if (a.turnK % 2 === 1 && a.wings.length >= 2) {
    // turnK counter-clockwise quarter turns are undone by as many clockwise ones
    const b = await analyseOnce(backend, cfg, rotate(img, a.turnK === 1 ? 270 : 90))
    return { ...b, turnK: a.turnK, pTurn: a.pTurn, checkedUpright: true }
  }
  return a
}
