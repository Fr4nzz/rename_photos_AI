/** Minimal RGBA image operations shared by the browser and the Node test harness. */

export interface RGBAImage {
  width: number
  height: number
  /** RGBA, row-major, 4 bytes per pixel */
  data: Uint8ClampedArray
}

export function makeImage(width: number, height: number, fill = 0): RGBAImage {
  const data = new Uint8ClampedArray(width * height * 4).fill(fill)
  if (fill !== 255) for (let i = 3; i < data.length; i += 4) data[i] = 255
  return { width, height, data }
}

export function crop(img: RGBAImage, x0: number, y0: number, x1: number, y1: number): RGBAImage {
  const w = x1 - x0, h = y1 - y0
  const out = makeImage(w, h)
  for (let y = 0; y < h; y++) {
    const src = ((y0 + y) * img.width + x0) * 4
    out.data.set(img.data.subarray(src, src + w * 4), y * w * 4)
  }
  return out
}

/** Bilinear resize with OpenCV INTER_LINEAR pixel-centre alignment. */
export function resize(img: RGBAImage, w: number, h: number): RGBAImage {
  const out = makeImage(w, h)
  const sx = img.width / w, sy = img.height / h
  const W = img.width, H = img.height, d = img.data, o = out.data
  for (let y = 0; y < h; y++) {
    let fy = (y + 0.5) * sy - 0.5
    if (fy < 0) fy = 0
    let y0 = Math.floor(fy)
    if (y0 > H - 1) y0 = H - 1
    const y1 = Math.min(y0 + 1, H - 1)
    const wy = fy - y0
    for (let x = 0; x < w; x++) {
      let fx = (x + 0.5) * sx - 0.5
      if (fx < 0) fx = 0
      let x0 = Math.floor(fx)
      if (x0 > W - 1) x0 = W - 1
      const x1 = Math.min(x0 + 1, W - 1)
      const wx = fx - x0
      const a = (y0 * W + x0) * 4, b = (y0 * W + x1) * 4, c = (y1 * W + x0) * 4, e = (y1 * W + x1) * 4
      const t = (y * w + x) * 4
      for (let k = 0; k < 3; k++) {
        o[t + k] = (d[a + k] * (1 - wx) + d[b + k] * wx) * (1 - wy) + (d[c + k] * (1 - wx) + d[e + k] * wx) * wy
      }
    }
  }
  return out
}

/** Rotate by 90 (counter-clockwise), 180 or 270 degrees. */
export function rotate(img: RGBAImage, ccw: 90 | 180 | 270): RGBAImage {
  const { width: W, height: H, data: d } = img
  const out = ccw === 180 ? makeImage(W, H) : makeImage(H, W)
  const o = out.data
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let nx: number, ny: number
      if (ccw === 90) { nx = y; ny = W - 1 - x } else if (ccw === 270) { nx = H - 1 - y; ny = x } else { nx = W - 1 - x; ny = H - 1 - y }
      const s = (y * W + x) * 4, t = (ny * out.width + nx) * 4
      o[t] = d[s]; o[t + 1] = d[s + 1]; o[t + 2] = d[s + 2]; o[t + 3] = 255
    }
  }
  return out
}

type Pt = [number, number]

/**
 * Rectified crop of a quadrilateral (top-left, top-right, bottom-right, bottom-left), as PIL's
 * QUAD transform (bilinear corner mapping, bicubic-like sampling approximated bilinearly), plus a
 * 5 px white border.
 */
export function quadCrop(img: RGBAImage, quad: Pt[]): RGBAImage {
  const [tl, tr, br, bl] = quad
  const dist = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1])
  const w = Math.max(1, Math.ceil(Math.max(dist(tl, tr), dist(bl, br))))
  const h = Math.max(1, Math.ceil(Math.max(dist(tl, bl), dist(tr, br))))
  const B = 5
  const out = makeImage(w + 2 * B, h + 2 * B, 255)
  const W = img.width, H = img.height, d = img.data, o = out.data
  for (let y = 0; y < h; y++) {
    const v = (y + 0.5) / h
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) / w
      const sx = tl[0] + (tr[0] - tl[0]) * u + (bl[0] - tl[0]) * v + (br[0] - bl[0] - tr[0] + tl[0]) * u * v - 0.5
      const sy = tl[1] + (tr[1] - tl[1]) * u + (bl[1] - tl[1]) * v + (br[1] - bl[1] - tr[1] + tl[1]) * u * v - 0.5
      const t = ((y + B) * out.width + x + B) * 4
      if (sx < -0.5 || sy < -0.5 || sx > W - 0.5 || sy > H - 0.5) continue // outside: stays white
      const x0 = Math.max(0, Math.min(W - 1, Math.floor(sx))), y0 = Math.max(0, Math.min(H - 1, Math.floor(sy)))
      const x1 = Math.min(x0 + 1, W - 1), y1 = Math.min(y0 + 1, H - 1)
      const wx = Math.min(1, Math.max(0, sx - x0)), wy = Math.min(1, Math.max(0, sy - y0))
      const a = (y0 * W + x0) * 4, b = (y0 * W + x1) * 4, c = (y1 * W + x0) * 4, e = (y1 * W + x1) * 4
      for (let k = 0; k < 3; k++) {
        o[t + k] = (d[a + k] * (1 - wx) + d[b + k] * wx) * (1 - wy) + (d[c + k] * (1 - wx) + d[e + k] * wx) * wy
      }
    }
  }
  return out
}
