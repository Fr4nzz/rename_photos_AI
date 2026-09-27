/**
 * Largest embedded JPEG preview of a TIFF-based RAW file (CR2, NEF, ARW, DNG, PEF, SRW, TIFF).
 *
 * RAW files carry camera-rendered JPEG previews, often full size (Canon CR2: IFD0 strip; Nikon NEF
 * and DNG: a SubIFD 'JpgFromRaw'). Using the largest one gives a sharp, correctly processed image
 * without decoding the raw sensor data. Returns null when none is found (ORF keeps its preview in
 * maker notes, which is not parsed here; callers fall back to exifr's small thumbnail).
 */

import { bmffBoxes, isCr3 } from './orientation'

const HEAD_BYTES = 1024 * 1024
const T = { subIfds: 0x014a, stripOffsets: 0x0111, stripByteCounts: 0x0117, jpegOffset: 0x0201, jpegLength: 0x0202 }

interface Candidate { offset: number; length: number }

function readValues(view: DataView, entry: number, le: boolean, base: number): number[] {
  const type = view.getUint16(entry + 2, le)
  const count = view.getUint32(entry + 4, le)
  const size = type === 3 ? 2 : type === 4 || type === 13 ? 4 : 0
  if (!size || count === 0 || count > 4096) return []
  const inline = size * count <= 4
  const at = inline ? entry + 8 : base + view.getUint32(entry + 8, le)
  if (at + size * count > view.byteLength) return []
  const out: number[] = []
  for (let i = 0; i < count; i++) out.push(size === 2 ? view.getUint16(at + 2 * i, le) : view.getUint32(at + 4 * i, le))
  return out
}

export function findJpegPreviews(head: ArrayBuffer): Candidate[] {
  const view = new DataView(head)
  if (view.byteLength < 16) return []
  const bo = view.getUint16(0)
  const le = bo === 0x4949
  if (!le && bo !== 0x4d4d) return []
  const magic = view.getUint16(2, le)
  if (magic !== 42 && magic !== 0x4f52 && magic !== 0x5352 && magic !== 0x55) return []

  const seen = new Set<number>()
  const queue: number[] = [view.getUint32(4, le)]
  const found: Candidate[] = []
  while (queue.length && seen.size < 64) {
    const ifd = queue.shift()!
    if (!ifd || seen.has(ifd) || ifd + 2 > view.byteLength) continue
    seen.add(ifd)
    const n = view.getUint16(ifd, le)
    if (ifd + 2 + 12 * n + 4 > view.byteLength) continue
    const tags = new Map<number, number[]>()
    for (let i = 0; i < n; i++) {
      const entry = ifd + 2 + 12 * i
      const tag = view.getUint16(entry, le)
      if (tag === T.subIfds || tag === T.stripOffsets || tag === T.stripByteCounts || tag === T.jpegOffset || tag === T.jpegLength) {
        tags.set(tag, readValues(view, entry, le, 0))
      }
    }
    queue.push(...(tags.get(T.subIfds) ?? []))
    const next = view.getUint32(ifd + 2 + 12 * n, le)
    if (next) queue.push(next)
    const jo = tags.get(T.jpegOffset)?.[0]
    const jl = tags.get(T.jpegLength)?.[0]
    if (jo && jl) found.push({ offset: jo, length: jl })
    const so = tags.get(T.stripOffsets)
    const sl = tags.get(T.stripByteCounts)
    if (so?.length === 1 && sl?.length === 1) found.push({ offset: so[0], length: sl[0] })
  }
  return found
}

/** CR3: the PRVW box (in uuid eaf42b5e…) holds a 1620 px camera JPEG. */
function findCr3Preview(view: DataView): Candidate[] {
  for (const [type, start, end] of bmffBoxes(view, 0, view.byteLength)) {
    if (!type.startsWith('uuid:eaf42b5e')) continue
    for (const [t2, s2, e2] of bmffBoxes(view, start + 8, end)) {
      if (t2 !== 'PRVW') continue
      for (let i = s2; i + 1 < Math.min(e2, view.byteLength); i++) {
        if (view.getUint8(i) === 0xff && view.getUint8(i + 1) === 0xd8) return [{ offset: i, length: e2 - i }]
      }
    }
  }
  return []
}

export async function extractLargestJpeg(file: Blob): Promise<Blob | null> {
  const head = await file.slice(0, HEAD_BYTES).arrayBuffer()
  const view = new DataView(head)
  const candidates = (isCr3(view) ? findCr3Preview(view) : findJpegPreviews(head))
    .filter((c) => c.offset + c.length <= file.size)
    .sort((a, b) => b.length - a.length)
  for (const c of candidates) {
    const sig = new Uint8Array(await file.slice(c.offset, c.offset + 2).arrayBuffer())
    if (sig[0] === 0xff && sig[1] === 0xd8) return file.slice(c.offset, c.offset + c.length, 'image/jpeg')
  }
  return null
}
