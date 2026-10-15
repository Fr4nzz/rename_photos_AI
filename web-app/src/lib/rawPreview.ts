/**
 * Largest embedded JPEG preview of a TIFF-based RAW file (CR2, NEF, ARW, DNG, PEF, SRW, TIFF).
 *
 * RAW files carry camera-rendered JPEG previews, often full size (Canon CR2: IFD0 strip; Nikon NEF
 * and DNG: a SubIFD 'JpgFromRaw'; Olympus ORF: in the maker notes' camera settings). Using the
 * largest one gives a sharp, correctly processed image without decoding the raw sensor data. The
 * sensor data itself may be stored as a lossless JPEG (CR2's last IFD), which browsers cannot
 * decode, so only baseline and progressive JPEGs count. Returns null when none is found (callers
 * fall back to exifr's small thumbnail).
 */

import { bmffBoxes, isCr3 } from './orientation'

const HEAD_BYTES = 1024 * 1024
const T = { subIfds: 0x014a, stripOffsets: 0x0111, stripByteCounts: 0x0117, jpegOffset: 0x0201, jpegLength: 0x0202, exifIfd: 0x8769, makerNote: 0x927c }
/** Olympus maker notes: the camera settings IFD and its preview's start and length */
const OLYMPUS = { cameraSettings: 0x2020, previewStart: 0x0101, previewLength: 0x0102 }

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
      if (tag === T.subIfds || tag === T.stripOffsets || tag === T.stripByteCounts || tag === T.jpegOffset || tag === T.jpegLength || tag === T.exifIfd) {
        tags.set(tag, readValues(view, entry, le, 0))
      } else if (tag === T.makerNote && view.getUint32(entry + 4, le) > 4) {
        found.push(...olympusPreview(view, view.getUint32(entry + 8, le)))
      }
    }
    queue.push(...(tags.get(T.subIfds) ?? []), ...(tags.get(T.exifIfd) ?? []))
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

/** IFD entries' first values by tag (SHORT, LONG or IFD), or null when the IFD lies outside the head. */
function ifdValues(view: DataView, ifd: number, le: boolean, wanted: number[]): Map<number, number> | null {
  if (ifd + 2 > view.byteLength) return null
  const n = view.getUint16(ifd, le)
  if (ifd + 2 + 12 * n > view.byteLength) return null
  const out = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    const entry = ifd + 2 + 12 * i
    const tag = view.getUint16(entry, le)
    if (wanted.includes(tag)) out.set(tag, readValues(view, entry, le, 0)[0] ?? 0)
  }
  return out
}

/**
 * ORF: the camera's JPEG is named in the Olympus maker notes, in the camera settings IFD (0x2020).
 * New-style notes ('OLYMPUS\0' + byte order) count offsets from the notes' start; old-style
 * ('OLYMP\0') from the TIFF header.
 */
function olympusPreview(view: DataView, at: number): Candidate[] {
  if (at + 16 > view.byteLength) return []
  const sig = String.fromCharCode(...new Uint8Array(view.buffer, view.byteOffset + at, 8))
  let base: number, ifd: number, le: boolean
  if (sig === 'OLYMPUS\0') {
    le = view.getUint16(at + 8) === 0x4949
    base = at
    ifd = at + 12
  } else if (sig.startsWith('OLYMP\0')) {
    le = view.getUint16(0) === 0x4949
    base = 0
    ifd = at + 8
  } else {
    return []
  }
  const settings = ifdValues(view, ifd, le, [OLYMPUS.cameraSettings])?.get(OLYMPUS.cameraSettings)
  if (!settings) return []
  const preview = ifdValues(view, base + settings, le, [OLYMPUS.previewStart, OLYMPUS.previewLength])
  const start = preview?.get(OLYMPUS.previewStart), length = preview?.get(OLYMPUS.previewLength)
  return start && length ? [{ offset: base + start, length }] : []
}

/**
 * Whether the bytes begin a JPEG a browser can draw: its frame header (SOF) is baseline (C0),
 * extended (C1) or progressive (C2), not lossless (C3, a RAW's sensor data) or arithmetic coded.
 */
export function isDisplayableJpeg(bytes: Uint8Array): boolean {
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) return false
  let i = 2
  while (i + 4 <= bytes.length) {
    if (bytes[i] !== 0xff) return false
    const marker = bytes[i + 1]
    if (marker === 0xff) { i++; continue }
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) return marker <= 0xc2
    if (marker === 0xda || marker === 0xd9) return false
    i += 2 + ((bytes[i + 2] << 8) | bytes[i + 3])
  }
  return false
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
    // the frame header follows the tables, within the first kilobytes (EXIF in a preview included)
    const start = new Uint8Array(await file.slice(c.offset, c.offset + Math.min(c.length, 128 * 1024)).arrayBuffer())
    if (isDisplayableJpeg(start)) return file.slice(c.offset, c.offset + c.length, 'image/jpeg')
  }
  return null
}
