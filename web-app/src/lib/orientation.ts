/**
 * Lossless rotation: read and change the EXIF Orientation tag (0x0112) in place.
 *
 * Rotating a photo for viewers only means changing this one number; the pixels stay untouched.
 * This is what ExifTool's `-Orientation=N` does, but here only the tag's 2 bytes are overwritten
 * (the file keeps its size and every other byte), so no backend, no re-encoding and no backup
 * copies are needed. Undo = write the previous value back.
 *
 * Supported: JPEG (APP1 Exif), TIFF-based RAW (CR2, NEF, ARW, DNG, PEF, SRW, ORF, RW2, TIFF) and
 * Canon CR3 (the TIFF block in the CMT1 box). Not supported (returns null): PNG, HEIC/HEIF, RAF.
 */

import { ANGLE_TO_ORIENTATION, ORIENTATION_TO_ANGLE } from './constants'

export interface OrientationSlot {
  /** current Orientation value (1-8) */
  value: number
  /** absolute byte offset of the tag's 2-byte value in the file */
  offset: number
  littleEndian: boolean
}

/** Bytes read from the start of a file to find IFD0; the tag always sits well inside this. */
export const ORIENTATION_SCAN_BYTES = 256 * 1024

const TAG_ORIENTATION = 0x0112
const TYPE_SHORT = 3

/** Parse a TIFF structure starting at `base` and return the Orientation entry of IFD0. */
function findInTiff(view: DataView, base: number): OrientationSlot | null {
  if (base + 8 > view.byteLength) return null
  const bo = view.getUint16(base)
  const le = bo === 0x4949 // 'II'
  if (!le && bo !== 0x4d4d) return null // 'MM'
  const magic = view.getUint16(base + 2, le)
  // 42 = TIFF (CR2, NEF, ARW, DNG, PEF, SRW); ORF uses 'RO' / 'SR'; RW2 uses 0x55
  if (magic !== 42 && magic !== 0x4f52 && magic !== 0x5352 && magic !== 0x55) return null
  const ifd = base + view.getUint32(base + 4, le)
  if (ifd + 2 > view.byteLength) return null
  const count = view.getUint16(ifd, le)
  for (let i = 0; i < count; i++) {
    const entry = ifd + 2 + 12 * i
    if (entry + 12 > view.byteLength) return null
    if (view.getUint16(entry, le) !== TAG_ORIENTATION) continue
    if (view.getUint16(entry + 2, le) !== TYPE_SHORT || view.getUint32(entry + 4, le) !== 1) return null
    return { value: view.getUint16(entry + 8, le), offset: entry + 8, littleEndian: le }
  }
  return null
}

/** Walk JPEG markers to the APP1 'Exif' segment and read its TIFF block. */
function findInJpeg(view: DataView): OrientationSlot | null {
  let pos = 2
  while (pos + 4 <= view.byteLength) {
    if (view.getUint8(pos) !== 0xff) return null
    const marker = view.getUint8(pos + 1)
    if (marker === 0xd9 || marker === 0xda) return null // end of image / start of scan
    const length = view.getUint16(pos + 2)
    if (marker === 0xe1 && pos + 10 <= view.byteLength &&
        view.getUint32(pos + 4) === 0x45786966 && view.getUint16(pos + 8) === 0) { // 'Exif\0\0'
      return findInTiff(view, pos + 10)
    }
    pos += 2 + length
  }
  return null
}

/** ISO-BMFF boxes (CR3): [type, payload start, box end] of the children of [start, end). */
export function* bmffBoxes(view: DataView, start: number, end: number): Generator<[string, number, number]> {
  let pos = start
  while (pos + 8 <= Math.min(end, view.byteLength)) {
    let size = view.getUint32(pos)
    let header = 8
    const type = String.fromCharCode(view.getUint8(pos + 4), view.getUint8(pos + 5), view.getUint8(pos + 6), view.getUint8(pos + 7))
    if (size === 1 && pos + 16 <= view.byteLength) { size = Number(view.getBigUint64(pos + 8)); header = 16 }
    if (size === 0) size = end - pos
    if (size < header) return
    if (type === 'uuid') header += 16
    yield [type === 'uuid' ? `uuid:${uuidAt(view, pos + header - 16)}` : type, pos + header, pos + size]
    pos += size
  }
}

function uuidAt(view: DataView, at: number): string {
  let s = ''
  for (let i = 0; i < 16 && at + i < view.byteLength; i++) s += view.getUint8(at + i).toString(16).padStart(2, '0')
  return s
}

export const isCr3 = (view: DataView) =>
  view.byteLength >= 12 && view.getUint32(4) === 0x66747970 /* ftyp */ && view.getUint32(8) === 0x63727820 /* 'crx ' */

/** CR3: moov -> uuid 85c0b687… (Canon metadata) -> CMT1 = IFD0 as a TIFF block. */
function findInCr3(view: DataView): OrientationSlot | null {
  for (const [type, start, end] of bmffBoxes(view, 0, view.byteLength)) {
    if (type !== 'moov') continue
    for (const [t2, s2, e2] of bmffBoxes(view, start, end)) {
      if (!t2.startsWith('uuid:85c0b687')) continue
      for (const [t3, s3] of bmffBoxes(view, s2, e2)) if (t3 === 'CMT1') return findInTiff(view, s3)
    }
  }
  return null
}

/** Locate the Orientation tag in the first bytes of a file. */
export function findOrientation(head: ArrayBuffer): OrientationSlot | null {
  const view = new DataView(head)
  if (view.byteLength < 12) return null
  if (view.getUint16(0) === 0xffd8) return findInJpeg(view)
  if (isCr3(view)) return findInCr3(view)
  return findInTiff(view, 0)
}

/** Current rotation of a file in degrees (counter-clockwise, as ORIENTATION_TO_ANGLE), or null. */
export async function readOrientation(file: Blob): Promise<OrientationSlot | null> {
  return findOrientation(await file.slice(0, ORIENTATION_SCAN_BYTES).arrayBuffer())
}

/** Orientation value after rotating a photo that currently has `current` by `angle` degrees. */
export function rotatedOrientation(current: number, angle: number): number {
  const now = ORIENTATION_TO_ANGLE[current] ?? 0
  return ANGLE_TO_ORIENTATION[(((now + angle) % 360) + 360) % 360]
}

/**
 * Write a new Orientation value into a file in place (File System Access API).
 * Returns the previous value (for undo), or null when the file has no writable Orientation tag.
 */
export async function writeOrientation(handle: FileSystemFileHandle, value: number): Promise<number | null> {
  const slot = await readOrientation(await handle.getFile())
  if (!slot) return null
  if (slot.value === value) return slot.value
  const bytes = new Uint8Array(2)
  new DataView(bytes.buffer).setUint16(0, value, slot.littleEndian)
  const writable = await handle.createWritable({ keepExistingData: true })
  try {
    await writable.seek(slot.offset)
    await writable.write(bytes)
  } finally {
    await writable.close()
  }
  const check = await readOrientation(await handle.getFile())
  if (!check || check.value !== value) throw new Error('Orientation write could not be verified')
  return slot.value
}

/** Rotate a file losslessly by `angle` degrees; returns {before, after} or null if unsupported. */
export async function rotateLossless(
  handle: FileSystemFileHandle,
  angle: number,
): Promise<{ before: number; after: number } | null> {
  const slot = await readOrientation(await handle.getFile())
  if (!slot) return null
  const after = rotatedOrientation(slot.value, angle)
  const before = await writeOrientation(handle, after)
  return before === null ? null : { before, after }
}
