import { describe, expect, it } from 'vitest'
import { readFileSync, existsSync } from 'node:fs'
import { findOrientation, rotatedOrientation, ORIENTATION_SCAN_BYTES } from '../orientation'

const head = (p: string) => {
  const b = readFileSync(p)
  return b.buffer.slice(b.byteOffset, b.byteOffset + Math.min(b.length, ORIENTATION_SCAN_BYTES))
}

describe('rotatedOrientation', () => {
  it('composes rotations', () => {
    expect(rotatedOrientation(1, 90)).toBe(8)
    expect(rotatedOrientation(1, 180)).toBe(3)
    expect(rotatedOrientation(3, 180)).toBe(1)
    expect(rotatedOrientation(6, 90)).toBe(1)
    expect(rotatedOrientation(8, -90)).toBe(1)
  })
})

// Real camera files (kept outside the repo); skipped when absent.
describe.skipIf(!existsSync('/tmp/rawtest/sample.CR2'))('real files', () => {
  it('finds the tag in a Canon CR2 (same offset the Python prototype patched)', () => {
    expect(findOrientation(head('/tmp/rawtest/sample.CR2'))).toEqual({ value: 3, offset: 110, littleEndian: true })
  })
  it('finds the tag in a camera JPEG', () => {
    const slot = findOrientation(head('/tmp/rawtest/sample.jpg'))
    expect(slot).not.toBeNull()
    expect(slot!.value).toBeGreaterThanOrEqual(1)
  })
  it('returns null for data without EXIF', () => {
    expect(findOrientation(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 0, 0, 0, 0, 0]).buffer)).toBeNull()
  })
})
