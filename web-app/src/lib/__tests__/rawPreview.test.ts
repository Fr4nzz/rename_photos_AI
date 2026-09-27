import { describe, expect, it } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { extractLargestJpeg } from '../rawPreview'

describe.skipIf(!existsSync('/tmp/rawtest/sample.CR2'))('RAW preview', () => {
  it('extracts the full-size JPEG from a Canon CR2', async () => {
    const b = readFileSync('/tmp/rawtest/sample.CR2')
    const jpeg = await extractLargestJpeg(new Blob([b]))
    expect(jpeg).not.toBeNull()
    const bytes = new Uint8Array(await jpeg!.arrayBuffer())
    expect(bytes[0]).toBe(0xff)
    expect(bytes[1]).toBe(0xd8)
    expect(jpeg!.size).toBeGreaterThan(1_000_000) // full-size preview, not the 160 px thumbnail
    // read the JPEG's frame size (SOF marker)
    let pos = 2, w = 0
    while (pos < bytes.length) {
      const marker = bytes[pos + 1], len = (bytes[pos + 2] << 8) | bytes[pos + 3]
      if (marker >= 0xc0 && marker <= 0xc2) { w = (bytes[pos + 7] << 8) | bytes[pos + 8]; break }
      pos += 2 + len
    }
    expect(w).toBeGreaterThan(3000)
  })
})
