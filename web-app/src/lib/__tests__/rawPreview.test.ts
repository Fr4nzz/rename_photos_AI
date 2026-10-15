import { describe, expect, it } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { extractLargestJpeg, isDisplayableJpeg } from '../rawPreview'

/** a JPEG's frame width (baseline or progressive SOF), 0 when it has none a browser can draw */
function frameWidth(bytes: Uint8Array): number {
  let pos = 2
  while (pos + 9 < bytes.length) {
    const marker = bytes[pos + 1], len = (bytes[pos + 2] << 8) | bytes[pos + 3]
    if (marker >= 0xc0 && marker <= 0xc2) return (bytes[pos + 7] << 8) | bytes[pos + 8]
    if (marker === 0xc3 || marker === 0xda) return 0
    pos += 2 + len
  }
  return 0
}

// RAW samples are not in the repository: a Canon CR2 (EOS 550D: its sensor data is a lossless JPEG
// larger than the preview) and an Olympus ORF (TG-6: the preview is in the maker notes)
for (const [name, minWidth] of [['canon.CR2', 3000], ['olympus.ORF', 1000]] as const) {
  describe.skipIf(!existsSync(`/tmp/rawtest/${name}`))(`${name} preview`, () => {
    it('extracts the camera JPEG a browser can draw', async () => {
      const jpeg = await extractLargestJpeg(new Blob([readFileSync(`/tmp/rawtest/${name}`)]))
      expect(jpeg).not.toBeNull()
      const bytes = new Uint8Array(await jpeg!.arrayBuffer())
      expect(isDisplayableJpeg(bytes)).toBe(true)
      expect(frameWidth(bytes)).toBeGreaterThan(minWidth) // not the 160 px thumbnail
    })
  })
}

describe('isDisplayableJpeg', () => {
  const jpeg = (sof: number) => new Uint8Array([0xff, 0xd8, 0xff, 0xc4, 0, 4, 0, 0, 0xff, sof, 0, 11, 8, 0, 16, 0, 16, 1, 1, 0x11, 0])
  it('takes baseline and progressive frames and refuses lossless ones', () => {
    expect(isDisplayableJpeg(jpeg(0xc0))).toBe(true)
    expect(isDisplayableJpeg(jpeg(0xc2))).toBe(true)
    expect(isDisplayableJpeg(jpeg(0xc3))).toBe(false)
    expect(isDisplayableJpeg(new Uint8Array([0x49, 0x49, 0x2a, 0]))).toBe(false)
  })
})

describe.skipIf(!existsSync('/tmp/rawtest/sample.CR3'))('CR3 preview', () => {
  it('extracts the PRVW JPEG', async () => {
    const jpeg = await extractLargestJpeg(new Blob([readFileSync('/tmp/rawtest/sample.CR3')]))
    expect(jpeg).not.toBeNull()
    const bytes = new Uint8Array(await jpeg!.arrayBuffer())
    expect([bytes[0], bytes[1]]).toEqual([0xff, 0xd8])
    expect(jpeg!.size).toBeGreaterThan(100_000)
  })
})
