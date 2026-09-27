import { describe, expect, it } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { decodeHeic } from '../heic'

describe.skipIf(!existsSync('/tmp/rawtest/sample.heic'))('HEIC', () => {
  it('decodes an iPhone HEIC photo to upright RGBA pixels', async () => {
    const img = await decodeHeic(new Blob([readFileSync('/tmp/rawtest/sample.heic')]))
    expect([img.width, img.height]).toEqual([4032, 3024]) // same as pillow-heif
    expect(img.data.length).toBe(4032 * 3024 * 4)
    let sum = 0
    for (let i = 0; i < img.data.length; i += 4001) sum += img.data[i]
    expect(sum).toBeGreaterThan(0) // not blank
  }, 60_000)
})
