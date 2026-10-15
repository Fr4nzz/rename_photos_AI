import { describe, expect, it } from 'vitest'
import { turnBox, wingsBox } from '../photoCrops'
import type { WingCheck } from '../wings/checks'

describe('parts of a photo', () => {
  it('turns a box with its photo', () => {
    // the top-left quarter: after a quarter clockwise it is the top-right one
    expect(turnBox([0, 0, 0.5, 0.5], 90)).toEqual([0.5, 0, 1, 0.5])
    expect(turnBox([0, 0, 0.5, 0.5], 180)).toEqual([0.5, 0.5, 1, 1])
    turnBox(turnBox([0.1, 0.2, 0.3, 0.6], 90), 270).forEach((v, i) => expect(v).toBeCloseTo([0.1, 0.2, 0.3, 0.6][i]))
  })
  it('the wings in the frame the photo was read in, also when the file had been turned', () => {
    const check = { wings: [{ box: [100, 100, 300, 200] }, { box: [500, 150, 700, 400] }], size: [1000, 500], applied: 0, turnK: 0, pTurn: 1 } as unknown as WingCheck
    expect(wingsBox(check)!.map((v) => Math.round(v * 100) / 100)).toEqual([0.06, 0.16, 0.74, 0.84])
    // checked after 180 was written: the same box turned back
    const turned = wingsBox({ ...check, applied: 180 })!
    expect(turned.map((v) => Math.round(v * 100) / 100)).toEqual([0.26, 0.16, 0.94, 0.84])
    expect(wingsBox({ ...check, wings: [] })).toBeNull()
  })
})
