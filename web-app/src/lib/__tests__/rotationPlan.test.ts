import { describe, expect, it } from 'vitest'
import { pendingRotation, uprightRotation } from '../rotationPlan'
import type { PhotoReading } from '../ocr/pipeline'

const reading = (turned: number, turn?: 90 | 270): PhotoReading => ({
  camid: 'CAM070001', conf: 0.99, candidates: [{ id: 'CAM070001', conf: 0.99 }], source: 'envelope', envelope: null,
  turned, size: [1600, 1067], lines: [{ box: [], text: 'CAM070001', conf: 0.99, ...(turn ? { turn } : {}) }],
})
describe('uprightRotation', () => {
  it('turns the photo the way its CAMID line had to be turned to be read', () => {
    expect(uprightRotation(reading(0), 'CAM070001')).toBe(0)
    expect(uprightRotation(reading(180), 'CAM070001')).toBe(180)
    expect(uprightRotation(reading(0, 90), 'CAM070001')).toBe(270) // read after turning 90° counter-clockwise
    expect(uprightRotation(reading(0, 270), 'CAM070001')).toBe(90)
    expect(uprightRotation(reading(0), null)).toBeNull()
  })
})

describe('pendingRotation', () => {
  it('is what is chosen but not yet written', () => {
    expect(pendingRotation({ rotChosen: '270', rotApplied: '90' })).toBe(180)
    expect(pendingRotation({ rotChosen: '', rotApplied: '' })).toBe(0)
  })
})
