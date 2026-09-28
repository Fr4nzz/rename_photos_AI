import { describe, expect, it } from 'vitest'
import { applyLearnedOffsets, pendingRotation, uprightRotation } from '../rotationPlan'
import type { PhotoReading } from '../ocr/pipeline'
import type { PhotoRow } from '@/types'

const reading = (turned: number, turn?: 90 | 270): PhotoReading => ({
  camid: 'CAM070001', conf: 0.99, candidates: [{ id: 'CAM070001', conf: 0.99 }], source: 'envelope', envelope: null,
  turned, size: [1600, 1067], lines: [{ box: [], text: 'CAM070001', conf: 0.99, ...(turn ? { turn } : {}) }],
})
const row = (i: number, minute: number, rotSuggested: string, rotChosen = rotSuggested, rotSource: PhotoRow['rotSource'] = 'ocr'): PhotoRow => ({
  from: `IMG_${i}.jpg`, currentPath: `IMG_${i}.jpg`, photoId: i, mainValue: '', co: '', n: '', skip: '', to: '', suffix: '',
  batchNumber: 0, captureDate: new Date(Date.UTC(2026, 0, 1, 10, minute)).toISOString(), status: 'Original', review: '',
  suggest: '', rotSuggested, rotChosen, rotSource, rotApplied: '',
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

describe('applyLearnedOffsets', () => {
  it('keeps a single correction as an exception', () => {
    const rows = [row(1, 0, '0', '90', 'manual'), row(2, 1, '0'), row(3, 2, '180')]
    expect(applyLearnedOffsets(rows).map((r) => [r.rotChosen, r.rotSource])).toEqual([['90', 'manual'], ['0', 'ocr'], ['180', 'ocr']])
  })
  it('applies an offset confirmed by two corrections to the rest of the session only', () => {
    const rows = [row(1, 0, '0', '90', 'manual'), row(2, 1, '180', '270', 'manual'), row(3, 2, '0'), row(4, 3, '180'),
                  row(5, 120, '0')] // two hours later: another session
    expect(applyLearnedOffsets(rows).map((r) => [r.rotChosen, r.rotSource])).toEqual(
      [['90', 'manual'], ['270', 'manual'], ['90', 'learned'], ['270', 'learned'], ['0', 'ocr']])
  })
  it('goes back to the reader when the corrections are undone', () => {
    const learned = applyLearnedOffsets([row(1, 0, '0', '90', 'manual'), row(2, 1, '0', '90', 'manual'), row(3, 2, '0')])
    const undone = applyLearnedOffsets(learned.map((r, i) => (i < 2 ? { ...r, rotChosen: '0' } : r)))
    expect(undone[2]).toMatchObject({ rotChosen: '0', rotSource: 'ocr' })
  })
})

describe('pendingRotation', () => {
  it('is what is chosen but not yet written', () => {
    expect(pendingRotation({ rotChosen: '270', rotApplied: '90' })).toBe(180)
    expect(pendingRotation({ rotChosen: '', rotApplied: '' })).toBe(0)
  })
})
