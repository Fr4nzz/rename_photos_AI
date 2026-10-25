import { describe, expect, it } from 'vitest'
import { envelopeTurn, pendingRotation, uprightRotation } from '../rotationPlan'
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

describe('envelopeTurn', () => {
  const textLine = (text: string, turn: 0 | 90 | 180 | 270) => ({ box: [], text, conf: 0.9, logp: -1, turn })
  const withText = (turn: 0 | 90 | 180 | 270, scores: Partial<Record<0 | 90 | 180 | 270, number>>, text = 'Ithomia salapia'): PhotoReading => ({
    ...reading(0), camid: null, lines: [],
    text: { turn, score: scores[turn] ?? 0, lines: [textLine(text, turn), textLine('♀', turn)], scores, extraReads: 0 },
  })
  it('takes a sure CAM ID line’s turn, in any direction, an ID ending a longer line too', () => {
    expect(envelopeTurn(reading(0, 90), 'CAM070001')).toBe(270)
    expect(envelopeTurn(reading(180), 'CAM070001')).toBe(180)
    const inLine = { ...reading(0, 270), lines: [{ box: [], text: 'CAM070001♀', conf: 0.98, turn: 270 as const }] }
    expect(envelopeTurn(inLine, 'CAM070001')).toBe(90)
  })
  it('an unsure CAM ID line, or an ID not to trust, gives way to the envelope’s whole text', () => {
    const unsure = { ...reading(0, 90), lines: [{ box: [], text: 'CAM070001', conf: 0.9, turn: 90 as const }] }
    expect(envelopeTurn(unsure, 'CAM070001')).toBeNull()
    expect(envelopeTurn(reading(0, 90), null)).toBeNull()
    expect(envelopeTurn({ ...withText(270, { 90: -0.4, 270: -0.1 }), lines: unsure.lines }, 'CAM070001')).toBe(90)
  })
  it('takes the whole text’s turn when clearly better than the other way, with enough read', () => {
    expect(envelopeTurn(withText(90, { 90: -0.2, 270: -0.5 }), null)).toBe(270)
    expect(envelopeTurn(withText(180, { 0: -0.5, 180: -0.03 }), null)).toBe(180)
    // read upright only (it read well enough not to try upside down)
    expect(envelopeTurn(withText(0, { 0: -0.05 }), null)).toBe(0)
    expect(envelopeTurn(withText(0, { 0: -0.3 }), null)).toBeNull()
    // too close to call, or too little text
    expect(envelopeTurn(withText(90, { 90: -0.2, 270: -0.25 }), null)).toBeNull()
    expect(envelopeTurn(withText(90, { 90: -0.2, 270: -0.5 }, '12'), null)).toBeNull()
  })
})
