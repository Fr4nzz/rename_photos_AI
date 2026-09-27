import { describe, expect, it } from 'vitest'
import { decide, prefill, type PhotoInput } from '../decide'
import type { PhotoReading } from '../pipeline'

const reading = (texts: string[]): PhotoReading => {
  const lines = texts.map((text) => ({ box: [], text, conf: 0.99 }))
  const candidates = texts.map((t) => ({ id: t.replace(/\s+/g, '').toUpperCase(), conf: 0.99 })).filter((c) => /^CAM\d{6}$/.test(c.id))
  return { camid: candidates[0]?.id ?? null, conf: 0.99, candidates, source: 'envelope', envelope: null, turned: 0, lines, size: [1600, 1067] }
}
const photo = (name: string, texts: string[], t: number): PhotoInput => ({ name, reading: reading(texts), capturedAt: t })

describe('prefill', () => {
  it('fixes handwriting lookalikes and keeps the replacement of a merged crossed-out ID', () => {
    expect(prefill('CAM07107L')).toBe('CAM071072')
    expect(prefill('CAM07e533')).toBe('CAM078533')
    expect(prefill('CAM77101077426')).toBe('CAM077426')
  })
})

describe('decide', () => {
  const known = new Set(['CAM078740', 'CAM078741', 'CAM078742', 'CAM078743', 'CAM073741', 'CAM078744'])
  it('renames consistent readings automatically', () => {
    const d = decide([photo('a', ['CAM078740'], 1), photo('b', ['CAM078741'], 2), photo('c', ['CAM078742'], 3)], known)
    expect(d.every((x) => x.auto)).toBe(true)
  })
  it('sends an out-of-sequence reading to review', () => {
    const d = decide([photo('a', ['CAM078740'], 1), photo('b', ['CAM078741'], 2), photo('c', ['CAM073741'], 3),
                      photo('d', ['CAM078742'], 4), photo('e', ['CAM078743'], 5)], known)
    expect(d[2].auto).toBe(false)
    expect(d[2].reasons).toEqual(['out-of-sequence'])
  })
  it('sends IDs missing from the database to review with nearby suggestions', () => {
    const d = decide([photo('a', ['CAM078740'], 1), photo('b', ['CAM07874L'], 2), photo('c', ['CAM078742'], 3)], known)
    expect(d[1].auto).toBe(false)
    expect(d[1].reasons).toEqual(['no-reading']) // 'CAM07874L' is not a whole CAMID reading
    expect(d[1].prefill).toBe('CAM078742')
    expect(d[1].candidates[0]).toBe('CAM078742')
  })
  it('flags an ID repeated on photos that are not taken back to back', () => {
    const d = decide([photo('a', ['CAM078740'], 1), photo('b', ['CAM078740'], 2), photo('c', ['CAM078741'], 3),
                      photo('d', ['CAM078742'], 4), photo('e', ['CAM078741'], 5)], known)
    expect(d[0].auto && d[1].auto).toBe(true) // a dorsal/ventral pair, back to back
    expect(d[2].reasons).toEqual(['repeated-id'])
    expect(d[4].reasons).toEqual(['repeated-id'])
  })
  it('suggests the missing number between the neighbouring photos', () => {
    const d = decide([photo('a', ['CAM078740'], 1), photo('b', ['xx'], 2), photo('c', ['CAM078742'], 3)], known)
    expect(d[1].candidates[0]).toBe('CAM078741')
  })
})
