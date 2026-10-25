import { describe, expect, it } from 'vitest'
import { planTurns, planTurnsByCamera } from '../rotationPlan'
import type { PhotoRow } from '@/types'

/** `rot`: the envelope's own sure turn ('' none), as runLocalOcr fills it */
const row = (from: string, minute: number, rot: string): PhotoRow => ({
  from, currentPath: from, photoId: minute, mainValue: '', co: '', n: '', skip: '', to: '', suffix: '', batchNumber: 0,
  captureDate: new Date(Date.UTC(2026, 0, 1, 10, minute)).toISOString(), status: 'Original', review: '', suggest: '',
  rotSuggested: rot, rotChosen: rot, rotSource: rot === '' ? '' : 'ocr', rotApplied: '',
})
const zero = (rows: PhotoRow[]) => new Map<string, number | null>(rows.map((r) => [r.from, 0]))
const wide = (rows: PhotoRow[], w: boolean | null = true) => new Map<string, boolean | null>(rows.map((r) => [r.from, w]))
const turns = (out: PhotoRow[]) => out.map((r) => [r.rotChosen, r.rotSource, r.review])

describe('planTurns', () => {
  it('a photo with a sure envelope takes its own turn, sideways included, and ends standing if it must', () => {
    const rows = [row('a', 0, '90'), row('b', 1, '270'), row('c', 2, '180'), row('d', 3, '0')]
    expect(turns(planTurns(rows, wide(rows), zero(rows)))).toEqual([
      ['90', 'ocr', ''], ['270', 'ocr', ''], ['180', 'ocr', ''], ['0', 'ocr', ''],
    ])
  })
  it('sure wings give a photo its own turn without an envelope, and win over an envelope that disagrees, flagged', () => {
    const rows = [row('a', 0, '90'), row('b', 1, '0'), row('c', 2, ''), row('d', 3, '180')]
    const wings = new Map<string, number | null>([['a', 0], ['b', 0], ['c', 270], ['d', null]])
    expect(turns(planTurns(rows, wide(rows), zero(rows), wings))).toEqual([
      // an envelope laid sideways beside upright wings: the wings, and a look
      ['0', 'ocr', 'turn-disagree'], ['0', 'ocr', ''], ['270', 'ocr', ''],
      // wings not sure: the envelope
      ['180', 'ocr', ''],
    ])
  })
  it('an envelope laid another way on a copy stand is no news when the wings give the camera’s usual turn', () => {
    const rows = [row('a', 0, '270'), row('b', 1, '0'), row('c', 2, '0'), row('d', 3, ''), row('e', 4, '90')]
    const upright = new Map<string, number | null>(rows.map((r) => [r.from, 0]))
    expect(turns(planTurns(rows, wide(rows), zero(rows), upright))).toEqual([
      ['0', 'ocr', ''], ['0', 'ocr', ''], ['0', 'ocr', ''], ['0', 'ocr', ''], ['0', 'ocr', ''],
    ])
    // the wings of one photo say otherwise than the camera's usual turn: its envelope's disagreement is news
    const odd = new Map([...upright, ['e', 180]])
    expect(turns(planTurns(rows, wide(rows), zero(rows), odd))[4]).toEqual(['180', 'ocr', 'turn-disagree'])
  })
  it('an envelope alone against the camera’s sure usual turn is followed, and flagged', () => {
    const rows = [row('a', 0, '0'), row('b', 1, '0'), row('c', 2, '0'), row('d', 3, '90')]
    expect(turns(planTurns(rows, wide(rows), zero(rows)))[3]).toEqual(['90', 'ocr', 'turn-camera'])
    // with no usual turn to go against (too few photos), nothing to say
    expect(turns(planTurns(rows.slice(2), wide(rows), zero(rows)))[1]).toEqual(['90', 'ocr', ''])
  })
  it('a photo saying nothing takes its camera’s usual turn from the others, kept landscape, in the camera frame', () => {
    const rows = [row('a', 0, '180'), row('b', 1, '180'), row('c', 2, '0'), row('d', 3, '180'), row('e', 4, '180'), row('f', 5, ''), row('g', 6, '')]
    const tags = new Map<string, number | null>([['a', 0], ['b', 0], ['c', 0], ['d', 0], ['e', 0], ['f', 90], ['g', 0]])
    const out = planTurns(rows, new Map([...wide(rows)].map(([k]) => [k, k !== 'f'])), tags)
    // c keeps its own turn (flagged: against the others); f shows standing because its tag turns it a
    // quarter: in the camera frame it needs 180 too, so 90 from how it shows. Four of five alike: sure.
    expect(turns(out)).toEqual([
      ['180', 'ocr', ''], ['180', 'ocr', ''], ['0', 'ocr', 'turn-camera'], ['180', 'ocr', ''], ['180', 'ocr', ''],
      ['90', 'neighbours', ''], ['180', 'neighbours', ''],
    ])
  })
  it('own turns that leave a photo standing say nothing of the camera’s usual turn; without a clear one, a look', () => {
    const rows = [row('a', 0, '270'), row('b', 1, '90'), row('c', 2, ''), row('d', 3, '180')]
    expect(turns(planTurns(rows, wide(rows), zero(rows)))).toEqual([
      ['270', 'ocr', ''], ['90', 'ocr', ''], ['180', 'neighbours', 'turn-unsure'], ['180', 'ocr', ''],
    ])
  })
  it('a batch with nothing sure stays as shot, flagged; manual turns and unknown shapes are left alone', () => {
    const rows = [row('a', 0, ''), row('b', 1, ''), { ...row('c', 2, '90'), rotSource: 'manual' as const }, row('d', 3, '')]
    const out = planTurns(rows, new Map([...wide(rows)].map(([k]) => [k, k === 'd' ? null : true])), zero(rows))
    expect(turns(out)).toEqual([['0', 'neighbours', 'turn-unsure'], ['0', 'neighbours', 'turn-unsure'], ['90', 'manual', ''], ['', '', '']])
  })
  it('drops the old turn flags before deciding again', () => {
    const rows = [{ ...row('a', 0, '0'), review: 'rotation-outlier,turn-unsure,turn-disagree,turn-camera,no-reading' }]
    expect(planTurns(rows, wide(rows), zero(rows))[0].review).toBe('no-reading')
  })
})

describe('planTurnsByCamera', () => {
  it('gives the photos of each camera their own usual turn', () => {
    const rows = ['a', 'b', 'c', 'd', 'e', 'f', 'g'].map((n, i): PhotoRow => row(n, i, ''))
    const wideAll = new Map<string, boolean | null>(rows.map((r) => [r.from, true]))
    const tags = new Map<string, number | null>(rows.map((r) => [r.from, 0]))
    // three photos of one camera upright by their wings, a fourth saying nothing; two of another upside down and a third saying nothing
    const wings = new Map<string, number | null>([['a', 0], ['b', 0], ['c', 0], ['d', null], ['e', 180], ['f', 180], ['g', null]])
    const camera = new Map([['a', 'Canon'], ['b', 'Canon'], ['c', 'Canon'], ['d', 'Canon'], ['e', 'OLYMPUS TG-6'], ['f', 'OLYMPUS TG-6'], ['g', 'OLYMPUS TG-6']])
    const out = planTurnsByCamera(rows, wideAll, tags, camera, wings)
    expect(out.map((r) => r.rotChosen)).toEqual(['0', '0', '0', '0', '180', '180', '180'])
    // two photos are too few to be sure of a camera's turn: the one taking it is left to look at
    expect(out.map((r) => r.review)).toEqual(['', '', '', '', '', '', 'turn-unsure'])
  })
})
