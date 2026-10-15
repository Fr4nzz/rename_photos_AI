import { describe, expect, it } from 'vitest'
import { oneTurn, oneTurnByCamera } from '../rotationPlan'
import type { PhotoRow } from '@/types'

const row = (from: string, minute: number, rot: string): PhotoRow => ({
  from, currentPath: from, photoId: minute, mainValue: '', co: '', n: '', skip: '', to: '', suffix: '', batchNumber: 0,
  captureDate: new Date(Date.UTC(2026, 0, 1, 10, minute)).toISOString(), status: 'Original', review: '', suggest: '',
  rotSuggested: rot, rotChosen: rot, rotSource: rot === '' ? '' : 'ocr', rotApplied: '',
})

describe('oneTurn', () => {
  const zero = (rows: PhotoRow[]) => new Map<string, number | null>(rows.map((r) => [r.from, 0]))
  const wide = (rows: PhotoRow[], w: boolean | null = true) => new Map<string, boolean | null>(rows.map((r) => [r.from, w]))
  it('gives every photo the turn most envelopes read upright or upside down say, in the camera frame', () => {
    const rows = [row('a', 0, '180'), row('b', 1, '180'), row('c', 2, '0'), row('d', 3, '180'), row('e', 4, '180'), row('f', 5, ''), row('g', 90, '270')]
    const tags = new Map<string, number | null>([['a', 0], ['b', 0], ['c', 0], ['d', 0], ['e', 0], ['f', 90], ['g', 0]])
    const out = oneTurn(rows, new Map([...wide(rows)].map(([k]) => [k, k !== 'f'])), tags)
    // f shows standing because its tag turns it a quarter: in the camera frame it needs 180 too, so 90 from how it shows.
    expect(out.map((r) => r.rotChosen)).toEqual(['180', '180', '180', '180', '180', '90', '180'])
    expect(out.every((r) => r.review === '')).toBe(true)
  })
  it('a batch of sideways envelopes stays as shot, landscape, flagged', () => {
    const rows = [row('a', 0, '270'), row('b', 1, '270'), row('c', 2, '90')]
    const out = oneTurn(rows, wide(rows), zero(rows))
    expect(out.map((r) => [r.rotChosen, r.review])).toEqual([['0', 'turn-unsure'], ['0', 'turn-unsure'], ['0', 'turn-unsure']])
  })
  it('a split batch is flagged; manual turns and unknown shapes are left alone', () => {
    const rows = [row('a', 0, '180'), row('b', 1, '0'), row('c', 2, '180'), { ...row('d', 3, '90'), rotSource: 'manual' as const }, row('e', 4, '0')]
    const out = oneTurn(rows, new Map([...wide(rows)].map(([k]) => [k, k === 'e' ? null : true])), zero(rows))
    expect(out.slice(0, 3).map((r) => [r.rotChosen, r.review])).toEqual([['180', 'turn-unsure'], ['180', 'turn-unsure'], ['180', 'turn-unsure']])
    expect(out[3].rotChosen).toBe('90')
    expect(out[4].rotChosen).toBe('0')
  })
  it('the wings settle what the envelopes leave unsure, and win over envelopes laid down turned', () => {
    // envelopes sideways (no say), the camera's tags random: the wings ask for 180 in the camera frame
    const rows = [row('a', 0, '270'), row('b', 1, '90'), row('c', 2, ''), row('d', 3, '')]
    const tags = new Map<string, number | null>([['a', 0], ['b', 180], ['c', 0], ['d', 180]])
    const wings = new Map<string, number | null>([['a', 180], ['b', 0], ['c', 180], ['d', null]])
    const out = oneTurn(rows, wide(rows), tags, wings)
    expect(out.map((r) => [r.rotChosen, r.review])).toEqual([['180', ''], ['0', ''], ['180', ''], ['0', '']])
    // three envelopes read upright, the wings of four photos upside down: the wings
    const turned = [row('a', 0, '0'), row('b', 1, '0'), row('c', 2, '0'), row('d', 3, '')]
    const all180 = new Map<string, number | null>(turned.map((r) => [r.from, 180]))
    expect(oneTurn(turned, wide(turned), zero(turned), all180).map((r) => r.rotChosen)).toEqual(['180', '180', '180', '180'])
  })
})

describe('oneTurnByCamera', () => {
  it('gives the photos of each camera their own turn', () => {
    const rows = ['a', 'b', 'c', 'd', 'e', 'f'].map((n, i): PhotoRow => ({ ...row(n, i, ''), rotSource: '' }))
    const wide = new Map<string, boolean | null>(rows.map((r) => [r.from, true]))
    const tags = new Map<string, number | null>(rows.map((r) => [r.from, 0]))
    // four photos of one camera upright, two of another upside down (as the wings say)
    const wings = new Map<string, number | null>([['a', 0], ['b', 0], ['c', 0], ['d', 0], ['e', 180], ['f', 180]])
    const camera = new Map([['a', 'Canon'], ['b', 'Canon'], ['c', 'Canon'], ['d', 'Canon'], ['e', 'OLYMPUS TG-6'], ['f', 'OLYMPUS TG-6']])
    const out = oneTurnByCamera(rows, wide, tags, camera, wings)
    expect(out.map((r) => r.rotChosen)).toEqual(['0', '0', '0', '0', '180', '180'])
    // two photos are too few to be sure of a turn: left to look at (the wing check clears it photo by photo)
    expect(out.map((r) => r.review)).toEqual(['', '', '', '', 'turn-unsure', 'turn-unsure'])
  })
})
