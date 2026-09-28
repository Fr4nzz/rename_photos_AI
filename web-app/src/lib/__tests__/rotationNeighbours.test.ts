import { describe, expect, it } from 'vitest'
import { fillFromNeighbours } from '../rotationPlan'
import type { PhotoRow } from '@/types'

const row = (from: string, minute: number, rot: string): PhotoRow => ({
  from, currentPath: from, photoId: minute, mainValue: '', co: '', n: '', skip: '', to: '', suffix: '', batchNumber: 0,
  captureDate: new Date(Date.UTC(2026, 0, 1, 10, minute)).toISOString(), status: 'Original', review: '', suggest: '',
  rotSuggested: rot, rotChosen: rot, rotSource: rot === '' ? '' : 'ocr', rotApplied: '',
})

describe('fillFromNeighbours', () => {
  it('uses the sensor frame, not the flipping orientation tag', () => {
    // same camera frame for all: photo 2's tag says 90 cw, so it needs 90 less than its neighbours
    const rows = [row('a', 0, '180'), row('b', 1, ''), row('c', 2, '180')]
    const tags = new Map<string, number | null>([['a', 0], ['b', 90], ['c', 0]])
    expect(fillFromNeighbours(rows, tags)[1]).toMatchObject({ rotChosen: '90', rotSource: 'neighbours' })
  })

  it('follows the nearer run at a change of setup and stays within the session', () => {
    const rows = [row('a', 0, '0'), row('b', 1, '0'), row('x', 2, ''), row('c', 3, '90'), row('d', 60, '270'), row('y', 61, '')]
    const tags = new Map<string, number | null>(rows.map((r) => [r.from, 0]))
    const out = fillFromNeighbours(rows, tags)
    expect(out[2].rotChosen).toBe('0') // two of three nearest read photos say 0
    expect(out[5].rotChosen).toBe('270') // only its own session counts
  })

  it('leaves manual choices and mirrored tags alone', () => {
    const rows = [row('a', 0, '90'), { ...row('b', 1, ''), rotSource: 'manual' as const, rotChosen: '180' }, row('c', 2, '')]
    const out = fillFromNeighbours(rows, new Map<string, number | null>([['a', 0], ['b', 0], ['c', null]]))
    expect(out[1].rotChosen).toBe('180')
    expect(out[2].rotChosen).toBe('')
  })
})
