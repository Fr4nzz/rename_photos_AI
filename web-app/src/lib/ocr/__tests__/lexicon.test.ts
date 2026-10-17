import { describe, expect, it } from 'vitest'
import { ctcLogProb, editDistance, lexiconMargin, lexiconPosteriors, MARGIN_MIN, type IdView } from '../lexicon'
import { decide, type PhotoInput } from '../decide'
import type { PhotoReading } from '../pipeline'

const COLS = ['_', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'A', 'M']

/** Posteriors from one {char: p} per step ('_' = blank); unnamed columns 0. */
const post = (steps: Record<string, number>[]) => {
  const p = new Float32Array(steps.length * COLS.length)
  steps.forEach((s, t) => { for (const [c, v] of Object.entries(s)) p[t * COLS.length + COLS.indexOf(c)] = v })
  return p
}
/** A clean reading of `text` ('_' for a blank step), with the step at `at` split between two digits. */
const view = (id: string, text: string, at = -1, split: [string, number, string, number] = ['3', 1, '8', 0]): IdView => {
  const steps = [...text].map((c, t) => (t === at ? { [split[0]]: split[1], [split[2]]: split[3] } : { [c]: 1 }))
  return { id, steps: steps.length, probs: post(steps) }
}

/** Reference: the CTC forward algorithm in log space (no rescaling). */
function refLogProb(p: Float32Array, steps: number, id: string): number {
  const lse = (...xs: number[]) => { const m = Math.max(...xs); return m === -Infinity ? m : m + Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0)) }
  const ext = [0]
  for (const c of id) ext.push(COLS.indexOf(c), 0)
  const lp = (t: number, s: number) => Math.log(p[t * COLS.length + ext[s]])
  let a = ext.map((_, s) => (s < 2 ? lp(0, s) : -Infinity))
  for (let t = 1; t < steps; t++) {
    a = ext.map((e, s) => lse(a[s], s >= 1 ? a[s - 1] : -Infinity, s >= 3 && s % 2 === 1 && e !== ext[s - 2] ? a[s - 2] : -Infinity) + lp(t, s))
  }
  return lse(a[a.length - 1], a[a.length - 2])
}

describe('ctcLogProb', () => {
  it('matches the log-space forward algorithm on random posteriors', () => {
    let seed = 7
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    const T = 30
    const p = new Float32Array(T * COLS.length)
    for (let t = 0; t < T; t++) {
      const row = COLS.map(() => rand() ** 3)
      const sum = row.reduce((a, b) => a + b)
      row.forEach((v, k) => { p[t * COLS.length + k] = v / sum })
    }
    for (const id of ['CAM036800', 'CAM111111', 'CAM078740']) expect(ctcLogProb(p, T, id)).toBeCloseTo(refLogProb(p, T, id), 9)
  })
  it('gives 0 to the only possible label and about -690 to an impossible one', () => {
    const v = view('CAM036800', 'CAM03680_0')
    expect(ctcLogProb(v.probs, v.steps, 'CAM036800')).toBeCloseTo(0, 9)
    expect(ctcLogProb(v.probs, v.steps, 'CAM036801')).toBeLessThan(-600)
  })
})

describe('lexiconPosteriors', () => {
  it('keeps blank, the digits and C, A, M in the scorer order', () => {
    const chars = ['blank', 'A', '0', 'C', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'M', 'x']
    const full = Float32Array.from(chars.map((_, k) => k / 100))
    const red = lexiconPosteriors(full, 1, chars.length, chars)
    expect([...red].map((v) => Math.round(v * 100))).toEqual([0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 1, 13])
  })
})

describe('lexiconMargin', () => {
  // "CAM036800": the 3 (step 4) is read 0.55 / 0.45 against an 8
  const text = 'CAM03680_0' // reads CAM036800
  const doubtful = view('CAM036800', text, 4, ['3', 0.55, '8', 0.45])
  it('is log P(read) - log P(best other database ID within 2 edits)', () => {
    const r = lexiconMargin([doubtful], 'CAM036800', new Set(['CAM036800', 'CAM086800', 'CAM036811', 'CAM555555']))
    expect(r!.rival).toBe('CAM086800')
    expect(r!.margin).toBeCloseTo(Math.log(0.55 / 0.45), 5)
    expect(r!.margin).toBeLessThan(MARGIN_MIN)
  })
  it('counts a rival 60 nats down when no other database ID is within 2 edits', () => {
    const r = lexiconMargin([doubtful], 'CAM036800', new Set(['CAM036800', 'CAM086888', 'CAM999999', 'not an id']))
    expect(r!.rival).toBeNull()
    expect(r!.margin).toBeCloseTo(Math.log(0.55) + 60, 5)
  })
  it('compares only IDs within 2 edits, insertions and deletions included', () => {
    expect(editDistance('036800', '368000')).toBe(2)
    const r = lexiconMargin([doubtful], 'CAM036800', new Set(['CAM036800', 'CAM368000', 'CAM086811']))
    expect(r!.rival).toBe('CAM368000')
  })
  it('takes the best of the views that read the ID, and none for a joined ID', () => {
    const clean = view('CAM036800', text)
    const known = new Set(['CAM036800', 'CAM086800'])
    expect(lexiconMargin([doubtful, clean], 'CAM036800', known)!.margin).toBeGreaterThan(600)
    expect(lexiconMargin([view('CAM086800', text)], 'CAM036800', known)).toBeNull()
  })
})

describe('decide with the margin', () => {
  const text = 'CAM03680_0' // reads CAM036800
  const photo = (name: string, id: string, t: number, idViews?: IdView[]): PhotoInput => {
    const reading: PhotoReading = { camid: id, conf: 0.99, candidates: [{ id, conf: 0.99 }], source: 'envelope', envelope: null,
      turned: 0, lines: [{ box: [], text: id, conf: 0.99 }], size: [1600, 1067], idViews }
    return { name, reading, capturedAt: t }
  }
  const known = new Set(['CAM036799', 'CAM036800', 'CAM086800', 'CAM036801'])
  const batch = (views?: IdView[]) => [photo('a', 'CAM036799', 1), photo('b', 'CAM036800', 2, views), photo('c', 'CAM036801', 3)]
  it('sends a read ID with a close, almost as likely database ID to review, that ID suggested', () => {
    const d = decide(batch([view('CAM036800', text, 4, ['3', 0.55, '8', 0.45])]), known)
    expect(d[1]).toMatchObject({ camid: null, auto: false, reasons: ['uncertain-read'], prefill: 'CAM036800' })
    expect(d[1].candidates[0]).toBe('CAM086800')
    expect(d[0].auto && d[2].auto).toBe(true)
  })
  it('renames when the read ID is clearly more likely', () => {
    expect(decide(batch([view('CAM036800', text, 4, ['3', 0.9, '8', 0.1])]), known)[1].auto).toBe(true)
  })
  it('has no margin without the database or the posteriors', () => {
    const views = [view('CAM036800', text, 4, ['3', 0.55, '8', 0.45])]
    expect(decide(batch(views), null)[1].camid).toBe('CAM036800')
    expect(decide(batch(), known)[1].auto).toBe(true)
  })
})
