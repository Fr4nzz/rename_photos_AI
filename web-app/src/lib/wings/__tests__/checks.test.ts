import { describe, expect, it } from 'vitest'
import type { PhotoRow } from '@/types'
import { wingFindings, wingTurn, withWingReasons, type WingCheck } from '../checks'
import { POSITIONS, type Side, type Wing } from '../model'

const row = (from: string, id: string, suffix: string, extra: Partial<PhotoRow> = {}): PhotoRow => ({
  from, currentPath: from, photoId: 1, mainValue: id, co: '', n: '', skip: '', to: '', suffix, batchNumber: 0,
  captureDate: null, status: 'Original', review: '', suggest: '', rotSuggested: '', rotChosen: '', rotSource: '', rotApplied: '', ...extra,
})
const sideOf = (p: number): Side => (p >= 0.95 ? 'dorsal' : p <= 0.05 ? 'ventral' : 'unsure')
/** A photo's check: P(dorsal) of its forewings and hindwings, its wings (4 unless said), its turn. */
function check(fore: number, hind: number, extra: Partial<WingCheck> = {}): WingCheck {
  const wings = POSITIONS.map((p): Wing => ({ box: [0, 0, 1, 1], pDorsal: p.startsWith('fore') ? fore : hind, pForewing: p.startsWith('fore') ? 1 : 0, kind: p.startsWith('fore') ? 'forewing' : 'hindwing', half: p.endsWith('left') ? 'left' : 'right' }))
  const both = (fore + hind) / 2
  const mixed = sideOf(fore) !== sideOf(hind) && sideOf(fore) !== 'unsure' && sideOf(hind) !== 'unsure'
  return {
    side: mixed ? 'mixed' : sideOf(both), p: both, fore: { side: sideOf(fore), p: fore }, hind: { side: sideOf(hind), p: hind },
    wings, missing: [], turnK: 0, pTurn: 0.99, size: [1600, 1067], applied: 0, ...extra,
  }
}
const texts = (f: ReturnType<typeof wingFindings>, name: string) => (f.get(name) ?? []).map((x) => x.text)

describe('the wing check of a batch', () => {
  const pair = [row('a.JPG', 'CAM000001', 'd'), row('b.JPG', 'CAM000001', 'v')]

  it('says nothing of a pair named as its wings show', () => {
    const f = wingFindings(pair, new Map([['a.JPG', check(0.99, 0.99)], ['b.JPG', check(0.01, 0.02)]]), 'Standard')
    expect(f.size).toBe(0)
  })

  it('a pair whose photos are each unsure but clearly apart is settled by the pair', () => {
    const f = wingFindings(pair, new Map([['a.JPG', check(0.9, 0.9)], ['b.JPG', check(0.1, 0.1)]]), 'Standard')
    expect(f.size).toBe(0)
  })

  it('d and v the other way round', () => {
    const f = wingFindings(pair, new Map([['a.JPG', check(0.01, 0.01)], ['b.JPG', check(0.99, 0.99)]]), 'Standard')
    expect(texts(f, 'a.JPG')[0]).toMatch(/look swapped.*underside/)
    expect(texts(f, 'b.JPG')[0]).toMatch(/look swapped.*upper side/)
  })

  it('both photos of one side', () => {
    const f = wingFindings(pair, new Map([['a.JPG', check(0.99, 0.99)], ['b.JPG', check(0.98, 0.99)]]), 'Standard')
    expect(texts(f, 'b.JPG')).toEqual(['Both photos of this ID show the upper side of the wings'])
  })

  it('hindwings not turned over: both photos flagged once, not again as mixed', () => {
    const f = wingFindings(pair, new Map([['a.JPG', check(0.99, 0.99)], ['b.JPG', check(0.01, 0.99)]]), 'Standard')
    expect(texts(f, 'a.JPG')).toEqual(['Some wings were not turned over between the two photos: hindwings upper side up in both'])
    expect(texts(f, 'b.JPG')).toEqual(texts(f, 'a.JPG'))
  })

  it('a single photo named against its wings, and a mixed one', () => {
    const rows = [row('a.JPG', 'CAM000002', 'd'), row('c.JPG', 'CAM000003', 'v')]
    const f = wingFindings(rows, new Map([['a.JPG', check(0.01, 0.02)], ['c.JPG', check(0.99, 0.01)]]), 'Standard')
    expect(texts(f, 'a.JPG')).toEqual(['Named d but the wings show their underside'])
    expect(texts(f, 'c.JPG')[0]).toMatch(/^Mixed sides: forewings show their upper side, hindwings their underside/)
  })

  it('a wing fewer in one photo of a pair, and a photo without wings', () => {
    const three = check(0.01, 0.01)
    three.wings = three.wings.slice(0, 3); three.missing = ['hindwing_right']
    const rows = [...pair, row('e.JPG', 'CAM000004', 'd')]
    const f = wingFindings(rows, new Map([['a.JPG', check(0.99, 0.99)], ['b.JPG', three], ['e.JPG', { ...check(0.5, 0.5), wings: [] }]]), 'Standard')
    expect(texts(f, 'b.JPG')).toEqual(['3 wings here, 4 in the d photo (no right hindwing)'])
    expect(texts(f, 'a.JPG')).toEqual([])
    expect(texts(f, 'e.JPG')).toEqual(['No wing found'])
  })

  it('which way up: the wings against the turn still to be written', () => {
    const upside = { turnK: 2 }
    const rows = [
      row('a.JPG', 'CAM000001', 'd', { rotChosen: '180' }), // turned as the wings say
      row('b.JPG', 'CAM000001', 'v'), // not turned
      row('c.JPG', 'CAM000005', 'd', { rotChosen: '180', rotApplied: '180' }), // written after the check
    ]
    const checks = new Map([['a.JPG', check(0.99, 0.99, upside)], ['b.JPG', check(0.01, 0.01, upside)], ['c.JPG', check(0.99, 0.99, upside)]])
    const f = wingFindings(rows, checks, 'Standard')
    expect(texts(f, 'a.JPG')).toEqual([])
    expect(texts(f, 'b.JPG')).toEqual(['The wings look upside down: forewings should be on top (turn 180°)'])
    expect(texts(f, 'c.JPG')).toEqual([])
    expect(wingTurn(check(0.9, 0.9, { turnK: 1, pTurn: 0.6 }), rows[1])).toBeNull()
  })

  it('wing clips are not checked; skipped photos neither', () => {
    const checks = new Map([['a.JPG', check(0.01, 0.01)], ['b.JPG', check(0.99, 0.99)]])
    expect(wingFindings(pair, checks, 'Wing Clips').size).toBe(0)
    expect(wingFindings([pair[0], { ...pair[1], skip: 'x' }], new Map([['a.JPG', check(0.99, 0.99)], ['b.JPG', { ...check(0.5, 0.5), wings: [] }]]), 'Standard').size).toBe(0)
  })

  it("a finding's code joins the photo's reasons once", () => {
    const checks = new Map([['a.JPG', check(0.01, 0.01)], ['b.JPG', check(0.99, 0.99)]])
    const f = wingFindings(pair, checks, 'Standard')
    const rows = withWingReasons([{ ...pair[0], review: 'out-of-sequence' }, pair[1]], f, checks)
    expect(rows.map((r) => r.review)).toEqual(['out-of-sequence,wing-side', 'wing-side'])
    expect(withWingReasons(rows, f, checks)[0]).toBe(rows[0])
  })

  it('which way up is settled by wings that agree with the turn, not by unsure ones', () => {
    const rows = [{ ...pair[0], review: 'turn-unsure' }, { ...pair[1], review: 'turn-unsure,not-in-database' }]
    const checks = new Map([['a.JPG', check(0.99, 0.99)], ['b.JPG', check(0.01, 0.01, { pTurn: 0.6 })]])
    const out = withWingReasons(rows, wingFindings(rows, checks, 'Standard'), checks)
    expect(out.map((r) => r.review)).toEqual(['', 'turn-unsure,not-in-database'])
  })
})
