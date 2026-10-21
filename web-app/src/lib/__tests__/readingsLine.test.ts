import { describe, expect, it } from 'vitest'
import type { PhotoRow } from '@/types'
import { readingsLine } from '../readingsLine'
import type { PhotoReading } from '../ocr/pipeline'
import type { WingCheck } from '../wings/checks'
import { POSITIONS, type Position, type Side, type Wing } from '../wings/model'

const row = (suffix: string, extra: Partial<PhotoRow> = {}): PhotoRow => ({
  from: 'a.JPG', currentPath: 'a.JPG', photoId: 1, mainValue: 'CAM000001', co: '', n: '', skip: '', to: '', suffix, batchNumber: 0,
  captureDate: null, status: 'Original', review: '', suggest: '', rotSuggested: '', rotChosen: '', rotSource: '', rotApplied: '', ...extra,
})
const sideOf = (p: number): Side => (p >= 0.95 ? 'dorsal' : p <= 0.05 ? 'ventral' : 'unsure')
/** A photo's check: P(dorsal) of its forewings and hindwings, its wings (all four unless some are missing), its turn. */
function check(fore: number, hind: number, extra: Partial<WingCheck> = {}, missing: Position[] = []): WingCheck {
  const wings = POSITIONS.filter((p) => !missing.includes(p)).map((p): Wing => ({ box: [0, 0, 1, 1], pDorsal: p.startsWith('fore') ? fore : hind, pForewing: p.startsWith('fore') ? 1 : 0, kind: p.startsWith('fore') ? 'forewing' : 'hindwing', half: p.endsWith('left') ? 'left' : 'right' }))
  const both = (fore + hind) / 2
  const mixed = sideOf(fore) !== sideOf(hind) && sideOf(fore) !== 'unsure' && sideOf(hind) !== 'unsure'
  return {
    side: mixed ? 'mixed' : sideOf(both), p: both, fore: { side: sideOf(fore), p: fore }, hind: { side: sideOf(hind), p: hind },
    wings, missing, turnK: 0, pTurn: 0.97, size: [1600, 1067], applied: 0, ...extra,
  }
}
const reading = (camid: string, conf: number) => ({ camid, conf }) as PhotoReading
const text = (l: ReturnType<typeof readingsLine>) => l!.parts.map((p) => p.text).join(' · ')
const warned = (l: ReturnType<typeof readingsLine>) => l!.parts.filter((p) => p.warn).map((p) => p.text)

describe('the readings line under a photo', () => {
  it('nothing before the photo is read', () => {
    expect(readingsLine(row('d'), undefined, undefined, 'Standard')).toBeNull()
  })

  it('a photo as named and turned: every part grey', () => {
    const l = readingsLine(row('d'), check(0.995, 0.99), reading('CAM000001', 0.938), 'Standard')
    expect(text(l)).toBe('dorsal 99 % · 4 wings · upright 97 % · ID 0.93')
    expect(warned(l)).toEqual(['ID 0.93'])
    expect(warned(readingsLine(row('v'), check(0.01, 0.02), reading('CAM000001', 0.99), 'Standard'))).toEqual([])
    expect(text(readingsLine(row('v'), check(0.01, 0.02), undefined, 'Standard'))).toBe('ventral 98 % · 4 wings · upright 97 %')
  })

  it('the side against the name, mixed, unsure and no wings', () => {
    expect(warned(readingsLine(row('d'), check(0.01, 0.02), undefined, 'Standard'))).toEqual(['ventral 98 %'])
    expect(warned(readingsLine(row('v'), check(0.99, 0.01), undefined, 'Standard'))).toEqual(['mixed sides'])
    expect(text(readingsLine(row('d'), check(0.3, 0.3), undefined, 'Standard'))).toMatch(/^unsure \(ventral 70 %\)/)
    expect(warned(readingsLine(row('d'), check(0.3, 0.3), undefined, 'Standard'))).toEqual([])
    expect(text(readingsLine(row('d'), check(0.99, 0.99, { wings: [], missing: POSITIONS }), undefined, 'Standard'))).toBe('no wings')
    // a skipped photo is renamed by nothing: nothing to disagree with
    expect(warned(readingsLine(row('d', { skip: 'x' }), check(0.01, 0.02), undefined, 'Standard'))).toEqual([])
  })

  it('a missing wing, and each wing in the title', () => {
    const l = readingsLine(row('d'), check(0.99, 0.97, {}, ['hindwing_right']), undefined, 'Standard')!
    expect(text(l)).toContain('3 of 4 wings')
    expect(l.title).toContain('left forewing: P(dorsal) 0.99')
    expect(l.title).toContain('left hindwing: P(dorsal) 0.97')
    expect(l.title).toContain('Missing: right hindwing')
  })

  it('the turn as the card shows the photo, against the rotation that will be written', () => {
    // the wings see the photo turned a quarter counter-clockwise: 90° clockwise chosen puts it upright
    const k1 = check(0.99, 0.99, { turnK: 1, pTurn: 0.94 })
    expect(text(readingsLine(row('d', { rotChosen: '90' }), k1, undefined, 'Standard'))).toContain('upright 94 %')
    const off = readingsLine(row('d'), k1, undefined, 'Standard')
    expect(warned(off)).toEqual(['turned 90° ccw 94 %'])
    // already written to the file when checked: the turn was in the file
    expect(warned(readingsLine(row('d', { rotChosen: '90', rotApplied: '90' }), { ...k1, applied: 90, turnK: 0 }, undefined, 'Standard'))).toEqual([])
    expect(warned(readingsLine(row('d', { rotChosen: '180' }), check(0.99, 0.99), undefined, 'Standard'))).toEqual(['upside down 97 %'])
    // the wings' turn is not used when unsure: shown, not marked
    expect(warned(readingsLine(row('d'), { ...k1, pTurn: 0.6 }, undefined, 'Standard'))).toEqual([])
  })

  it('the ID confidence only for the ID the photo has', () => {
    expect(text(readingsLine(row('d', { mainValue: 'CAM000002' }), check(0.99, 0.99), reading('CAM000001', 0.99), 'Standard'))).not.toContain('ID')
    expect(readingsLine(row('d', { mainValue: 'CAM000002' }), undefined, reading('CAM000001', 0.99), 'Standard')).toBeNull()
    expect(text(readingsLine(row('d'), undefined, reading('CAM000001', 0.999), 'Standard'))).toBe('ID 0.99')
  })
})
