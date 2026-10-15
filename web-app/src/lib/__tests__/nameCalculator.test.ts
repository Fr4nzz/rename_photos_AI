import { describe, expect, it } from 'vitest'
import type { PhotoRow } from '@/types'
import { calculateFinalNames, viewsFromWings } from '../nameCalculator'

const row = (n: number, id: string, extra: Partial<PhotoRow> = {}): PhotoRow => ({
  from: `IMG_${n}.JPG`, currentPath: `IMG_${n}.JPG`, photoId: n, mainValue: id, co: '', n: '', skip: '', to: '', suffix: '',
  batchNumber: 0, captureDate: null, status: 'Original', review: '', suggest: '', rotSuggested: '', rotChosen: '', rotSource: '', rotApplied: '', ...extra,
})
const views = (rows: PhotoRow[]) => rows.map((r) => `${r.suffix}${r.viewBy ? `(${r.viewBy})` : ''}`)

describe('the views of an ID\'s photos', () => {
  it('follow the order taken without wings, and the wings when they settle them', () => {
    expect(views(calculateFinalNames([row(1, 'CAM1'), row(2, 'CAM1')], 'CAM', 'Standard', ''))).toEqual(['d', 'v'])
    // shot ventral side first
    const ventralFirst = [row(1, 'CAM1', { pDorsal: '0.01' }), row(2, 'CAM1', { pDorsal: '0.99' })]
    const named = calculateFinalNames(ventralFirst, 'CAM', 'Standard', '')
    expect(views(named)).toEqual(['v(wings)', 'd(wings)'])
    expect(named.map((r) => r.to)).toEqual(['CAM1v.JPG', 'CAM1d.JPG'])
    // the wings agreeing with the order: nothing to say
    expect(views(calculateFinalNames([row(1, 'CAM1', { pDorsal: '0.98' }), row(2, 'CAM1', { pDorsal: '0.03' })], 'CAM', 'Standard', ''))).toEqual(['d', 'v'])
  })

  it('a CRISPR set: dorsal ones d, d2, ventral ones v, v2, whatever the order', () => {
    const rows = [row(1, 'CAM2', { pDorsal: '0.99' }), row(2, 'CAM2', { pDorsal: '0.98' }), row(3, 'CAM2', { pDorsal: '0.02' }), row(4, 'CAM2', { pDorsal: '0.01' })]
    expect(calculateFinalNames(rows, 'CAM', 'Standard', '').map((r) => r.suffix)).toEqual(['d', 'd2', 'v', 'v2'])
    // one side unclear: the order taken
    expect(viewsFromWings([{ pDorsal: '0.99' }, { pDorsal: '0.6' }, { pDorsal: '0.01' }])).toBeNull()
    // both photos of one side: the order (the wing check flags it)
    expect(viewsFromWings([{ pDorsal: '0.99' }, { pDorsal: '0.97' }])).toBeNull()
  })

  it('a view typed stays, the ID\'s other photos take the views left; not on wing clips', () => {
    const rows = [row(1, 'CAM3', { suffix: 'v', viewBy: 'manual' }), row(2, 'CAM3', { pDorsal: '0.01' }), row(3, 'CAM3')]
    expect(views(calculateFinalNames(rows, 'CAM', 'Standard', ''))).toEqual(['v(manual)', 'd', 'd2'])
    const clips = [row(1, 'CAM4', { pDorsal: '0.01' }), row(2, 'CAM4', { pDorsal: '0.99' })]
    expect(calculateFinalNames(clips, 'CAM', 'Wing Clips', '').map((r) => r.suffix)).toEqual(['v1', 'v2'])
  })
})
