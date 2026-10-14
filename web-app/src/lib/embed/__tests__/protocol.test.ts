import { describe, expect, it } from 'vitest'
import type { PhotoRow } from '@/types'
import { applyEdits, snapshotRows, undoEdits, type HostChange } from '../protocol'
import { calculateFinalNames } from '../../nameCalculator'

const names = { mainColumn: 'CAM', suffixMode: 'Standard' as const, customSuffixes: '' }
const row = (n: number, id: string, extra: Partial<PhotoRow> = {}): PhotoRow => ({
  from: `IMG_${n}.JPG`, currentPath: `IMG_${n}.JPG`, photoId: n, mainValue: id, co: '', n: '', skip: '', to: '', suffix: '',
  batchNumber: 0, captureDate: `2026-07-20T10:${String(n).padStart(2, '0')}:00.000Z`, status: 'Original', review: '', suggest: '',
  rotSuggested: '0', rotChosen: '0', rotSource: 'ocr', rotApplied: '', ...extra,
})
const batch = () => calculateFinalNames([
  row(1, 'CAM077001'), row(2, 'CAM077001'),
  row(3, 'CAM077002'), row(4, 'CAM077008', { review: 'out-of-sequence', suggest: 'CAM077002 CAM077003' }),
  row(5, 'CAM077003'), row(6, 'CAM077003'), row(7, 'CAM077003'),
  row(8, '', { review: 'no-reading' }),
], 'CAM', 'Standard', '')

describe('the batch as its host sees it', () => {
  it('names each photo by its file, with its ID, view, flags and rotation', () => {
    const rows = snapshotRows(batch(), new Map(), new Map())
    expect(rows[3]).toMatchObject({ file: 'IMG_4.JPG', n: 4, id: 'CAM077008', view: 'd', to: 'CAM077008d.JPG', review: ['out-of-sequence'], suggest: ['CAM077002', 'CAM077003'], rotation: 0, rotationPending: 0 })
    expect(rows[6]).toMatchObject({ id: 'CAM077003', view: 'd2', to: 'CAM077003d2.JPG' })
    expect(rows[7]).toMatchObject({ id: '', to: '', review: ['no-reading'] })
  })
})

describe("the host's corrections", () => {
  it('sets an ID (the photo is confirmed and the names follow), skips, turns and leaves a photo for a person', () => {
    const before = batch()
    const { rows, changes, problems } = applyEdits(before, [
      { file: 'IMG_4.JPG', id: 'cam077002' },
      { file: 'img_7.jpg', skip: true },
      { file: 'IMG_8.JPG', leave: 'the envelope is out of the frame' },
      { file: 'IMG_1.JPG', rotate: 180 },
      { file: 'IMG_99.JPG', id: 'CAM077009' },
      { file: 'IMG_2.JPG', rotate: 45 },
    ], names)
    expect(rows[3]).toMatchObject({ mainValue: 'CAM077002', review: '', suffix: 'v', to: 'CAM077002v.JPG' })
    expect(rows[6]).toMatchObject({ skip: 'x', to: '' })
    expect(rows[7]).toMatchObject({ review: 'no-reading,assistant', n: 'the envelope is out of the frame' })
    expect(rows[0]).toMatchObject({ rotChosen: '180', rotSource: 'manual' })
    expect(changes.map((c) => [c.file, c.what])).toEqual([
      ['IMG_4.JPG', ['ID CAM077008 → CAM077002', 'confirmed']],
      ['IMG_7.JPG', ['skipped']],
      ['IMG_8.JPG', ['left for you', 'note']],
      ['IMG_1.JPG', ['turned 180°']],
    ])
    expect(problems).toEqual([
      { file: 'IMG_99.JPG', error: 'no photo of that name in the batch' },
      { file: 'IMG_2.JPG', error: 'rotate is 90, 180 or 270 (clockwise)' },
    ])
    // the rows given are left as they were
    expect(before[3].mainValue).toBe('CAM077008')

    // taken back: every row as before, flags and names too
    expect(undoEdits(rows, changes, names)).toEqual(before)
    const shown = snapshotRows(rows, new Map(), new Map(changes.map((c): [string, HostChange] => [c.file, c])))
    expect(shown[3].byHost).toEqual(['ID CAM077008 → CAM077002', 'confirmed'])
    expect(shown[1].byHost).toBeUndefined()
  })

  it('keeps a photo flagged when asked, and sets a view by hand', () => {
    const { rows, changes } = applyEdits(batch(), [
      { file: 'IMG_4.JPG', id: 'CAM077002', confirm: false },
      { file: 'IMG_6.JPG', view: 'v' },
      { file: 'IMG_7.JPG', view: 'v2' },
    ], names)
    expect(rows[3]).toMatchObject({ mainValue: 'CAM077002', review: 'out-of-sequence' })
    expect(rows[5]).toMatchObject({ suffix: 'v', to: 'CAM077003v.JPG' })
    expect(rows[6]).toMatchObject({ suffix: 'v2', to: 'CAM077003v2.JPG' })
    expect(changes.map((c) => c.what)).toEqual([['ID CAM077008 → CAM077002'], ['view v2']])
  })

  it('takes two rounds of changes back to the first state', () => {
    const before = batch()
    const one = applyEdits(before, [{ file: 'IMG_4.JPG', id: 'CAM077003' }], names)
    const two = applyEdits(one.rows, [{ file: 'IMG_4.JPG', id: 'CAM077002' }, { file: 'IMG_3.JPG', skip: true }], names)
    expect(undoEdits(two.rows, [...one.changes, ...two.changes], names)).toEqual(before)
  })
})
