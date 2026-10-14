import { describe, expect, it } from 'vitest'
import type { PhotoRow } from '@/types'
import { groupSpecimens, specimenNote } from '../specimens'

const row = (n: number, id: string, extra: Partial<PhotoRow> = {}): PhotoRow => ({
  from: `IMG_${n}.JPG`, currentPath: `IMG_${n}.JPG`, photoId: n, mainValue: id, co: '', n: '', skip: '', to: '', suffix: '',
  batchNumber: 0, captureDate: `2026-07-20T10:${String(n).padStart(2, '0')}:00.000Z`, status: 'Original', review: '', suggest: '',
  rotSuggested: '', rotChosen: '', rotSource: '', rotApplied: '', ...extra,
})

describe('the photos of a batch by specimen', () => {
  const rows = [
    row(1, 'CAM077001'), row(2, 'CAM077001'),
    row(3, 'CAM077002'),
    row(4, 'CAM077003'), row(5, 'CAM077003', { review: 'out-of-sequence' }),
    row(6, ''), row(7, ''),
    row(8, 'CAM077004'), row(9, 'CAM077004'), row(10, 'CAM077004', { skip: 'x' }),
    row(11, 'CAM077001'),
  ]
  it('groups them by ID in the order taken, with what each still needs', () => {
    const groups = groupSpecimens([...rows].reverse())
    expect(groups.map((g) => [g.id, g.rows.map((r) => r.photoId), g.state, specimenNote(g)])).toEqual([
      // a third photo of an ID taken later joins its specimen: three photos to look at
      ['CAM077001', [1, 2, 11], 'count', '3 photos'],
      ['CAM077002', [3], 'count', 'Only 1 photo'],
      ['CAM077003', [4, 5], 'review', 'Needs review'],
      // photos without an ID are not one specimen
      ['', [6], 'no-id', 'No ID'],
      ['', [7], 'no-id', 'No ID'],
      // a skipped photo does not count
      ['CAM077004', [8, 9, 10], 'ok', '2 photos'],
    ])
    expect(groups.map((g) => g.key)).toEqual([1, 3, 4, 6, 7, 8])
  })
  it('counts photos only in the standard naming (wing clips have as many as there are)', () => {
    expect(groupSpecimens(rows, 'Wing Clips').map((g) => g.state)).toEqual(['ok', 'ok', 'review', 'no-id', 'no-id', 'ok'])
  })
})
