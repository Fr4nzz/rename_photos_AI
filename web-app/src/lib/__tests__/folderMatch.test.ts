import { describe, expect, it } from 'vitest'
import type { FileEntry, PhotoRow } from '@/types'
import { matchFolder } from '../folderMatch'

const row = (from: string, currentPath = from): PhotoRow => ({
  from, currentPath, photoId: 1, mainValue: '', co: '', n: '', skip: '', to: '', suffix: '', batchNumber: 0,
  captureDate: null, status: 'Original', review: '', suggest: '', rotSuggested: '', rotChosen: '', rotSource: '', rotApplied: '',
})
const file = (name: string): FileEntry => ({ name, path: name, file: new File([], name), extension: name.slice(name.lastIndexOf('.')).toLowerCase() })

describe('the results against the folder', () => {
  it('finds the photos gone and those added, a RAW going with its photo', () => {
    const rows = [row('IMG_1.JPG', 'CAM1d.JPG'), row('IMG_2.JPG'), row('IMG_3.JPG')]
    const files = ['CAM1d.JPG', 'CAM1d.CR2', 'IMG_2.JPG', 'IMG_2.CR2', 'IMG_9.JPG', 'IMG_9.CR2', 'IMG_10.CR2'].map(file)
    const { missing, unread } = matchFolder(rows, files)
    expect(missing.map((r) => r.from)).toEqual(['IMG_3.JPG'])
    expect(unread.map((f) => f.name)).toEqual(['IMG_9.JPG', 'IMG_10.CR2'])
  })
  it('says nothing without a folder', () => {
    expect(matchFolder([row('IMG_1.JPG')], [])).toEqual({ missing: [], unread: [] })
  })
})
