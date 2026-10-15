import { describe, expect, it } from 'vitest'
import { photosToRead, rowsWithRaws } from '../selection'
import type { FileEntry, PhotoRow } from '@/types'

const entry = (name: string): FileEntry => ({ name, path: name, file: new File([], name), extension: name.slice(name.lastIndexOf('.')).toLowerCase() })

describe('RAW files with a JPG', () => {
  const files = ['IMG_5350.CR2', 'IMG_5350.JPG', 'IMG_5351.cr2', 'IMG_5352.JPG'].map(entry)

  it('are not read: their JPG is', () => {
    expect(photosToRead(files).map((f) => f.name)).toEqual(['IMG_5350.JPG', 'IMG_5351.cr2', 'IMG_5352.JPG'])
  })

  it('show their JPG reading and new name', () => {
    const row = { from: 'IMG_5350.JPG', currentPath: 'CAM079894v.JPG', mainValue: 'CAM079894' } as PhotoRow
    const rows = rowsWithRaws([row], files)
    expect(rows.get('IMG_5350.CR2')).toMatchObject({ mainValue: 'CAM079894', currentPath: 'CAM079894v.CR2' })
    expect(rows.has('IMG_5351.cr2')).toBe(false)
  })
})
