import { describe, expect, it } from 'vitest'
import { restoreSidecar, saveSidecar } from '../reviewSidecar'
import { useProcessingStore } from '@/stores/processingStore'
import type { PhotoReading } from '../ocr/pipeline'
import type { PhotoRow } from '@/types'

// A folder opened again resumes as it was left: what the CSV does not hold is kept beside it.

/** A folder in memory, as much of the File System Access API as the app uses. */
function folder() {
  const files = new Map<string, string>()
  const dir = (): FileSystemDirectoryHandle =>
    ({
      getDirectoryHandle: async () => dir(),
      getFileHandle: async (name: string, opts?: { create?: boolean }) => {
        if (!files.has(name) && !opts?.create) throw new DOMException('missing', 'NotFoundError')
        return {
          createWritable: async () => ({ write: async (text: string) => void files.set(name, text), close: async () => {} }),
          getFile: async () => ({ text: async () => files.get(name) ?? '' }),
        }
      },
    }) as unknown as FileSystemDirectoryHandle
  return { dir: dir(), files }
}
const row = (from: string, extra: Partial<PhotoRow> = {}): PhotoRow => ({
  from, currentPath: from, photoId: 1, mainValue: 'CAM000001', co: '', n: '', skip: '', to: '', suffix: '', batchNumber: 0,
  captureDate: null, status: 'Original', review: '', suggest: '', rotSuggested: '', rotChosen: '', rotSource: '', rotApplied: '', ...extra,
})

describe('the review kept beside its CSV', () => {
  it("keeps the reader's results, the photos' times and the assistant's changes, and puts them back", async () => {
    const { dir, files } = folder()
    const reading = { camid: 'CAM000001', conf: 0.97, candidates: [], source: 'envelope', envelope: [1, 2, 3, 4], turned: 0, lines: [], size: [1600, 1067] } as PhotoReading
    const store = useProcessingStore.getState()
    store.setPhotoRows([row('IMG_1.JPG', { captureDate: '2026-10-14T10:00:00.000Z' }), row('IMG_2.JPG')])
    store.setOcrReadings(new Map([['IMG_1.JPG', reading]]))
    store.clearHostChanges()
    store.addHostChanges([{ file: 'IMG_2.JPG', what: ['id'], before: { mainValue: 'CAM000007' }, id: { from: 'CAM000007', to: 'CAM000001' } }], 'Claude')
    await saveSidecar(dir, 'batch.csv')
    expect([...files.keys()]).toEqual(['batch.csv.review.json'])

    // Closed and opened again: the store empty, the rows from the CSV.
    store.setOcrReadings(new Map())
    store.clearHostChanges()
    const rows = await restoreSidecar(dir, 'batch.csv', [row('IMG_1.JPG'), row('IMG_2.JPG')])
    const now = useProcessingStore.getState()
    expect(rows[0].captureDate).toBe('2026-10-14T10:00:00.000Z')
    expect(now.ocrReadings.get('IMG_1.JPG')?.conf).toBe(0.97)
    expect(now.hostChanges.get('IMG_2.JPG')?.id).toEqual({ from: 'CAM000007', to: 'CAM000001' })
    expect(now.hostName).toBe('Claude')
  })
  it('without anything kept, the rows stay as they are and no earlier changes are left over', async () => {
    const { dir } = folder()
    useProcessingStore.getState().addHostChanges([{ file: 'x', what: ['id'], before: {} }], 'Claude')
    const rows = [row('IMG_1.JPG')]
    expect(await restoreSidecar(dir, 'other.csv', rows)).toBe(rows)
    expect(useProcessingStore.getState().hostChanges.size).toBe(0)
  })
})
