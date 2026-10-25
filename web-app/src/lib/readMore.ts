/**
 * Photos added to a folder after its reading: read them (the IDs, the wings, their turn) and add
 * them to the batch, the rows already there left as they are.
 */
import { toast } from 'sonner'
import type { FileEntry, PhotoRow } from '@/types'
import { useProcessingStore } from '@/stores/processingStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { runLocalOcr } from './ocr/runLocal'
import { calculateFinalNames } from './nameCalculator'
import { appendRotationLog, listFolder, writeRotations } from './rotationWrite'
import { applyWingChecks, checkPhotos } from './wings/runWings'
import { wingTurn } from './wings/checks'
import { planTurnsByCamera } from './rotationPlan'
import { saveBatch } from './saveBatch'
import { getErrorMessage } from './errors'

export async function readMore(entries: FileEntry[]): Promise<void> {
  const s = useProcessingStore.getState()
  const settings = useSettingsStore.getState()
  if (!entries.length || s.isProcessing) return
  const controller = new AbortController()
  s.setProcessing(true)
  s.setProgress(0, `Reading ${entries.length} added photo(s)…`)
  try {
    const first = Math.max(0, ...s.photoRows.map((r) => r.photoId)) + 1
    const base: PhotoRow[] = entries.map((e, i) => ({
      from: e.name, currentPath: e.name, photoId: first + i, mainValue: '', co: '', n: '', skip: '', to: '', suffix: '', batchNumber: 0,
      captureDate: null, status: 'New', review: '', suggest: '', rotSuggested: '', rotChosen: '', rotSource: '', rotApplied: '',
    }))
    s.setFileMap(new Map([...useProcessingStore.getState().fileMap, ...entries.map((e) => [e.name, e.file] as const)]))
    const result = await runLocalOcr(entries, base, (done, total) => s.setProgress(Math.floor((done / total) * 50), `Read ${done}/${total} added photos`), controller.signal)
    const withWings = settings.suffixMode !== 'Wing Clips'
    const checks = withWings
      ? await checkPhotos(result.unturned, (done, total) => s.setProgress(50 + Math.floor((done / total) * 50), `Checked the wings of ${done}/${total} added photos`), controller.signal)
      : new Map()
    // their turns with the wings, as a first reading; none without them (wing clips)
    let added: PhotoRow[] = withWings
      ? planTurnsByCamera(result.unturned, result.landscape, result.tagCw, result.camera,
        new Map(result.unturned.map((r) => [r.from, checks.has(r.from) ? wingTurn(checks.get(r.from)!, r) : null])))
      : result.unturned.map((r) => ({ ...r, rotSuggested: '', rotChosen: '', rotSource: '' }))
    const dir = useProcessingStore.getState().dirHandle
    if (settings.autoRotate && dir) {
      const written = await writeRotations(dir, added, await listFolder(dir), true)
      await appendRotationLog(dir, written.entries)
      added = written.rows
    }
    const now = useProcessingStore.getState()
    let rows = [...now.photoRows, ...added]
    if (withWings) ({ rows } = applyWingChecks(rows, new Map([...now.wingChecks, ...checks]), new Set(added.map((r) => r.from))))
    else rows = calculateFinalNames(rows, settings.mainColumn, settings.suffixMode, settings.customSuffixes)
    now.setOcrReadings(new Map([...now.ocrReadings, ...result.readings]))
    now.setPhotoRows(rows)
    await saveBatch(rows)
    const review = added.filter((r) => rows.find((x) => x.photoId === r.photoId)?.review).length
    now.setProgress(100, `Added ${added.length} photo(s): ${added.length - review} ready, ${review} to review`)
    toast.success(`Added ${added.length} photo(s) to the batch: ${added.length - review} ready, ${review} to review.`)
  } catch (e) {
    toast.error(`Reading the added photos failed: ${getErrorMessage(e)}`)
  } finally {
    useProcessingStore.getState().setProcessing(false)
  }
}
