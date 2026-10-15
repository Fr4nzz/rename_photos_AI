/**
 * The wing check of a batch, after the reading: the wing models in their own workers (the readers'
 * are idle by then), one photo per worker at a time, so the IDs read are in Review first and the
 * check fills in. Its findings join the reasons to look at a photo, and its results are kept with
 * the batch (beside the CSV) and told to the hosting page's assistant.
 */
import { SUPPORTED_RAW_EXTENSIONS } from '../constants'
import { saveCsvToFolder, saveCsvToStorage, toCsvString } from '../csvHandler'
import { saveSidecar } from '../reviewSidecar'
import { useProcessingStore } from '@/stores/processingStore'
import type { PhotoRow } from '@/types'
import { useSettingsStore } from '@/stores/settingsStore'
import { toast } from 'sonner'
import { wingFindings, withWingReasons } from './checks'
import type { WingRequest, WingResponse } from './wingWorker'

const BASE = import.meta.env.BASE_URL

class WingPool {
  private workers: Worker[] = []
  private ready: Promise<void>[] = []
  private pending = new Map<number, (r: WingResponse) => void>()
  private nextId = 1

  constructor(size: number) {
    for (let i = 0; i < size; i++) {
      const w = new Worker(new URL('./wingWorker.ts', import.meta.url), { type: 'module' })
      this.ready.push(new Promise((resolve, reject) => {
        const onInit = (e: MessageEvent<WingResponse>) => {
          if (e.data.id !== -1) return
          w.removeEventListener('message', onInit)
          if ('ready' in e.data && e.data.ready) resolve()
          else reject(new Error('error' in e.data ? e.data.error : 'the wing check failed to start'))
        }
        w.addEventListener('message', onInit)
      }))
      w.addEventListener('message', (e: MessageEvent<WingResponse>) => {
        if (e.data.id === -1) return
        this.pending.get(e.data.id)?.(e.data)
        this.pending.delete(e.data.id)
      })
      w.postMessage({ init: new URL(BASE, location.href).href })
      this.workers.push(w)
    }
  }

  get size() { return this.workers.length }
  async whenReady() { await Promise.all(this.ready) }

  check(worker: number, file: File): Promise<WingResponse> {
    const id = this.nextId++
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
    return new Promise((resolve) => {
      this.pending.set(id, resolve)
      this.workers[worker].postMessage({ id, file, raw: SUPPORTED_RAW_EXTENSIONS.has(ext) } satisfies WingRequest)
    })
  }
}

let pool: WingPool | null = null

/** A row's photo: by the name it had when read, or the one it has now (renamed since). */
export const fileOf = (row: PhotoRow): File | undefined => {
  const files = useProcessingStore.getState().fileMap
  return files.get(row.from) ?? files.get(row.currentPath)
}
let running: AbortController | null = null

/** Stops a check under way (a new reading starts, or another folder opens). */
export function stopWingCheck() {
  running?.abort()
  running = null
  useProcessingStore.getState().setWingRun(null)
}

/**
 * Checks the wings of the batch's photos not checked yet (all of them with `again`), then adds the
 * findings to the rows and saves. Returns the number of photos with a finding (null: stopped).
 */
export async function checkWings({ again = false } = {}): Promise<number | null> {
  stopWingCheck()
  const controller = new AbortController()
  running = controller
  const store = useProcessingStore.getState()
  // another folder opened meanwhile: its photos may have the same names
  const moved = () => useProcessingStore.getState().dirHandle !== store.dirHandle
  const done = new Map(again ? [] : store.wingChecks)
  const todo = store.photoRows.filter((r) => !done.has(r.from) && fileOf(r))
  const applied = new Map(store.photoRows.map((r) => [r.from, Number(r.rotApplied || 0)]))
  if (todo.length) {
    store.setWingRun({ done: 0, total: todo.length })
    pool ??= new WingPool((navigator.hardwareConcurrency ?? 2) >= 6 ? 2 : 1)
    await pool.whenReady()
    let next = 0, count = 0
    const lane = async (worker: number) => {
      while (next < todo.length && !controller.signal.aborted) {
        const row = todo[next++]
        const file = fileOf(row)
        const res = file ? await pool!.check(worker, file) : null
        if (controller.signal.aborted || moved()) return controller.abort()
        if (res && 'ok' in res && res.ok) done.set(row.from, { ...res.result, applied: applied.get(row.from) ?? 0 })
        const s = useProcessingStore.getState()
        s.setWingChecks(new Map(done))
        s.setWingRun({ done: ++count, total: todo.length })
      }
    }
    await Promise.all(Array.from({ length: pool.size }, (_, i) => lane(i)))
    if (controller.signal.aborted) {
      if (moved()) useProcessingStore.getState().setWingRun(null)
      return null
    }
  }
  running = null
  const s = useProcessingStore.getState()
  const { suffixMode, mainColumn } = useSettingsStore.getState()
  s.setWingChecks(done)
  s.setWingRun(null)
  const findings = wingFindings(s.photoRows, done, suffixMode)
  const rows = withWingReasons(s.photoRows, findings, done)
  if (rows.some((r, i) => r !== s.photoRows[i])) {
    s.setPhotoRows(rows)
    const review = rows.filter((r) => r.review).length
    if (!s.isProcessing && s.progress.message.startsWith('Done:')) s.setProgress(100, `Done: ${rows.length - review} ready, ${review} to review`)
  }
  if (s.currentCsvName) {
    const text = toCsvString(rows, mainColumn)
    await saveCsvToStorage(s.currentCsvName, text)
    if (s.dirHandle) {
      await saveCsvToFolder(s.dirHandle, s.currentCsvName, text)
      await saveSidecar(s.dirHandle, s.currentCsvName)
    }
  }
  return findings.size
}

/** The check, and a word when it is done: on its own after a reading, or from Review's button. */
export async function runWingCheck(again = false) {
  try {
    const flagged = await checkWings({ again })
    if (flagged === null) return
    toast.info(flagged ? `Wing check: ${flagged} photo(s) to look at (named d/v against their wings, a wing fewer, or upside down).` : 'Wing check: every photo matches its name.')
  } catch (e) {
    toast.error(`The wing check failed: ${e instanceof Error ? e.message : String(e)}`)
  }
}
