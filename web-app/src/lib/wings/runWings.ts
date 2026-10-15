/**
 * The wing check: the wing models in their own workers (the readers' are idle by then), one photo per
 * worker at a time. A reading runs it after the IDs (its findings join the reasons to look at a photo,
 * and the wings vote on the batch's turn); Review's button runs it on photos not checked yet. The
 * results are kept with the batch (beside the CSV) and told to the hosting page's assistant.
 */
import { SUPPORTED_RAW_EXTENSIONS } from '../constants'
import { saveCsvToFolder, saveCsvToStorage, toCsvString } from '../csvHandler'
import { saveSidecar } from '../reviewSidecar'
import { useProcessingStore } from '@/stores/processingStore'
import type { PhotoRow } from '@/types'
import { useSettingsStore } from '@/stores/settingsStore'
import { toast } from 'sonner'
import { wingFindings, withWingReasons, type WingCheck } from './checks'
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

/** Stops a check started from Review's button (another folder opens, or a new reading starts). */
export function stopWingCheck() {
  running?.abort()
  running = null
  useProcessingStore.getState().setWingRun(null)
}

/**
 * The wings of `rows`' photos, by original name; `onProgress` after each photo. A photo that cannot
 * be checked is left out. Stops early with the signal (its results so far are returned).
 */
export async function checkPhotos(rows: PhotoRow[], onProgress: (done: number, total: number) => void, signal: AbortSignal): Promise<Map<string, WingCheck>> {
  const out = new Map<string, WingCheck>()
  const todo = rows.filter((r) => fileOf(r))
  if (!todo.length) return out
  pool ??= new WingPool((navigator.hardwareConcurrency ?? 2) >= 6 ? 2 : 1)
  await pool.whenReady()
  let next = 0, count = 0
  const lane = async (worker: number) => {
    while (next < todo.length && !signal.aborted) {
      const row = todo[next++]
      const file = fileOf(row)
      const res = file ? await pool!.check(worker, file) : null
      if (signal.aborted) return
      if (res && 'ok' in res && res.ok) out.set(row.from, { ...res.result, applied: Number(row.rotApplied || 0) })
      onProgress(++count, todo.length)
    }
  }
  await Promise.all(Array.from({ length: pool.size }, (_, i) => lane(i)))
  return out
}

/** The rows with the wings' findings among their reasons, the results kept in the store. */
export function applyWingChecks(rows: PhotoRow[], checks: Map<string, WingCheck>): { rows: PhotoRow[]; flagged: number } {
  const findings = wingFindings(rows, checks, useSettingsStore.getState().suffixMode)
  useProcessingStore.getState().setWingChecks(checks)
  return { rows: withWingReasons(rows, findings, checks), flagged: findings.size }
}

/**
 * Review's button: checks the photos not checked yet (a batch read before the check existed, or
 * one stopped), adds the findings to the rows and saves. Returns the photos flagged (null: stopped).
 */
export async function checkWings(): Promise<number | null> {
  stopWingCheck()
  const controller = new AbortController()
  running = controller
  const store = useProcessingStore.getState()
  // another folder opened meanwhile: its photos may have the same names
  const moved = () => useProcessingStore.getState().dirHandle !== store.dirHandle
  const done = new Map(store.wingChecks)
  store.setWingRun({ done: 0, total: 0 })
  const fresh = await checkPhotos(store.photoRows.filter((r) => !done.has(r.from)), (n, total) => {
    if (moved()) controller.abort()
    useProcessingStore.getState().setWingRun({ done: n, total })
  }, controller.signal)
  if (controller.signal.aborted) return null
  running = null
  for (const [k, v] of fresh) done.set(k, v)
  const s = useProcessingStore.getState()
  s.setWingRun(null)
  const { rows, flagged } = applyWingChecks(s.photoRows, done)
  s.setPhotoRows(rows)
  const { mainColumn } = useSettingsStore.getState()
  if (s.currentCsvName) {
    const text = toCsvString(rows, mainColumn)
    await saveCsvToStorage(s.currentCsvName, text)
    if (s.dirHandle) {
      await saveCsvToFolder(s.dirHandle, s.currentCsvName, text)
      await saveSidecar(s.dirHandle, s.currentCsvName)
    }
  }
  return flagged
}

/** What the check found, in a few words. */
export const wingSummary = (flagged: number) =>
  flagged ? `${flagged} photo(s) flagged by their wings (named d/v against them, a wing fewer, or turned)` : 'the wings match every name'

/** Review's button, with a word when it is done. */
export async function runWingCheck() {
  try {
    const flagged = await checkWings()
    if (flagged !== null) toast.info(`Wing check: ${wingSummary(flagged)}.`)
  } catch (e) {
    toast.error(`The wing check failed: ${e instanceof Error ? e.message : String(e)}`)
  }
}
