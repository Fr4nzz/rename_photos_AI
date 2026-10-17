/**
 * The wing check: the wing models in the shared photo workers (photoPool.ts). A reading checks the
 * wings from the same decoding of each photo (readPhotos; its findings join the reasons to look at a
 * photo, and the wings vote on the batch's turn); Review's button runs it on photos not checked yet.
 * The results are kept with the batch (beside the CSV) and told to the hosting page's assistant.
 */
import { saveBatch } from '../saveBatch'
import { calculateFinalNames } from '../nameCalculator'
import { getPhotoPool } from '../photoPool'
import { useProcessingStore } from '@/stores/processingStore'
import type { PhotoRow } from '@/types'
import { useSettingsStore } from '@/stores/settingsStore'
import { toast } from 'sonner'
import { wingFindings, withWingReasons, type WingCheck } from './checks'

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
  const pool = await getPhotoPool()
  let next = 0, count = 0
  const lane = async () => {
    while (next < todo.length && !signal.aborted) {
      const row = todo[next++]
      const file = fileOf(row)
      const res = file ? await pool.photo(file, { read: false, wings: true }) : null
      if (signal.aborted) return
      if (res?.ok && res.wings) out.set(row.from, { ...res.wings, applied: Number(row.rotApplied || 0) })
      onProgress(++count, todo.length)
    }
  }
  await Promise.all(Array.from({ length: pool.lanes }, lane))
  return out
}

/**
 * The rows with their wings' P(dorsal), their views worked out again (the wings name a pair shot
 * ventral side first, or a CRISPR set), and the wings' findings among their reasons (of the photos
 * in `only`, when given); the results kept in the store.
 */
export function applyWingChecks(rows: PhotoRow[], checks: Map<string, WingCheck>, only?: Set<string>): { rows: PhotoRow[]; flagged: number } {
  const { suffixMode, mainColumn, customSuffixes } = useSettingsStore.getState()
  const sided = rows.map((r) => {
    const p = checks.get(r.from)?.p
    return { ...r, pDorsal: p === null || p === undefined ? '' : String(Math.round(p * 10000) / 10000) }
  })
  const named = calculateFinalNames(sided, mainColumn, suffixMode, customSuffixes)
  const findings = wingFindings(named, checks, suffixMode)
  useProcessingStore.getState().setWingChecks(checks)
  const flagged = only ? [...findings.keys()].filter((k) => only.has(k)).length : findings.size
  return { rows: withWingReasons(named, findings, checks, only), flagged }
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
  const { rows, flagged } = applyWingChecks(s.photoRows, done, new Set(fresh.keys()))
  s.setPhotoRows(rows)
  await saveBatch(rows)
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
