/** Write chosen rotations to photos and their RAW files, and keep the per-folder rotation log. */
import { SUPPORTED_RAW_EXTENSIONS } from './constants'
import { getRotationLog, saveRotationLog } from './csvHandler'
import { getErrorMessage } from './errors'
import { logger } from './logger'
import { rotateLossless } from './orientation'
import { companionsOf, indexByStem } from './renamePlan'
import { pendingRotation } from './rotationPlan'
import { useProcessingStore } from '@/stores/processingStore'
import type { PhotoRow, RotationLogEntry } from '@/types'

export async function listFolder(dirHandle: FileSystemDirectoryHandle): Promise<string[]> {
  const names: string[] = []
  for await (const [name, entry] of dirHandle.entries()) if (entry.kind === 'file') names.push(name)
  return names
}

export /**
 * Write each row's pending rotation (chosen minus already applied) to its photo and RAW companions,
 * losslessly (EXIF Orientation tag only). Returns the updated rows, log entries and the files that
 * could not be rotated (no Orientation tag, e.g. HEIC/PNG).
 */
async function writeRotations(
  dirHandle: FileSystemDirectoryHandle,
  rows: PhotoRow[],
  folderNames: string[],
  withCompanions: boolean,
): Promise<{ rows: PhotoRow[]; entries: RotationLogEntry[]; failed: string[] }> {
  const stems = indexByStem(folderNames)
  const entries: RotationLogEntry[] = []
  const failed: string[] = []
  const out = rows.map((r) => ({ ...r }))
  for (const r of out) {
    const pending = pendingRotation(r)
    if (!pending || r.skip === 'x') continue
    const names = [r.currentPath, ...(withCompanions ? companionsOf(r.currentPath, stems, SUPPORTED_RAW_EXTENSIONS) : [])]
    let mainDone = false
    for (const name of names) {
      try {
        // the UI angle is clockwise; orientation angles are counter-clockwise
        const res = await rotateLossless(await dirHandle.getFileHandle(name), -pending)
        if (!res) { failed.push(name); continue }
        entries.push({ original: name, method: 'tag', before: res.before, after: res.after, angle: pending, timestamp: new Date().toISOString() })
        if (name === r.currentPath) mainDone = true
      } catch (err: unknown) {
        failed.push(name)
        logger.warn(`Could not rotate ${name}: ${getErrorMessage(err)}`)
      }
    }
    if (mainDone) r.rotApplied = String((Number(r.rotApplied || 0) + pending) % 360)
  }
  return { rows: out, entries, failed }
}

/** Add entries to the folder's rotation log (so Undo and Restore can revert them). */
export async function appendRotationLog(dirHandle: FileSystemDirectoryHandle, entries: RotationLogEntry[]) {
  if (!entries.length) return
  const log = [...(await getRotationLog(dirHandle)), ...entries]
  await saveRotationLog(log, dirHandle)
  useProcessingStore.getState().setRotationLog(log)
}
