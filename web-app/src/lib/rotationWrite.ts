/** Write chosen rotations to photos and their RAW files, and keep the per-folder rotation log. */
import { SUPPORTED_RAW_EXTENSIONS } from './constants'
import { getRotationLog, saveRotationLog } from './csvHandler'
import { getErrorMessage } from './errors'
import { logger } from './logger'
import { readOrientation, rotateLossless, writeOrientation } from './orientation'
import { companionsOf, indexByStem } from './renamePlan'
import { pendingRotation } from './rotationPlan'
import { useProcessingStore } from '@/stores/processingStore'
import type { PhotoRow, RotationLogEntry } from '@/types'

export async function listFolder(dirHandle: FileSystemDirectoryHandle): Promise<string[]> {
  const names: string[] = []
  for await (const [name, entry] of dirHandle.entries()) if (entry.kind === 'file') names.push(name)
  return names
}

/**
 * Give a photo's RAW companions the photo's own orientation tag. The JPEG and the RAW of one shot
 * share the sensor frame, but the camera's gyro-based tags can disagree; the tag the JPEG ends up
 * with (checked by the reader or by you) is the one to trust.
 */
export async function syncCompanions(
  dirHandle: FileSystemDirectoryHandle,
  name: string,
  companions: string[],
  angle: number,
): Promise<{ entries: RotationLogEntry[]; failed: string[] }> {
  const entries: RotationLogEntry[] = []
  const failed: string[] = []
  const tag = (await readOrientation(await (await dirHandle.getFileHandle(name)).getFile()))?.value
  if (!tag) return { entries, failed: companions }
  for (const other of companions) {
    try {
      const before = await writeOrientation(await dirHandle.getFileHandle(other), tag)
      if (before === null) failed.push(other)
      else if (before !== tag) entries.push({ original: other, method: 'tag', before, after: tag, angle, timestamp: new Date().toISOString() })
    } catch (err: unknown) {
      failed.push(other)
      logger.warn(`Could not rotate ${other}: ${getErrorMessage(err)}`)
    }
  }
  return { entries, failed }
}

/**
 * Write each row's pending rotation (chosen minus already applied) to its photo, losslessly (EXIF
 * Orientation tag only), and give its RAW companions the same tag. Rows whose orientation is known
 * but needs no turn still get their RAW files synced. Returns the updated rows, log entries and the
 * files that could not be rotated (no Orientation tag, e.g. HEIC/PNG).
 */
export async function writeRotations(
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
    if (r.skip === 'x' || r.rotChosen === '') continue
    const pending = pendingRotation(r)
    if (pending) {
      try {
        // the UI angle is clockwise; orientation angles are counter-clockwise
        const res = await rotateLossless(await dirHandle.getFileHandle(r.currentPath), -pending)
        if (!res) { failed.push(r.currentPath); continue }
        entries.push({ original: r.currentPath, method: 'tag', before: res.before, after: res.after, angle: pending, timestamp: new Date().toISOString() })
        r.rotApplied = String((Number(r.rotApplied || 0) + pending) % 360)
      } catch (err: unknown) {
        failed.push(r.currentPath)
        logger.warn(`Could not rotate ${r.currentPath}: ${getErrorMessage(err)}`)
        continue
      }
    }
    if (withCompanions) {
      const synced = await syncCompanions(dirHandle, r.currentPath, companionsOf(r.currentPath, stems, SUPPORTED_RAW_EXTENSIONS), pending)
      entries.push(...synced.entries)
      failed.push(...synced.failed)
    }
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
