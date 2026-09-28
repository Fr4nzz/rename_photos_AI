/**
 * Auto rotate: read each photo's CAMID and return the clockwise rotation that puts the envelope
 * text upright, or null when unsure (no ID read, or an ID that is not in the database, which may be
 * a misread of an upside-down line).
 */
import { loadCamidSet } from './database'
import { readPhotos } from './engine'
import { chooseCamid, type PhotoReading } from './pipeline'
import { uprightRotation } from '../rotationPlan'

export async function suggestRotations(
  files: File[],
  onProgress?: (done: number, total: number) => void,
  signal?: AbortSignal,
): Promise<Map<string, number | null>> {
  const database = loadCamidSet()
  const readings = new Map<string, PhotoReading | undefined>()
  await readPhotos(files, (r) => {
    readings.set(r.name, r.reading)
    onProgress?.(readings.size, files.length)
  }, signal)
  const { ids } = await database
  return new Map([...readings].map(([name, reading]) =>
    [name, reading ? uprightRotation(reading, chooseCamid(reading, ids)?.id ?? null) : null]))
}
