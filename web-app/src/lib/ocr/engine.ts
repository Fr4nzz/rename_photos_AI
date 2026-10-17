/**
 * Main-thread API of the local CAMID reader: the photos go to the shared photo workers (photoPool.ts),
 * which can check their wings from the same decoding; a queue and cancellation.
 * The models (~24 MB) are fetched once per worker and then served from the browser cache.
 */
import { getPhotoPool } from '../photoPool'
import type { WingResult } from '../wings/model'
import type { PhotoReading } from './pipeline'

export interface OcrResult { name: string; reading?: PhotoReading; error?: string; ms?: number }

/**
 * Read many photos, as many at a time as the workers take. Results arrive through `onResult` in
 * completion order; stop early with the abort signal. With `onWings`, each photo's wings are checked
 * from the same decoding (null: the check failed on that photo).
 */
export async function readPhotos(
  files: File[], onResult: (r: OcrResult) => void, signal?: AbortSignal,
  onWings?: (name: string, wings: WingResult | null) => void,
): Promise<void> {
  const pool = await getPhotoPool()
  let index = 0
  const lane = async () => {
    while (index < files.length && !signal?.aborted) {
      const file = files[index++]
      const res = await pool.photo(file, { read: true, wings: !!onWings })
      if (res.ok && res.reading) onResult({ name: file.name, reading: res.reading, ms: res.ms })
      else onResult({ name: file.name, error: res.ok ? res.readError ?? 'unknown error' : res.error })
      onWings?.(file.name, res.ok ? res.wings ?? null : null)
    }
  }
  await Promise.all(Array.from({ length: pool.lanes }, lane))
}
