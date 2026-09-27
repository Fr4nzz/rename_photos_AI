/**
 * Main-thread API of the local CAMID reader: a small pool of OCR workers, a queue, cancellation.
 * The models (~24 MB) are fetched once per worker and then served from the browser cache.
 */
import { SUPPORTED_RAW_EXTENSIONS } from '../constants'
import type { OcrRequest, OcrResponse } from './ocrWorker'
import type { PhotoReading } from './pipeline'

export interface OcrResult { name: string; reading?: PhotoReading; error?: string; ms?: number }

const BASE = import.meta.env.BASE_URL

class OcrPool {
  private workers: Worker[] = []
  private ready: Promise<void>[] = []
  private pending = new Map<number, (r: OcrResponse) => void>()
  private nextId = 1
  private next = 0

  constructor(size: number) {
    for (let i = 0; i < size; i++) {
      const w = new Worker(new URL('./ocrWorker.ts', import.meta.url), { type: 'module' })
      this.ready.push(new Promise((resolve, reject) => {
        const onInit = (e: MessageEvent<OcrResponse>) => {
          if (e.data.id !== -1) return
          w.removeEventListener('message', onInit)
          if ('ready' in e.data && e.data.ready) resolve()
          else reject(new Error('error' in e.data ? e.data.error : 'OCR worker failed to start'))
        }
        w.addEventListener('message', onInit)
      }))
      w.addEventListener('message', (e: MessageEvent<OcrResponse>) => {
        if (e.data.id === -1) return
        this.pending.get(e.data.id)?.(e.data)
        this.pending.delete(e.data.id)
      })
      w.postMessage({ init: new URL(BASE, location.href).href })
      this.workers.push(w)
    }
  }

  async whenReady() { await Promise.all(this.ready) }

  read(file: File): Promise<OcrResponse> {
    const id = this.nextId++
    const worker = this.workers[this.next++ % this.workers.length]
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
    return new Promise((resolve) => {
      this.pending.set(id, resolve)
      worker.postMessage({ id, file, raw: SUPPORTED_RAW_EXTENSIONS.has(ext) } satisfies OcrRequest)
    })
  }

  terminate() { this.workers.forEach((w) => w.terminate()) }
}

let pool: OcrPool | null = null

/** Start (or reuse) the worker pool: 2 workers, or 1 on machines with few cores. */
export function getOcrPool(): OcrPool {
  pool ??= new OcrPool((navigator.hardwareConcurrency ?? 2) >= 6 ? 2 : 1)
  return pool
}

/**
 * Read many photos, at most `concurrency` at a time per worker. Results arrive through `onResult`
 * in completion order; stop early with the abort signal.
 */
export async function readPhotos(files: File[], onResult: (r: OcrResult) => void, signal?: AbortSignal): Promise<void> {
  const ocr = getOcrPool()
  await ocr.whenReady()
  let index = 0
  const lane = async () => {
    while (index < files.length && !signal?.aborted) {
      const file = files[index++]
      const res = await ocr.read(file)
      if ('ok' in res && res.ok) onResult({ name: file.name, reading: res.reading, ms: res.ms })
      else onResult({ name: file.name, error: 'error' in res ? res.error : 'unknown error' })
    }
  }
  await Promise.all([lane(), lane(), lane()])
}
