/**
 * Main-thread side of the photo workers (photoWorker.ts): a photo goes to the least busy worker, at
 * most two at a time each (one is decoded while the other is read). How many workers, and the device
 * of each model, are chosen for this computer (devicePlan.ts).
 */
import { SUPPORTED_RAW_EXTENSIONS } from './constants'
import type { PhotoInit, PhotoReady, PhotoRequest, PhotoResponse } from './photoWorker'
import type { RuntimePlan } from './runtime'
import { choosePlan } from './devicePlan'
import { useSettingsStore } from '@/stores/settingsStore'

export interface PoolPlan {
  workers: number
  /** photos in flight per worker */
  perWorker: number
  runtime: RuntimePlan
}

const BASE = import.meta.env.BASE_URL

export class PhotoPool {
  private workers: { worker: Worker; busy: number }[] = []
  private pending = new Map<number, (r: PhotoResponse) => void>()
  private waiting: (() => void)[] = []
  private nextId = 1
  readonly ready: Promise<void>
  readonly plan: PoolPlan

  constructor(plan: PoolPlan) {
    this.plan = plan
    const starts: Promise<void>[] = []
    // the first worker (two of six) runs the large line detections of the others (photoWorker.ts)
    const big = Math.max(1, Math.floor(plan.workers / 3))
    const serves: MessagePort[][] = Array.from({ length: big }, () => [])
    const sends: (MessagePort | undefined)[] = []
    for (let i = big; i < plan.workers; i++) {
      const channel = new MessageChannel()
      serves[i % big].push(channel.port2)
      sends[i] = channel.port1
    }
    for (let i = 0; i < plan.workers; i++) {
      const worker = new Worker(new URL('./photoWorker.ts', import.meta.url), { type: 'module' })
      starts.push(new Promise((resolve, reject) => {
        const onInit = (e: MessageEvent<PhotoReady>) => {
          if (e.data.id !== -1) return
          worker.removeEventListener('message', onInit)
          if (e.data.ready) resolve()
          else reject(new Error(e.data.error))
        }
        worker.addEventListener('message', onInit)
      }))
      worker.addEventListener('message', (e: MessageEvent<PhotoResponse | PhotoReady>) => {
        const r = e.data
        if ('ready' in r) return
        this.pending.get(r.id)?.(r)
        this.pending.delete(r.id)
      })
      const ports = { bigLines: sends[i], serves: serves[i] }
      worker.postMessage({ init: new URL(BASE, location.href).href, plan: plan.runtime, ...ports } satisfies PhotoInit,
        [...(ports.bigLines ? [ports.bigLines] : []), ...(ports.serves ?? [])])
      this.workers.push({ worker, busy: 0 })
    }
    this.ready = Promise.all(starts).then(() => {})
  }

  /** callers' lanes that keep every worker busy */
  get lanes() { return this.plan.workers * this.plan.perWorker }

  /** Reads and/or checks the wings of one photo, when a worker has room. */
  async photo(file: File, tasks: { read: boolean; wings: boolean }): Promise<PhotoResponse> {
    await this.ready
    let slot = this.free()
    while (!slot) {
      await new Promise<void>((resolve) => this.waiting.push(resolve))
      slot = this.free()
    }
    slot.busy++
    const id = this.nextId++
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
    try {
      return await new Promise<PhotoResponse>((resolve) => {
        this.pending.set(id, resolve)
        slot.worker.postMessage({ id, file, raw: SUPPORTED_RAW_EXTENSIONS.has(ext), ...tasks } satisfies PhotoRequest)
      })
    } finally {
      slot.busy--
      this.waiting.shift()?.()
    }
  }

  /** no photo in a worker */
  get idle() { return this.workers.every((w) => w.busy === 0) }

  private free() {
    const least = this.workers.reduce((a, b) => (b.busy < a.busy ? b : a))
    return least.busy < this.plan.perWorker ? least : null
  }

  terminate() { this.workers.forEach((w) => w.worker.terminate()) }
}

let pool: Promise<PhotoPool> | null = null
/** the settings' workers the pool was started with */
let poolChosen = 0

/**
 * The shared pool, started on first use with this computer's plan (devicePlan.ts) and the settings'
 * workers; started again when they changed (once its photos are done) or for a different plan
 * (benchmarks).
 */
export async function getPhotoPool(plan?: PoolPlan): Promise<PhotoPool> {
  const chosen = useSettingsStore.getState().photoWorkers
  if (pool) {
    const p = await pool.catch(() => null)
    const stale = plan ? JSON.stringify(plan) !== JSON.stringify(p?.plan) : chosen !== poolChosen
    if (p && stale && p.idle) {
      p.terminate()
      pool = null
    }
  }
  if (!pool) {
    poolChosen = chosen
    pool = (async () => {
      const p = new PhotoPool(plan ?? await choosePlan(chosen))
      await p.ready
      return p
    })()
    // a pool that failed to start is started again next time
    pool.catch(() => { pool = null })
  }
  return pool
}
