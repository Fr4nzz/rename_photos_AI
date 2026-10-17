/**
 * Which device runs each model, and how many photo workers, on this computer. A graphics card that
 * WebGPU offers runs a model when it is faster there than the processor's share of it: the first time
 * on a computer (and again when the browser or the card changes), each model is timed on the card,
 * which also compiles its programs, and the processor's speed from two small models; the choice is
 * kept in the browser (localStorage). A weak card can lose to a fast processor, a slow processor
 * gains from almost any card.
 */
import type { PoolPlan } from './photoPool'
import type { PhotoProbe, ProbeResult } from './photoWorker'
import { MODEL_KEYS, type Device, type ModelKey, type ProbeTimes } from './runtime'

const KEY = 'rename-photos-device-plan'
/** bump to time again on every computer (new models, a new rule) */
const VERSION = 1

/**
 * Processor time of each model against the envelope detector's (onnxruntime-web 1.30, WebAssembly,
 * one thread, 17 Oct 2026: 154, 79, 50, 1,105 and 62 ms); convolution networks keep their ratios
 * from one processor to another. The rest of a photo's work (decoding, shrinking, the crops) is
 * about 2.8 envelope detections.
 */
const CPU_RATIO: Record<ModelKey, number> = { envelope: 1, line: 0.51, rec: 0.32, wingseg: 7.2, wingcls: 0.4 }
const OTHER_WORK = 2.8
/** runs of each model per photo (a reading stops at its first sure line: ~5 recogniser runs) */
const RUNS: Record<ModelKey, number> = { envelope: 1, line: 1.1, rec: 4.7, wingseg: 1, wingcls: 1 }

/**
 * Workers on the processor alone: one per logical processor (on two cores with SMT, four workers read
 * 10 % faster than two), as many as the memory allows: a worker holds ~250 MB, one running the large
 * line detections ~1 GB. Six at most, and only on processors with 12 threads (which come with 16 GB):
 * the browser does not tell more than 8 GB.
 */
export function cpuPlan(threads = navigator.hardwareConcurrency || 2, memoryGB?: number): PoolPlan {
  memoryGB ??= (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4
  const cap = memoryGB >= 8 ? (threads >= 12 ? 6 : 4) : 2
  return { workers: Math.max(2, Math.min(threads, cap)), perWorker: 2, runtime: { devices: {}, threads: 1 } }
}

/** With the graphics card: two workers keep it busy (one decodes and prepares while the other runs). */
function gpuPlan(devices: Partial<Record<ModelKey, Device>>): PoolPlan {
  return { workers: 2, perWorker: 2, runtime: { devices, threads: 1 } }
}

/**
 * The plan with the shortest estimated time per photo: on the processor alone, its workers share the
 * work; with the card, the card runs its models one photo after another while the workers do the rest.
 * The card runs every model it can unless another split is clearly (10 %) faster: the estimate is rough.
 */
export function planFromTimes(times: ProbeTimes, threads = navigator.hardwareConcurrency || 2, memoryGB?: number): PoolPlan {
  const cpu = cpuPlan(threads, memoryGB)
  const envelope = times.cpu.envelope
  if (envelope === undefined) return cpu
  const cpuMs = (m: ModelKey) => RUNS[m] * (times.cpu[m] ?? envelope * CPU_RATIO[m])
  const other = OTHER_WORK * envelope
  const perPhoto = (onGpu: ModelKey[], workers: number) => {
    const card = onGpu.reduce((t, m) => t + RUNS[m] * times.gpu[m]!, 0)
    const rest = MODEL_KEYS.filter((m) => !onGpu.includes(m)).reduce((t, m) => t + cpuMs(m), other)
    return Math.max(card, rest / Math.min(workers, Math.max(1, threads / 2)))
  }
  const usable = MODEL_KEYS.filter((m) => times.gpu[m] !== undefined)
  const all = { ms: perPhoto(usable, gpuPlan({}).workers), onGpu: usable }
  let best = { ms: perPhoto([], cpu.workers), onGpu: [] as ModelKey[] }
  for (let mask = 1; mask < 1 << usable.length; mask++) {
    const onGpu = usable.filter((_, i) => mask & (1 << i))
    const ms = perPhoto(onGpu, gpuPlan({}).workers)
    if (ms < best.ms) best = { ms, onGpu }
  }
  if (usable.length && best.ms >= 0.9 * all.ms) best = all
  return best.onGpu.length ? gpuPlan(Object.fromEntries(best.onGpu.map((m) => [m, 'gpu' as const]))) : cpu
}

async function adapterInfo(): Promise<string | null> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(o?: object): Promise<{ info?: Record<string, unknown>; isFallbackAdapter?: boolean } | null> } }).gpu
  if (!gpu) return null
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null)
  // a software adapter (no real card) is far slower than the processor's WebAssembly
  if (!adapter || adapter.isFallbackAdapter || adapter.info?.isFallbackAdapter) return null
  const i = adapter.info ?? {}
  return [i.vendor, i.architecture, i.device, i.description].join('|')
}

function probe(): Promise<ProbeTimes> {
  const worker = new Worker(new URL('./photoWorker.ts', import.meta.url), { type: 'module' })
  return new Promise<ProbeTimes>((resolve, reject) => {
    worker.onmessage = (e: MessageEvent<ProbeResult>) => {
      if (e.data.id !== -2) return
      if ('times' in e.data) resolve(e.data.times)
      else reject(new Error(e.data.error))
    }
    worker.onerror = (e) => reject(new Error(e.message))
    worker.postMessage({ probe: new URL(import.meta.env.BASE_URL, location.href).href } satisfies PhotoProbe)
  }).finally(() => worker.terminate())
}

/** This computer's plan: kept from before, timed now (a few seconds, once), or the processor's. */
export async function choosePlan(): Promise<PoolPlan> {
  const threads = navigator.hardwareConcurrency || 2
  const gpu = await adapterInfo().catch(() => null)
  if (!gpu) return cpuPlan(threads)
  const signature = JSON.stringify([VERSION, navigator.userAgent, gpu, threads])
  try {
    const kept = JSON.parse(localStorage.getItem(KEY) ?? 'null')
    if (kept?.signature === signature) return planFromTimes(kept.times, threads)
  } catch {
    // no storage (private window) or an old entry: timed again
  }
  try {
    const times = await probe()
    try {
      localStorage.setItem(KEY, JSON.stringify({ signature, times }))
    } catch {
      // not kept: timed again next time
    }
    return planFromTimes(times, threads)
  } catch (e) {
    console.warn('timing the graphics card failed; the processor reads the photos', e)
    return cpuPlan(threads)
  }
}
