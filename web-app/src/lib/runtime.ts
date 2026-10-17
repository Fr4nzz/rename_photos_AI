/**
 * The models of a photo worker on onnxruntime-web: each on the device its plan gives it (the graphics
 * card through WebGPU, or the processor through WebAssembly), loaded on first use. A model that
 * fails on the graphics card (an operator its driver cannot run) falls back to the processor.
 * The WebGPU build of onnxruntime is loaded only when a model runs on the graphics card: it is twice
 * the download, and its own WebAssembly runs the processor's models as fast as the plain build.
 */
export type ModelKey = 'envelope' | 'line' | 'rec' | 'wingseg' | 'wingcls'
export type Device = 'gpu' | 'cpu'
export const MODEL_KEYS: ModelKey[] = ['envelope', 'line', 'rec', 'wingseg', 'wingcls']

export interface RuntimePlan {
  /** the device of each model (missing: the processor) */
  devices: Partial<Record<ModelKey, Device>>
  /** WebAssembly threads of this worker (used only when the page is cross-origin isolated) */
  threads: number
}

export interface Output { data: Float32Array; dims: number[] }

export interface Runtime {
  run(model: ModelKey, data: Float32Array, dims: number[]): Promise<Output>
  /** every output, in the model's order */
  runAll(model: ModelKey, data: Float32Array, dims: number[]): Promise<Output[]>
  inputShape(model: ModelKey): Promise<readonly (number | string)[] | undefined>
  /** the device each loaded model runs on */
  devices(): Partial<Record<ModelKey, Device>>
  /** the device a model runs (or will run) on */
  deviceOf(model: ModelKey): Device
}

type Ort = typeof import('onnxruntime-web/wasm')
type Session = Awaited<ReturnType<Ort['InferenceSession']['create']>>

/** The model files; the wing models' names come from wingside.json. */
async function modelFiles(base: string): Promise<Record<ModelKey, string>> {
  const cfg = await (await fetch(`${base}models/wingside.json`)).json()
  return { envelope: 'envelope_det.onnx', line: 'line_det.onnx', rec: 'camid_rec.onnx', wingseg: cfg.segmentation.file, wingcls: cfg.classifier.file }
}

export async function openRuntime(plan: RuntimePlan, base: string): Promise<Runtime> {
  const gpu = MODEL_KEYS.some((k) => plan.devices[k] === 'gpu') && 'gpu' in navigator
  const ort: Ort = gpu ? await import('onnxruntime-web/webgpu') : await import('onnxruntime-web/wasm')
  ort.env.logLevel = 'error' // the converted models log harmless 'unused initializer' warnings
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, plan.threads) : 1
  const files = modelFiles(base)
  const sessions = new Map<ModelKey, Promise<Session>>()
  const placed: Partial<Record<ModelKey, Device>> = {}

  // fetched again for a fall back (from the browser's cache): onnxruntime keeps its own copy
  const fileBytes = async (model: ModelKey) => new Uint8Array(await (await fetch(`${base}models/${(await files)[model]}`)).arrayBuffer())
  const create = async (model: ModelKey, device: Device) => {
    const s = await ort.InferenceSession.create(await fileBytes(model), {
      executionProviders: device === 'gpu' ? [{ name: 'webgpu', powerPreference: 'high-performance' } as never] : ['wasm'],
      graphOptimizationLevel: 'all', logSeverityLevel: 3,
    })
    placed[model] = device
    return s
  }
  const session = (model: ModelKey): Promise<Session> => {
    if (!sessions.has(model)) {
      const device = gpu ? plan.devices[model] ?? 'cpu' : 'cpu'
      sessions.set(model, create(model, device).catch((e) => {
        if (device === 'cpu') throw e
        console.warn(`${model} cannot load on the graphics card, on the processor instead:`, e)
        return create(model, 'cpu')
      }))
    }
    return sessions.get(model)!
  }
  // one run at a time: the WebGPU build cannot start a run while another waits for the card (it
  // hangs), and on the processor runs take turns anyway; a second photo is decoded meanwhile
  let turn: Promise<unknown> = Promise.resolve()
  const runAll = (model: ModelKey, data: Float32Array, dims: number[]): Promise<Output[]> => {
    const mine = turn.then(() => runNow(model, data, dims))
    turn = mine.catch(() => {})
    return mine
  }
  const runNow = async (model: ModelKey, data: Float32Array, dims: number[]): Promise<Output[]> => {
    const s = await session(model)
    const feed = { [s.inputNames[0]]: new ort.Tensor('float32', data, dims) }
    let out
    try {
      out = await s.run(feed)
    } catch (e) {
      if (placed[model] !== 'gpu') throw e
      // a kernel the driver cannot run shows only on the first run: the model moves to the processor
      console.warn(`${model} failed on the graphics card, on the processor instead:`, e)
      sessions.delete(model)
      void s.release()
      const cpu = create(model, 'cpu')
      sessions.set(model, cpu)
      out = await (await cpu).run({ [s.inputNames[0]]: new ort.Tensor('float32', data, dims) })
    }
    return s.outputNames.map((n) => ({ data: out[n].data as Float32Array, dims: out[n].dims as number[] }))
  }
  return {
    runAll,
    async run(model, data, dims) { return (await runAll(model, data, dims))[0] },
    async inputShape(model) {
      const m = (await session(model)).inputMetadata[0]
      return m?.isTensor ? m.shape : undefined
    },
    devices: () => ({ ...placed }),
    deviceOf: (model) => placed[model] ?? (gpu ? plan.devices[model] ?? 'cpu' : 'cpu'),
  }
}

/** Input shapes for timing each model (an envelope's text lines; a photo's four wings). */
const PROBE_SHAPES: Record<ModelKey, number[]> = {
  envelope: [1, 3, 512, 512], line: [1, 3, 480, 480], rec: [1, 3, 48, 320], wingseg: [1, 3, 640, 640], wingcls: [4, 3, 192, 192],
}

export interface ProbeTimes {
  /** ms of a warm run of each model on the graphics card (missing: it cannot run there) */
  gpu: Partial<Record<ModelKey, number>>
  /** ms of a warm run on the processor, one thread, of the models timed there */
  cpu: Partial<Record<ModelKey, number>>
}

/**
 * Times each model on the graphics card and two small ones on the processor (the others are in a
 * fixed ratio to them on any processor: the same kind of network). The first runs on the card
 * compile its programs; the best of the next two counts.
 */
export async function probeDevices(base: string): Promise<ProbeTimes> {
  const ort: Ort = await import('onnxruntime-web/webgpu')
  ort.env.logLevel = 'error'
  ort.env.wasm.numThreads = 1
  const files = await modelFiles(base)
  const out: ProbeTimes = { gpu: {}, cpu: {} }
  const time = async (model: ModelKey, device: Device, runs: number) => {
    const bytes = new Uint8Array(await (await fetch(`${base}models/${files[model]}`)).arrayBuffer())
    const s = await ort.InferenceSession.create(bytes, {
      executionProviders: device === 'gpu' ? [{ name: 'webgpu', powerPreference: 'high-performance' } as never] : ['wasm'],
      graphOptimizationLevel: 'all', logSeverityLevel: 3,
    })
    try {
      const dims = PROBE_SHAPES[model]
      const x = new ort.Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b)).fill(0.5), dims)
      let best = Infinity
      for (let i = 0; i < runs; i++) {
        const t0 = performance.now()
        const o = await s.run({ [s.inputNames[0]]: x })
        for (const n of s.outputNames) await o[n].getData()
        if (i) best = Math.min(best, performance.now() - t0)
      }
      return best
    } finally {
      await s.release()
    }
  }
  for (const model of MODEL_KEYS) {
    try {
      out.gpu[model] = await time(model, 'gpu', 3)
    } catch {
      // stays off the graphics card
    }
  }
  for (const model of ['envelope', 'rec'] as const) out.cpu[model] = await time(model, 'cpu', 2)
  return out
}
