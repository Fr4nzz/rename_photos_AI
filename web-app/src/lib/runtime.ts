/**
 * The models of a photo worker on onnxruntime-web (WebAssembly), each loaded on first use.
 */
export type ModelKey = 'envelope' | 'line' | 'rec' | 'wingseg' | 'wingcls'

export interface RuntimePlan {
  /** WebAssembly threads of this worker (used only when the page is cross-origin isolated) */
  threads: number
}

export interface Output { data: Float32Array; dims: number[] }

export interface Runtime {
  run(model: ModelKey, data: Float32Array, dims: number[]): Promise<Output>
  /** every output, in the model's order */
  runAll(model: ModelKey, data: Float32Array, dims: number[]): Promise<Output[]>
  inputShape(model: ModelKey): Promise<readonly (number | string)[] | undefined>
}

type Ort = typeof import('onnxruntime-web/wasm')
type Session = Awaited<ReturnType<Ort['InferenceSession']['create']>>

/** The model files; the wing models' names come from wingside.json. */
async function modelFiles(base: string): Promise<Record<ModelKey, string>> {
  const cfg = await (await fetch(`${base}models/wingside.json`)).json()
  return { envelope: 'envelope_det.onnx', line: 'line_det.onnx', rec: 'camid_rec.onnx', wingseg: cfg.segmentation.file, wingcls: cfg.classifier.file }
}

export async function openRuntime(plan: RuntimePlan, base: string): Promise<Runtime> {
  const ort: Ort = await import('onnxruntime-web/wasm')
  ort.env.logLevel = 'error' // the converted models log harmless 'unused initializer' warnings
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, plan.threads) : 1
  const files = modelFiles(base)
  const sessions = new Map<ModelKey, Promise<Session>>()
  const create = async (model: ModelKey) => {
    // the file's bytes are not kept: onnxruntime keeps its own copy
    const bytes = new Uint8Array(await (await fetch(`${base}models/${(await files)[model]}`)).arrayBuffer())
    return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3 })
  }
  const session = (model: ModelKey): Promise<Session> => {
    if (!sessions.has(model)) sessions.set(model, create(model))
    return sessions.get(model)!
  }
  const runAll = async (model: ModelKey, data: Float32Array, dims: number[]): Promise<Output[]> => {
    const s = await session(model)
    const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', data, dims) })
    return s.outputNames.map((n) => ({ data: out[n].data as Float32Array, dims: out[n].dims as number[] }))
  }
  return {
    runAll,
    async run(model, data, dims) { return (await runAll(model, data, dims))[0] },
    async inputShape(model) {
      const m = (await session(model)).inputMetadata[0]
      return m?.isTensor ? m.shape : undefined
    },
  }
}

