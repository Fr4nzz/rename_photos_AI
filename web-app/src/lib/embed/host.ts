/**
 * Embedded mode, the browser part: the app in a frame of another page of the same origin
 * (`…/index.html?embed`). It tells that page the batch as it changes, takes the database's IDs from
 * it, and does what it asks: small copies of photos, corrections to rows, undoing them.
 *
 * Messages (window.postMessage, same origin only):
 *   to the host:    { source: 'rename-photos', type: 'ready' | 'state' | 'result', … }
 *   from the host:  { source: 'rename-host', type: 'camids', ids }
 *                   { source: 'rename-host', type: 'command', id, name, args }
 * Commands (each answered by a `result` with the same id: { ok, data } or { ok: false, error }):
 *   state                         the batch now (HostState)
 *   view  { files, part, sheet }  photos as small JPEGs: part 'photo' (whole) or 'envelope' (the envelope,
 *                                 upright); `sheet`: up to six a picture, each with its name and ID under it
 *   edit  { edits, by }           corrections (HostEdit[]); `by` names who made them, shown on the cards
 *   undo  {}                      every correction of the host taken back
 */
import { useProcessingStore } from '@/stores/processingStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { canvasToBlob, loadImagePreview, rotateCanvas } from '../imageProcessing'
import { pendingRotation } from '../rotationPlan'
import { setHostCamids, hostDatabase } from '../ocr/database'
import { PROTOCOL_VERSION, applyEdits, folderBytes, snapshotRows, undoEdits, type HostEdit, type HostState } from './protocol'
import { wingFindings } from '../wings/checks'
import type { PhotoRow } from '@/types'

export const isEmbedded = () =>
  typeof window !== 'undefined' && window.parent !== window && new URLSearchParams(window.location.search).has('embed')

const post = (message: Record<string, unknown>, transfer: Transferable[] = []) =>
  window.parent.postMessage({ source: 'rename-photos', ...message }, window.location.origin, transfer)

const namesOf = () => {
  const { mainColumn, suffixMode, customSuffixes } = useSettingsStore.getState()
  return { mainColumn, suffixMode, customSuffixes }
}

/** The open folder's size, counted once when it is opened. */
let sized: { dir: FileSystemDirectoryHandle; bytes: number } | null = null

export function hostState(): HostState {
  const s = useProcessingStore.getState()
  const database = hostDatabase()
  return {
    version: PROTOCOL_VERSION,
    folder: s.dirHandle?.name ?? null,
    canWrite: !!s.dirHandle,
    files: s.imageFiles.length,
    folderBytes: sized && sized.dir === s.dirHandle ? sized.bytes : null,
    reading: { on: s.isProcessing, percent: Math.round(s.progress.percent), message: s.progress.message },
    database,
    wingCheck: { on: !!s.wingRun, done: s.wingRun?.done ?? 0, total: s.wingRun?.total ?? 0, checked: s.wingChecks.size },
    rows: snapshotRows(s.photoRows, s.ocrReadings, s.hostChanges, {
      checks: s.wingChecks, findings: wingFindings(s.photoRows, s.wingChecks, useSettingsStore.getState().suffixMode),
    }),
  }
}

const MAX_VIEW = 24
const SHEET = { cols: 3, per: 6, cellW: 620, cellH: 440, caption: 30 }

function fit(source: HTMLCanvasElement, maxSide: number): HTMLCanvasElement {
  const scale = Math.min(1, maxSide / Math.max(source.width, source.height))
  if (scale === 1) return source
  const out = document.createElement('canvas')
  out.width = Math.max(1, Math.round(source.width * scale))
  out.height = Math.max(1, Math.round(source.height * scale))
  out.getContext('2d')!.drawImage(source, 0, 0, out.width, out.height)
  return out
}

/** A photo as its card shows it (turned by the rotation still to be written), or its envelope upright. */
async function render(row: PhotoRow, file: File, part: 'photo' | 'envelope'): Promise<HTMLCanvasElement> {
  const preview = await loadImagePreview(file, 1600)
  const reading = useProcessingStore.getState().ocrReadings.get(row.from)
  if (part === 'photo' || !reading?.envelope) return fit(rotateCanvas(preview, pendingRotation(row)), 1100)
  // the envelope's box refers to the photo as it was read: rotations written since are undone first
  const applied = Number(row.rotApplied || 0) % 360
  const read = applied ? rotateCanvas(preview, 360 - applied) : preview
  const s = read.width / reading.size[0]
  const [x1, y1, x2, y2] = reading.envelope
  const pad = 0.04 * Math.max(x2 - x1, y2 - y1)
  const x = Math.max(0, (x1 - pad) * s), y = Math.max(0, (y1 - pad) * s)
  const w = Math.min(read.width, (x2 + pad) * s) - x, h = Math.min(read.height, (y2 + pad) * s) - y
  if (w <= 4 || h <= 4) return fit(rotateCanvas(preview, pendingRotation(row)), 1100)
  const crop = document.createElement('canvas')
  crop.width = Math.round(w)
  crop.height = Math.round(h)
  crop.getContext('2d')!.drawImage(read, x, y, w, h, 0, 0, crop.width, crop.height)
  return fit(rotateCanvas(crop, Number(row.rotChosen || 0) % 360), 900)
}

const caption = (row: PhotoRow) =>
  `#${row.photoId} ${row.currentPath || row.from} → ${row.mainValue.trim() || '(no ID)'}${row.suffix ? ` ${row.suffix}` : ''}${row.skip === 'x' ? ' · skipped' : ''}`

/** Several photos on one picture, each with its caption under it. */
function sheet(cells: { canvas: HTMLCanvasElement; text: string }[]): HTMLCanvasElement {
  const cols = Math.min(SHEET.cols, cells.length)
  const lines = Math.ceil(cells.length / cols)
  const out = document.createElement('canvas')
  out.width = cols * SHEET.cellW
  out.height = lines * (SHEET.cellH + SHEET.caption)
  const ctx = out.getContext('2d')!
  ctx.fillStyle = 'white'
  ctx.fillRect(0, 0, out.width, out.height)
  ctx.font = '600 17px system-ui, sans-serif'
  ctx.textBaseline = 'middle'
  cells.forEach(({ canvas, text }, i) => {
    const left = (i % cols) * SHEET.cellW, top = Math.floor(i / cols) * (SHEET.cellH + SHEET.caption)
    const scale = Math.min((SHEET.cellW - 8) / canvas.width, (SHEET.cellH - 8) / canvas.height)
    const w = canvas.width * scale, h = canvas.height * scale
    ctx.drawImage(canvas, left + (SHEET.cellW - w) / 2, top + (SHEET.cellH - h) / 2, w, h)
    ctx.fillStyle = '#111'
    ctx.fillText(text, left + 8, top + SHEET.cellH + SHEET.caption / 2, SHEET.cellW - 16)
    ctx.strokeStyle = '#bbb'
    ctx.strokeRect(left + 0.5, top + 0.5, SHEET.cellW - 1, SHEET.cellH + SHEET.caption - 1)
  })
  return out
}

interface ViewArgs { files?: string[]; part?: 'photo' | 'envelope'; sheet?: boolean }

async function view(args: ViewArgs) {
  const { photoRows, fileMap } = useProcessingStore.getState()
  const part = args.part === 'photo' ? 'photo' : 'envelope'
  const wanted = (args.files ?? []).slice(0, MAX_VIEW).map((f) => String(f).trim().toLowerCase())
  const missing: string[] = []
  const found: { row: PhotoRow; file: File }[] = []
  for (const name of wanted) {
    const row = photoRows.find((r) => r.from.toLowerCase() === name || r.currentPath.toLowerCase() === name)
    const file = row && (fileMap.get(row.from) ?? fileMap.get(row.currentPath))
    if (row && file) found.push({ row, file })
    else missing.push(name)
  }
  const images: { files: string[]; blob: Blob }[] = []
  if (args.sheet) {
    for (let i = 0; i < found.length; i += SHEET.per) {
      const group = found.slice(i, i + SHEET.per)
      const cells = await Promise.all(group.map(async ({ row, file }) => ({ canvas: await render(row, file, part), text: caption(row) })))
      images.push({ files: group.map((g) => g.row.from), blob: await canvasToBlob(sheet(cells), 'image/jpeg', 0.85) })
    }
  } else {
    for (const { row, file } of found) images.push({ files: [row.from], blob: await canvasToBlob(await render(row, file, part), 'image/jpeg', 0.85) })
  }
  return { images, missing }
}

function edit(args: { edits?: HostEdit[]; by?: string }) {
  const store = useProcessingStore.getState()
  const outcome = applyEdits(store.photoRows, Array.isArray(args.edits) ? args.edits.slice(0, 2000) : [], namesOf())
  if (outcome.changes.length) {
    store.setPhotoRows(outcome.rows)
    store.addHostChanges(outcome.changes, String(args.by ?? '').slice(0, 40) || 'the assistant')
  }
  const touched = new Set(outcome.changes.map((c) => c.file))
  return {
    changed: outcome.changes.map((c) => ({ file: c.file, what: c.what })),
    problems: outcome.problems,
    rows: hostState().rows.filter((r) => touched.has(r.file)),
  }
}

/** Every correction of the host taken back (also the review toolbar's button). */
export function undoHostChanges(): number {
  const store = useProcessingStore.getState()
  const changes = store.hostLog
  if (!changes.length) return 0
  store.setPhotoRows(undoEdits(store.photoRows, changes, namesOf()))
  store.clearHostChanges()
  return new Set(changes.map((c) => c.file)).size
}

async function run(name: string, args: Record<string, unknown>): Promise<unknown> {
  if (name === 'state') return hostState()
  if (name === 'view') return view(args as ViewArgs)
  if (name === 'edit') return edit(args as { edits?: HostEdit[]; by?: string })
  if (name === 'undo') return { rowsRestored: undoHostChanges() }
  throw new Error(`unknown command ${name}`)
}

/** Starts talking to the page hosting the app. Called once at startup when embedded. */
export function startHostBridge() {
  let timer: ReturnType<typeof setTimeout> | undefined
  let last = 0
  const tell = () => {
    timer = undefined
    last = Date.now()
    post({ type: 'state', state: hostState() })
  }
  // a reading in progress changes several times a second: told at most once a second
  const soon = () => {
    if (timer) return
    timer = setTimeout(tell, Math.max(250, 1000 - (Date.now() - last)))
  }
  useProcessingStore.subscribe((now, before) => {
    if (now.dirHandle !== before.dirHandle && now.dirHandle) {
      const dir = now.dirHandle
      folderBytes(dir).then((bytes) => {
        sized = { dir, bytes }
        soon()
      }, () => {})
    }
    if (now.photoRows !== before.photoRows || now.isProcessing !== before.isProcessing || now.progress !== before.progress ||
        now.dirHandle !== before.dirHandle || now.imageFiles !== before.imageFiles || now.hostChanges !== before.hostChanges ||
        now.wingChecks !== before.wingChecks || now.wingRun !== before.wingRun) soon()
  })
  window.addEventListener('message', (event) => {
    if (event.origin !== window.location.origin || event.source !== window.parent) return
    const m = event.data as { source?: string; type?: string; id?: string; name?: string; args?: Record<string, unknown>; ids?: string[] } | null
    if (!m || m.source !== 'rename-host') return
    if (m.type === 'camids' && Array.isArray(m.ids)) {
      setHostCamids(m.ids)
      soon()
    }
    if (m.type === 'hello') tell()
    // The page asks to show a part of the app: the review, when it starts a chat about the batch.
    if (m.type === 'show') window.dispatchEvent(new CustomEvent('rename:show', { detail: (m as { view?: string }).view }))
    if (m.type === 'command' && typeof m.id === 'string' && typeof m.name === 'string') {
      run(m.name, m.args ?? {}).then(
        (data) => post({ type: 'result', id: m.id, ok: true, data }),
        (error: unknown) => post({ type: 'result', id: m.id, ok: false, error: error instanceof Error ? error.message : String(error) }),
      )
    }
  })
  post({ type: 'ready', version: PROTOCOL_VERSION })
  tell()
}
