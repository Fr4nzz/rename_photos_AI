/**
 * Embedded mode, the part without a browser: what the batch looks like to the page hosting the app
 * (`snapshot`), and the corrections that page can make to it (`applyEdits`, `undoEdits`).
 *
 * The host is another page of the same origin showing this app in a frame (address ending in
 * `?embed`): a database app whose assistant double-checks the readings before the files are
 * renamed. It never touches files: it reads rows, asks for small copies of photos, and changes
 * rows, exactly what a person does in the review. Renaming stays the person's click.
 */
import type { PhotoRow, SuffixMode } from '@/types'
import type { PhotoReading } from '../ocr/pipeline'
import { calculateFinalNames } from '../nameCalculator'
import { pendingRotation } from '../rotationPlan'
import { wingTurn, type WingCheck, type WingFinding } from '../wings/checks'
import type { Side } from '../wings/model'

export const PROTOCOL_VERSION = 1

/** A row as the host sees it. */
export interface HostRow {
  /** the file's name when the folder was opened (how the host names a photo) */
  file: string
  /** its name now, when it was renamed since */
  now?: string
  n: number
  id: string
  /** d, v, d2…: by the order the photos of one ID were taken in, or by their wings (`viewFrom`), or as set */
  view: string
  viewFrom?: 'wings' | 'manual'
  to: string
  taken: string | null
  status: PhotoRow['status']
  skip?: true
  /** why the reader left it for a person ('' when sure, or confirmed) */
  review: string[]
  suggest: string[]
  otherIdsRead?: string[]
  note?: string
  /** how sure the reader was of `id` (0–1), and where it read it */
  conf?: number
  readFrom?: PhotoReading['source']
  /** clockwise degrees: chosen for the photo, and still to be written to the file */
  rotation: number
  rotationPending: number
  rotationFrom: PhotoRow['rotSource']
  /** changed by the host: what it changed */
  byHost?: string[]
  /** the wing check (when it ran on this photo): the side the wings show (dorsal: upper side up), P(dorsal),
   *  each wing group's side, the wings found and the places without one, and the clockwise turn that puts
   *  the forewings on top from how the file shows now (when the wings are sure of it) */
  wings?: { side: Side; pDorsal: number | null; forewings: Side; hindwings: Side; count: number; missing?: string[]; turn?: number }
  /** what the wings say against the name or the turn (a reason `wing-…` among `review` while it waits) */
  wingNotes?: string[]
}

/** The total size of a folder's files, photos and RAW files alike (what an upload of it takes). */
/** A whole photo's longest side for `view`: 1100 px unless asked, between 300 and 1600. */
export function viewSide(side: unknown): number {
  const n = Math.round(Number(side))
  return Number.isFinite(n) && n > 0 ? Math.min(1600, Math.max(300, n)) : 1100
}

export async function folderBytes(dir: FileSystemDirectoryHandle): Promise<number> {
  let bytes = 0
  for await (const entry of (dir as unknown as { values(): AsyncIterable<FileSystemHandle> }).values()) {
    if (entry.kind === 'file') bytes += (await (entry as FileSystemFileHandle).getFile()).size
  }
  return bytes
}

export interface HostState {
  version: number
  folder: string | null
  /** the folder can be written (rotating and renaming in place) */
  canWrite: boolean
  /** photos in the folder, and those with a row (read, or loaded from a results file) */
  files: number
  /** the size of everything in the folder, RAW files too (null until it is counted) */
  folderBytes: number | null
  reading: { on: boolean; percent: number; message: string }
  database: { ids: number; source: string } | null
  /** the wing check: running (photos done of total), and how many photos it has checked */
  wingCheck: { on: boolean; done: number; total: number; checked: number }
  rows: HostRow[]
}

/** One correction. `file`: the photo, by its name (the original one or the one it has now). */
export interface HostEdit {
  file: string
  /** the ID on the envelope */
  id?: string
  /** true (default when `id` is given): the photo no longer waits for a person */
  confirm?: boolean
  /** left for a person, with the reason shown on its card */
  leave?: string
  skip?: boolean
  /** turn the photo clockwise by 90, 180 or 270 from how it shows now */
  rotate?: number
  /** d, v… instead of the one its order gives */
  view?: string
  note?: string
}

/** What a host's edit changed in a row, kept to take it back. */
export interface HostChange {
  file: string
  what: string[]
  before: Partial<PhotoRow>
  /** the ID it replaced: what the reader (or the person) had, and what the host set */
  id?: { from: string; to: string }
}

const lower = (s: string) => s.trim().toLowerCase()
const split = (s: string, by: string | RegExp) => s.split(by).map((x) => x.trim()).filter(Boolean)
const norm = (a: number) => ((a % 360) + 360) % 360
export const HOST_REASON = 'assistant'

export function snapshotRows(
  rows: PhotoRow[],
  readings: Map<string, PhotoReading>,
  changed: Map<string, HostChange>,
  wings: { checks: Map<string, WingCheck>; findings: Map<string, WingFinding[]> } = { checks: new Map(), findings: new Map() },
): HostRow[] {
  const round = (p: number | null) => (p === null ? null : Math.round(p * 1000) / 1000)
  return rows.map((r) => {
    const reading = readings.get(r.from)
    const others = split(r.co, /\s+/)
    const check = wings.checks.get(r.from)
    const turn = check ? wingTurn(check, r) : null
    const notes = wings.findings.get(r.from)?.map((f) => f.text) ?? []
    return {
      file: r.from,
      ...(r.currentPath && r.currentPath !== r.from ? { now: r.currentPath } : {}),
      n: r.photoId,
      id: r.mainValue.trim(),
      view: r.suffix,
      ...(r.viewBy ? { viewFrom: r.viewBy } : {}),
      to: r.to,
      taken: r.captureDate,
      status: r.status,
      ...(r.skip === 'x' ? { skip: true as const } : {}),
      review: split(r.review, ','),
      suggest: split(r.suggest, /\s+/),
      ...(others.length ? { otherIdsRead: others } : {}),
      ...(r.n.trim() ? { note: r.n.trim() } : {}),
      ...(reading && reading.camid === r.mainValue.trim() ? { conf: Math.round(reading.conf * 100) / 100 } : {}),
      ...(reading ? { readFrom: reading.source } : {}),
      rotation: Number(r.rotChosen || 0),
      rotationPending: pendingRotation(r),
      rotationFrom: r.rotSource,
      ...(changed.has(r.from) ? { byHost: changed.get(r.from)!.what } : {}),
      ...(check ? {
        wings: {
          side: check.side, pDorsal: round(check.p), forewings: check.fore.side, hindwings: check.hind.side, count: check.wings.length,
          ...(check.missing.length ? { missing: check.missing } : {}), ...(turn !== null ? { turn } : {}),
        },
      } : {}),
      ...(notes.length ? { wingNotes: notes } : {}),
    }
  })
}

export interface EditOutcome {
  rows: PhotoRow[]
  /** the changes made, one per row touched (kept by the caller for undo) */
  changes: HostChange[]
  /** edits that could not be made, and why */
  problems: { file: string; error: string }[]
}

/** The rows after the host's corrections; final names follow, as after a person's edit. */
export function applyEdits(
  rows: PhotoRow[],
  edits: HostEdit[],
  names: { mainColumn: string; suffixMode: SuffixMode; customSuffixes: string },
): EditOutcome {
  let out = rows.map((r) => ({ ...r }))
  const changes: HostChange[] = []
  const problems: EditOutcome['problems'] = []
  const views = new Map<number, string>()
  let rename = false
  for (const e of edits) {
    const wanted = lower(String(e?.file ?? ''))
    const found = out.filter((r) => lower(r.from) === wanted || lower(r.currentPath) === wanted)
    if (found.length !== 1) {
      problems.push({ file: String(e?.file ?? ''), error: found.length ? 'more than one photo has that name' : 'no photo of that name in the batch' })
      continue
    }
    const row = found[0]
    const before: Partial<PhotoRow> = {}
    const what: string[] = []
    let idChange: HostChange['id']
    const set = <K extends keyof PhotoRow>(key: K, value: PhotoRow[K], label: string) => {
      if (row[key] === value) return
      if (!(key in before)) before[key] = row[key]
      row[key] = value
      what.push(label)
    }
    if (e.id !== undefined) {
      const id = String(e.id).replace(/\s+/g, '').toUpperCase()
      if (!id) {
        problems.push({ file: row.from, error: 'an empty ID: skip the photo instead' })
        continue
      }
      if (id !== row.mainValue.trim()) {
        idChange = { from: row.mainValue.trim(), to: id }
        set('mainValue', id, `ID ${row.mainValue.trim() || '(none)'} → ${id}`)
        rename = true
      }
    }
    if (e.skip !== undefined && (row.skip === 'x') !== !!e.skip) {
      set('skip', e.skip ? 'x' : '', e.skip ? 'skipped' : 'not skipped')
      rename = true
    }
    if (e.rotate !== undefined) {
      const turn = norm(Number(e.rotate))
      if (![90, 180, 270].includes(turn)) {
        problems.push({ file: row.from, error: 'rotate is 90, 180 or 270 (clockwise)' })
        continue
      }
      set('rotChosen', String(norm(Number(row.rotChosen || 0) + turn)), `turned ${turn}°`)
      set('rotSource', 'manual', '')
    }
    if (e.note !== undefined) set('n', String(e.note).slice(0, 300), 'note')
    if (e.leave !== undefined) {
      const reasons = split(row.review, ',')
      if (!reasons.includes(HOST_REASON)) set('review', [...reasons, HOST_REASON].join(','), 'left for you')
      const text = String(e.leave).slice(0, 300)
      if (text) set('n', text, 'note')
    } else if ((e.confirm ?? e.id !== undefined) && row.review) {
      set('review', '', 'confirmed')
    }
    if (e.view !== undefined && String(e.view).trim()) views.set(row.photoId, String(e.view).trim().slice(0, 6))
    if (what.length || views.has(row.photoId)) changes.push({ file: row.from, what: what.filter(Boolean), before, ...(idChange ? { id: idChange } : {}) })
  }
  // a view set stays as set (like one typed); the ID's other photos follow when the names are worked out
  for (const [photoId, view] of views) {
    const row = out.find((r) => r.photoId === photoId)!
    const c = changes.find((x) => x.file === row.from)!
    if ((row.suffix === view && row.viewBy === 'manual') || !row.mainValue.trim()) continue
    if (!('suffix' in c.before)) Object.assign(c.before, { suffix: row.suffix, to: row.to, viewBy: row.viewBy ?? '' })
    row.suffix = view
    row.viewBy = 'manual'
    c.what.push(`view ${view}`)
    rename = true
  }
  if (rename) {
    const named = calculateFinalNames(out, names.mainColumn, names.suffixMode, names.customSuffixes)
    out = named.map((r, i) => {
      const c = changes.find((x) => x.file === r.from)
      // names of other rows move too (a third photo of an ID becomes d2): theirs are not "changes"
      if (c && !('suffix' in c.before) && r.suffix !== rows[i].suffix) Object.assign(c.before, { suffix: rows[i].suffix, to: rows[i].to, viewBy: rows[i].viewBy ?? '' })
      return r
    })
  }
  return { rows: out, changes: changes.filter((c) => c.what.length), problems }
}

/** The rows with the host's changes taken back: each row as it was before the first of them. */
export function undoEdits(
  rows: PhotoRow[],
  changes: HostChange[],
  names: { mainColumn: string; suffixMode: SuffixMode; customSuffixes: string },
): PhotoRow[] {
  // a later change's `before` is an earlier change's result: per field, the earliest one wins
  const first = new Map<string, Partial<PhotoRow>>()
  for (const c of changes) first.set(c.file, { ...c.before, ...(first.get(c.file) ?? {}) })
  const restored = rows.map((r) => (first.has(r.from) ? { ...r, ...first.get(r.from) } : r))
  const views = new Map(restored.filter((r) => 'suffix' in (first.get(r.from) ?? {})).map((r) => [r.photoId, { suffix: r.suffix, to: r.to, viewBy: r.viewBy }]))
  // the names of the other photos of the IDs concerned follow, as after a person's edit
  const named = calculateFinalNames(restored, names.mainColumn, names.suffixMode, names.customSuffixes)
  return named.map((r) => (views.has(r.photoId) && r.mainValue.trim() ? { ...r, ...views.get(r.photoId)! } : r))
}
