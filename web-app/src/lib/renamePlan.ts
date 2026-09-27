/**
 * Rename planning, done before any file is touched.
 *
 * - Names are compared case-insensitively (Windows and macOS folders are case-insensitive).
 * - An operation is refused when its target already exists and is not itself being renamed away
 *   (an unrelated file), or when two operations want the same target.
 * - Renames whose target is currently held by another file of the batch (swaps, cycles, case-only
 *   changes) go through a unique temporary name, so no file is ever overwritten.
 */

export interface RenameOp {
  src: string
  dst: string
}

export interface PlannedStep {
  from: string
  to: string
  /** true for the intermediate move to a temporary name */
  temp: boolean
  /** the original name of the file this step moves (for the undo log) */
  original: string
}

export interface RenamePlan {
  steps: PlannedStep[]
  refused: { op: RenameOp; reason: 'target-exists' | 'duplicate-target' | 'missing-source' }[]
}

const key = (name: string) => name.toLowerCase()

export function planRenames(ops: RenameOp[], existingNames: Iterable<string>): RenamePlan {
  const existing = new Set([...existingNames].map(key))
  const refused: RenamePlan['refused'] = []
  const active = ops.filter((op) => op.src !== op.dst)

  const targetCount = new Map<string, number>()
  for (const op of active) targetCount.set(key(op.dst), (targetCount.get(key(op.dst)) ?? 0) + 1)
  const sources = new Set(active.map((op) => key(op.src)))

  const accepted: RenameOp[] = []
  for (const op of active) {
    if (!existing.has(key(op.src))) refused.push({ op, reason: 'missing-source' })
    else if ((targetCount.get(key(op.dst)) ?? 0) > 1) refused.push({ op, reason: 'duplicate-target' })
    else if (existing.has(key(op.dst)) && !sources.has(key(op.dst))) refused.push({ op, reason: 'target-exists' })
    else accepted.push(op)
  }
  // A refused op leaves its source in place, which may block another op's target (repeat for chains).
  let finalOps = accepted
  for (;;) {
    const staying = new Set(refused.map((r) => key(r.op.src)))
    const blocked = finalOps.filter((op) => staying.has(key(op.dst)))
    if (blocked.length === 0) break
    for (const op of blocked) refused.push({ op, reason: 'target-exists' })
    finalOps = finalOps.filter((op) => !blocked.includes(op))
  }

  const steps: PlannedStep[] = []
  const via = new Map<string, string>()
  // Phase 1: a source that is also someone's target (or only changes case) moves to a temp name.
  finalOps.forEach((op, i) => {
    const blocking = finalOps.some((o) => o !== op && key(o.dst) === key(op.src)) || key(op.src) === key(op.dst)
    if (blocking) {
      const temp = `${op.src}.renaming-${i}.tmp`
      steps.push({ from: op.src, to: temp, temp: true, original: op.src })
      via.set(op.src, temp)
    }
  })
  // Phase 2: every file goes to its final name; targets are now free.
  for (const op of finalOps) {
    steps.push({ from: via.get(op.src) ?? op.src, to: op.dst, temp: false, original: op.src })
  }
  return { steps, refused }
}

/** Index a folder listing: lower-case stem -> file names with that stem (any extension, any case). */
export function indexByStem(names: Iterable<string>): Map<string, string[]> {
  const index = new Map<string, string[]>()
  for (const name of names) {
    const dot = name.lastIndexOf('.')
    const stem = key(dot > 0 ? name.slice(0, dot) : name)
    const list = index.get(stem)
    if (list) list.push(name)
    else index.set(stem, [name])
  }
  return index
}

/** RAW companions of a photo (same stem, RAW extension, any case), excluding the photo itself. */
export function companionsOf(fileName: string, stemIndex: Map<string, string[]>, rawExtensions: Set<string>): string[] {
  const dot = fileName.lastIndexOf('.')
  const stem = key(dot > 0 ? fileName.slice(0, dot) : fileName)
  return (stemIndex.get(stem) ?? []).filter((n) => n !== fileName && rawExtensions.has(key(n.slice(n.lastIndexOf('.')))))
}
