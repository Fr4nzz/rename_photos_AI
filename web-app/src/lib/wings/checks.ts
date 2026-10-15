/**
 * What the wings say about a batch's names: a specimen is photographed with its wings' upper side up
 * (d) and turned over (v), so the side the wings show should match the name's letter, the two photos
 * should show the same wings, and the forewings should be on top once the photo is turned as chosen.
 * Rules as in the models' archive check (ithomiini_maps, scripts/dorsal_ventral/12_evaluate.py).
 */
import type { PhotoRow, SuffixMode } from '@/types'
import { pendingRotation } from '../rotationPlan'
import type { Side, WingResult } from './model'

export interface WingCheck extends WingResult {
  /** clockwise degrees already written to the file when it was checked (rotApplied then) */
  applied: number
}

export type WingCode = 'wing-side' | 'wing-count' | 'wing-turn'
export interface WingFinding { code: WingCode; text: string }

/** a joint pair probability below this calls a wing group's two photos swapped */
const SWAP_Q = 0.05
/** the wings' turn is trusted with this many wings, this sure */
const TURN_WINGS = 3
const TURN_SURE = 0.9

const norm = (a: number) => ((a % 360) + 360) % 360
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
const logit = (p: number) => Math.log((p + 1e-12) / (1 - p + 1e-12))
const decided = (s: Side) => s === 'dorsal' || s === 'ventral'
const LETTER: Record<string, 'dorsal' | 'ventral'> = { d: 'dorsal', v: 'ventral' }
const PLACE: Record<string, string> = {
  forewing_left: 'left forewing', forewing_right: 'right forewing', hindwing_left: 'left hindwing', hindwing_right: 'right hindwing',
}

/** Clockwise turn the wings ask for, from how the photo shows now (null: they do not say). */
export function wingTurn(check: WingCheck, row: Pick<PhotoRow, 'rotApplied'>): number | null {
  if (check.wings.length < TURN_WINGS || check.pTurn < TURN_SURE) return null
  // the photo was checked with `applied` written; what was written since already turned it
  return norm(90 * check.turnK - (Number(row.rotApplied || 0) - check.applied))
}

/** The findings of each photo, by its original name. */
export function wingFindings(rows: PhotoRow[], checks: Map<string, WingCheck>, suffixMode: SuffixMode): Map<string, WingFinding[]> {
  const out = new Map<string, WingFinding[]>()
  if (suffixMode === 'Wing Clips') return out
  const add = (row: PhotoRow, code: WingCode, text: string) => {
    const list = out.get(row.from) ?? []
    if (!list.some((f) => f.text === text)) list.push({ code, text })
    out.set(row.from, list)
  }

  for (const row of rows) {
    const c = checks.get(row.from)
    if (!c || row.skip === 'x') continue
    if (!c.wings.length) { add(row, 'wing-count', 'No wing found'); continue }
    const turn = wingTurn(c, row)
    if (turn !== null && turn !== pendingRotation(row)) {
      const more = norm(turn - pendingRotation(row))
      add(row, 'wing-turn', more === 180
        ? 'The wings look upside down: forewings should be on top (turn 180°)'
        : `The wings look turned: forewings should be on top (turn ${more}° clockwise)`)
    }
  }
  if (suffixMode !== 'Standard') return out

  const byId = new Map<string, PhotoRow[]>()
  for (const row of rows) {
    const id = row.mainValue.trim()
    if (!id || row.skip === 'x' || !checks.get(row.from)?.wings.length) continue
    byId.set(id, [...(byId.get(id) ?? []), row])
  }
  for (const group of byId.values()) {
    const d = group.filter((r) => r.suffix === 'd'), v = group.filter((r) => r.suffix === 'v')
    const pairSettled = new Set<string>()
    if (group.length === 2 && d.length === 1 && v.length === 1) {
      const dc = checks.get(d[0].from)!, vc = checks.get(v[0].from)!
      // each wing group: the same side in both photos, the other way round, or as named
      const verdicts: [string, string][] = []
      for (const [label, key] of [['forewings', 'fore'], ['hindwings', 'hind']] as const) {
        const a = dc[key], b = vc[key]
        if (a.p === null || b.p === null) continue
        const q = sigmoid(logit(a.p) - logit(b.p))
        if (a.side === b.side && decided(a.side)) verdicts.push([label, `same-${a.side}`])
        else if (q < SWAP_Q) verdicts.push([label, 'swapped'])
        else verdicts.push([label, q >= 0.5 ? 'ok' : 'unsure'])
      }
      const kinds = new Set(verdicts.map(([, x]) => x))
      if (verdicts.length && kinds.size === 1 && kinds.has('swapped')) {
        add(d[0], 'wing-side', 'The d and v photos look swapped: the wings here show their underside')
        add(v[0], 'wing-side', 'The d and v photos look swapped: the wings here show their upper side')
      } else if (verdicts.length && kinds.size === 1 && [...kinds][0].startsWith('same-')) {
        const side = [...kinds][0].slice(5)
        for (const r of [d[0], v[0]]) add(r, 'wing-side', `Both photos of this ID show the ${side === 'dorsal' ? 'upper side' : 'underside'} of the wings`)
      } else if (verdicts.some(([, x]) => x === 'swapped' || x.startsWith('same-'))) {
        const what = verdicts.filter(([, x]) => x === 'swapped' || x.startsWith('same-'))
          .map(([g, x]) => (x === 'swapped' ? `${g} the other way round` : `${g} ${x === 'same-dorsal' ? 'upper side' : 'underside'} up in both`)).join(', ')
        for (const r of [d[0], v[0]]) add(r, 'wing-side', `Some wings were not turned over between the two photos: ${what}`)
      } else if (verdicts.length && kinds.size === 1 && kinds.has('ok')) {
        pairSettled.add(d[0].from).add(v[0].from)
      }
      // the same wings in both photos: one fewer means a wing moved or fell between them
      if (dc.wings.length !== vc.wings.length) {
        for (const [r, c, o] of [[d[0], dc, vc], [v[0], vc, dc]] as const) {
          if (c.wings.length >= o.wings.length) continue
          const absent = c.missing.filter((p) => !o.missing.includes(p)).map((p) => PLACE[p])
          add(r, 'wing-count', `${c.wings.length} wing${c.wings.length === 1 ? '' : 's'} here, ${o.wings.length} in the ${r === d[0] ? 'v' : 'd'} photo${absent.length ? ` (no ${absent.join(', ')})` : ''}`)
        }
      }
    }
    for (const r of group) {
      const c = checks.get(r.from)!
      if (out.get(r.from)?.some((f) => f.code === 'wing-side')) continue
      if (c.side === 'mixed') {
        add(r, 'wing-side', `Mixed sides: forewings show their ${c.fore.side === 'dorsal' ? 'upper side' : 'underside'}, hindwings their ${c.hind.side === 'dorsal' ? 'upper side' : 'underside'}`)
        continue
      }
      const named = LETTER[r.suffix[0]]
      if (!named || pairSettled.has(r.from) || !decided(c.side) || c.side === named) continue
      add(r, 'wing-side', `Named ${r.suffix} but the wings show their ${c.side === 'dorsal' ? 'upper side' : 'underside'}`)
    }
  }
  return out
}

/**
 * The rows with each finding's code among the reasons to look at them (once). A photo left to check
 * which way up it is ('turn-unsure') no longer waits for that when its wings agree with its turn.
 */
export function withWingReasons(rows: PhotoRow[], findings: Map<string, WingFinding[]>, checks: Map<string, WingCheck>): PhotoRow[] {
  return rows.map((r) => {
    const codes = [...new Set((findings.get(r.from) ?? []).map((f) => f.code))]
    const reasons = r.review.split(',').filter(Boolean)
    const check = checks.get(r.from)
    const settled = reasons.includes('turn-unsure') && !!check && wingTurn(check, r) === pendingRotation(r)
    const kept = settled ? reasons.filter((x) => x !== 'turn-unsure') : reasons
    const fresh = codes.filter((c) => !kept.includes(c))
    return fresh.length || settled ? { ...r, review: [...kept, ...fresh].join(',') } : r
  })
}
