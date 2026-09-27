/**
 * From per-photo readings to renaming decisions (the rules evaluated on the sealed test half:
 * 94.9% renamed correctly, 0.31% wrong, 4.8% to a person).
 *
 *  1. Database rule: the most confident whole-line CAMID reading that exists in the database.
 *  2. Batch check: photos are taken in sessions covering nearby CAMIDs. In capture-time order, a
 *     proposal is sent to review when none of the proposals of the 3 photos on either side is
 *     within 15 of it (settings chosen on the round-2 training half).
 *  3. Repeated ID: a specimen's photos (dorsal, ventral) are taken back to back, so the same
 *     CAMID on photos that are not next to each other in capture order, or on more than two
 *     photos, sends those photos to review (it caught a misread the batch check could not).
 *  4. Everything without a confident, consistent proposal goes to review with a pre-filled
 *     suggestion: the most CAMID-like reading with handwriting lookalikes fixed, plus database IDs
 *     within two edits, closest to the neighbouring photos' IDs first.
 */
import { CAMID_RE, chooseCamid, normalizeReading, type PhotoReading } from './pipeline'

export type ReviewReason = 'no-reading' | 'not-in-database' | 'out-of-sequence' | 'repeated-id' | 'unreadable-photo'

export interface PhotoInput {
  name: string
  reading?: PhotoReading
  error?: string
  /** EXIF capture time (ms since epoch) or null */
  capturedAt: number | null
}

export interface Decision {
  name: string
  camid: string | null
  auto: boolean
  reasons: ReviewReason[]
  /** pre-filled text for the review box */
  prefill: string
  /** existing CAMIDs close to the reading, best first */
  candidates: string[]
  conf: number
}

const NEAR = 3
const GAP = 15

const LOOKALIKE: Record<string, string> = {
  O: '0', o: '0', D: '0', Q: '0', U: '0', I: '1', l: '1', i: '1', '|': '1', L: '2', Z: '2', z: '2',
  S: '5', s: '5', G: '6', b: '6', T: '7', B: '8', e: '8', g: '9', q: '9', A: '4',
}

export function camidLikeness(text: string): number {
  const t = text.replace(/\s+/g, '')
  const m = /^C[A4][MN]?/i.exec(t)
  if (!m) return 0
  const digits = [...t.slice(m[0].length)].filter((c) => /\d/.test(c) || c in LOOKALIKE).length
  return (m[0].toUpperCase().startsWith('CAM') ? 30 : 10) + Math.min(digits, 6)
}

export function prefill(text: string): string {
  const t = text.replace(/\s+/g, '')
  const m = /^C[A4][MN]?/i.exec(t)
  let body = [...(m ? t.slice(m[0].length) : t)].map((c) => (/\d/.test(c) ? c : LOOKALIKE[c] ?? '?')).join('')
  if (body.length >= 8) body = body.slice(-6) // a crossed-out ID run together with its replacement
  return 'CAM' + body
}

function editDistance(a: string, b: string): number {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j)
  for (let i = 1; i <= a.length; i++) {
    const cur = [i]
    for (let j = 1; j <= b.length; j++) {
      cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] || a[i - 1] === '?' ? 0 : 1)))
    }
    prev = cur
  }
  return prev[b.length]
}

const num = (id: string) => Number(id.slice(3))

export function decide(photos: PhotoInput[], known: Set<string> | null): Decision[] {
  // capture order (photos without a time keep their name order at the end)
  const order = photos.map((p, i) => ({ p, i })).sort((a, b) =>
    (a.p.capturedAt ?? Infinity) - (b.p.capturedAt ?? Infinity) || a.p.name.localeCompare(b.p.name))
  const proposal = new Map<number, { id: string; conf: number }>()
  for (const { p, i } of order) {
    const choice = p.reading ? chooseCamid(p.reading, known) : null
    if (choice) proposal.set(i, choice)
  }
  const seq = order.map((o) => o.i)
  const positions = new Map<string, number[]>()
  seq.forEach((i, pos) => {
    const id = proposal.get(i)?.id
    if (id) positions.set(id, [...(positions.get(id) ?? []), pos])
  })
  const repeated = new Set<string>()
  for (const [id, pos] of positions) {
    if (pos.length > 2 || (pos.length === 2 && pos[1] - pos[0] > 1)) repeated.add(id)
  }
  const decisions: Decision[] = new Array(photos.length)
  seq.forEach((i, pos) => {
    const p = photos[i]
    const neighbours = [...seq.slice(Math.max(0, pos - NEAR), pos), ...seq.slice(pos + 1, pos + 1 + NEAR)]
      .map((j) => proposal.get(j)?.id).filter((x): x is string => !!x)
    const mine = proposal.get(i)
    const reasons: ReviewReason[] = []
    if (p.error) reasons.push('unreadable-photo')
    else if (!mine) {
      const anyReading = p.reading?.candidates.length
      reasons.push(anyReading ? 'not-in-database' : 'no-reading')
    } else {
      if (neighbours.length >= 2 && Math.min(...neighbours.map((n) => Math.abs(num(n) - num(mine.id)))) > GAP) {
        reasons.push('out-of-sequence')
      }
      if (repeated.has(mine.id)) reasons.push('repeated-id')
    }
    // suggestion for review: the most CAMID-like line, lookalikes fixed; database IDs close to it
    let fill = mine?.id ?? ''
    let candidates: string[] = []
    if (!mine || reasons.length) {
      const texts = p.reading?.lines.map((l) => l.text) ?? []
      const best = texts.map((t) => ({ t, s: camidLikeness(t) })).sort((a, b) => b.s - a.s)[0]
      if (!fill) fill = best && best.s >= 10 ? prefill(best.t) : 'CAM'
      const body = normalizeReading(fill)
      if (known && body.length >= 8) {
        const centre = neighbours.length ? neighbours.map(num).sort((a, b) => a - b)[Math.floor(neighbours.length / 2)] : null
        candidates = [...known]
          .map((id) => ({ id, d: editDistance(body.slice(3), id.slice(3)) }))
          .filter((c) => c.d <= 2)
          .sort((a, b) => a.d - b.d || (centre === null ? 0 : Math.abs(num(a.id) - centre) - Math.abs(num(b.id) - centre)))
          .slice(0, 3).map((c) => c.id)
      }
      // a neighbour's ID is the likely answer for the second photo (dorsal/ventral) of a specimen
      for (const n of neighbours) if (!candidates.includes(n) && CAMID_RE.test(n) && candidates.length < 4 && editDistance(body.slice(3), n.slice(3)) <= 2) candidates.push(n)
    }
    decisions[i] = { name: p.name, camid: mine?.id ?? null, auto: !!mine && reasons.length === 0, reasons, prefill: fill, candidates, conf: mine?.conf ?? 0 }
  })
  return decisions
}
