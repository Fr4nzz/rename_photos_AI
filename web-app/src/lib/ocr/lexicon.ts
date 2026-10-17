/**
 * Lexicon margin of a read CAMID: how much more probable the recogniser makes it than the best other
 * database ID within two edits (CTC log-probability, nats). A confident misread of one round digit
 * (3/8, 9/8, 6/5) has a close database neighbour about as probable as itself; below 1 nat the photo
 * goes to a person (CAMID campaign of 14-15 Oct 2026, p3 weights: locked set 8 -> 2 wrong renames,
 * 13 -> 5 on photos without a CAMID, -0.11 pp correct). Port of its decode.py (`ctc_lexicon_logp`,
 * `lev_all`, `accepted_margin`).
 */

/** Below this margin (nats) a read ID is not taken without a person. */
export const MARGIN_MIN = 1

/** Posterior columns kept for the scorer: blank, 0-9, C, A, M. */
const LEX_CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'A', 'M']
const COL: Record<string, number> = Object.fromEntries(LEX_CHARS.map((c, i) => [c, i + 1]))
const K = LEX_CHARS.length + 1
/** a rival's log-probability when no other database ID is within two edits (as decode.py) */
const NO_RIVAL = -60
const CAMID = /^CAM[0-9]{6}$/

/** One line view whose greedy reading is CAM + 6 digits, with its per-step posteriors. */
export interface IdView {
  id: string
  steps: number
  /** steps x 14 (blank, 0-9, C, A, M) */
  probs: Float32Array
}

/** The recogniser's softmax [steps, classes] cut to the scorer's columns (the dictionary has blank at 0). */
export function lexiconPosteriors(probs: Float32Array, steps: number, classes: number, chars: string[]): Float32Array {
  const cols = [0, ...LEX_CHARS.map((c) => chars.indexOf(c))]
  const out = new Float32Array(steps * K)
  for (let t = 0; t < steps; t++) for (let k = 0; k < K; k++) out[t * K + k] = probs[t * classes + cols[k]]
  return out
}

/**
 * log P(id | x) by the CTC forward algorithm, in probability space rescaled at each step (as
 * decode.py's `ctc_lexicon_logp`, so an impossible label ends near -690 rather than -Infinity).
 */
export function ctcLogProb(p: Float32Array, steps: number, id: string): number {
  const label = [...id].map((c) => COL[c])
  const S = 2 * label.length + 1
  const ext = new Int32Array(S) // blanks (0) between and around the labels
  label.forEach((l, i) => { ext[2 * i + 1] = l })
  if (!steps) return Math.log(1e-300)
  let alpha = new Float64Array(S), next = new Float64Array(S)
  alpha[0] = p[ext[0]]; alpha[1] = p[ext[1]]
  let logscale = 0
  for (let t = 1; t < steps; t++) {
    let m = 0
    for (let s = 0; s < S; s++) {
      let a = alpha[s]
      if (s >= 1) a += alpha[s - 1]
      // skip a blank only between two different labels
      if (s >= 3 && s % 2 === 1 && ext[s] !== ext[s - 2]) a += alpha[s - 2]
      next[s] = a * p[t * K + ext[s]]
      if (next[s] > m) m = next[s]
    }
    if (m === 0) m = 1e-300
    for (let s = 0; s < S; s++) next[s] /= m
    logscale += Math.log(m)
    ;[alpha, next] = [next, alpha]
  }
  return Math.log(Math.max(alpha[S - 1] + alpha[S - 2], 1e-300)) + logscale
}

/** Edit distance; a '?' in `a` (an unreadable character) matches anything. */
export function editDistance(a: string, b: string): number {
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

/**
 * Margin of a CAMID taken by the database rule: the best, over the line views that read it exactly,
 * of log P(id) - log P(best other database ID within 2 edits of it), with that rival. Null when no
 * view reads it whole (an ID joined from two pieces): the margin does not apply.
 */
export function lexiconMargin(views: IdView[], id: string, known: Set<string>): { margin: number; rival: string | null } | null {
  const mine = views.filter((v) => v.id === id)
  if (!mine.length) return null
  const body = id.slice(3)
  const others = [...known].filter((k) => k !== id && CAMID.test(k) && editDistance(body, k.slice(3)) <= 2)
  let best: { margin: number; rival: string | null } | null = null
  for (const v of mine) {
    let rival: string | null = null, other = -Infinity
    for (const o of others) {
      const lp = ctcLogProb(v.probs, v.steps, o)
      if (lp > other) { other = lp; rival = o }
    }
    if (rival === null) other = NO_RIVAL
    const margin = ctcLogProb(v.probs, v.steps, id) - other
    if (!best || margin > best.margin) best = { margin, rival }
  }
  return best
}
