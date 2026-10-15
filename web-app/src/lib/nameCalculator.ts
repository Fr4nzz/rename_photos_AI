import type { PhotoRow, SuffixMode } from '@/types'

function generateSuffixes(
  groupSize: number,
  mode: SuffixMode,
  customPattern: string[]
): string[] {
  if (mode === 'Wing Clips') {
    return Array.from({ length: groupSize }, (_, i) => `v${i + 1}`)
  }

  let pattern: string[]
  if (mode === 'Standard') {
    pattern = ['d', 'v']
  } else if (mode === 'Custom') {
    pattern = customPattern.length > 0 ? customPattern : []
    if (pattern.length === 0) {
      return Array.from({ length: groupSize }, (_, i) => `_${i + 1}`)
    }
  } else {
    return Array.from({ length: groupSize }, (_, i) => `_${i + 1}`)
  }

  const suffixes: string[] = []
  for (let i = 0; i < groupSize; i++) {
    const roundNum = Math.floor(i / pattern.length)
    const patIdx = i % pattern.length
    let suffix = pattern[patIdx]
    if (roundNum > 0) suffix += String(roundNum + 1)
    suffixes.push(suffix)
  }
  return suffixes
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
const logit = (p: number) => Math.log((p + 1e-12) / (1 - p + 1e-12))
/** sure of a pair's sides: the joint probability that the first is the dorsal one, this far from ½ */
const PAIR_SURE = 0.95
const SIDE_SURE = 0.95

/**
 * The views of one ID's photos from their wings (standard naming), in the order given, or null when
 * the wings do not settle them: a pair whose wings tell one photo's upper side from the other's;
 * more photos when each one's side is clear and both sides are there (dorsal ones d, d2…, ventral v, v2…).
 */
export function viewsFromWings(rows: Pick<PhotoRow, 'pDorsal'>[]): string[] | null {
  const ps = rows.map((r) => (r.pDorsal ? Number(r.pDorsal) : NaN))
  if (rows.length < 2 || ps.some((p) => !Number.isFinite(p))) return null
  if (rows.length === 2) {
    const [a, b] = ps
    const q = sigmoid(logit(a) - logit(b))
    if (q >= PAIR_SURE && a > 0.5 && b < 0.5) return ['d', 'v']
    if (q <= 1 - PAIR_SURE && b > 0.5 && a < 0.5) return ['v', 'd']
    return null
  }
  const sides = ps.map((p) => (p >= SIDE_SURE ? 'd' : p <= 1 - SIDE_SURE ? 'v' : null))
  if (sides.includes(null) || !sides.includes('d') || !sides.includes('v')) return null
  const seen = { d: 0, v: 0 }
  return sides.map((x) => {
    const n = ++seen[x as 'd' | 'v']
    return n === 1 ? x! : `${x}${n}`
  })
}

/**
 * Each photo's view and final name, by ID (skipped photos and those without an ID get none). A view
 * typed by hand stays; the others follow the mode's pattern in the order given, or, in the standard
 * naming, the wings when they settle them (a butterfly photographed ventral side first, or a CRISPR
 * one with more views).
 */
export function calculateFinalNames(
  rows: PhotoRow[],
  _mainColumn: string,
  suffixMode: SuffixMode,
  customSuffixesStr: string
): PhotoRow[] {
  if (rows.length === 0) return rows

  const customPattern = customSuffixesStr
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)

  const manual = (r: PhotoRow) => r.viewBy === 'manual' && !!r.suffix.trim()
  const result = rows.map((r) => (manual(r) ? { ...r, to: '' } : { ...r, to: '', suffix: '', viewBy: '' as const }))

  // Group by main value (skip rows marked to skip)
  const groups = new Map<string, number[]>()
  for (let i = 0; i < result.length; i++) {
    const row = result[i]
    if (row.skip === 'x' || !row.mainValue?.trim()) continue
    const key = row.mainValue.trim()
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(i)
  }

  for (const [identifier, indices] of groups) {
    const typed = indices.filter((i) => manual(result[i]))
    const auto = indices.filter((i) => !manual(result[i]))
    const byOrder = generateSuffixes(indices.length, suffixMode, customPattern)
    const wings = suffixMode === 'Standard' && !typed.length ? viewsFromWings(auto.map((i) => result[i])) : null
    // the pattern's views not typed on another photo, in order
    const used = new Set(typed.map((i) => result[i].suffix.trim()))
    const free = [...byOrder.filter((x) => !used.has(x)), ...generateSuffixes(indices.length * 2, suffixMode, customPattern).filter((x) => !used.has(x) && !byOrder.includes(x))]
    auto.forEach((idx, k) => {
      const suffix = wings ? wings[k] : free[k]
      result[idx].suffix = suffix
      if (wings && wings[k] !== byOrder[k]) result[idx].viewBy = 'wings'
    })
    for (const idx of indices) result[idx].to = `${identifier}${result[idx].suffix}${getExtension(result[idx].from)}`
  }

  return result
}

function getExtension(filename: string): string {
  const dot = filename.lastIndexOf('.')
  return dot >= 0 ? filename.slice(dot) : ''
}
