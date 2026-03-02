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

  const result = rows.map((r) => ({ ...r, to: '', suffix: '' }))

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
    const suffixes = generateSuffixes(indices.length, suffixMode, customPattern)
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i]
      result[idx].suffix = suffixes[i]
      const ext = getExtension(result[idx].from)
      result[idx].to = `${identifier}${suffixes[i]}${ext}`
    }
  }

  return result
}

function getExtension(filename: string): string {
  const dot = filename.lastIndexOf('.')
  return dot >= 0 ? filename.slice(dot) : ''
}
