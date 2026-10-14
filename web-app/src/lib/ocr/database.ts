/**
 * The set of existing CAMIDs, for the database rule. Read live from the specimen database's
 * Google Sheet (public CSV export; Collection, Insectary and CRISPR tabs, CAM_ID* columns), cached
 * for a day, with the copy bundled at build time as fallback when offline.
 */
const SHEET = '1QZj6YgHAJ9NmFXFPCtu-i-1NDuDmAdMF2Wogts7S2_4'
const TABS = { Collection_data: '900206579', Insectary_data: '402580526', CRISPR: '952436162' }
const CACHE_KEY = 'camid-set-v1'
const MAX_AGE_MS = 24 * 3600 * 1000
const CAMID = /^CAM\d{6}$/

/** Minimal RFC 4180 CSV parser (quoted fields, embedded commas, quotes and newlines). */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = [], field = '', quoted = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++ }
      else if (c === '"') quoted = false
      else field += c
    } else if (c === '"') quoted = true
    else if (c === ',') { row.push(field); field = '' }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(field); rows.push(row); row = []; field = ''
    } else field += c
  }
  if (field || row.length) { row.push(field); rows.push(row) }
  return rows
}

/** CAMIDs from every column whose header starts with CAM_ID. */
export function camidsFromCsv(text: string): string[] {
  const [header, ...rows] = parseCsv(text.replace(/^\uFEFF/, ''))
  const cols = (header ?? []).map((h, i) => (/^CAM_ID/i.test(h.trim()) ? i : -1)).filter((i) => i >= 0)
  const out: string[] = []
  for (const r of rows) for (const c of cols) {
    const v = (r[c] ?? '').trim()
    if (CAMID.test(v)) out.push(v)
  }
  return out
}

async function fetchLive(): Promise<string[]> {
  const texts = await Promise.all(Object.values(TABS).map(async (gid) => {
    const res = await fetch(`https://docs.google.com/spreadsheets/d/${SHEET}/export?format=csv&gid=${gid}`)
    if (!res.ok) throw new Error(`database tab ${gid}: HTTP ${res.status}`)
    return res.text()
  }))
  const ids = [...new Set(texts.flatMap(camidsFromCsv))].sort()
  if (ids.length < 1000) throw new Error('database export looks incomplete')
  return ids
}

export interface CamidSet { ids: Set<string>; source: 'live' | 'cache' | 'bundled' | 'host'; updated: string }

// Embedded in a database app (lib/embed/host.ts): the IDs come from that app, as it has them now.
let hostIds: { ids: Set<string>; updated: string } | null = null
let hostWaiting: (() => void)[] = []
export function setHostCamids(ids: string[]) {
  hostIds = { ids: new Set(ids.map((id) => String(id).trim().toUpperCase()).filter(Boolean)), updated: new Date().toISOString() }
  hostWaiting.forEach((resolve) => resolve())
  hostWaiting = []
}
/** What the host gave, for the state told back to it (null: not embedded, or not given yet). */
export const hostDatabase = () => (hostIds ? { ids: hostIds.ids.size, source: 'host' } : null)
const embedded = () => typeof window !== 'undefined' && window.parent !== window && new URLSearchParams(window.location.search).has('embed')

export async function loadCamidSet(): Promise<CamidSet> {
  if (embedded()) {
    // the host sends them when the frame says it is ready; a moment's wait covers a slow answer
    if (!hostIds) await Promise.race([new Promise<void>((resolve) => hostWaiting.push(resolve)), new Promise((resolve) => setTimeout(resolve, 8000))])
    if (hostIds) return { ids: hostIds.ids, source: 'host', updated: hostIds.updated }
  }
  try {
    const cached = JSON.parse(localStorage.getItem(CACHE_KEY) ?? 'null') as { ids: string[]; t: number } | null
    if (cached && Date.now() - cached.t < MAX_AGE_MS) {
      return { ids: new Set(cached.ids), source: 'cache', updated: new Date(cached.t).toISOString() }
    }
    const ids = await fetchLive()
    localStorage.setItem(CACHE_KEY, JSON.stringify({ ids, t: Date.now() }))
    return { ids: new Set(ids), source: 'live', updated: new Date().toISOString() }
  } catch {
    const ids = (await (await fetch(`${import.meta.env.BASE_URL}db/camids.json`)).json()) as string[]
    return { ids: new Set(ids), source: 'bundled', updated: '2026-09-25' }
  }
}
