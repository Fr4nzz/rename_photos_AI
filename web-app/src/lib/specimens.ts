/**
 * The review by specimen: the photos of a batch grouped by the ID read on their envelopes, in the
 * order they were taken, and what each group still needs. A specimen is photographed twice, dorsal
 * then ventral, so in the standard naming a group with another number of photos is worth a look.
 */
import type { PhotoRow, SuffixMode } from '@/types'

export interface Specimen {
  /** stable while its ID is being typed: the first photo's number */
  key: number
  id: string
  /** its photos in the order taken, skipped ones too */
  rows: PhotoRow[]
  /** ok, or why it needs a person: a photo waiting for review, no ID, or not two photos */
  state: 'ok' | 'review' | 'no-id' | 'count'
  /** photos that will be renamed (not skipped) */
  photos: number
}

const taken = (r: PhotoRow) => (r.captureDate ? Date.parse(r.captureDate) : NaN)

export function groupSpecimens(rows: PhotoRow[], suffixMode: SuffixMode = 'Standard'): Specimen[] {
  const ordered = [...rows].sort((a, b) => (taken(a) || Infinity) - (taken(b) || Infinity) || a.photoId - b.photoId)
  const byId = new Map<string, PhotoRow[]>()
  const groups: PhotoRow[][] = []
  for (const row of ordered) {
    const id = row.mainValue.trim()
    // a photo without an ID stands alone: nothing says which specimen it is
    if (!id) { groups.push([row]); continue }
    const group = byId.get(id)
    if (group) group.push(row)
    else { const fresh = [row]; byId.set(id, fresh); groups.push(fresh) }
  }
  return groups.map((group) => {
    const id = group[0].mainValue.trim()
    const active = group.filter((r) => r.skip !== 'x')
    const state: Specimen['state'] = !id ? 'no-id'
      : active.some((r) => r.review) ? 'review'
      : suffixMode === 'Standard' && active.length !== 2 ? 'count'
      : 'ok'
    return { key: group[0].photoId, id, rows: group, state, photos: active.length }
  })
}

/** What a specimen's state says, in a few words. */
export function specimenNote(s: Specimen): string {
  if (s.state === 'no-id') return 'No ID'
  if (s.state === 'review') return 'Needs review'
  if (s.state === 'count') return s.photos === 0 ? 'Every photo skipped' : s.photos === 1 ? 'Only 1 photo' : `${s.photos} photos`
  return `${s.photos} photos`
}
