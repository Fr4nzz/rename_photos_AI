import type { PhotoRow } from '@/types'

/**
 * Every photo's CAMID in shooting order (last digits), coloured by state: gaps, repeats and
 * outliers stand out at a glance. Click a chip to jump to its card.
 */
export function IdTimeline({ rows, onJump }: { rows: PhotoRow[]; onJump: (photoId: number) => void }) {
  if (rows.length < 2) return null
  const ordered = [...rows].sort((a, b) =>
    (a.captureDate ?? '').localeCompare(b.captureDate ?? '') || a.photoId - b.photoId)
  return (
    <div className="flex gap-1 overflow-x-auto border-b px-3 py-1.5" aria-label="CAMIDs in shooting order">
      {ordered.map((r) => {
        const id = r.mainValue.trim()
        const state = r.skip === 'x' ? 'skip' : r.review ? 'review' : id ? 'ok' : 'empty'
        const tone = {
          ok: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
          review: 'border-amber-500 bg-amber-500/15 text-amber-700 dark:text-amber-300',
          empty: 'border-dashed text-muted-foreground',
          skip: 'border-transparent text-muted-foreground/60 line-through',
        }[state]
        return (
          <button
            key={r.photoId}
            type="button"
            onClick={() => onJump(r.photoId)}
            title={`${r.from}${id ? ` → ${id}` : ''}`}
            className={`shrink-0 rounded border px-1.5 py-0.5 font-mono text-[11px] leading-4 hover:ring-1 hover:ring-primary ${tone}`}
          >
            {id ? id.slice(-4) : '····'}
          </button>
        )
      })}
    </div>
  )
}
