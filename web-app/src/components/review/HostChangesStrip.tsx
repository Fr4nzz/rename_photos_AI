import { ArrowRight, Sparkles } from 'lucide-react'
import { useProcessingStore } from '@/stores/processingStore'
import type { PhotoRow } from '@/types'

/**
 * Embedded mode: what the hosting page's assistant changed in the batch, a chip per photo (the ID
 * the reader had and the one it set, or what else it did); the photos it left for a person first.
 * A click goes to the photo's card.
 */
export function HostChangesStrip({ rows, onJump }: { rows: PhotoRow[]; onJump: (photoId: number) => void }) {
  const changes = useProcessingStore((s) => s.hostChanges)
  const name = useProcessingStore((s) => s.hostName)
  if (changes.size === 0) return null
  const items = rows.filter((r) => changes.has(r.from)).map((row) => ({ row, change: changes.get(row.from)! }))
    .sort((a, b) => Number(!!b.row.review) - Number(!!a.row.review) || a.row.photoId - b.row.photoId)
  return (
    <div className="flex items-center gap-1.5 overflow-x-auto border-b bg-sky-500/5 px-3 py-1.5 text-[11px]" aria-label={`Changed by ${name}`} data-host-strip>
      <span className="flex flex-shrink-0 items-center gap-1 font-medium text-sky-800 dark:text-sky-300">
        <Sparkles className="h-3.5 w-3.5" /> {name} · {items.length}
      </span>
      {items.map(({ row, change }) => {
        const other = [...new Set(change.what.filter((w) => !w.startsWith('ID ') && w !== 'note'))]
        return (
          <button key={row.photoId} type="button" onClick={() => onJump(row.photoId)} title={`${row.currentPath}: ${change.what.join(' · ')}`}
            className={`flex flex-shrink-0 items-center gap-1 rounded border px-1.5 py-0.5 hover:ring-1 hover:ring-primary ${row.review ? 'border-amber-500 bg-amber-500/10' : 'border-sky-500/40 bg-background'}`}>
            <span className="text-muted-foreground">{row.currentPath.replace(/\.[^.]+$/, '')}</span>
            {change.id && (
              <span className="flex items-center gap-1 font-mono">
                <span className="line-through">{change.id.from ? change.id.from.slice(-4) : '····'}</span>
                <ArrowRight className="h-3 w-3" />
                <span className="font-semibold">{change.id.to.slice(-4)}</span>
              </span>
            )}
            {other.length > 0 && <span>{other.join(' · ')}</span>}
          </button>
        )
      })}
    </div>
  )
}
