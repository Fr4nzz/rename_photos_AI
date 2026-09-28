import { useEffect, useRef, useState } from 'react'
import { Check } from 'lucide-react'
import { loadImagePreview, canvasToBlobUrl } from '@/lib/imageProcessing'
import { getErrorMessage } from '@/lib/errors'
import type { FileEntry, PhotoRow } from '@/types'

interface CardProps {
  entry: FileEntry
  selected: boolean
  onToggle: () => void
  /** the reading of this photo, once read */
  row?: PhotoRow
  onOpen?: (photoId: number) => void
}

function ImageSelectionCard({ entry, selected, onToggle, row, onOpen }: CardProps) {
  const [thumb, setThumb] = useState<string | null>(null)
  const [nearView, setNearView] = useState(false)
  const ref = useRef<HTMLButtonElement>(null)

  // decode thumbnails only for cards near the viewport (large folders stay light)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setNearView(true); io.disconnect() } }, { rootMargin: '600px' })
    io.observe(el)
    return () => io.disconnect()
  }, [])

  useEffect(() => {
    if (!nearView) return
    let cancelled = false
    let url: string | null = null
    loadImagePreview(entry.file, 360)
      .then(canvasToBlobUrl)
      .then((nextUrl) => {
        if (cancelled) {
          URL.revokeObjectURL(nextUrl)
          return
        }
        url = nextUrl
        setThumb(nextUrl)
      })
      .catch((error: unknown) => {
        if (!cancelled) setThumb(null)
        console.warn(`Could not preview ${entry.name}: ${getErrorMessage(error)}`)
      })

    return () => {
      cancelled = true
      if (url) URL.revokeObjectURL(url)
    }
  }, [entry, nearView])

  return (
    <button
      ref={ref}
      type="button"
      onClick={onToggle}
      className={`overflow-hidden rounded border bg-card text-left transition ${selected ? 'border-primary ring-1 ring-primary' : 'border-border hover:border-primary/60'}${row?.review ? ' outline outline-2 outline-offset-1 outline-amber-500' : ''}`}
    >
      <div className="relative">
        {thumb ? (
          <img src={thumb} alt={entry.name} className="h-32 w-full object-contain bg-muted" />
        ) : (
          <div className="flex h-32 items-center justify-center bg-muted text-xs text-muted-foreground">
            No preview
          </div>
        )}
        <div
          aria-label={`Select ${entry.name}`}
          aria-checked={selected}
          role="checkbox"
          className={`absolute left-2 top-2 flex size-7 items-center justify-center rounded border ${selected ? 'border-primary bg-primary text-primary-foreground' : 'border-border bg-background/85 text-transparent'}`}
        >
          <Check className="size-4" />
        </div>
      </div>
      <div className="space-y-0.5 p-2">
        <div className="truncate text-xs font-medium">{entry.name}</div>
        {row ? (
          <span
            role="link"
            tabIndex={0}
            onClick={(e) => { e.stopPropagation(); onOpen?.(row.photoId) }}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.stopPropagation(); onOpen?.(row.photoId) } }}
            className={`block truncate font-mono text-[11px] hover:underline ${row.review ? 'text-amber-600 dark:text-amber-400' : row.mainValue ? 'text-emerald-600 dark:text-emerald-400' : 'text-muted-foreground'}`}
            title="Open in review"
          >
            {row.status === 'Renamed' ? row.currentPath : row.mainValue ? `${row.mainValue}${row.suffix}` : '·······'}
          </span>
        ) : (
          <div className="text-[10px] text-muted-foreground">
            {entry.extension.toUpperCase()} · {new Date(entry.file.lastModified).toLocaleDateString()}
          </div>
        )}
      </div>
    </button>
  )
}

interface Props {
  files: FileEntry[]
  selectedNames: Set<string>
  onToggle: (name: string) => void
  rows?: Map<string, PhotoRow>
  onOpen?: (photoId: number) => void
}

export function ImageSelectionGrid({ files, selectedNames, onToggle, rows, onOpen }: Props) {
  if (files.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
        No images match the current filters.
      </div>
    )
  }

  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3 p-3">
      {files.map((entry) => (
        <ImageSelectionCard
          key={entry.name}
          entry={entry}
          selected={selectedNames.has(entry.name)}
          onToggle={() => onToggle(entry.name)}
          row={rows?.get(entry.name)}
          onOpen={onOpen}
        />
      ))}
    </div>
  )
}
