import { useEffect, useState } from 'react'
import { Check } from 'lucide-react'
import { loadImagePreview, canvasToBlobUrl } from '@/lib/imageProcessing'
import { getErrorMessage } from '@/lib/errors'
import type { FileEntry } from '@/types'

interface CardProps {
  entry: FileEntry
  selected: boolean
  onToggle: () => void
}

function ImageSelectionCard({ entry, selected, onToggle }: CardProps) {
  const [thumb, setThumb] = useState<string | null>(null)

  useEffect(() => {
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
  }, [entry])

  return (
    <button
      type="button"
      onClick={onToggle}
      className={`overflow-hidden rounded border bg-card text-left transition ${selected ? 'border-primary ring-1 ring-primary' : 'border-border hover:border-primary/60'}`}
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
        <div className="text-[10px] text-muted-foreground">
          {entry.extension.toUpperCase()} · {new Date(entry.file.lastModified).toLocaleDateString()}
        </div>
      </div>
    </button>
  )
}

interface Props {
  files: FileEntry[]
  selectedNames: Set<string>
  onToggle: (name: string) => void
}

export function ImageSelectionGrid({ files, selectedNames, onToggle }: Props) {
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
        />
      ))}
    </div>
  )
}
