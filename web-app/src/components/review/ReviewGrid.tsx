import { ScrollArea } from '@/components/ui/scroll-area'
import { ReviewCard } from './ReviewCard'
import type { PhotoRow } from '@/types'

interface Props {
  rows: PhotoRow[]
  onUpdateRow: (photoId: number, updates: Partial<PhotoRow>) => void
  duplicatePairs: Set<string>
}

export function ReviewGrid({ rows, onUpdateRow, duplicatePairs }: Props) {
  if (rows.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
        No data to display. Process images first or load a CSV.
      </div>
    )
  }

  return (
    <ScrollArea className="flex-1 min-h-0">
      <div className="grid grid-cols-1 gap-2 p-3 md:grid-cols-2">
        {rows.map((row) => {
          const key = `${row.mainValue?.trim()}__${row.suffix?.trim()}`
          const isDuplicate = !!(row.mainValue?.trim() && row.suffix?.trim() && duplicatePairs.has(key))
          return (
            <ReviewCard
              key={row.photoId}
              row={row}
              onUpdate={onUpdateRow}
              isDuplicate={isDuplicate}
            />
          )
        })}
      </div>
    </ScrollArea>
  )
}
