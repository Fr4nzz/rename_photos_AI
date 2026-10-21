import { Fragment } from 'react'
import { useProcessingStore } from '@/stores/processingStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { readingsLine } from '@/lib/readingsLine'
import type { PhotoRow } from '@/types'

/** What the models read on a photo, in one quiet line; the part against the card in amber, each wing in the title. */
export function ReadingsLine({ row }: { row: PhotoRow }) {
  const check = useProcessingStore((s) => s.wingChecks.get(row.from))
  const reading = useProcessingStore((s) => s.ocrReadings.get(row.from))
  const mode = useSettingsStore((s) => s.suffixMode)
  const line = readingsLine(row, check, reading, mode)
  if (!line) return null
  return (
    <div className="truncate text-[11px] text-muted-foreground" title={line.title} data-readings>
      {line.parts.map((p, i) => (
        <Fragment key={i}>
          {i > 0 && ' · '}
          <span className={p.warn ? 'text-amber-700 dark:text-amber-400' : undefined} data-warn={p.warn ? '1' : undefined}>{p.text}</span>
        </Fragment>
      ))}
    </div>
  )
}
