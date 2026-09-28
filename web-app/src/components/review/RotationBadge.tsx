import { ArrowLeftRight, Check, Hand, RotateCcw, RotateCw, ScanText, Wand2 } from 'lucide-react'
import type { PhotoRow } from '@/types'
import { pendingRotation } from '@/lib/rotationPlan'

const SOURCE = {
  ocr: { Icon: ScanText, text: 'suggested from the envelope text' },
  manual: { Icon: Hand, text: 'set by you (click to go back to the suggestion)' },
  learned: { Icon: Wand2, text: 'learned from corrections (earlier version)' },
  neighbours: { Icon: ArrowLeftRight, text: 'no CAMID read: taken from the photos shot just before and after' },
  '': { Icon: null, text: 'no suggestion (no CAMID read)' },
} as const

/**
 * Rotation shown on each review card: the clockwise angle used for this photo, where it came from
 * (reader / neighbouring photos / you), and whether it is already written to the file. Buttons turn it.
 */
export function RotationBadge({ row, onChange }: { row: PhotoRow; onChange: (updates: Partial<PhotoRow>) => void }) {
  const chosen = Number(row.rotChosen || 0)
  const applied = row.rotApplied !== '' && pendingRotation(row) === 0 && chosen !== 0
  const source = SOURCE[row.rotSource]
  const turn = (delta: number) => onChange({ rotChosen: String((((chosen + delta) % 360) + 360) % 360), rotSource: 'manual' })
  const title = `Rotated ${chosen}° clockwise · ${source.text}${applied ? ' · written to the file' : pendingRotation(row) ? ' · applied when you rename' : ''}`

  return (
    <>
      <button
        type="button"
        title={title}
        aria-label={title}
        onClick={() => row.rotSource === 'manual' && row.rotSuggested !== '' && onChange({ rotChosen: row.rotSuggested, rotSource: 'ocr' })}
        className={`absolute bottom-1 left-1 flex items-center gap-1 rounded bg-background/90 px-1.5 py-0.5 text-[11px] font-medium shadow-sm ${chosen ? 'text-primary' : 'text-muted-foreground'}`}
      >
        <RotateCw className="h-3 w-3" />
        {chosen}°
        {source.Icon && <source.Icon className="h-3 w-3" />}
        {applied && <Check className="h-3 w-3 text-emerald-600" />}
      </button>
      <div className="absolute right-1 top-1 flex gap-0.5">
        <button type="button" title="Rotate 90° counter-clockwise" aria-label="Rotate 90° counter-clockwise" onClick={() => turn(-90)}
          className="rounded bg-background/90 p-1 shadow-sm hover:bg-accent">
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
        <button type="button" title="Rotate 90° clockwise" aria-label="Rotate 90° clockwise" onClick={() => turn(90)}
          className="rounded bg-background/90 p-1 shadow-sm hover:bg-accent">
          <RotateCw className="h-3.5 w-3.5" />
        </button>
      </div>
    </>
  )
}
