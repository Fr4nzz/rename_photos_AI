import { useEffect, useState } from 'react'
import { ArrowRight, Check, ChevronDown, ChevronRight, Sparkles, TriangleAlert } from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useProcessingStore } from '@/stores/processingStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { loadImagePreview, rotateCanvas } from '@/lib/imageProcessing'
import { pendingRotation } from '@/lib/rotationPlan'
import { REASON_LABEL } from '@/lib/ocr/runLocal'
import { specimenNote, type Specimen } from '@/lib/specimens'
import type { PhotoRow } from '@/types'
import { IdLineZoom } from './IdLineZoom'
import { RotationBadge } from './RotationBadge'

/**
 * The review by specimen: one card per ID with its photos side by side (dorsal, ventral), the ID
 * line of the envelope enlarged, and the ID in large type. In the order taken (the IDs around a doubtful
 * one often say what it is) the specimens that need a person are open among the others, a line each,
 * opened with a click (or all at once: «Show all photos»); «To look at first» puts them first. What is rarely touched (suffix,
 * crossed out, notes, a photo's own ID) sits behind each photo's «more».
 */
interface Props {
  specimens: Specimen[]
  /** specimens opened by hand, or by a jump from the strip above */
  open: Set<number>
  onToggle: (key: number) => void
  onUpdateRow: (photoId: number, updates: Partial<PhotoRow>) => void
  onUpdateRows: (photoIds: number[], updates: Partial<PhotoRow>) => void
}

const VALID = /^CAM\d{6}$/
/** Cards drawn at first; the rest wait for «Show more» (each loads its photos). */
const FIRST = 40

function useThumb(row: PhotoRow, file: File | undefined) {
  const [url, setUrl] = useState<string | null>(null)
  const pending = pendingRotation(row)
  useEffect(() => {
    if (!file) return
    let cancelled = false
    let made: string | null = null
    ;(async () => {
      try {
        const canvas = await loadImagePreview(file, 800)
        if (cancelled) return
        ;(pending ? rotateCanvas(canvas, pending) : canvas).toBlob((blob) => {
          if (cancelled || !blob) return
          made = URL.createObjectURL(blob)
          setUrl(made)
        }, 'image/jpeg', 0.85)
      } catch {
        if (!cancelled) setUrl(null)
      }
    })()
    return () => {
      cancelled = true
      if (made) URL.revokeObjectURL(made)
    }
  }, [file, pending])
  return url
}

function PhotoCell({ row, onUpdate }: { row: PhotoRow; onUpdate: (updates: Partial<PhotoRow>) => void }) {
  const file = useProcessingStore((s) => s.fileMap.get(row.from) ?? s.fileMap.get(row.currentPath))
  const hostChange = useProcessingStore((s) => s.hostChanges.get(row.from))
  const hostName = useProcessingStore((s) => s.hostName)
  const [more, setMore] = useState(false)
  const url = useThumb(row, file)
  const skipped = row.skip === 'x'
  const other = hostChange?.what.filter((w) => !w.startsWith('ID ')) ?? []

  return (
    <div id={`card-${row.photoId}`} className={`min-w-0 max-w-[50%] flex-1 space-y-1 ${skipped ? 'opacity-50' : ''}`}>
      <div className="relative overflow-hidden rounded-sm border bg-muted/40 leading-none">
        {url ? <img src={url} alt={row.from} className="max-h-44 w-full object-contain" /> : <div className="h-28" />}
        {file && <RotationBadge row={row} onChange={onUpdate} />}
      </div>
      <div className="flex items-baseline gap-1.5 text-[11px]" title={row.to ? `${row.currentPath} → ${row.to}` : row.currentPath}>
        <span className="rounded bg-muted px-1 font-mono font-semibold">{skipped ? 'skip' : row.suffix || '–'}</span>
        <span className="truncate text-muted-foreground">{row.currentPath}</span>
        {row.status === 'Renamed' && <Check className="h-3 w-3 flex-shrink-0 text-emerald-600" aria-label="Renamed" />}
      </div>
      {row.review && !skipped && (
        <div className="space-y-1 rounded-md bg-amber-500/10 p-1 text-[11px] text-amber-700 dark:text-amber-400">
          <div className="flex items-center gap-1">
            <TriangleAlert className="h-3 w-3 flex-shrink-0" />
            <span className="flex-1 truncate" title={row.review.split(',').map((r) => REASON_LABEL[r] ?? r).join(' · ')}>
              {row.review.split(',').map((r) => REASON_LABEL[r] ?? r).join(' · ')}
            </span>
            <Button size="icon" variant="outline" className="h-5 w-5" title="Confirm this photo's ID" aria-label="Confirm this photo's ID"
              disabled={!VALID.test(row.mainValue.trim())} onClick={() => onUpdate({ review: '' })}>
              <Check className="h-3 w-3" />
            </Button>
          </div>
          {row.suggest && (
            <div className="flex flex-wrap gap-1">
              {row.suggest.split(' ').filter(Boolean).map((id) => (
                <button key={id} type="button" title="Use this ID for this photo" onClick={() => onUpdate({ mainValue: id, review: '' })}
                  className="rounded border bg-background px-1 font-mono text-[11px] text-foreground hover:bg-accent">
                  {id}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {hostChange && (
        <div className="rounded-md bg-sky-500/10 p-1 text-[11px] text-sky-800 dark:text-sky-300" data-host-change>
          <div className="flex flex-wrap items-center gap-1">
            <Sparkles className="h-3 w-3 flex-shrink-0" />
            <span className="font-medium">{hostName}</span>
            {hostChange.id && (
              <span className="flex items-center gap-1 font-mono">
                <span className="line-through decoration-sky-700/60" title="What the reader had">{hostChange.id.from || '(none)'}</span>
                <ArrowRight className="h-3 w-3" />
                <span className="font-semibold">{hostChange.id.to}</span>
              </span>
            )}
          </div>
          {other.length > 0 && <div>{[...new Set(other)].join(' · ')}</div>}
          {row.n.trim() && <div className="italic">{row.n}</div>}
        </div>
      )}
      <div className="flex items-center gap-2 text-[11px]">
        <Checkbox id={`skip-${row.photoId}`} checked={skipped} onCheckedChange={(c) => onUpdate({ skip: c ? 'x' : '' })} />
        <Label htmlFor={`skip-${row.photoId}`} className="text-[11px]">Skip</Label>
        <button type="button" className="ml-auto flex items-center text-muted-foreground hover:text-foreground" aria-expanded={more} onClick={() => setMore(!more)}>
          {more ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />} more
        </button>
      </div>
      {more && (
        <div className="grid grid-cols-2 gap-1">
          {([['ID of this photo', 'mainValue', 'font-mono'], ['Suffix', 'suffix', ''], ['Crossed out', 'co', ''], ['Notes', 'n', '']] as const).map(([label, key, cls]) => (
            <label key={key} className="space-y-0.5 text-[10px] text-muted-foreground">
              {label}
              <Input value={row[key]} onChange={(e) => onUpdate({ [key]: e.target.value })} className={`h-6 text-[11px] text-foreground ${cls}`} />
            </label>
          ))}
          {row.to && <div className="col-span-2 truncate text-[10px] text-muted-foreground">→ {row.to}</div>}
        </div>
      )}
    </div>
  )
}

function SpecimenCard({ specimen, onUpdateRow, onUpdateRows, onClose }: {
  specimen: Specimen
  onUpdateRow: Props['onUpdateRow']
  onUpdateRows: Props['onUpdateRows']
  onClose?: () => void
}) {
  const first = specimen.rows[0]
  const file = useProcessingStore((s) => s.fileMap.get(first.from) ?? s.fileMap.get(first.currentPath))
  const ids = specimen.rows.map((r) => r.photoId)
  const waiting = specimen.rows.filter((r) => r.review && r.skip !== 'x').map((r) => r.photoId)
  const ok = specimen.state === 'ok'
  // dorsal and ventral: only the standard naming expects two photos of a specimen
  const pairs = useSettingsStore((s) => s.suffixMode === 'Standard')
  return (
    <div className={`space-y-2 rounded-lg border bg-card p-2.5 ${ok ? '' : 'ring-2 ring-amber-500/70'}`} data-specimen={specimen.id || 'none'}>
      <div className="flex items-center gap-2">
        <Input
          value={specimen.id}
          placeholder="CAM······"
          aria-label="ID of these photos"
          data-review={ok ? undefined : '1'}
          onChange={(e) => onUpdateRows(ids, { mainValue: e.target.value })}
          onKeyDown={(e) => {
            if (e.key !== 'Enter' || !VALID.test(specimen.id)) return
            if (waiting.length) onUpdateRows(waiting, { review: '' })
            // keyboard flow: on to the next specimen that needs a look
            const inputs = [...document.querySelectorAll<HTMLInputElement>('input[data-review="1"]')]
            const next = inputs[inputs.indexOf(e.currentTarget) + 1] ?? inputs.find((el) => el !== e.currentTarget)
            next?.focus()
            next?.scrollIntoView({ behavior: 'smooth', block: 'center' })
          }}
          className="h-8 w-36 font-mono text-base font-semibold"
        />
        {file && <div className="min-w-0 flex-1"><IdLineZoom name={first.from} file={file} camid={specimen.id} applied={Number(first.rotApplied || 0)} /></div>}
        <span className={`ml-auto flex-shrink-0 rounded px-1.5 py-0.5 text-[11px] ${ok ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300' : 'bg-amber-500/15 text-amber-700 dark:text-amber-300'}`}>
          {specimenNote(specimen)}
        </span>
        {waiting.length > 0 && (
          <Button size="sm" variant="outline" className="h-7 gap-1 px-2 text-xs" disabled={!VALID.test(specimen.id)}
            title="The ID is right: confirm its photos (Enter)" onClick={() => onUpdateRows(waiting, { review: '' })}>
            <Check className="h-3.5 w-3.5" /> Confirm
          </Button>
        )}
        {onClose && (
          <button type="button" className="text-muted-foreground hover:text-foreground" title="Close" aria-label="Close" onClick={onClose}>
            <ChevronDown className="h-4 w-4" />
          </button>
        )}
      </div>
      <div className="flex gap-2">
        {specimen.rows.map((row) => <PhotoCell key={row.photoId} row={row} onUpdate={(u) => onUpdateRow(row.photoId, u)} />)}
        {/* a specimen has two photos: the place of the one that is not there */}
        {pairs && specimen.id && specimen.rows.length === 1 && (
          <div className="flex min-h-28 flex-1 items-center justify-center rounded-sm border border-dashed text-center text-[11px] text-muted-foreground">
            No second photo<br />(dorsal and ventral)
          </div>
        )}
      </div>
    </div>
  )
}

const ALL_OPEN_KEY = 'review:all-open'
const ORDER_KEY = 'review:order'

/** A ready butterfly as one line: its ID, views and files; a click opens its card. */
function SpecimenLine({ specimen: s, onOpen }: { specimen: Specimen; onOpen: () => void }) {
  return (
    <button type="button" onClick={onOpen} data-specimen-line={s.id}
      className="flex w-full items-center gap-2 rounded-md border bg-card px-2 py-1 text-left text-xs hover:bg-accent">
      <Check className="h-3.5 w-3.5 flex-shrink-0 text-emerald-600" />
      <span className="font-mono font-semibold">{s.id}</span>
      <span className="text-muted-foreground">{s.rows.filter((r) => r.skip !== 'x').map((r) => r.suffix).join(' · ')}</span>
      <span className="min-w-0 flex-1 truncate text-muted-foreground">{s.rows.map((r) => r.currentPath).join(', ')}</span>
      <ChevronRight className="h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" />
    </button>
  )
}

export function SpecimenGrid({ specimens, open, onToggle, onUpdateRow, onUpdateRows }: Props) {
  const [shown, setShown] = useState(FIRST)
  // Every butterfly's photos, the ready ones too (to look through the whole batch): kept for next time.
  const [allOpen, setAllOpen] = useState(() => localStorage.getItem(ALL_OPEN_KEY) === '1')
  // As taken (the doubtful ones among the IDs around them, which often say what a misread ID is), or the doubtful first.
  const [order, setOrder] = useState<'taken' | 'attention'>(() => (localStorage.getItem(ORDER_KEY) === 'attention' ? 'attention' : 'taken'))
  const [shownReady, setShownReady] = useState(FIRST)
  const showAll = (on: boolean) => {
    setAllOpen(on)
    localStorage.setItem(ALL_OPEN_KEY, on ? '1' : '0')
  }
  const sortBy = (value: 'taken' | 'attention') => {
    setOrder(value)
    localStorage.setItem(ORDER_KEY, value)
  }
  if (specimens.length === 0) {
    return <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">No data to display. Read a folder's photos first.</div>
  }
  const attention = specimens.filter((s) => s.state !== 'ok')
  const ready = specimens.filter((s) => s.state === 'ok')
  const card = (s: Specimen, closable: boolean) => (
    <SpecimenCard key={s.key} specimen={s} onUpdateRow={onUpdateRow} onUpdateRows={onUpdateRows} {...(closable ? { onClose: () => onToggle(s.key) } : {})} />
  )
  const controls = (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      <span className="font-medium text-amber-700 dark:text-amber-300">To look at · {attention.length}</span>
      <span className="text-emerald-700 dark:text-emerald-300">Ready · {ready.length}</span>
      <span className="ml-auto flex items-center gap-1 text-muted-foreground">
        Order:
        {([['taken', 'As taken'], ['attention', 'To look at first']] as const).map(([value, label]) => (
          <button key={value} type="button" data-order={value} aria-pressed={order === value}
            className={`rounded px-1.5 py-0.5 ${order === value ? 'bg-secondary font-medium text-foreground' : 'hover:text-foreground'}`}
            onClick={() => sortBy(value)}>{label}</button>
        ))}
      </span>
      <button type="button" data-show-all className="text-muted-foreground underline-offset-2 hover:text-foreground hover:underline" onClick={() => showAll(!allOpen)}>
        {allOpen ? 'Show ready as lines' : 'Show all photos'}
      </button>
    </div>
  )
  return (
    <ScrollArea className="min-h-0 flex-1">
      <div className="space-y-3 p-3">
        {controls}
        {order === 'taken' ? (
          // One list in the order taken: the doubtful ones open (outlined), the ready ones a line each unless all are open.
          <div className="space-y-1.5" data-order-list="taken">
            {specimens.slice(0, allOpen ? shownReady + attention.length : specimens.length).map((s) =>
              s.state !== 'ok' ? card(s, false) : allOpen || open.has(s.key) ? card(s, !allOpen) : <SpecimenLine key={s.key} specimen={s} onOpen={() => onToggle(s.key)} />,
            )}
            {allOpen && specimens.length > shownReady + attention.length && (
              <Button variant="outline" size="sm" className="text-xs" onClick={() => setShownReady(shownReady + FIRST)}>Show more</Button>
            )}
          </div>
        ) : (
          <>
            {attention.length > 0 && (
              <section className="space-y-2" aria-label="To look at">
                <div className="grid grid-cols-1 gap-2 xl:grid-cols-2 2xl:grid-cols-3">
                  {attention.slice(0, shown).map((s) => card(s, false))}
                </div>
                {attention.length > shown && (
                  <Button variant="outline" size="sm" className="text-xs" onClick={() => setShown(shown + FIRST)}>Show {Math.min(FIRST, attention.length - shown)} more</Button>
                )}
              </section>
            )}
            {ready.length > 0 && (
              <section className="space-y-1.5" aria-label="Ready">
                {allOpen ? (
                  <>
                    <div className="grid grid-cols-1 gap-2 xl:grid-cols-2 2xl:grid-cols-3">{ready.slice(0, shownReady).map((s) => card(s, false))}</div>
                    {ready.length > shownReady && (
                      <Button variant="outline" size="sm" className="text-xs" onClick={() => setShownReady(shownReady + FIRST)}>Show {Math.min(FIRST, ready.length - shownReady)} more</Button>
                    )}
                  </>
                ) : (
                  <div className="grid grid-cols-1 gap-1.5 xl:grid-cols-2 2xl:grid-cols-3">
                    {ready.map((s) => (open.has(s.key) ? card(s, true) : <SpecimenLine key={s.key} specimen={s} onOpen={() => onToggle(s.key)} />))}
                  </div>
                )}
              </section>
            )}
          </>
        )}
      </div>
    </ScrollArea>
  )
}
