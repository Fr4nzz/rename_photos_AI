import { useEffect, useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { loadImagePreview, rotateCanvas } from '@/lib/imageProcessing'

/** Long side of the large view: enough to read an envelope and see a wing's edge. */
const SIDE = 2400

/**
 * A photo large, turned as it will be (the turn still to be written applied), with its RAW file a
 * click away: to look at the wings, the envelope, or whether the RAW is turned like its JPG.
 */
export function PhotoViewer({ open, onOpenChange, title, file, raw, pending }: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  file: File
  raw?: File
  /** clockwise turn still to be written */
  pending: number
}) {
  const [showRaw, setShowRaw] = useState(false)
  const [url, setUrl] = useState<string | null>(null)
  const [zoomed, setZoomed] = useState(false)
  const shown = showRaw && raw ? raw : file

  useEffect(() => {
    if (!open) return
    let cancelled = false
    let made: string | null = null
    ;(async () => {
      const canvas = await loadImagePreview(shown, SIDE)
      if (cancelled) return
      ;(pending ? rotateCanvas(canvas, pending) : canvas).toBlob((blob) => {
        if (cancelled || !blob) return
        made = URL.createObjectURL(blob)
        setUrl(made)
      }, 'image/jpeg', 0.92)
    })().catch(() => !cancelled && setUrl(null))
    return () => {
      cancelled = true
      if (made) URL.revokeObjectURL(made)
      setUrl(null)
    }
  }, [open, shown, pending])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[96vw] gap-2 p-3 sm:max-w-[96vw]">
        <DialogHeader className="flex-row items-center gap-2 space-y-0 pr-8">
          <DialogTitle className="truncate text-sm">{showRaw && raw ? raw.name : title}</DialogTitle>
          {raw && (
            <div className="ml-auto flex rounded-md border p-0.5" role="radiogroup" aria-label="File">
              {([['JPG', false], ['RAW', true]] as const).map(([label, value]) => (
                <Button key={label} size="sm" variant={showRaw === value ? 'secondary' : 'ghost'} role="radio" aria-checked={showRaw === value}
                  className="h-6 px-2 text-xs" onClick={() => setShowRaw(value)}>{label}</Button>
              ))}
            </div>
          )}
        </DialogHeader>
        <div className={`max-h-[84vh] ${zoomed ? 'overflow-auto' : 'flex items-center justify-center overflow-hidden'}`}>
          {url ? (
            <img src={url} alt={shown.name} onClick={() => setZoomed(!zoomed)} title={zoomed ? 'Click to fit' : 'Click to see it at full size'}
              className={zoomed ? 'max-w-none cursor-zoom-out' : 'max-h-[84vh] max-w-full cursor-zoom-in object-contain'} />
          ) : (
            <div className="flex h-[60vh] items-center justify-center text-sm text-muted-foreground">Loading…</div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
