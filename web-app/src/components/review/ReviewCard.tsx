import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { useProcessingStore } from '@/stores/processingStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { loadImagePreview, cropCanvas } from '@/lib/imageProcessing'
import type { PhotoRow } from '@/types'
import { Check, TriangleAlert } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { REASON_LABEL } from '@/lib/ocr/runLocal'
import { IdLineZoom } from './IdLineZoom'

interface Props {
  row: PhotoRow
  onUpdate: (photoId: number, updates: Partial<PhotoRow>) => void
  isDuplicate?: boolean
}

export function ReviewCard({ row, onUpdate, isDuplicate }: Props) {
  const fileMap = useProcessingStore((s) => s.fileMap)
  const { reviewCropEnabled, reviewThumbSize, cropSettings } = useSettingsStore()
  const [thumbUrl, setThumbUrl] = useState<string | null>(null)
  const file = fileMap.get(row.from)

  useEffect(() => {
    if (!file) return

    let cancelled = false
    let urlToRevoke: string | null = null

    // Use loadImagePreview (worker + IDB cache) instead of full-res decode
    ;(async () => {
      try {
        const canvas = await loadImagePreview(file, 800)
        if (cancelled) return

        const output = reviewCropEnabled && cropSettings.zoom
          ? cropCanvas(canvas, cropSettings)
          : canvas

        output.toBlob((blob) => {
          if (cancelled || !blob) return
          const url = URL.createObjectURL(blob)
          urlToRevoke = url
          setThumbUrl(url)
        }, 'image/jpeg', 0.85)
      } catch {
        if (!cancelled) setThumbUrl(null)
      }
    })()

    return () => {
      cancelled = true
      if (urlToRevoke) URL.revokeObjectURL(urlToRevoke)
    }
  }, [file, reviewCropEnabled, cropSettings])

  const statusColor =
    row.status === 'Renamed'
      ? 'bg-green-500/10 text-green-600'
      : row.status === 'Missing'
        ? 'bg-red-500/10 text-red-600'
        : 'bg-muted text-muted-foreground'

  return (
    <Card id={`card-${row.photoId}`} className={`overflow-hidden${row.review ? ' ring-2 ring-amber-500' : isDuplicate ? ' ring-2 ring-amber-500/50' : ''}`}>
      <CardHeader className="px-3 py-2 pb-1">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="truncate text-xs font-medium">
            {row.from}
          </CardTitle>
          <div className="flex items-center gap-1 flex-shrink-0">
            {isDuplicate && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-amber-500 text-amber-600">
                Duplicate
              </Badge>
            )}
            {row.batchNumber > 0 && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                Msg {row.batchNumber}
              </Badge>
            )}
            <Badge className={`text-[10px] px-1.5 py-0 ${statusColor}`}>
              {row.status}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex gap-3 px-3 pb-2">
        {/* Thumbnail caps at the card width, so large slider values wait for wider layouts. */}
        {file && thumbUrl && (
          <div className="flex-shrink-0" style={{ maxWidth: '50%' }}>
            <img
              src={thumbUrl}
              alt={row.from}
              style={{ maxHeight: reviewThumbSize, maxWidth: '100%', height: 'auto', width: 'auto' }}
              className="rounded-sm border"
            />
          </div>
        )}

        {/* Fields */}
        <div className="min-w-0 flex-1 space-y-1.5">
          {file && <IdLineZoom name={row.from} file={file} camid={row.mainValue.trim()} />}
          {row.review && (
            <div className="space-y-1 rounded-md bg-amber-500/10 p-1.5">
              <div className="flex items-center gap-1.5 text-[11px] text-amber-700 dark:text-amber-400">
                <TriangleAlert className="h-3.5 w-3.5 flex-shrink-0" />
                <span className="flex-1 truncate">{row.review.split(',').map((r) => REASON_LABEL[r] ?? r).join(' · ')}</span>
                <Button
                  size="icon"
                  variant="outline"
                  className="h-6 w-6"
                  title="Confirm this CAMID (Enter)"
                  aria-label="Confirm this CAMID"
                  disabled={!/^CAM\d{6}$/.test(row.mainValue.trim())}
                  onClick={() => onUpdate(row.photoId, { review: '' })}
                >
                  <Check className="h-3.5 w-3.5" />
                </Button>
              </div>
              {row.suggest && (
                <div className="flex flex-wrap gap-1">
                  {row.suggest.split(' ').filter(Boolean).map((id) => (
                    <button
                      key={id}
                      type="button"
                      onClick={() => onUpdate(row.photoId, { mainValue: id, review: '' })}
                      className="rounded border bg-background px-1.5 py-0.5 font-mono text-[11px] hover:bg-accent"
                      title="Use this CAMID"
                    >
                      {id}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {/* Main value + Suffix */}
          <div className="flex gap-1.5">
            <div className="flex-1 space-y-0.5">
              <Label className="text-[10px] text-muted-foreground">CAM</Label>
              <Input
                value={row.mainValue}
                onChange={(e) => onUpdate(row.photoId, { mainValue: e.target.value })}
                data-review={row.review ? '1' : undefined}
                onKeyDown={(e) => {
                  if (e.key !== 'Enter' || !/^CAM\d{6}$/.test(row.mainValue.trim())) return
                  if (row.review) onUpdate(row.photoId, { review: '' })
                  // keyboard flow: move on to the next photo still waiting for review
                  const inputs = [...document.querySelectorAll<HTMLInputElement>('input[data-review="1"]')]
                  const next = inputs[inputs.indexOf(e.currentTarget) + 1] ?? inputs.find((el) => el !== e.currentTarget)
                  next?.focus()
                  next?.scrollIntoView({ behavior: 'smooth', block: 'center' })
                }}
                className="h-7 font-mono text-xs"
              />
            </div>
            <div className="w-14 space-y-0.5">
              <Label className="text-[10px] text-muted-foreground">Suffix</Label>
              <Input
                value={row.suffix}
                onChange={(e) => onUpdate(row.photoId, { suffix: e.target.value })}
                className="h-7 text-xs"
              />
            </div>
          </div>

          {/* To */}
          <div className="space-y-0.5">
            <Label className="text-[10px] text-muted-foreground">To</Label>
            <Input
              value={row.to}
              readOnly
              className="h-7 bg-muted text-xs"
            />
          </div>

          {/* Crossed out + Notes */}
          <div className="flex gap-1.5">
            <div className="flex-1 space-y-0.5">
              <Label className="text-[10px] text-muted-foreground">Crossed Out</Label>
              <Input
                value={row.co}
                onChange={(e) => onUpdate(row.photoId, { co: e.target.value })}
                className="h-7 text-xs"
              />
            </div>
            <div className="flex-1 space-y-0.5">
              <Label className="text-[10px] text-muted-foreground">Notes</Label>
              <Input
                value={row.n}
                onChange={(e) => onUpdate(row.photoId, { n: e.target.value })}
                className="h-7 text-xs"
              />
            </div>
          </div>

          {/* Skip checkbox */}
          <div className="flex items-center gap-2 pt-0.5">
            <Checkbox
              id={`skip-${row.photoId}`}
              checked={row.skip === 'x'}
              onCheckedChange={(c) =>
                onUpdate(row.photoId, { skip: c ? 'x' : '' })
              }
            />
            <Label htmlFor={`skip-${row.photoId}`} className="text-xs">
              Skip
            </Label>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
