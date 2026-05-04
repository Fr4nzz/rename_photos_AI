import { useCallback, useRef, useState } from 'react'
import type { PointerEvent } from 'react'
import { VisuallyHidden } from 'radix-ui'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { useSettingsStore } from '@/stores/settingsStore'
import type { Previews } from '@/hooks/useProcessTab'

interface Props {
  previews: Previews
}

const PREVIEW_LABELS: { key: keyof Previews; label: string }[] = [
  { key: 'original', label: 'Original' },
  { key: 'rotated', label: 'Rotated' },
  { key: 'processed', label: 'Processed' },
  { key: 'grid', label: 'Grid Preview' },
]

export function PreviewPanel({ previews }: Props) {
  const [enlarged, setEnlarged] = useState<string | null>(null)
  const previewTileHeight = useSettingsStore((s) => s.previewTileHeight)
  const updateSetting = useSettingsStore((s) => s.updateSetting)
  const dragStart = useRef<{ y: number; height: number } | null>(null)

  const setPreviewTileHeight = useCallback(
    (height: number) => {
      updateSetting('previewTileHeight', Math.min(520, Math.max(120, height)))
    },
    [updateSetting]
  )

  const startResize = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      dragStart.current = { y: event.clientY, height: previewTileHeight }
      event.currentTarget.setPointerCapture(event.pointerId)
    },
    [previewTileHeight]
  )

  const resize = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (!dragStart.current) return
      const delta = event.clientY - dragStart.current.y
      setPreviewTileHeight(dragStart.current.height + delta)
    },
    [setPreviewTileHeight]
  )

  const stopResize = useCallback((event: PointerEvent<HTMLDivElement>) => {
    dragStart.current = null
    event.currentTarget.releasePointerCapture(event.pointerId)
  }, [])

  return (
    <>
      <div className="space-y-1.5">
        <div className="grid grid-cols-2 gap-2">
          {PREVIEW_LABELS.map(({ key, label }) => (
            <Card key={key} className="overflow-hidden">
              <CardHeader className="px-2 py-1.5">
                <CardTitle className="text-xs font-medium">{label}</CardTitle>
              </CardHeader>
              <CardContent className="p-1">
                {previews[key] ? (
                  <img
                    src={previews[key]!}
                    alt={label}
                    className="w-full cursor-pointer rounded object-contain bg-muted"
                    style={{ height: previewTileHeight }}
                    onClick={() => setEnlarged(previews[key])}
                  />
                ) : (
                  <div
                    className="flex items-center justify-center rounded bg-muted text-xs text-muted-foreground"
                    style={{ height: previewTileHeight }}
                  >
                    No preview
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>

        <div
          role="separator"
          aria-label="Resize preview images"
          tabIndex={0}
          className="group flex cursor-row-resize items-center gap-2 py-1"
          onPointerDown={startResize}
          onPointerMove={resize}
          onPointerUp={stopResize}
          onPointerCancel={stopResize}
        >
          <div className="h-px flex-1 bg-border transition-colors group-hover:bg-primary" />
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Preview height
          </span>
          <div className="h-px flex-1 bg-border transition-colors group-hover:bg-primary" />
        </div>

        <input
          type="range"
          min={120}
          max={520}
          step={10}
          value={previewTileHeight}
          onChange={(event) => setPreviewTileHeight(Number(event.target.value))}
          className="h-1 w-full cursor-pointer accent-primary"
          aria-label="Preview image height"
        />
      </div>

      <Dialog open={!!enlarged} onOpenChange={() => setEnlarged(null)}>
        <DialogContent className="max-w-4xl p-2" aria-describedby={undefined}>
          <VisuallyHidden.Root>
            <DialogTitle>Enlarged preview</DialogTitle>
          </VisuallyHidden.Root>
          {enlarged && (
            <img
              src={enlarged}
              alt="Enlarged preview"
              className="w-full rounded object-contain"
            />
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}
