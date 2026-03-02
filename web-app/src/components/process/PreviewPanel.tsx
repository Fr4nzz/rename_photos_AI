import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent } from '@/components/ui/dialog'
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

  return (
    <>
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
                  className="h-40 w-full cursor-pointer rounded object-contain bg-muted"
                  onClick={() => setEnlarged(previews[key])}
                />
              ) : (
                <div className="flex h-40 items-center justify-center rounded bg-muted text-xs text-muted-foreground">
                  No preview
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      <Dialog open={!!enlarged} onOpenChange={() => setEnlarged(null)}>
        <DialogContent className="max-w-4xl p-2">
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
