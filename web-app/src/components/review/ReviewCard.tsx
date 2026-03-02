import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { useProcessingStore } from '@/stores/processingStore'
import type { PhotoRow } from '@/types'

interface Props {
  row: PhotoRow
  onUpdate: (photoId: number, updates: Partial<PhotoRow>) => void
}

export function ReviewCard({ row, onUpdate }: Props) {
  const fileMap = useProcessingStore((s) => s.fileMap)
  const [thumbUrl, setThumbUrl] = useState<string | null>(null)

  useEffect(() => {
    const file = fileMap.get(row.from)
    if (!file) {
      setThumbUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setThumbUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [fileMap, row.from])

  const statusColor =
    row.status === 'Renamed'
      ? 'bg-green-500/10 text-green-600'
      : row.status === 'Missing'
        ? 'bg-red-500/10 text-red-600'
        : 'bg-muted text-muted-foreground'

  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-3 py-2 pb-1">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="truncate text-xs font-medium">
            {row.from}
          </CardTitle>
          <div className="flex items-center gap-1 flex-shrink-0">
            <Badge variant="outline" className="text-[10px] px-1.5 py-0">
              Msg {row.batchNumber}
            </Badge>
            <Badge className={`text-[10px] px-1.5 py-0 ${statusColor}`}>
              {row.status}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex gap-3 px-3 pb-2">
        {/* Thumbnail */}
        {thumbUrl && (
          <div className="flex-shrink-0">
            <img
              src={thumbUrl}
              alt={row.from}
              className="h-28 w-28 rounded-sm border object-cover"
            />
          </div>
        )}

        {/* Fields */}
        <div className="min-w-0 flex-1 space-y-1.5">
          {/* Main value + Suffix */}
          <div className="flex gap-1.5">
            <div className="flex-1 space-y-0.5">
              <Label className="text-[10px] text-muted-foreground">CAM</Label>
              <Input
                value={row.mainValue}
                onChange={(e) => onUpdate(row.photoId, { mainValue: e.target.value })}
                className="h-7 text-xs"
              />
            </div>
            <div className="w-16 space-y-0.5">
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
