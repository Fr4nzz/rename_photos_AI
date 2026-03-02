import type { FileEntry } from '@/types'
import { useSettingsStore } from '@/stores/settingsStore'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'

interface Props {
  imageFiles: FileEntry[]
  selectedImageIndex: number
  onSelectImage: (index: number) => void
  selectedGridIndex: number
  onSelectGrid: (index: number) => void
  gridCount: number
}

export function PreviewSelector({
  imageFiles,
  selectedImageIndex,
  onSelectImage,
  selectedGridIndex,
  onSelectGrid,
  gridCount,
}: Props) {
  const { previewRaw, updateSetting } = useSettingsStore()

  return (
    <div className="space-y-2">
      <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        Preview Selection
      </label>

      <Select
        value={String(selectedImageIndex)}
        onValueChange={(v) => onSelectImage(Number(v))}
        disabled={imageFiles.length === 0}
      >
        <SelectTrigger className="h-8 text-xs">
          <SelectValue placeholder="Select image..." />
        </SelectTrigger>
        <SelectContent>
          {imageFiles.map((f, i) => (
            <SelectItem key={f.name} value={String(i)} className="text-xs">
              {f.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select
        value={String(selectedGridIndex)}
        onValueChange={(v) => onSelectGrid(Number(v))}
        disabled={gridCount === 0}
      >
        <SelectTrigger className="h-8 text-xs">
          <SelectValue placeholder="Select grid..." />
        </SelectTrigger>
        <SelectContent>
          {Array.from({ length: gridCount }, (_, i) => (
            <SelectItem key={i} value={String(i)} className="text-xs">
              Grid {i + 1}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <div className="flex items-center gap-2">
        <Checkbox
          id="previewRaw"
          checked={previewRaw}
          onCheckedChange={(c) => updateSetting('previewRaw', !!c)}
        />
        <Label htmlFor="previewRaw" className="text-xs">
          Include RAW images
        </Label>
      </div>
    </div>
  )
}
