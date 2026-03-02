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
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { RotateCcw } from 'lucide-react'

const ROTATION_OPTIONS = [
  { label: '0° (No Change)', value: 0 },
  { label: '90° CCW', value: 90 },
  { label: '180°', value: 180 },
  { label: '90° CW', value: 270 },
]

interface Props {
  backendConnected: boolean
}

export function RotationSettings({ backendConnected }: Props) {
  const { rotationAngle, useExif, updateSetting } = useSettingsStore()

  return (
    <div className="space-y-2">
      <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        Image Rotation
      </label>

      <Select
        value={String(rotationAngle)}
        onValueChange={(v) => updateSetting('rotationAngle', Number(v))}
      >
        <SelectTrigger className="h-8 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {ROTATION_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={String(opt.value)} className="text-xs">
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <div className="flex items-center gap-2">
        <Checkbox
          id="useExif"
          checked={useExif}
          onCheckedChange={(c) => updateSetting('useExif', !!c)}
        />
        <Label htmlFor="useExif" className="text-xs">
          Use EXIF rotation
        </Label>
      </div>

      <Tooltip>
        <TooltipTrigger asChild>
          <div>
            <Button
              variant="outline"
              size="sm"
              className="w-full gap-1.5 text-xs"
              disabled={!backendConnected}
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Apply Rotation to Files
            </Button>
          </div>
        </TooltipTrigger>
        {!backendConnected && (
          <TooltipContent>
            Requires local backend for RAW/HEIC file rotation
          </TooltipContent>
        )}
      </Tooltip>
    </div>
  )
}
