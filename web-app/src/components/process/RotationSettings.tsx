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

const ROTATION_OPTIONS = [
  { label: '0° (No Change)', value: 0 },
  { label: '90° CCW', value: 90 },
  { label: '180°', value: 180 },
  { label: '90° CW', value: 270 },
]

export function RotationSettings() {
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

      <p className="text-xs text-muted-foreground">
        File rotation is handled in Select Images so it can use the current selection and undo log.
      </p>
    </div>
  )
}
