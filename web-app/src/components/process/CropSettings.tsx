import { useSettingsStore } from '@/stores/settingsStore'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'

export function CropSettings() {
  const { cropSettings, updateCropSetting } = useSettingsStore()

  return (
    <div className="space-y-2">
      <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        Cropping & Filters
      </label>

      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <Checkbox
            id="enableCrop"
            checked={cropSettings.zoom}
            onCheckedChange={(c) => updateCropSetting('zoom', !!c)}
          />
          <Label htmlFor="enableCrop" className="text-xs">Enable Cropping</Label>
        </div>

        <div className="flex items-center gap-2">
          <Checkbox
            id="grayscale"
            checked={cropSettings.grayscale}
            onCheckedChange={(c) => updateCropSetting('grayscale', !!c)}
          />
          <Label htmlFor="grayscale" className="text-xs">Convert to Grayscale</Label>
        </div>

        <div className="flex items-center gap-2">
          <Checkbox
            id="prerotate"
            checked={cropSettings.prerotate}
            onCheckedChange={(c) => updateCropSetting('prerotate', !!c)}
          />
          <Label htmlFor="prerotate" className="text-xs">Pre-rotate for AI</Label>
        </div>
      </div>

      {cropSettings.zoom && (
        <div className="grid grid-cols-2 gap-2">
          {(['top', 'bottom', 'left', 'right'] as const).map((side) => (
            <div key={side} className="space-y-0.5">
              <Label className="text-xs capitalize text-muted-foreground">{side}</Label>
              <Input
                type="number"
                step="0.05"
                min="0"
                max="0.9"
                value={cropSettings[side]}
                onChange={(e) =>
                  updateCropSetting(side, parseFloat(e.target.value) || 0)
                }
                className="h-7 text-xs"
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
