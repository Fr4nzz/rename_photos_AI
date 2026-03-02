import { useSettingsStore } from '@/stores/settingsStore'
import { useApiKeysStore } from '@/stores/apiKeysStore'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

export function ApiSettings() {
  const settings = useSettingsStore()
  const { availableModels } = useApiKeysStore()

  return (
    <div className="space-y-2">
      <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        API Settings
      </label>

      <div className="space-y-1.5">
        <Label className="text-xs text-muted-foreground">Model</Label>
        <Select
          value={settings.modelName}
          onValueChange={(v) => settings.updateSetting('modelName', v)}
        >
          <SelectTrigger className="h-8 text-xs">
            <SelectValue placeholder="Select a model..." />
          </SelectTrigger>
          <SelectContent>
            {availableModels.map((m) => (
              <SelectItem key={m} value={m} className="text-xs">
                {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-0.5">
          <Label className="text-xs text-muted-foreground">Grids/Message</Label>
          <Input
            type="number"
            min={1}
            value={settings.imagesPerPrompt}
            onChange={(e) =>
              settings.updateSetting('imagesPerPrompt', parseInt(e.target.value) || 1)
            }
            className="h-7 text-xs"
          />
        </div>
        <div className="space-y-0.5">
          <Label className="text-xs text-muted-foreground">Merged Height</Label>
          <Input
            type="number"
            min={480}
            step={120}
            value={settings.mergedImgHeight}
            onChange={(e) =>
              settings.updateSetting('mergedImgHeight', parseInt(e.target.value) || 1080)
            }
            className="h-7 text-xs"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-0.5">
          <Label className="text-xs text-muted-foreground">Grid Rows</Label>
          <Input
            type="number"
            min={1}
            max={10}
            value={settings.gridRows}
            onChange={(e) =>
              settings.updateSetting('gridRows', parseInt(e.target.value) || 3)
            }
            className="h-7 text-xs"
          />
        </div>
        <div className="space-y-0.5">
          <Label className="text-xs text-muted-foreground">Grid Cols</Label>
          <Input
            type="number"
            min={1}
            max={10}
            value={settings.gridCols}
            onChange={(e) =>
              settings.updateSetting('gridCols', parseInt(e.target.value) || 3)
            }
            className="h-7 text-xs"
          />
        </div>
      </div>

      <div className="space-y-0.5">
        <Label className="text-xs text-muted-foreground">Main Column</Label>
        <Input
          value={settings.mainColumn}
          onChange={(e) => settings.updateSetting('mainColumn', e.target.value)}
          className="h-7 text-xs"
        />
      </div>
    </div>
  )
}
