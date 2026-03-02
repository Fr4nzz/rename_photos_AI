import { useSettingsStore } from '@/stores/settingsStore'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { RotateCcw, Save } from 'lucide-react'

export function PromptEditor() {
  const { promptText, updateSetting, resetPrompt } = useSettingsStore()

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Prompt
        </label>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 gap-1 text-xs"
          onClick={resetPrompt}
        >
          <RotateCcw className="h-3 w-3" />
          Restore Default
        </Button>
      </div>
      <Textarea
        value={promptText}
        onChange={(e) => updateSetting('promptText', e.target.value)}
        className="min-h-[120px] resize-y font-mono text-xs"
      />
    </div>
  )
}
