import { useProcessTab } from '@/hooks/useProcessTab'
import { PreviewSelector } from './PreviewSelector'
import { RotationSettings } from './RotationSettings'
import { CropSettings } from './CropSettings'
import { ApiSettings } from './ApiSettings'
import { PromptEditor } from './PromptEditor'
import { PreviewPanel } from './PreviewPanel'
import { ProcessingControls } from './ProcessingControls'
import { useSettingsStore } from '@/stores/settingsStore'

export function ProcessTab() {
  const hook = useProcessTab()
  const local = useSettingsStore((s) => s.engine) === 'local'

  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1">
        {/* Left panel: Settings */}
        <div className="w-80 flex-shrink-0 overflow-y-auto border-r p-3 space-y-3">
          <div className="rounded border bg-card p-3">
            <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Selected Images
            </div>
            <div className="mt-1 text-2xl font-semibold">{hook.imageFiles.length}</div>
            <div className="text-xs text-muted-foreground">
              Use Select Images to change the active set.
            </div>
          </div>
          <PreviewSelector
            imageFiles={hook.imageFiles}
            selectedImageIndex={hook.selectedImageIndex}
            onSelectImage={hook.setSelectedImageIndex}
            selectedGridIndex={hook.selectedGridIndex}
            onSelectGrid={hook.setSelectedGridIndex}
            gridCount={hook.gridCount}
          />
          {!local && (<>
            <RotationSettings />
            <CropSettings />
            <ApiSettings />
          </>)}
        </div>

        {/* Right panel: Previews + Prompt */}
        <div className="flex min-w-0 flex-1 flex-col overflow-y-auto p-3 space-y-3">
          {local ? (
            <div className="rounded border bg-card p-4 text-sm text-muted-foreground">
              Photos are read on this computer by the built-in CAMID reader and checked against the
              specimen database. Uncertain photos go to Review with a suggestion.
            </div>
          ) : (<>
            <PreviewPanel previews={hook.previews} />
            <PromptEditor />
          </>)}
        </div>
      </div>

      {/* Bottom bar: Processing controls */}
      <ProcessingControls
        onStart={hook.startProcessing}
        onStop={hook.stopProcessing}
        hasImages={hook.imageFiles.length > 0}
      />
    </div>
  )
}
