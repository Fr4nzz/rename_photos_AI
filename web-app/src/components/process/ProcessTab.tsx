import { useProcessTab } from '@/hooks/useProcessTab'
import { FolderSelector } from './FolderSelector'
import { PreviewSelector } from './PreviewSelector'
import { RotationSettings } from './RotationSettings'
import { CropSettings } from './CropSettings'
import { ApiSettings } from './ApiSettings'
import { PromptEditor } from './PromptEditor'
import { PreviewPanel } from './PreviewPanel'
import { ProcessingControls } from './ProcessingControls'

export function ProcessTab() {
  const hook = useProcessTab()

  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1">
        {/* Left panel: Settings */}
        <div className="w-80 flex-shrink-0 overflow-y-auto border-r p-3 space-y-3">
          <FolderSelector
            folderName={hook.folderName}
            onSelect={hook.selectDirectory}
            onFileInput={hook.handleFileInput}
          />
          <PreviewSelector
            imageFiles={hook.imageFiles}
            selectedImageIndex={hook.selectedImageIndex}
            onSelectImage={hook.setSelectedImageIndex}
            selectedGridIndex={hook.selectedGridIndex}
            onSelectGrid={hook.setSelectedGridIndex}
            gridCount={hook.gridCount}
          />
          <RotationSettings backendConnected={hook.backendConnected} />
          <CropSettings />
          <ApiSettings />
        </div>

        {/* Right panel: Previews + Prompt */}
        <div className="flex min-w-0 flex-1 flex-col overflow-y-auto p-3 space-y-3">
          <PreviewPanel previews={hook.previews} />
          <PromptEditor />
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
