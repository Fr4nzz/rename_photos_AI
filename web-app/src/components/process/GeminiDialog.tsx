import { Settings2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { ApiKeysTab } from '@/components/api-keys/ApiKeysTab'
import type { useProcessTab } from '@/hooks/useProcessTab'
import { PreviewSelector } from './PreviewSelector'
import { RotationSettings } from './RotationSettings'
import { CropSettings } from './CropSettings'
import { ApiSettings } from './ApiSettings'
import { PromptEditor } from './PromptEditor'
import { PreviewPanel } from './PreviewPanel'

/** Settings of the Gemini reader (prompt, grids, model, API keys), out of the way of the local flow. */
export function GeminiDialog({ hook }: { hook: ReturnType<typeof useProcessTab> }) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="Gemini settings">
          <Settings2 className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-5xl">
        <DialogHeader>
          <DialogTitle>Gemini settings</DialogTitle>
        </DialogHeader>
        <div className="grid gap-3 md:grid-cols-[18rem_1fr]">
          <div className="space-y-3">
            <PreviewSelector
              imageFiles={hook.imageFiles}
              selectedImageIndex={hook.selectedImageIndex}
              onSelectImage={hook.setSelectedImageIndex}
              selectedGridIndex={hook.selectedGridIndex}
              onSelectGrid={hook.setSelectedGridIndex}
              gridCount={hook.gridCount}
            />
            <RotationSettings />
            <CropSettings />
            <ApiSettings />
          </div>
          <div className="min-w-0 space-y-3">
            <PreviewPanel previews={hook.previews} />
            <PromptEditor />
            <ApiKeysTab />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
