import { Feather, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useProcessingStore } from '@/stores/processingStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { runWingCheck, stopWingCheck } from '@/lib/wings/runWings'

/** Review's button for the wing check of the photos not checked yet, and its progress while it runs. */
export function WingCheckButton() {
  const run = useProcessingStore((s) => s.wingRun)
  const unchecked = useProcessingStore((s) => s.photoRows.filter((r) => (s.fileMap.has(r.from) || s.fileMap.has(r.currentPath)) && !s.wingChecks.has(r.from)).length)
  const clips = useSettingsStore((s) => s.suffixMode === 'Wing Clips')
  if (run) {
    return (
      <Button variant="outline" size="sm" className="h-8 gap-1.5 text-xs" onClick={stopWingCheck}
        title="Checking which side of the wings each photo shows, how many wings and which way up. Click to stop.">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Wings {run.done}/{run.total}
      </Button>
    )
  }
  if (!unchecked || clips) return null
  return (
    <Button variant="outline" size="sm" className="h-8 gap-1.5 text-xs" onClick={() => runWingCheck()}
      title="Check the wings of the photos: the side they show against d/v, a wing fewer in one photo of a pair, and which way up. Runs on this computer.">
      <Feather className="h-3.5 w-3.5" />
      Check wings ({unchecked})
    </Button>
  )
}
