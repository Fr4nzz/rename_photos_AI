import { useState, useEffect } from 'react'
import { useSettingsStore } from '@/stores/settingsStore'
import { useApiKeysStore } from '@/stores/apiKeysStore'
import { useProcessingStore } from '@/stores/processingStore'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Play, Square } from 'lucide-react'
import { listStoredCsvs } from '@/lib/csvHandler'
import type { RunMode } from '@/types'

interface Props {
  onStart: () => void
  onStop: () => void
  hasImages: boolean
}

export function ProcessingControls({ onStart, onStop, hasImages }: Props) {
  const { modelName } = useSettingsStore()
  const { apiKeys } = useApiKeysStore()
  const {
    isProcessing,
    progress,
    runMode,
    retryMessages,
    continueCsvName,
    setRunMode,
    setRetryMessages,
    setContinueCsvName,
  } = useProcessingStore()

  const [csvFiles, setCsvFiles] = useState<string[]>([])

  useEffect(() => {
    if (runMode === 'continue') {
      listStoredCsvs().then(setCsvFiles)
    }
  }, [runMode])

  const canStart = hasImages && apiKeys.length > 0 && !!modelName && !isProcessing

  return (
    <div className="border-t bg-card px-4 py-2 space-y-2">
      <div className="flex items-center gap-3">
        <Select
          value={runMode}
          onValueChange={(v) => setRunMode(v as RunMode)}
          disabled={isProcessing}
        >
          <SelectTrigger className="h-8 w-48 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="start_over" className="text-xs">Start Over</SelectItem>
            <SelectItem value="continue" className="text-xs">Continue from CSV</SelectItem>
            <SelectItem value="retry_specific" className="text-xs">Retry Specific Messages</SelectItem>
          </SelectContent>
        </Select>

        {runMode === 'continue' && (
          <Select
            value={continueCsvName}
            onValueChange={setContinueCsvName}
            disabled={isProcessing}
          >
            <SelectTrigger className="h-8 w-56 text-xs">
              <SelectValue placeholder="Select CSV..." />
            </SelectTrigger>
            <SelectContent>
              {csvFiles.map((f) => (
                <SelectItem key={f} value={f} className="text-xs">{f}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {runMode === 'retry_specific' && (
          <Input
            placeholder="e.g. 1,3,5-7"
            value={retryMessages}
            onChange={(e) => setRetryMessages(e.target.value)}
            className="h-8 w-36 text-xs"
            disabled={isProcessing}
          />
        )}

        <div className="flex-1" />

        {isProcessing ? (
          <Button variant="destructive" size="sm" onClick={onStop} className="gap-1.5">
            <Square className="h-3.5 w-3.5" />
            Stop
          </Button>
        ) : (
          <Button size="sm" onClick={onStart} disabled={!canStart} className="gap-1.5">
            <Play className="h-3.5 w-3.5" />
            Ask AI (Start)
          </Button>
        )}
      </div>

      {(isProcessing || progress.percent > 0) && (
        <div className="space-y-1">
          <Progress value={progress.percent} className="h-2" />
          <p className="text-xs text-muted-foreground">{progress.message}</p>
        </div>
      )}
    </div>
  )
}
