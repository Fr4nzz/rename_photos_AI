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
import { Cpu, Play, ScanText, Sparkles, Square, Wand2, SlidersHorizontal } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { listStoredCsvs } from '@/lib/csvHandler'
import { cpuPlan } from '@/lib/devicePlan'
import type { RunMode } from '@/types'
import type { ReactNode } from 'react'
import type { useProcessTab } from '@/hooks/useProcessTab'
import { GeminiDialog } from './GeminiDialog'

interface Props {
  onStart: () => void
  onStop: () => void
  hasImages: boolean
  /** folder controls shown at the start of the bar */
  children?: ReactNode
  geminiHook: ReturnType<typeof useProcessTab>
  /** «More options» is in the review's controls, in this bar */
  optionsElsewhere?: boolean
}

export function ProcessingControls({ onStart, onStop, hasImages, children, geminiHook, optionsElsewhere = false }: Props) {
  const { modelName, engine, autoRotate, photoWorkers, updateSetting } = useSettingsStore()
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

  const local = engine === 'local'
  // the choice of reader sits behind the review's «More options» (hidden at first inside another app)
  const advanced = useProcessingStore((s) => s.advanced)
  const canStart = hasImages && !isProcessing && (local || (apiKeys.length > 0 && !!modelName))
  const threads = navigator.hardwareConcurrency || 2
  const autoWorkers = cpuPlan(threads).workers

  return (
    <div className="space-y-2 border-b bg-card px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        {children}
        <div className="flex-1" />
        {!advanced && local && !optionsElsewhere && (
          <Button variant="ghost" size="icon" className="h-7 w-7" aria-label="More options" title="More options: the other reader, results files, copies"
            onClick={() => useProcessingStore.getState().setAdvanced(true)}>
            <SlidersHorizontal className="h-3.5 w-3.5" />
          </Button>
        )}
        {(advanced || !local) && (
        <div className="flex rounded-md border p-0.5" role="radiogroup" aria-label="Reader">
          {([['local', Cpu, 'On this computer: built-in CAMID reader, no API key, photos stay local'],
             ['gemini', Sparkles, 'Gemini: sends photo grids to Google with your API key']] as const).map(([value, Icon, tip]) => (
            <Tooltip key={value}>
              <TooltipTrigger asChild>
                <Button
                  variant={engine === value ? 'secondary' : 'ghost'}
                  size="sm"
                  role="radio"
                  aria-checked={engine === value}
                  aria-label={tip}
                  disabled={isProcessing}
                  onClick={() => updateSetting('engine', value)}
                  className="h-7 w-9 px-0"
                >
                  <Icon className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{tip}</TooltipContent>
            </Tooltip>
          ))}
        </div>
        )}
        {advanced && local && (
          <Tooltip>
            <TooltipTrigger asChild>
              <div>
                <Select value={String(photoWorkers)} onValueChange={(v) => updateSetting('photoWorkers', Number(v))} disabled={isProcessing}>
                  <SelectTrigger className="h-8 w-[9.5rem] text-xs" aria-label="Photos at once">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0" className="text-xs">{`At once: auto (${autoWorkers})`}</SelectItem>
                    {Array.from({ length: threads }, (_, i) => i + 1).map((n) => (
                      <SelectItem key={n} value={String(n)} className="text-xs">{`At once: ${n}`}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              {`Photos read at the same time on this computer. Auto: ${autoWorkers} here (2 with less than 8 GB of memory). More is faster on processors with many threads (this one has ${threads}), but each takes about 0.5 GB of memory, up to 1 GB on large photos: pick fewer if other programs need the memory or the page reloads on its own. With the graphics card, 2 are enough.`}
            </TooltipContent>
          </Tooltip>
        )}
        {!local && (<>
        <GeminiDialog hook={geminiHook} />
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

        </>)}

        {isProcessing ? (
          <Button variant="destructive" size="sm" onClick={onStop} className="gap-1.5">
            <Square className="h-3.5 w-3.5" />
            Stop
          </Button>
        ) : (<>
          {local && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={autoRotate ? 'secondary' : 'ghost'}
                  size="sm"
                  role="switch"
                  aria-checked={autoRotate}
                  aria-label="Auto-rotate"
                  onClick={() => updateSetting('autoRotate', !autoRotate)}
                  className={`gap-1.5 text-xs ${autoRotate ? '' : 'text-muted-foreground line-through'}`}
                >
                  <Wand2 className="h-3.5 w-3.5" />
                  Auto-rotate
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {autoRotate
                  ? 'On: after reading, each photo (and its RAW) is turned so its forewings are on top, by its wings or envelope when sure, else as its camera\'s other photos. Not on wing clips. Lossless; Undo or Restore reverts it.'
                  : 'Off: rotations are only suggested in Review and written when you rename.'}
              </TooltipContent>
            </Tooltip>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" onClick={onStart} disabled={!canStart} className="gap-1.5">
                {local ? <ScanText className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                {local ? 'Read photos' : 'Ask AI (Start)'}
              </Button>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              {local
                ? 'On this computer: reads the CAMID on each envelope, checks the wings (the side they show against d/v, the same wings in both photos, which way up), and turns the batch upright with Auto-rotate.'
                : 'Sends the photos to Gemini to read their IDs.'}
            </TooltipContent>
          </Tooltip>
        </>)}
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
