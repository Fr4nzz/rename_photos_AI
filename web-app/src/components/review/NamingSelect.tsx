import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useSettingsStore } from '@/stores/settingsStore'
import type { SuffixMode } from '@/types'

/** How the photos of one ID are named: d/v (more views d2, v2…), wing clips (v1, v2…), or a pattern of one's own. */
export function NamingSelect() {
  const { suffixMode, customSuffixes, updateSetting } = useSettingsStore()
  return (
    <div className="flex items-center gap-1.5">
      <Tooltip>
        <TooltipTrigger asChild>
          <div>
            <Select value={suffixMode} onValueChange={(v) => updateSetting('suffixMode', v as SuffixMode)}>
              <SelectTrigger className="h-8 w-36 text-xs" aria-label="Naming">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Standard" className="text-xs">d, v (d2, v2…)</SelectItem>
                <SelectItem value="Wing Clips" className="text-xs">Wing clips: v1, v2…</SelectItem>
                <SelectItem value="Custom" className="text-xs">My pattern</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">
          How the photos of one ID are named. d, v: the upper side, then the underside, by the wings when they are sure and by the order taken otherwise; more photos of a side get d2, v2 (CRISPR). Wing clips: v1, v2… in the order taken. A view typed on a photo stays.
        </TooltipContent>
      </Tooltip>
      {suffixMode === 'Custom' && (
        <Input value={customSuffixes} onChange={(e) => updateSetting('customSuffixes', e.target.value)}
          placeholder="d,v,body" aria-label="Pattern" title="The views in the order taken, separated by commas" className="h-8 w-28 text-xs" />
      )}
    </div>
  )
}
