import { RefreshCw, Crop, ImageIcon, ScanSearch, CheckCheck, Undo2, Images, LayoutGrid, SlidersHorizontal } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useSettingsStore } from '@/stores/settingsStore'
import { useProcessingStore } from '@/stores/processingStore'
import { undoHostChanges } from '@/lib/embed/host'
import type { FilterType, SortOption } from '@/hooks/useReviewTab'
import { WingCheckButton } from './WingCheckButton'
import { NamingSelect } from './NamingSelect'

interface Props {
  csvFiles: string[]
  selectedCsv: string
  onSelectCsv: (name: string) => void
  onNewCsv: () => void
  onRefresh: () => void
  filter: FilterType
  onFilterChange: (f: FilterType) => void
  sortOption: SortOption
  onSortChange: (s: SortOption) => void
  selectedOnly: boolean
  onSelectedOnlyChange: (value: boolean) => void
  reviewCount: number
  onConfirmShown: () => void
  /** in the bar above, beside the folder's controls (no row of its own) */
  inline?: boolean
}

const FILTERS: { label: string; value: FilterType }[] = [
  { label: 'All', value: 'all' },
  { label: 'Needs review', value: 'review' },
  { label: 'Crossed Out', value: 'crossedOut' },
  { label: 'Has Notes', value: 'hasNotes' },
  { label: 'Skipped', value: 'skipped' },
  { label: 'Mismatches', value: 'mismatches' },
]

const SORT_OPTIONS: { label: string; value: SortOption }[] = [
  { label: 'File Name A-Z', value: 'name-asc' },
  { label: 'File Name Z-A', value: 'name-desc' },
  { label: 'Batch 1-N', value: 'batch-asc' },
  { label: 'Batch N-1', value: 'batch-desc' },
  { label: 'CAM A-Z', value: 'cam-asc' },
  { label: 'CAM Z-A', value: 'cam-desc' },
  { label: 'Date (New-Old)', value: 'date-desc' },
  { label: 'Date (Old-New)', value: 'date-asc' },
]

export function ReviewToolbar({
  csvFiles,
  selectedCsv,
  onSelectCsv,
  onNewCsv,
  onRefresh,
  filter,
  onFilterChange,
  sortOption,
  onSortChange,
  selectedOnly,
  onSelectedOnlyChange,
  reviewCount,
  onConfirmShown,
  inline = false,
}: Props) {
  const { reviewCropEnabled, reviewThumbSize, reviewLayout, reviewZoom, updateSetting } = useSettingsStore()
  // the controls few people need stay behind «More options» (hidden at first inside another app)
  const advanced = useProcessingStore((s) => s.advanced)
  const setAdvanced = useProcessingStore((s) => s.setAdvanced)
  // embedded mode: the rows the hosting page's assistant changed
  const hostChanged = useProcessingStore((s) => s.hostChanges.size)
  const hostName = useProcessingStore((s) => s.hostName)

  return (
    <div className={inline ? 'flex flex-wrap items-center gap-2' : 'flex flex-wrap items-center gap-2 border-b px-3 py-2'}>
      <div className="flex rounded-md border p-0.5" role="radiogroup" aria-label="Cards">
        {([['specimens', Images, 'One card per specimen: its photos side by side'], ['photos', LayoutGrid, 'One card per photo']] as const).map(([value, Icon, tip]) => (
          <Button key={value} variant={reviewLayout === value ? 'secondary' : 'ghost'} size="sm" role="radio" aria-checked={reviewLayout === value}
            aria-label={tip} title={tip} onClick={() => updateSetting('reviewLayout', value)} className="h-7 w-8 px-0">
            <Icon className="h-4 w-4" />
          </Button>
        ))}
      </div>

      {advanced && (<>
      <Select value={selectedCsv} onValueChange={onSelectCsv}>
        <SelectTrigger className="h-8 w-56 text-xs">
          <SelectValue placeholder="Select CSV..." />
        </SelectTrigger>
        <SelectContent>
          {csvFiles.map((f) => (
            <SelectItem key={f} value={f} className="text-xs">
              {f}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Button variant="outline" size="sm" className="text-xs" onClick={onNewCsv}>
        New CSV
      </Button>

      <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onRefresh}>
        <RefreshCw className="h-3.5 w-3.5" />
      </Button>

      <div className="mx-2 h-5 w-px bg-border" />
      </>)}

      <Select value={filter} onValueChange={(v) => onFilterChange(v as FilterType)}>
        <SelectTrigger className="h-8 w-36 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {[...FILTERS, ...(hostChanged > 0 ? [{ label: `Changed by ${hostName}`, value: 'byHost' as FilterType }] : [])].map((f) => (
            <SelectItem key={f.value} value={f.value} className="text-xs">
              {f.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {reviewCount > 0 && (
        <Button
          variant="outline"
          size="sm"
          className="h-8 gap-1.5 border-amber-500/60 text-xs"
          onClick={onConfirmShown}
          title="Confirm every shown photo whose CAM field holds a complete CAMID"
        >
          <CheckCheck className="h-3.5 w-3.5" />
          {reviewCount}
        </Button>
      )}

      {hostChanged > 0 && (
        <Button
          variant="outline"
          size="sm"
          className="h-8 gap-1.5 border-sky-500/60 text-xs"
          onClick={() => toast.success(`Took back the changes to ${undoHostChanges()} photo(s).`)}
          title={`Take back every change made by ${hostName} (${hostChanged} photos)`}
        >
          <Undo2 className="h-3.5 w-3.5" />
          {hostName}: {hostChanged}
        </Button>
      )}

      <NamingSelect />

      <div className="flex items-center gap-1.5">
        <div className="flex rounded-md border p-0.5" role="radiogroup" aria-label="Photos">
          {([['photo', ImageIcon, 'The whole photos'], ['wings', ScanSearch, 'Zoomed to the wings, the envelope beside: to check dorsal and ventral at a glance']] as const).map(([value, Icon, tip]) => (
            <Button key={value} variant={reviewZoom === value ? 'secondary' : 'ghost'} size="sm" role="radio" aria-checked={reviewZoom === value}
              aria-label={tip} title={tip} onClick={() => updateSetting('reviewZoom', value)} className="h-7 gap-1 px-2 text-xs">
              <Icon className="h-3.5 w-3.5" />{value === 'wings' ? 'Wings' : 'Photo'}
            </Button>
          ))}
        </div>
        <input type="range" min={120} max={640} step={20} value={reviewThumbSize} aria-label="Photo size" title={`Photo size: ${reviewThumbSize} px high`}
          onChange={(e) => updateSetting('reviewThumbSize', Number(e.target.value))} className="h-1 w-24 cursor-pointer accent-primary" />
      </div>

      <WingCheckButton />

      {reviewLayout === 'photos' && (
      <Select value={sortOption} onValueChange={(v) => onSortChange(v as SortOption)}>
        <SelectTrigger className="h-8 w-40 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {SORT_OPTIONS.map((s) => (
            <SelectItem key={s.value} value={s.value} className="text-xs">
              {s.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      )}

      {!inline && <div className="flex-1" />}
      <Button variant={advanced ? 'secondary' : 'ghost'} size="sm" className="h-8 gap-1.5 text-xs" aria-pressed={advanced}
        title="Results files, selected photos only, crop and thumbnail size; saving a copy or a ZIP" onClick={() => setAdvanced(!advanced)}>
        <SlidersHorizontal className="h-3.5 w-3.5" />
        More options
      </Button>

      {advanced && (<>
      <div className="flex items-center gap-1.5">
        <Checkbox
          id="review-selected"
          checked={selectedOnly}
          onCheckedChange={(checked) => onSelectedOnlyChange(!!checked)}
        />
        <Label htmlFor="review-selected" className="text-xs cursor-pointer">
          Selected only
        </Label>
      </div>

      <div className="mx-2 h-5 w-px bg-border" />

      {/* Crop toggle */}
      <div className="flex items-center gap-1.5">
        <Checkbox
          id="review-crop"
          checked={reviewCropEnabled}
          onCheckedChange={(c) => updateSetting('reviewCropEnabled', !!c)}
        />
        <Label htmlFor="review-crop" className="flex items-center gap-1 text-xs cursor-pointer">
          <Crop className="h-3 w-3" />
          Crop
        </Label>
      </div>

      </>)}
    </div>
  )
}
