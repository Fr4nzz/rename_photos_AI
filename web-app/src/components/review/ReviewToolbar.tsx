import { RefreshCw, Crop, ImageIcon } from 'lucide-react'
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
import type { FilterType, SortOption } from '@/hooks/useReviewTab'

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
}

const FILTERS: { label: string; value: FilterType }[] = [
  { label: 'All', value: 'all' },
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
}: Props) {
  const { reviewCropEnabled, reviewThumbSize, updateSetting } = useSettingsStore()

  return (
    <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
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

      <Select value={filter} onValueChange={(v) => onFilterChange(v as FilterType)}>
        <SelectTrigger className="h-8 w-36 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {FILTERS.map((f) => (
            <SelectItem key={f.value} value={f.value} className="text-xs">
              {f.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

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

      <div className="mx-2 h-5 w-px bg-border" />

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

      {/* Thumb size slider */}
      <div className="flex items-center gap-1.5">
        <ImageIcon className="h-3 w-3 text-muted-foreground" />
        <input
          type="range"
          min={80}
          max={640}
          step={10}
          value={reviewThumbSize}
          onChange={(e) => updateSetting('reviewThumbSize', Number(e.target.value))}
          className="h-1 w-40 cursor-pointer accent-primary"
        />
        <span className="text-[10px] text-muted-foreground w-8">{reviewThumbSize}</span>
      </div>
    </div>
  )
}
