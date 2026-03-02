import { RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { FilterType, SortOption } from '@/hooks/useReviewTab'

interface Props {
  csvFiles: string[]
  selectedCsv: string
  onSelectCsv: (name: string) => void
  onRefresh: () => void
  filter: FilterType
  onFilterChange: (f: FilterType) => void
  sortOption: SortOption
  onSortChange: (s: SortOption) => void
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
]

export function ReviewToolbar({
  csvFiles,
  selectedCsv,
  onSelectCsv,
  onRefresh,
  filter,
  onFilterChange,
  sortOption,
  onSortChange,
}: Props) {
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
    </div>
  )
}
