import type { ImageSortOption } from '@/lib/selection'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface Props {
  query: string
  onQueryChange: (value: string) => void
  extensionFilter: string
  onExtensionFilterChange: (value: string) => void
  extensions: string[]
  sortOption: ImageSortOption
  onSortOptionChange: (value: ImageSortOption) => void
  selectedCount: number
  totalCount: number
  onSelectAll: () => void
  onClear: () => void
  onInvert: () => void
  onReplaceMatches: () => void
  onAddMatches: () => void
}

export function ImageSelectionToolbar({
  query,
  onQueryChange,
  extensionFilter,
  onExtensionFilterChange,
  extensions,
  sortOption,
  onSortOptionChange,
  selectedCount,
  totalCount,
  onSelectAll,
  onClear,
  onInvert,
  onReplaceMatches,
  onAddMatches,
}: Props) {
  return (
    <div className="flex flex-wrap items-end gap-2 border-b p-3">
      <div className="space-y-1">
        <Label className="text-xs text-muted-foreground">Selected Images</Label>
        <div className="text-sm font-medium">{selectedCount} / {totalCount}</div>
      </div>

      <Button variant="outline" size="sm" onClick={selectedCount === totalCount ? onClear : onSelectAll}>
        {selectedCount === totalCount ? 'Unselect All' : 'Select All'}
      </Button>
      <Button variant="outline" size="sm" onClick={onInvert}>Invert</Button>

      <div className="min-w-48 flex-1 space-y-1">
        <Label className="text-xs text-muted-foreground">Filename contains</Label>
        <Input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          className="h-8 text-xs"
          placeholder="IMG_001, CAM, .JPG..."
        />
      </div>
      <Button variant="outline" size="sm" onClick={onReplaceMatches}>Select Matches</Button>
      <Button variant="outline" size="sm" onClick={onAddMatches}>Add Matches</Button>

      <Select value={extensionFilter} onValueChange={onExtensionFilterChange}>
        <SelectTrigger className="h-8 w-28 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all" className="text-xs">All Types</SelectItem>
          {extensions.map((ext) => (
            <SelectItem key={ext} value={ext} className="text-xs">{ext}</SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select value={sortOption} onValueChange={(value) => onSortOptionChange(value as ImageSortOption)}>
        <SelectTrigger className="h-8 w-36 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="name-asc" className="text-xs">Name A-Z</SelectItem>
          <SelectItem value="name-desc" className="text-xs">Name Z-A</SelectItem>
          <SelectItem value="date-desc" className="text-xs">Newest</SelectItem>
          <SelectItem value="date-asc" className="text-xs">Oldest</SelectItem>
          <SelectItem value="type-asc" className="text-xs">File Type</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}
