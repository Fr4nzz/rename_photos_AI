import { useReviewTab } from '@/hooks/useReviewTab'
import { ReviewToolbar } from './ReviewToolbar'
import { ReviewPagination } from './ReviewPagination'
import { ReviewGrid } from './ReviewGrid'
import { ReviewActionBar } from './ReviewActionBar'

export function ReviewTab() {
  const hook = useReviewTab()

  return (
    <div className="flex h-full min-h-0 flex-col">
      <ReviewToolbar
        csvFiles={hook.csvFiles}
        selectedCsv={hook.selectedCsv}
        onSelectCsv={hook.loadCsv}
        onRefresh={hook.refreshCsvList}
        filter={hook.filter}
        onFilterChange={hook.setFilter}
        sortOption={hook.sortOption}
        onSortChange={hook.setSortOption}
      />

      <ReviewPagination
        currentPage={hook.currentPage}
        totalPages={hook.totalPages}
        totalItems={hook.filteredRows.length}
        onPageChange={hook.setCurrentPage}
      />

      <ReviewGrid
        rows={hook.pagedRows}
        onUpdateRow={hook.updateRow}
        duplicatePairs={hook.duplicatePairs}
      />

      <ReviewActionBar
        onRecalculate={hook.recalculateNames}
        onSave={hook.saveChanges}
        onExportCsv={hook.exportCsv}
        hasData={hook.photoRows.length > 0}
      />
    </div>
  )
}
