import { useReviewTab } from '@/hooks/useReviewTab'
import { ReviewToolbar } from './ReviewToolbar'
import { ReviewPagination } from './ReviewPagination'
import { ReviewGrid } from './ReviewGrid'
import { ReviewActionBar } from './ReviewActionBar'
import { IdTimeline } from './IdTimeline'

export function ReviewTab() {
  const hook = useReviewTab()

  // jump from the timeline to a photo's card: show it (clearing a filter that hides it), then scroll
  const jumpTo = (photoId: number) => {
    let index = hook.filteredRows.findIndex((r) => r.photoId === photoId)
    if (index < 0) {
      hook.setFilter('all')
      index = [...hook.photoRows].sort((a, b) => a.from.localeCompare(b.from)).findIndex((r) => r.photoId === photoId)
    }
    hook.setCurrentPage(Math.floor(Math.max(0, index) / hook.itemsPerPage) + 1)
    setTimeout(() => document.getElementById(`card-${photoId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 150)
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <ReviewToolbar
        csvFiles={hook.csvFiles}
        selectedCsv={hook.selectedCsv}
        onSelectCsv={hook.loadCsv}
        onNewCsv={hook.createNewCsv}
        onRefresh={hook.refreshCsvList}
        filter={hook.filter}
        onFilterChange={hook.setFilter}
        sortOption={hook.sortOption}
        onSortChange={hook.setSortOption}
        selectedOnly={hook.reviewSelectedOnly}
        onSelectedOnlyChange={hook.setReviewSelectedOnly}
        reviewCount={hook.reviewCount}
        onConfirmShown={hook.confirmShown}
      />

      <IdTimeline rows={hook.photoRows} onJump={jumpTo} />

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
        onSave={hook.saveChanges}
        onExportCsv={hook.exportCsv}
        hasData={hook.photoRows.length > 0}
        rowsToActOn={hook.filteredRows}
      />
    </div>
  )
}
