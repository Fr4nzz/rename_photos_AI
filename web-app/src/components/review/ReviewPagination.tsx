import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface Props {
  currentPage: number
  totalPages: number
  totalItems: number
  onPageChange: (page: number) => void
}

export function ReviewPagination({
  currentPage,
  totalPages,
  totalItems,
  onPageChange,
}: Props) {
  return (
    <div className="flex items-center justify-between border-b px-3 py-1.5">
      <span className="text-xs text-muted-foreground">
        {totalItems} items
      </span>

      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          disabled={currentPage <= 1}
          onClick={() => onPageChange(currentPage - 1)}
        >
          <ChevronLeft className="h-3.5 w-3.5" />
        </Button>

        <Input
          type="number"
          min={1}
          max={totalPages}
          value={currentPage}
          onChange={(e) => {
            const n = parseInt(e.target.value, 10)
            if (n >= 1 && n <= totalPages) onPageChange(n)
          }}
          className="h-7 w-12 text-center text-xs"
        />

        <span className="text-xs text-muted-foreground">of {totalPages}</span>

        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          disabled={currentPage >= totalPages}
          onClick={() => onPageChange(currentPage + 1)}
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  )
}
