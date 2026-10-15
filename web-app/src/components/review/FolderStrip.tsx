import { useMemo } from 'react'
import { FileWarning, FilePlus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useProcessingStore } from '@/stores/processingStore'
import { matchFolder } from '@/lib/folderMatch'
import { readMore } from '@/lib/readMore'

const few = (names: string[]) => names.slice(0, 4).join(', ') + (names.length > 4 ? ` and ${names.length - 4} more` : '')

/** The results against the folder now: photos gone from it, and photos added to it since the reading (read with a click). */
export function FolderStrip() {
  const rows = useProcessingStore((s) => s.photoRows)
  const files = useProcessingStore((s) => s.imageFiles)
  const busy = useProcessingStore((s) => s.isProcessing)
  const { missing, unread } = useMemo(() => matchFolder(rows, files), [rows, files])
  if (!missing.length && !unread.length) return null
  return (
    <div className="space-y-1 border-b bg-amber-500/5 px-3 py-1.5 text-xs">
      {missing.length > 0 && (
        <div className="flex items-center gap-1.5 text-amber-800 dark:text-amber-300" data-folder-missing>
          <FileWarning className="h-3.5 w-3.5 flex-shrink-0" />
          <span>{missing.length} photo(s) of these results are no longer in the folder (moved or deleted): {few(missing.map((r) => r.currentPath))}. They are not renamed.</span>
        </div>
      )}
      {unread.length > 0 && (
        <div className="flex items-center gap-1.5" data-folder-unread>
          <FilePlus className="h-3.5 w-3.5 flex-shrink-0 text-sky-700 dark:text-sky-300" />
          <span>{unread.length} photo(s) in the folder are not in these results: {few(unread.map((f) => f.name))}.</span>
          <Button size="sm" variant="outline" className="h-6 px-2 text-xs" disabled={busy} onClick={() => readMore(unread)}>
            Read them and add them
          </Button>
        </div>
      )}
    </div>
  )
}
