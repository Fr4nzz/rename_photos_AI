import { Feather } from 'lucide-react'
import { useProcessingStore } from '@/stores/processingStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { wingFindings, type WingCheck, type WingFinding } from '@/lib/wings/checks'
import type { PhotoRow, SuffixMode } from '@/types'

// the findings of the whole batch, worked out once for all the cards drawn from the same rows
let memo: { rows: PhotoRow[]; checks: Map<string, WingCheck>; mode: SuffixMode; out: Map<string, WingFinding[]> } | null = null
function findingsOf(rows: PhotoRow[], checks: Map<string, WingCheck>, mode: SuffixMode) {
  if (!memo || memo.rows !== rows || memo.checks !== checks || memo.mode !== mode) memo = { rows, checks, mode, out: wingFindings(rows, checks, mode) }
  return memo.out
}

/** What the wings say about a photo as it is named and turned now (nothing when they agree). */
export function WingNotes({ row }: { row: PhotoRow }) {
  const rows = useProcessingStore((s) => s.photoRows)
  const checks = useProcessingStore((s) => s.wingChecks)
  const mode = useSettingsStore((s) => s.suffixMode)
  const notes = checks.size ? findingsOf(rows, checks, mode).get(row.from) : undefined
  if (!notes?.length || row.skip === 'x') return null
  return (
    <div className="space-y-0.5 rounded-md bg-amber-500/10 p-1 text-[11px] text-amber-800 dark:text-amber-300" data-wing-notes>
      {notes.map((n) => (
        <div key={n.text} className="flex items-start gap-1">
          <Feather className="mt-px h-3 w-3 flex-shrink-0" />
          <span>{n.text}</span>
        </div>
      ))}
    </div>
  )
}
