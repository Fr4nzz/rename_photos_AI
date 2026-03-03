import { useState } from 'react'
import { toast } from 'sonner'
import { useSettingsStore } from '@/stores/settingsStore'
import { useProcessingStore } from '@/stores/processingStore'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  getDownloadCapabilities,
  saveToFolder,
  downloadAsZip,
  getBrowserSaveAsInstructions,
} from '@/lib/fileDownload'
import {
  saveRenameLog,
  getRenameLog,
  type RenameLogEntry,
} from '@/lib/csvHandler'
import { supportsDirectoryPicker } from '@/lib/fileAccess'
import { logger } from '@/lib/logger'
import type { SuffixMode } from '@/types'
import {
  Calculator,
  Save,
  Download,
  FolderDown,
  FileDown,
  Info,
  FileEdit,
  Undo2,
} from 'lucide-react'

/**
 * Get a readwrite directory handle, reusing the stored one if available.
 * Falls back to showing a directory picker if no stored handle exists.
 */
async function getReadWriteDirHandle(): Promise<FileSystemDirectoryHandle> {
  const stored = useProcessingStore.getState().dirHandle
  if (stored) {
    // Ensure we have readwrite permission
    const perm = await (stored as any).queryPermission({ mode: 'readwrite' })
    if (perm === 'granted') return stored
    // Request readwrite if only read was granted
    const req = await (stored as any).requestPermission({ mode: 'readwrite' })
    if (req === 'granted') return stored
  }
  // Fallback: ask user to pick the folder
  return await (window as any).showDirectoryPicker({
    id: 'photo-rename',
    mode: 'readwrite',
  })
}

/**
 * Rename a file in-place within a directory.
 * Tries move() API first (efficient O(1) rename), falls back to copy-and-delete.
 */
async function renameFileInDir(
  dirHandle: FileSystemDirectoryHandle,
  oldName: string,
  newName: string
): Promise<void> {
  if (oldName === newName) return

  const sourceHandle = await dirHandle.getFileHandle(oldName)

  // Try native move() first (Chrome may support it for local files)
  if (typeof (sourceHandle as any).move === 'function') {
    try {
      await (sourceHandle as any).move(newName)
      return
    } catch {
      // move() not supported for local files — fall through to copy-and-delete
    }
  }

  // Fallback: read → write new → delete original
  const file = await sourceHandle.getFile()
  const destHandle = await dirHandle.getFileHandle(newName, { create: true })
  const writable = await destHandle.createWritable()
  await writable.write(file)
  await writable.close()
  await dirHandle.removeEntry(oldName)
}

interface Props {
  onRecalculate: () => void
  onSave: () => void
  onExportCsv: () => void
  hasData: boolean
}

export function ReviewActionBar({
  onRecalculate,
  onSave,
  onExportCsv,
  hasData,
}: Props) {
  const { suffixMode, customSuffixes, updateSetting } = useSettingsStore()
  const { photoRows, fileMap, setPhotoRows } = useProcessingStore()
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null)
  const [isRenaming, setIsRenaming] = useState(false)
  const { canSaveToFolder, tier } = getDownloadCapabilities()

  // Build named blobs from fileMap using the 'to' field
  function buildRenamedFiles(): { name: string; blob: Blob }[] {
    const files: { name: string; blob: Blob }[] = []
    for (const row of photoRows) {
      if (!row.to?.trim() || row.skip === 'x') continue
      const file = fileMap.get(row.from)
      if (!file) continue
      files.push({ name: row.to, blob: file })
    }
    return files
  }

  const handleSaveToFolder = async () => {
    const files = buildRenamedFiles()
    if (files.length === 0) {
      toast.error('No files to save. Make sure files are loaded and names are calculated.')
      return
    }

    try {
      setDownloadProgress(0)
      const count = await saveToFolder(files, (current, total) => {
        setDownloadProgress(Math.round((current / total) * 100))
      })
      toast.success(`Saved ${count} files to folder`)
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        toast.error(`Save failed: ${e.message}`)
      }
    } finally {
      setDownloadProgress(null)
    }
  }

  const handleDownloadZip = async () => {
    const files = buildRenamedFiles()
    if (files.length === 0) {
      toast.error('No files to download. Make sure files are loaded and names are calculated.')
      return
    }

    try {
      setDownloadProgress(0)
      await downloadAsZip(files, 'renamed-photos.zip', (percent) => {
        setDownloadProgress(percent)
      })
      toast.success(`ZIP with ${files.length} files ready`)
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        toast.error(`Download failed: ${e.message}`)
      }
    } finally {
      setDownloadProgress(null)
    }
  }

  // Rename files in-place using File System Access API
  const handleRenameFiles = async () => {
    if (!supportsDirectoryPicker()) {
      toast.error('File rename requires Chrome or Edge browser')
      return
    }

    const rowsToRename = photoRows.filter((r) => r.to?.trim() && r.skip !== 'x' && r.status !== 'Renamed')
    if (rowsToRename.length === 0) {
      toast.error('No files to rename. Calculate names first.')
      return
    }

    try {
      const dirHandle = await getReadWriteDirHandle()

      setIsRenaming(true)
      setDownloadProgress(0)

      const renameLog: RenameLogEntry[] = []
      let renamed = 0

      for (let i = 0; i < rowsToRename.length; i++) {
        const row = rowsToRename[i]
        try {
          await renameFileInDir(dirHandle, row.currentPath, row.to)

          renameLog.push({
            original: row.currentPath,
            renamed: row.to,
            timestamp: new Date().toISOString(),
          })
          renamed++
        } catch (err: any) {
          logger.warn(`Could not rename ${row.currentPath}: ${err.message}`)
        }

        setDownloadProgress(Math.round(((i + 1) / rowsToRename.length) * 100))
      }

      // Save rename log for restore
      const existingLog = await getRenameLog()
      await saveRenameLog([...existingLog, ...renameLog])

      // Update row statuses
      const renamedSet = new Set(renameLog.map((e) => e.original))
      const updatedRows = photoRows.map((r) => {
        if (renamedSet.has(r.currentPath)) {
          const entry = renameLog.find((e) => e.original === r.currentPath)
          return { ...r, status: 'Renamed' as const, currentPath: entry?.renamed ?? r.currentPath }
        }
        return r
      })
      setPhotoRows(updatedRows)

      toast.success(`Renamed ${renamed} files in-place`)
      logger.info(`Renamed ${renamed} files, ${rowsToRename.length - renamed} skipped`)
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        toast.error(`Rename failed: ${e.message}`)
      }
    } finally {
      setIsRenaming(false)
      setDownloadProgress(null)
    }
  }

  // Restore original names using rename log
  const handleRestore = async () => {
    const log = await getRenameLog()
    if (log.length === 0) {
      toast.error('No rename log found. Nothing to restore.')
      return
    }

    try {
      const dirHandle = await getReadWriteDirHandle()

      setIsRenaming(true)
      setDownloadProgress(0)

      let restored = 0
      for (let i = 0; i < log.length; i++) {
        const entry = log[i]
        try {
          await renameFileInDir(dirHandle, entry.renamed, entry.original)
          restored++
        } catch (err: any) {
          logger.warn(`Could not restore ${entry.renamed}: ${err.message}`)
        }

        setDownloadProgress(Math.round(((i + 1) / log.length) * 100))
      }

      // Clear rename log after restore
      await saveRenameLog([])

      // Update row statuses back to Original
      const restoredNames = new Set(log.map((e) => e.renamed))
      const updatedRows = photoRows.map((r) => {
        if (restoredNames.has(r.currentPath)) {
          const entry = log.find((e) => e.renamed === r.currentPath)
          return { ...r, status: 'Original' as const, currentPath: entry?.original ?? r.currentPath }
        }
        return r
      })
      setPhotoRows(updatedRows)

      toast.success(`Restored ${restored} files to original names`)
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        toast.error(`Restore failed: ${e.message}`)
      }
    } finally {
      setIsRenaming(false)
      setDownloadProgress(null)
    }
  }

  const saveAsInfo = getBrowserSaveAsInstructions()
  const showSaveAsTip = tier === 'zip-fallback'

  return (
    <div className="border-t bg-card px-3 py-2 space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {/* Suffix mode */}
        <Select
          value={suffixMode}
          onValueChange={(v) => updateSetting('suffixMode', v as SuffixMode)}
        >
          <SelectTrigger className="h-8 w-44 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="Standard" className="text-xs">
              Standard (d, v, d2, v2, ...)
            </SelectItem>
            <SelectItem value="Wing Clips" className="text-xs">
              Wing Clips (v1, v2, v3, ...)
            </SelectItem>
            <SelectItem value="Custom" className="text-xs">
              Custom
            </SelectItem>
          </SelectContent>
        </Select>

        {suffixMode === 'Custom' && (
          <Input
            value={customSuffixes}
            onChange={(e) => updateSetting('customSuffixes', e.target.value)}
            placeholder="d,v,body"
            className="h-8 w-28 text-xs"
          />
        )}

        <div className="mx-1 h-5 w-px bg-border" />

        <Button
          variant="outline"
          size="sm"
          className="gap-1 text-xs"
          onClick={onRecalculate}
          disabled={!hasData}
        >
          <Calculator className="h-3.5 w-3.5" />
          Recalculate
        </Button>

        <Button
          variant="outline"
          size="sm"
          className="gap-1 text-xs"
          onClick={onSave}
          disabled={!hasData}
        >
          <Save className="h-3.5 w-3.5" />
          Save
        </Button>

        <Button
          variant="outline"
          size="sm"
          className="gap-1 text-xs"
          onClick={onExportCsv}
          disabled={!hasData}
        >
          <FileDown className="h-3.5 w-3.5" />
          Export CSV
        </Button>

        <div className="flex-1" />

        {/* Rename & Restore (FSA API) */}
        {canSaveToFolder && (
          <>
            <Button
              size="sm"
              className="gap-1 text-xs"
              onClick={handleRenameFiles}
              disabled={!hasData || isRenaming}
            >
              <FileEdit className="h-3.5 w-3.5" />
              Rename Files
            </Button>

            <Button
              variant="outline"
              size="sm"
              className="gap-1 text-xs"
              onClick={handleRestore}
              disabled={isRenaming}
            >
              <Undo2 className="h-3.5 w-3.5" />
              Restore
            </Button>

            <div className="mx-1 h-5 w-px bg-border" />

            <Button
              variant="outline"
              size="sm"
              className="gap-1 text-xs"
              onClick={handleSaveToFolder}
              disabled={!hasData || isRenaming}
            >
              <FolderDown className="h-3.5 w-3.5" />
              Save to Folder
            </Button>
          </>
        )}

        <Button
          variant={canSaveToFolder ? 'outline' : 'default'}
          size="sm"
          className="gap-1 text-xs"
          onClick={handleDownloadZip}
          disabled={!hasData || isRenaming}
        >
          <Download className="h-3.5 w-3.5" />
          Download ZIP
        </Button>

        {showSaveAsTip && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <Info className="h-3.5 w-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs text-xs">
              <p className="font-medium">Want to choose where files are saved?</p>
              <p>
                Go to <code>{saveAsInfo.path}</code>
              </p>
              <p>{saveAsInfo.steps}</p>
            </TooltipContent>
          </Tooltip>
        )}
      </div>

      {downloadProgress !== null && (
        <Progress value={downloadProgress} className="h-1.5" />
      )}
    </div>
  )
}
