import { useState } from 'react'
import { toast } from 'sonner'
import { useSettingsStore } from '@/stores/settingsStore'
import { useProcessingStore } from '@/stores/processingStore'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { companionsOf, indexByStem, planRenames } from '@/lib/renamePlan'
import { writeOrientation } from '@/lib/orientation'
import { pendingRotation } from '@/lib/rotationPlan'
import { appendRotationLog, writeRotations } from '@/lib/rotationWrite'
import { getRotationLog, saveRotationLog } from '@/lib/csvHandler'
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
import { folderAccessHelp, supportsDirectoryPicker, getImageFilesFromHandle } from '@/lib/fileAccess'
import { SUPPORTED_RAW_EXTENSIONS } from '@/lib/constants'
import { getErrorMessage, getErrorName } from '@/lib/errors'
import { logger } from '@/lib/logger'
import type { PhotoRow, SuffixMode } from '@/types'
import {
  Save,
  Download,
  FolderDown,
  FileDown,
  Info,
  FileEdit,
  RotateCw,
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
    const perm = await stored.queryPermission?.({ mode: 'readwrite' })
    if (perm === 'granted') return stored
    // Request readwrite if only read was granted
    const req = await stored.requestPermission?.({ mode: 'readwrite' })
    if (req === 'granted') return stored
  }
  // Fallback: ask user to pick the folder
  if (!window.showDirectoryPicker) {
    throw new Error(folderAccessHelp())
  }
  return await window.showDirectoryPicker({
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
  if (sourceHandle.move) {
    try {
      await sourceHandle.move(newName)
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

/**
 * Re-read directory and rebuild fileMap so Review tab thumbnails
 * use fresh File objects after rename/restore.
 */
async function refreshFileMapFromDir(
  dirHandle: FileSystemDirectoryHandle,
  rows: PhotoRow[]
) {
  const entries = await getImageFilesFromHandle(dirHandle, 'all')
  const filesByName = new Map<string, File>()
  for (const e of entries) filesByName.set(e.name, e.file)
  const newFileMap = new Map<string, File>()
  for (const row of rows) {
    const file = filesByName.get(row.currentPath)
    if (file) newFileMap.set(row.from, file)
  }
  useProcessingStore.getState().setFileMap(newFileMap)
  logger.info(`Refreshed fileMap: ${newFileMap.size} entries`)
}

interface Props {
  onSave: () => void
  onExportCsv: () => void
  hasData: boolean
  rowsToActOn: PhotoRow[]
}

export function ReviewActionBar({
  onSave,
  onExportCsv,
  hasData,
  rowsToActOn,
}: Props) {
  const { suffixMode, customSuffixes, updateSetting } = useSettingsStore()
  const { photoRows, fileMap, renameCompanions, setPhotoRows, setRenameCompanions } = useProcessingStore()
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null)
  const [isRenaming, setIsRenaming] = useState(false)
  const { canSaveToFolder, tier } = getDownloadCapabilities()

  // Build named blobs from fileMap using the 'to' field
  function buildRenamedFiles(): { name: string; blob: Blob }[] {
    const files: { name: string; blob: Blob }[] = []
    for (const row of rowsToActOn) {
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
    } catch (e: unknown) {
      if (getErrorName(e) !== 'AbortError') {
        toast.error(`Save failed: ${getErrorMessage(e)}`)
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
    } catch (e: unknown) {
      if (getErrorName(e) !== 'AbortError') {
        toast.error(`Download failed: ${getErrorMessage(e)}`)
      }
    } finally {
      setDownloadProgress(null)
    }
  }

  // Rename files in-place using File System Access API
  // 'both': write waiting rotations, then rename; 'rename': names only
  const handleRenameFiles = async (mode: 'both' | 'rename' = 'both') => {
    if (!supportsDirectoryPicker()) {
      toast.error('File rename requires Chrome or Edge browser')
      return
    }

    // photos still waiting for review are never renamed
    const rowsToRename = rowsToActOn.filter((r) => r.to?.trim() && r.skip !== 'x' && r.status !== 'Renamed' && !r.review)
    const waiting = rowsToActOn.filter((r) => r.review && r.skip !== 'x').length
    if (waiting) toast.info(`${waiting} photo(s) still need review and will keep their names.`)
    if (rowsToRename.length === 0 && (mode === 'rename' || pendingCount === 0)) {
      toast.error('No files to rename. Calculate names first.')
      return
    }

    try {
      const dirHandle = await getReadWriteDirHandle()

      setIsRenaming(true)
      setDownloadProgress(0)

      // List the folder once: existing names and RAW companions (same stem, any extension case)
      const folderNames: string[] = []
      for await (const [name, entry] of dirHandle.entries()) if (entry.kind === 'file') folderNames.push(name)
      const stems = indexByStem(folderNames)

      // Every rotation still waiting in view is written first (to the current names, also for photos
      // renamed earlier or still in review), then files are renamed
      const inView = new Set(rowsToActOn.map((r) => r.photoId))
      const toTurn = mode === 'both' ? photoRows.filter((r) => inView.has(r.photoId)) : []
      const turned = toTurn.filter((r) => r.skip !== 'x' && pendingRotation(r)).length
      const rotated = await writeRotations(dirHandle, toTurn, folderNames, renameCompanions)
      if (rotated.entries.length) {
        await appendRotationLog(dirHandle, rotated.entries)
        const byId = new Map(rotated.rows.map((r) => [r.photoId, r]))
        setPhotoRows(useProcessingStore.getState().photoRows.map((r) => byId.get(r.photoId) ?? r))
      }
      if (rotated.failed.length) toast.warning(`${rotated.failed.length} file(s) have no orientation tag and were not rotated: ${rotated.failed.slice(0, 3).join(', ')}`)

      const ops: { src: string; dst: string }[] = []
      for (const row of rowsToRename) {
        ops.push({ src: row.currentPath, dst: row.to })
        if (renameCompanions) {
          const newStem = row.to.replace(/\.[^.]+$/, '')
          for (const rawName of companionsOf(row.currentPath, stems, SUPPORTED_RAW_EXTENSIONS)) {
            ops.push({ src: rawName, dst: newStem + rawName.slice(rawName.lastIndexOf('.')) })
          }
        }
      }
      const plan = planRenames(ops, folderNames)
      if (plan.refused.length) {
        const why = { 'target-exists': 'name already used by another file', 'duplicate-target': 'two files would get the same name', 'missing-source': 'file not found' }
        const sample = plan.refused.slice(0, 3).map((r) => `${r.op.src} → ${r.op.dst} (${why[r.reason]})`).join('\n')
        const proceed = window.confirm(`${plan.refused.length} rename(s) will be skipped:\n${sample}${plan.refused.length > 3 ? '\n…' : ''}\n\nRename the other ${ops.length - plan.refused.length}?`)
        if (!proceed) return
      }
      setDownloadProgress(10)

      // Execute: every completed step is logged; the log is saved even if a step fails midway
      const previousLog = await getRenameLog(dirHandle)
      const renameLog: RenameLogEntry[] = []
      let done = 0
      try {
        for (const step of plan.steps) {
          await renameFileInDir(dirHandle, step.from, step.to)
          if (!step.temp) renameLog.push({ original: step.original, renamed: step.to, timestamp: new Date().toISOString() })
          done++
          setDownloadProgress(10 + Math.round((done / plan.steps.length) * 85))
        }
      } catch (err: unknown) {
        toast.error(`Rename stopped after ${renameLog.length} file(s): ${getErrorMessage(err)}. Restore can undo them.`)
      } finally {
        await saveRenameLog([...previousLog, ...renameLog], dirHandle)
      }
      const renamed = renameLog.length

      // Update row statuses
      const renamedSet = new Set(renameLog.map((e) => e.original))
      const updatedRows = useProcessingStore.getState().photoRows.map((r) => {
        if (renamedSet.has(r.currentPath)) {
          const entry = renameLog.find((e) => e.original === r.currentPath)
          return { ...r, status: 'Renamed' as const, currentPath: entry?.renamed ?? r.currentPath }
        }
        return r
      })
      setPhotoRows(updatedRows)

      // Refresh file references so thumbnails use fresh File objects
      await refreshFileMapFromDir(dirHandle, updatedRows)

      if (renamed || turned) toast.success([renamed && `Renamed ${renamed} files`, turned && `rotated ${turned} photo(s)`].filter(Boolean).join(' · '))
      logger.info(`Renamed ${renamed} files, ${rowsToRename.length - renamed} skipped`)
    } catch (e: unknown) {
      if (getErrorName(e) !== 'AbortError') {
        toast.error(`Rename failed: ${getErrorMessage(e)}`)
      }
    } finally {
      setIsRenaming(false)
      setDownloadProgress(null)
    }
  }

  // Write the chosen rotations without renaming
  const pendingCount = rowsToActOn.filter((r) => pendingRotation(r) && r.skip !== 'x').length
  const handleApplyRotations = async () => {
    try {
      const dirHandle = await getReadWriteDirHandle()
      setIsRenaming(true)
      const folderNames: string[] = []
      for await (const [name, entry] of dirHandle.entries()) if (entry.kind === 'file') folderNames.push(name)
      const rotated = await writeRotations(dirHandle, rowsToActOn, folderNames, renameCompanions)
      await appendRotationLog(dirHandle, rotated.entries)
      const byId = new Map(rotated.rows.map((r) => [r.photoId, r]))
      const rows = useProcessingStore.getState().photoRows.map((r) => byId.get(r.photoId) ?? r)
      setPhotoRows(rows)
      await refreshFileMapFromDir(dirHandle, rows)
      toast.success(`Rotated ${rotated.entries.length} file(s)${rotated.failed.length ? ` · ${rotated.failed.length} without an orientation tag skipped` : ''}`)
    } catch (e: unknown) {
      if (getErrorName(e) !== 'AbortError') toast.error(`Rotation failed: ${getErrorMessage(e)}`)
    } finally {
      setIsRenaming(false)
    }
  }

  // Restore original names using rename log
  const handleRestore = async () => {
    const logDir = await getReadWriteDirHandle().catch(() => null)
    const log = await getRenameLog(logDir)
    const rotatedBefore = logDir ? (await getRotationLog(logDir)).some((e) => e.method === 'tag') : false
    if (log.length === 0 && !rotatedBefore) {
      toast.error('Nothing to restore: no renames or rotations logged in this folder.')
      return
    }

    try {
      const dirHandle = await getReadWriteDirHandle()

      setIsRenaming(true)
      setDownloadProgress(0)

      // Undo through the same planner: final name -> original name, never overwriting a file
      const folderNames: string[] = []
      for await (const [name, entry] of dirHandle.entries()) if (entry.kind === 'file') folderNames.push(name)
      const current = new Map<string, string>() // original -> latest name
      for (const e of log) current.set(e.original, e.renamed)
      const plan = planRenames([...current].map(([original, renamed]) => ({ src: renamed, dst: original })), folderNames)
      let restored = 0
      for (const step of plan.steps) {
        try {
          await renameFileInDir(dirHandle, step.from, step.to)
          if (!step.temp) restored++
        } catch (err: unknown) {
          logger.warn(`Could not restore ${step.from}: ${getErrorMessage(err)}`)
        }
        setDownloadProgress(Math.round((restored / Math.max(1, current.size)) * 100))
      }
      if (plan.refused.length) toast.warning(`${plan.refused.length} file(s) could not be restored (name taken or file missing)`)

      // Clear rename log after restore
      if (log.length) await saveRenameLog([], dirHandle)

      // Undo rotations written by this app (latest first); names are mapped back through the rename log
      const rotations = await getRotationLog(dirHandle)
      const originalOf = new Map(log.map((e) => [e.renamed, e.original]))
      let unrotated = 0
      for (const entry of [...rotations].reverse()) {
        if (entry.method !== 'tag' || entry.before === undefined) continue
        for (const name of [entry.original, originalOf.get(entry.original)]) {
          if (!name) continue
          try {
            await writeOrientation(await dirHandle.getFileHandle(name), entry.before)
            unrotated++
            break
          } catch { /* try the name before renaming */ }
        }
      }
      if (rotations.length) await saveRotationLog(rotations.filter((e) => e.method !== 'tag'), dirHandle)

      // Update row statuses back to Original
      const restoredNames = new Set(log.map((e) => e.renamed))
      const updatedRows = photoRows.map((r) => {
        if (restoredNames.has(r.currentPath)) {
          const entry = log.find((e) => e.renamed === r.currentPath)
          return { ...r, status: 'Original' as const, currentPath: entry?.original ?? r.currentPath }
        }
        return r
      }).map((r) => ({ ...r, rotApplied: '' }))
      setPhotoRows(updatedRows)

      // Refresh file references so thumbnails use fresh File objects
      await refreshFileMapFromDir(dirHandle, updatedRows)

      toast.success(`Restored ${restored} file names${unrotated ? ` and ${unrotated} rotations` : ''}`)
    } catch (e: unknown) {
      if (getErrorName(e) !== 'AbortError') {
        toast.error(`Restore failed: ${getErrorMessage(e)}`)
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

        <div className="flex items-center gap-1.5">
          <Checkbox
            id="rename-companions"
            checked={renameCompanions}
            onCheckedChange={(checked) => setRenameCompanions(!!checked)}
          />
          <Label htmlFor="rename-companions" className="text-xs cursor-pointer">
            Rename companion RAW/sidecar files
          </Label>
        </div>

        <div className="mx-1 h-5 w-px bg-border" />



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
            {pendingCount > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="gap-1 text-xs"
                onClick={handleApplyRotations}
                disabled={isRenaming}
                title="Write the chosen rotations to the photos (and their RAW files) without renaming"
              >
                <RotateCw className="h-3.5 w-3.5" />
                Rotate {pendingCount}
              </Button>
            )}

            {pendingCount > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="gap-1 text-xs"
                onClick={() => handleRenameFiles('rename')}
                disabled={!hasData || isRenaming}
                title="Rename only; the chosen rotations stay waiting"
              >
                <FileEdit className="h-3.5 w-3.5" />
                Rename
              </Button>
            )}

            <Button
              size="sm"
              className="gap-1 text-xs"
              onClick={() => handleRenameFiles('both')}
              disabled={!hasData || isRenaming}
              title={pendingCount ? 'Write the chosen rotations (photos and RAW files), then rename' : 'Rename the photos and their RAW files'}
            >
              <FileEdit className="h-3.5 w-3.5" />
              {pendingCount ? 'Rename & rotate' : 'Rename'}
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
