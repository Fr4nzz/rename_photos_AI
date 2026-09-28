import { useEffect, useMemo, useRef, useState, type InputHTMLAttributes } from 'react'
import { toast } from 'sonner'
import { ClipboardCheck, FolderOpen, LayoutGrid, Lock, RefreshCw, RotateCcw, RotateCw, Undo2 } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { useProcessingStore } from '@/stores/processingStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { BROWSER_ROTATABLE_EXTENSIONS, SUPPORTED_RAW_EXTENSIONS } from '@/lib/constants'
import { companionsOf, indexByStem } from '@/lib/renamePlan'
import { listFolder, syncCompanions } from '@/lib/rotationWrite'
import {
  getImageFilesFromHandle,
  folderAccessHelp,
  getImageFilesFromInput,
  openDirectoryPicker,
  supportsDirectoryPicker,
} from '@/lib/fileAccess'
import {
  clearPreviewCache,
  rotateBrowserImageFile,
} from '@/lib/imageProcessing'
import { getRotationLog, saveDirHandle, saveRotationLog } from '@/lib/csvHandler'
import { rotateLossless, writeOrientation } from '@/lib/orientation'
import { getErrorMessage, getErrorName } from '@/lib/errors'
import {
  invertImageSelection,
  matchImageNames,
  selectAllImageNames,
  sortImageFiles,
  type ImageSortOption,
} from '@/lib/selection'
import { ImageSelectionGrid } from '@/components/select/ImageSelectionGrid'
import { ImageSelectionToolbar } from '@/components/select/ImageSelectionToolbar'
import { ProcessingControls } from '@/components/process/ProcessingControls'
import { ReviewPane } from '@/components/review/ReviewPane'
import { FolderAccessBanner } from './FolderAccessBanner'
import { useProcessTab } from '@/hooks/useProcessTab'
import type { RotationLogEntry } from '@/types'

const ROTATION_BACKUP_DIR = 'rotation_backups'

async function getBackupDir(dirHandle: FileSystemDirectoryHandle): Promise<FileSystemDirectoryHandle> {
  return dirHandle.getDirectoryHandle(ROTATION_BACKUP_DIR, { create: true })
}

async function copyFileIntoDir(
  dirHandle: FileSystemDirectoryHandle,
  fileName: string,
  file: File | Blob
) {
  const handle = await dirHandle.getFileHandle(fileName, { create: true })
  const writable = await handle.createWritable()
  await writable.write(file)
  await writable.close()
}

function isPlaywrightPickerIntercept(error: unknown): boolean {
  return getErrorName(error) === 'AbortError'
    && getErrorMessage(error).includes('Intercepted by Page.setInterceptFileChooserDialog')
}

export function PhotosView() {
  const hook = useProcessTab()
  const photoRows = useProcessingStore((s) => s.photoRows)
  const isProcessing = useProcessingStore((s) => s.isProcessing)
  const [view, setView] = useState<'photos' | 'review'>('photos')
  const [focus, setFocus] = useState<{ photoId: number } | null>(null)
  const rowsByName = useMemo(() => new Map(photoRows.map((r) => [r.from, r])), [photoRows])
  const reviewCount = photoRows.filter((r) => r.review).length

  // a finished read opens the review
  const wasProcessing = useRef(false)
  useEffect(() => {
    if (wasProcessing.current && !isProcessing && photoRows.length) setView('review')
    wasProcessing.current = isProcessing
  }, [isProcessing, photoRows.length])

  const {
    imageFiles,
    selectedImageNames,
    dirHandle,
    setDirHandle,
    setImageFiles,
    setFileMap,
    setSelectedImageNames,
    toggleSelectedImage,
    rotationLog,
    setRotationLog,
  } = useProcessingStore()
  const { useExif, previewRaw, updateSetting } = useSettingsStore()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState('')
  const [inputFolderName, setInputFolderName] = useState('')
  const [extensionFilter, setExtensionFilter] = useState('all')
  const [sortOption, setSortOption] = useState<ImageSortOption>('name-asc')
  const [busy, setBusy] = useState(false)
  const directoryInputProps = {
    webkitdirectory: '',
  } as InputHTMLAttributes<HTMLInputElement>

  const visibleFiles = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    const typeFiltered = extensionFilter === 'all'
      ? imageFiles
      : imageFiles.filter((entry) => entry.extension === extensionFilter)
    const filtered = normalizedQuery
      ? typeFiltered.filter((entry) => entry.name.toLowerCase().includes(normalizedQuery))
      : typeFiltered
    return sortImageFiles(filtered, sortOption)
  }, [extensionFilter, imageFiles, query, sortOption])

  const extensions = useMemo(
    () => Array.from(new Set(imageFiles.map((entry) => entry.extension))).sort(),
    [imageFiles]
  )
  async function refreshFiles(handle = dirHandle) {
    if (!handle) return
    const files = await getImageFilesFromHandle(handle, previewRaw ? 'all' : 'compressed')
    setImageFiles(files, false)
    setSelectedImageNames(new Set(files.filter((entry) => selectedImageNames.has(entry.name)).map((entry) => entry.name)))
    setFileMap(new Map(files.map((entry) => [entry.name, entry.file])))
  }

  async function openFolder() {
    if (!supportsDirectoryPicker()) {
      toast.info('Your browser will ask to "upload" the folder: the photos stay on this computer, nothing is sent anywhere.', { duration: 8000 })
      fileInputRef.current?.click()
      return
    }

    try {
      const handle = await openDirectoryPicker()
      setDirHandle(handle)
      setInputFolderName('')
      await saveDirHandle(handle)
      clearPreviewCache()
      const files = await getImageFilesFromHandle(handle, previewRaw ? 'all' : 'compressed')
      setImageFiles(files, true)
      setFileMap(new Map(files.map((entry) => [entry.name, entry.file])))
      const log = await getRotationLog(handle)
      setRotationLog(log)
      toast.success(`Loaded ${files.length} image file(s).`)
    } catch (error: unknown) {
      if (isPlaywrightPickerIntercept(error)) {
        fileInputRef.current?.click()
      } else if (getErrorName(error) !== 'AbortError') {
        toast.error(`Could not open folder: ${getErrorMessage(error)}`)
      }
    }
  }

  function loadFolderFromInput(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return

    setDirHandle(null)
    clearPreviewCache()
    const files = getImageFilesFromInput(fileList, previewRaw ? 'all' : 'compressed')
    setImageFiles(files, true)
    setFileMap(new Map(files.map((entry) => [entry.name, entry.file])))
    setRotationLog([])
    const firstPath = files[0]?.path ?? ''
    setInputFolderName(firstPath.includes('/') ? firstPath.split('/')[0] : 'Selected files')
    toast.success(`Loaded ${files.length} image file(s).`)
  }

  async function setIncludeRaw(checked: boolean) {
    updateSetting('previewRaw', checked)
    if (!dirHandle) return
    clearPreviewCache()
    const files = await getImageFilesFromHandle(dirHandle, checked ? 'all' : 'compressed')
    setImageFiles(files, true)
    setFileMap(new Map(files.map((entry) => [entry.name, entry.file])))
  }

  function replaceMatches() {
    const matches = matchImageNames(imageFiles, query)
    setSelectedImageNames(matches)
    toast.info(`Selected ${matches.size} matching image(s).`)
  }

  function addMatches() {
    const matches = matchImageNames(imageFiles, query)
    setSelectedImageNames(new Set([...selectedImageNames, ...matches]))
    toast.info(`Added ${matches.size} matching image(s).`)
  }

  async function rotateSelectedImages(rotationAngle: number) {
    if (!dirHandle) {
      toast.error('Open a folder first.')
      return
    }

    const selected = imageFiles.filter((entry) => selectedImageNames.has(entry.name))
    if (selected.length === 0) return

    setBusy(true)
    const stems = indexByStem(await listFolder(dirHandle))
    const entries: RotationLogEntry[] = []
    let lossless = 0
    let reencoded = 0
    const failed: string[] = []
    let backupDir: FileSystemDirectoryHandle | null = null

    for (const entry of selected) {
      try {
        const fileHandle = await dirHandle.getFileHandle(entry.name)
        // Lossless first: change only the EXIF Orientation tag (JPEG and camera RAW files).
        // The UI angle is clockwise; orientation angles are counter-clockwise.
        const tag = await rotateLossless(fileHandle, -rotationAngle)
        if (tag) {
          entries.push({ original: entry.name, method: 'tag', before: tag.before, after: tag.after,
                         angle: rotationAngle, timestamp: new Date().toISOString() })
          lossless++
          // its RAW files (hidden unless RAW is shown) take the same orientation
          if (!SUPPORTED_RAW_EXTENSIONS.has(entry.extension)) {
            const others = companionsOf(entry.name, stems, SUPPORTED_RAW_EXTENSIONS).filter((n) => !selectedImageNames.has(n))
            const synced = await syncCompanions(dirHandle, entry.name, others, rotationAngle)
            entries.push(...synced.entries)
          }
          continue
        }
        // Files without an Orientation tag (PNG, some JPEGs): rewrite the pixels, keep a backup.
        if (!BROWSER_ROTATABLE_EXTENSIONS.has(entry.extension)) {
          failed.push(entry.name)
          continue
        }
        const file = await fileHandle.getFile()
        backupDir ??= await getBackupDir(dirHandle)
        const backupName = `${new Date().toISOString().replace(/[:.]/g, '-')}__${entry.name}`
        await copyFileIntoDir(backupDir, backupName, file)
        const rotatedBlob = await rotateBrowserImageFile(file, rotationAngle, useExif)
        if (!rotatedBlob) continue
        const writable = await fileHandle.createWritable()
        await writable.write(rotatedBlob)
        await writable.close()
        entries.push({ original: entry.name, method: 're-encode', backup: backupName, angle: rotationAngle,
                       timestamp: new Date().toISOString() })
        reencoded++
      } catch (error: unknown) {
        failed.push(entry.name)
        console.warn(`Could not rotate ${entry.name}: ${getErrorMessage(error)}`)
      }
    }

    const nextLog = [...rotationLog, ...entries]
    setRotationLog(nextLog)
    await saveRotationLog(nextLog, dirHandle)
    clearPreviewCache()
    await refreshFiles()
    setBusy(false)

    const parts = [`Rotated ${lossless + reencoded} file(s)`]
    if (reencoded) parts.push(`${reencoded} re-saved (no orientation tag; originals in ${ROTATION_BACKUP_DIR})`)
    if (failed.length) parts.push(`${failed.length} not rotatable: ${failed.slice(0, 3).join(', ')}${failed.length > 3 ? '…' : ''}`)
    ;(failed.length ? toast.warning : toast.success)(parts.join(' · '))
  }

  async function undoRotations() {
    if (!dirHandle || rotationLog.length === 0) {
      toast.info('No browser rotation log to undo.')
      return
    }

    setBusy(true)
    let restored = 0

    for (const entry of [...rotationLog].reverse()) {
      try {
        if (entry.method === 'tag' && entry.before !== undefined) {
          await writeOrientation(await dirHandle.getFileHandle(entry.original), entry.before)
        } else if (entry.backup) {
          const backupDir = await getBackupDir(dirHandle)
          const backupFile = await (await backupDir.getFileHandle(entry.backup)).getFile()
          await copyFileIntoDir(dirHandle, entry.original, backupFile)
        }
        restored++
      } catch (error: unknown) {
        console.warn(`Could not restore ${entry.original}: ${getErrorMessage(error)}`)
      }
    }

    setRotationLog([])
    await saveRotationLog([], dirHandle)
    clearPreviewCache()
    await refreshFiles()
    setBusy(false)
    toast.success(`Restored ${restored} file(s).`)
  }

  const rotateButtons = (
    <div className="flex items-center gap-1">
      {([[-90, RotateCcw, 'Rotate the selected photos 90° counter-clockwise'], [90, RotateCw, 'Rotate the selected photos 90° clockwise'], [180, RefreshCw, 'Rotate the selected photos 180°']] as const).map(([angle, Icon, tip]) => (
        <Tooltip key={angle}>
          <TooltipTrigger asChild>
            <Button variant="outline" size="icon" className="h-8 w-8" aria-label={tip} disabled={busy || !dirHandle || selectedImageNames.size === 0}
              onClick={() => rotateSelectedImages((angle + 360) % 360)}>
              <Icon className="h-3.5 w-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{tip} (lossless, RAW files too)</TooltipContent>
        </Tooltip>
      ))}
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="outline" size="icon" className="h-8 w-8" aria-label="Undo rotations" disabled={busy || rotationLog.length === 0} onClick={undoRotations}>
            <Undo2 className="h-3.5 w-3.5" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Undo every rotation written in this folder</TooltipContent>
      </Tooltip>
    </div>
  )

  return (
    <div className="flex h-full min-h-0 flex-col">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        {...directoryInputProps}
        onChange={(event) => loadFolderFromInput(event.target.files)}
      />

      <FolderAccessBanner />
      <ProcessingControls onStart={hook.startProcessing} onStop={hook.stopProcessing} hasImages={hook.imageFiles.length > 0} geminiHook={hook}>
        <Button variant="outline" size="sm" onClick={openFolder} className="gap-1.5">
          <FolderOpen className="h-3.5 w-3.5" />
          Open folder
        </Button>
        <div className="max-w-56 truncate text-sm text-muted-foreground">
          {dirHandle ? dirHandle.name : inputFolderName || 'No folder selected'}
        </div>
        {!dirHandle && inputFolderName && !supportsDirectoryPicker() && (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="flex items-center gap-1 rounded bg-amber-500/15 px-1.5 py-0.5 text-[11px] text-amber-700 dark:text-amber-400">
                <Lock className="h-3 w-3" />
                Read-only
              </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">{folderAccessHelp()}</TooltipContent>
          </Tooltip>
        )}
        <div className="flex items-center gap-1.5">
          <Checkbox id="select-raw" checked={previewRaw} onCheckedChange={(checked) => setIncludeRaw(!!checked)} />
          <Label htmlFor="select-raw" className="text-xs">RAW</Label>
        </div>
        {photoRows.length > 0 && (
          <div className="ml-2 flex rounded-md border p-0.5" role="radiogroup" aria-label="View">
            {([['photos', LayoutGrid, 'Photos: select and rotate', null], ['review', ClipboardCheck, 'Review CAMIDs and rename', reviewCount]] as const).map(([value, Icon, tip, count]) => (
              <Tooltip key={value}>
                <TooltipTrigger asChild>
                  <Button variant={view === value ? 'secondary' : 'ghost'} size="sm" role="radio" aria-checked={view === value} aria-label={tip}
                    onClick={() => setView(value)} className="h-7 gap-1 px-2 text-xs">
                    <Icon className="h-4 w-4" />
                    {count ? <span className="text-amber-600 dark:text-amber-400">{count}</span> : null}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{tip}</TooltipContent>
              </Tooltip>
            ))}
          </div>
        )}
      </ProcessingControls>

      {view === 'review' && photoRows.length > 0 ? (
        <ReviewPane focus={focus} />
      ) : (<>
        <ImageSelectionToolbar
          query={query}
          onQueryChange={setQuery}
          extensionFilter={extensionFilter}
          onExtensionFilterChange={setExtensionFilter}
          extensions={extensions}
          sortOption={sortOption}
          onSortOptionChange={setSortOption}
          selectedCount={selectedImageNames.size}
          totalCount={imageFiles.length}
          onSelectAll={() => setSelectedImageNames(selectAllImageNames(imageFiles))}
          onClear={() => setSelectedImageNames(new Set())}
          onInvert={() => setSelectedImageNames(invertImageSelection(imageFiles, selectedImageNames))}
          onReplaceMatches={replaceMatches}
          onAddMatches={addMatches}
          trailing={rotateButtons}
        />
        <div className="min-h-0 flex-1 overflow-auto">
          <ImageSelectionGrid
            files={visibleFiles}
            selectedNames={selectedImageNames}
            onToggle={toggleSelectedImage}
            rows={rowsByName}
            onOpen={(photoId) => { setFocus({ photoId }); setView('review') }}
          />
        </div>
      </>)}
    </div>
  )
}
