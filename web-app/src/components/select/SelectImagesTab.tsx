import { useMemo, useState } from 'react'
import { toast } from 'sonner'
import { FolderOpen, RotateCcw, Undo2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { useProcessingStore } from '@/stores/processingStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { BROWSER_ROTATABLE_EXTENSIONS } from '@/lib/constants'
import { getImageFilesFromHandle, openDirectoryPicker, supportsDirectoryPicker } from '@/lib/fileAccess'
import { clearPreviewCache, rotateBrowserImageFile } from '@/lib/imageProcessing'
import { getRotationLog, saveDirHandle, saveRotationLog } from '@/lib/csvHandler'
import { getErrorMessage, getErrorName } from '@/lib/errors'
import {
  invertImageSelection,
  matchImageNames,
  selectAllImageNames,
  sortImageFiles,
  type ImageSortOption,
} from '@/lib/selection'
import { ImageSelectionGrid } from './ImageSelectionGrid'
import { ImageSelectionToolbar } from './ImageSelectionToolbar'
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

export function SelectImagesTab() {
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
  const { rotationAngle, useExif, previewRaw, updateSetting } = useSettingsStore()
  const [query, setQuery] = useState('')
  const [extensionFilter, setExtensionFilter] = useState('all')
  const [sortOption, setSortOption] = useState<ImageSortOption>('name-asc')
  const [busy, setBusy] = useState(false)

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
      toast.error('Folder selection requires Chrome, Edge, or another File System Access browser.')
      return
    }

    try {
      const handle = await openDirectoryPicker()
      setDirHandle(handle)
      await saveDirHandle(handle)
      clearPreviewCache()
      const files = await getImageFilesFromHandle(handle, previewRaw ? 'all' : 'compressed')
      setImageFiles(files, true)
      setFileMap(new Map(files.map((entry) => [entry.name, entry.file])))
      const log = await getRotationLog()
      setRotationLog(log)
      toast.success(`Loaded ${files.length} image file(s).`)
    } catch (error: unknown) {
      if (getErrorName(error) !== 'AbortError') {
        toast.error(`Could not open folder: ${getErrorMessage(error)}`)
      }
    }
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

  async function rotateSelectedImages() {
    if (!dirHandle) {
      toast.error('Open a folder first.')
      return
    }
    if (rotationAngle === 0) {
      toast.info('Rotation is set to 0°.')
      return
    }

    const selected = imageFiles.filter((entry) => selectedImageNames.has(entry.name))
    const rotatable = selected.filter((entry) => BROWSER_ROTATABLE_EXTENSIONS.has(entry.extension))
    const skipped = selected.length - rotatable.length
    if (rotatable.length === 0) {
      toast.error('No selected JPEG or PNG files can be rotated in the browser.')
      return
    }

    const confirmed = window.confirm(
      `Rotate ${rotatable.length} selected JPEG/PNG file(s) in place? ` +
      'Original copies will be saved in rotation_backups so this can be undone.'
    )
    if (!confirmed) return

    setBusy(true)
    const backupDir = await getBackupDir(dirHandle)
    const entries: RotationLogEntry[] = []
    let rotated = 0

    for (const entry of rotatable) {
      try {
        const fileHandle = await dirHandle.getFileHandle(entry.name)
        const file = await fileHandle.getFile()
        const stamp = new Date().toISOString().replace(/[:.]/g, '-')
        const backupName = `${stamp}__${entry.name}`
        await copyFileIntoDir(backupDir, backupName, file)

        const rotatedBlob = await rotateBrowserImageFile(file, rotationAngle, useExif)
        if (!rotatedBlob) continue
        const writable = await fileHandle.createWritable()
        await writable.write(rotatedBlob)
        await writable.close()
        entries.push({ original: entry.name, backup: backupName, angle: rotationAngle, timestamp: new Date().toISOString() })
        rotated++
      } catch (error: unknown) {
        console.warn(`Could not rotate ${entry.name}: ${getErrorMessage(error)}`)
      }
    }

    const nextLog = [...rotationLog, ...entries]
    setRotationLog(nextLog)
    await saveRotationLog(nextLog)
    clearPreviewCache()
    await refreshFiles()
    setBusy(false)

    toast.success(`Rotated ${rotated} file(s).${skipped ? ` Skipped ${skipped} non-JPEG/PNG file(s).` : ''}`)
  }

  async function undoRotations() {
    if (!dirHandle || rotationLog.length === 0) {
      toast.info('No browser rotation log to undo.')
      return
    }

    const confirmed = window.confirm(`Restore ${rotationLog.length} rotated file(s) from rotation_backups?`)
    if (!confirmed) return

    setBusy(true)
    const backupDir = await getBackupDir(dirHandle)
    let restored = 0

    for (const entry of [...rotationLog].reverse()) {
      try {
        const backupHandle = await backupDir.getFileHandle(entry.backup)
        const backupFile = await backupHandle.getFile()
        await copyFileIntoDir(dirHandle, entry.original, backupFile)
        restored++
      } catch (error: unknown) {
        console.warn(`Could not restore ${entry.original}: ${getErrorMessage(error)}`)
      }
    }

    setRotationLog([])
    await saveRotationLog([])
    clearPreviewCache()
    await refreshFiles()
    setBusy(false)
    toast.success(`Restored ${restored} file(s).`)
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-3 border-b p-3">
        <Button variant="outline" size="sm" onClick={openFolder} className="gap-1.5">
          <FolderOpen className="h-3.5 w-3.5" />
          Open Input Folder
        </Button>
        <div className="text-sm text-muted-foreground">
          {dirHandle ? dirHandle.name : 'No folder selected'}
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          <Checkbox
            id="select-raw"
            checked={previewRaw}
            onCheckedChange={(checked) => setIncludeRaw(!!checked)}
          />
          <Label htmlFor="select-raw" className="text-xs">Include RAW images</Label>
        </div>
        <Button size="sm" onClick={rotateSelectedImages} disabled={busy || selectedImageNames.size === 0} className="gap-1.5">
          <RotateCcw className="h-3.5 w-3.5" />
          Rotate Selected
        </Button>
        <Button variant="outline" size="sm" onClick={undoRotations} disabled={busy || rotationLog.length === 0} className="gap-1.5">
          <Undo2 className="h-3.5 w-3.5" />
          Undo Rotation
        </Button>
      </div>

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
      />

      <div className="min-h-0 flex-1 overflow-auto">
        <ImageSelectionGrid
          files={visibleFiles}
          selectedNames={selectedImageNames}
          onToggle={toggleSelectedImage}
        />
      </div>
    </div>
  )
}
