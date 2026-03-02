import JSZip from 'jszip'
import type { DownloadTier } from '@/types'

export interface NamedBlob {
  name: string
  blob: Blob
}

export function getDownloadCapabilities(): {
  canSaveToFolder: boolean
  canPickSaveLocation: boolean
  tier: DownloadTier
} {
  const canSaveToFolder = 'showDirectoryPicker' in window
  const canPickSaveLocation = 'showSaveFilePicker' in window
  return {
    canSaveToFolder,
    canPickSaveLocation,
    tier: canSaveToFolder
      ? 'directory'
      : canPickSaveLocation
        ? 'zip-picker'
        : 'zip-fallback',
  }
}

/**
 * Tier 1: Write files directly to a user-chosen folder.
 */
export async function saveToFolder(
  files: NamedBlob[],
  onProgress?: (current: number, total: number) => void
): Promise<number> {
  const dirHandle = await (window as any).showDirectoryPicker({
    id: 'photo-output',
    mode: 'readwrite',
    startIn: 'pictures',
  })

  for (let i = 0; i < files.length; i++) {
    const { name, blob } = files[i]
    const fileHandle = await dirHandle.getFileHandle(name, { create: true })
    const writable = await fileHandle.createWritable()
    await writable.write(blob)
    await writable.close()
    onProgress?.(i + 1, files.length)
  }

  return files.length
}

/**
 * Tier 2/3: Download as ZIP using STORE compression (no recompression for images).
 */
export async function downloadAsZip(
  files: NamedBlob[],
  zipName = 'renamed-photos.zip',
  onProgress?: (percent: number) => void
): Promise<void> {
  const zip = new JSZip()
  for (const { name, blob } of files) {
    zip.file(name, blob)
  }

  const zipBlob = await zip.generateAsync(
    { type: 'blob', compression: 'STORE' },
    (metadata) => onProgress?.(Math.round(metadata.percent))
  )

  await saveBlob(zipBlob, zipName)
}

/**
 * Save a blob using File System Access API when available, fallback to <a download>.
 */
async function saveBlob(blob: Blob, suggestedName: string): Promise<void> {
  if ('showSaveFilePicker' in window) {
    try {
      const handle = await (window as any).showSaveFilePicker({
        suggestedName,
        types: [
          {
            description: 'ZIP Archive',
            accept: { 'application/zip': ['.zip'] },
          },
        ],
        startIn: 'downloads',
      })
      const writable = await handle.createWritable()
      await writable.write(blob)
      await writable.close()
      return
    } catch (err: any) {
      if (err.name === 'AbortError') return
    }
  }

  // Fallback: <a download>
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = suggestedName
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  setTimeout(() => {
    URL.revokeObjectURL(url)
    a.remove()
  }, 1000)
}

/**
 * Get browser-specific instructions for enabling "Ask where to save".
 */
export function getBrowserSaveAsInstructions(): {
  browser: string
  path: string
  steps: string
} {
  const ua = navigator.userAgent.toLowerCase()
  if (ua.includes('firefox')) {
    return {
      browser: 'Firefox',
      path: 'Settings > General > Files and Applications',
      steps: 'Select "Always ask you where to save files"',
    }
  }
  if (ua.includes('edg/')) {
    return {
      browser: 'Edge',
      path: 'Settings > Downloads',
      steps: 'Toggle ON "Ask me what to do with each download"',
    }
  }
  return {
    browser: 'Chrome',
    path: 'Settings > Downloads',
    steps: 'Toggle ON "Ask where to save each file before downloading"',
  }
}
