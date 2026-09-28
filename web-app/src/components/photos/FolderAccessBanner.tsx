import { useState } from 'react'
import { toast } from 'sonner'
import { Copy, TriangleAlert, X } from 'lucide-react'
import { supportsDirectoryPicker } from '@/lib/fileAccess'

const FLAG = 'brave://flags/#file-system-access-api'

/**
 * Shown when the browser cannot write to folders. Brave hides the folder API behind a flag; web
 * pages may not open brave:// addresses, so the banner offers the address to copy instead.
 */
export function FolderAccessBanner() {
  const [hidden, setHidden] = useState(() => sessionStorage.getItem('folder-banner-hidden') === '1')
  if (hidden || supportsDirectoryPicker()) return null
  const brave = 'brave' in navigator
  const copy = async () => {
    await navigator.clipboard.writeText(FLAG)
    toast.success('Copied. Paste it into the address bar of a new tab.')
  }
  return (
    <div className="flex flex-wrap items-center gap-2 border-b bg-amber-500/10 px-3 py-1.5 text-xs text-amber-800 dark:text-amber-300">
      <TriangleAlert className="h-3.5 w-3.5 flex-shrink-0" />
      {brave ? (<>
        <span>Brave can only read folders, so renaming and rotating are off. To turn them on, click</span>
        <button type="button" onClick={copy} title="Copy (Brave does not let websites open its settings pages)"
          className="inline-flex items-center gap-1 rounded bg-background/80 px-1.5 py-0.5 font-mono text-foreground underline decoration-dotted hover:bg-background">
          {FLAG}
          <Copy className="h-3 w-3" />
        </button>
        <span>to copy it, paste it into a new tab, set it to <b>Enabled</b> and click <b>Relaunch</b>.</span>
      </>) : (
        <span>This browser can only read folders, so renaming and rotating are off. Use Chrome or Edge to change files in place.</span>
      )}
      <button type="button" aria-label="Hide" className="ml-auto rounded p-0.5 hover:bg-amber-500/20"
        onClick={() => { sessionStorage.setItem('folder-banner-hidden', '1'); setHidden(true) }}>
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}
