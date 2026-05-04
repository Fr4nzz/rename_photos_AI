import { useRef } from 'react'
import type { InputHTMLAttributes } from 'react'
import { FolderOpen } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { supportsDirectoryPicker } from '@/lib/fileAccess'

interface Props {
  folderName: string
  onSelect: () => void
  onFileInput: (files: FileList) => void
}

export function FolderSelector({ folderName, onSelect, onFileInput }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const directoryInputProps: InputHTMLAttributes<HTMLInputElement> & {
    webkitdirectory: string
    directory: string
  } = {
    webkitdirectory: '',
    directory: '',
  }

  const handleClick = () => {
    if (supportsDirectoryPicker()) {
      onSelect()
    } else {
      inputRef.current?.click()
    }
  }

  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        Input Folder
      </label>
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={handleClick} className="gap-1.5">
          <FolderOpen className="h-3.5 w-3.5" />
          Browse
        </Button>
        <span className="truncate text-sm text-muted-foreground">
          {folderName || '(No folder selected)'}
        </span>
      </div>
      {/* Hidden fallback for non-Chromium browsers */}
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        {...directoryInputProps}
        onChange={(e) => e.target.files && onFileInput(e.target.files)}
      />
    </div>
  )
}
