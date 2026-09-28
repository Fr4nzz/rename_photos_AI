import { Info, Sun, Moon } from 'lucide-react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/components/ThemeProviderHook'
import { GITHUB_REPO_URL } from '@/lib/constants'

export function Header() {
  const { resolved, setTheme } = useTheme()
  const toggleTheme = () => setTheme(resolved === 'dark' ? 'light' : 'dark')

  return (
    <header className="flex items-center justify-between border-b px-4 py-2">
      <a
        href={GITHUB_REPO_URL}
        className="text-lg font-semibold tracking-tight transition-colors hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
      >
        AI Photo Processor
      </a>
      <div className="flex items-center gap-1">
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="ghost" size="icon" aria-label="About the CAMID reader">
              <Info className="h-4 w-4" />
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>About the CAMID reader</DialogTitle>
            </DialogHeader>
            <div className="space-y-3 text-sm text-muted-foreground">
              <p>
                Photos are read on this computer and checked against the specimen database. Nothing is
                uploaded. Uncertain photos are flagged for review with suggestions.
              </p>
              <p className="text-xs">
                Envelope segmentation: a fine-tuned{' '}
                <a className="underline" href="https://github.com/ultralytics/ultralytics">Ultralytics YOLO</a> model (AGPL-3.0).
                Text lines and CAMIDs: fine-tuned{' '}
                <a className="underline" href="https://github.com/PaddlePaddle/PaddleOCR">PaddleOCR</a> PP-OCRv5 models (Apache-2.0).
                Runs with <a className="underline" href="https://onnxruntime.ai">ONNX Runtime Web</a> (MIT);
                HEIC photos are decoded with <a className="underline" href="https://github.com/strukturag/libheif">libheif</a> (LGPL-3.0).
              </p>
            </div>
          </DialogContent>
        </Dialog>
        <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label="Toggle theme">
          {resolved === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </div>
    </header>
  )
}
