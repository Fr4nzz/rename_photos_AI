import { useEffect, useRef } from 'react'
import { Download, Sun, Moon, Wifi, WifiOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useTheme } from '@/components/ThemeProviderHook'
import { useBackendStore } from '@/stores/backendStore'
import { BACKEND_DOWNLOAD_URL_WINDOWS, BACKEND_RELEASES_URL } from '@/lib/constants'

function isWindowsClient() {
  const nav = window.navigator as Navigator & {
    userAgentData?: { platform?: string }
  }
  const platform = nav.userAgentData?.platform || nav.platform || nav.userAgent
  return /win/i.test(platform)
}

export function Header() {
  const { resolved, setTheme } = useTheme()
  const status = useBackendStore((s) => s.status)
  const isWindows = isWindowsClient()
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  // Single check on mount, then poll with backoff; stop after 3 failures to avoid console noise
  useEffect(() => {
    const schedule = () => {
      const { checkHealth } = useBackendStore.getState()
      checkHealth().then(() => {
        const { failCount, status: s } = useBackendStore.getState()
        if (s.connected) {
          timerRef.current = setTimeout(schedule, 60_000)
        } else if (failCount < 3) {
          timerRef.current = setTimeout(schedule, 10_000)
        }
        // After 3 failures, stop polling to avoid ERR_CONNECTION_REFUSED noise
      })
    }
    schedule()
    return () => clearTimeout(timerRef.current)
  }, [])

  const toggleTheme = () => setTheme(resolved === 'dark' ? 'light' : 'dark')

  return (
    <header className="flex items-center justify-between border-b px-4 py-2">
      <h1 className="text-lg font-semibold tracking-tight">
        AI Photo Processor
      </h1>

      <div className="flex items-center gap-2">
        {!status.connected && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="sm" asChild>
                <a
                  href={isWindows ? BACKEND_DOWNLOAD_URL_WINDOWS : BACKEND_RELEASES_URL}
                  download={isWindows ? 'AIPhotoProcessor-Backend.exe' : undefined}
                >
                  <Download className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">
                    {isWindows ? 'Download RAW Rotation Backend' : 'RAW Rotation Backend for Windows'}
                  </span>
                </a>
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {isWindows
                ? 'Only needed if you want to rotate RAW files such as CR2 or ORF'
                : 'RAW rotation backend is currently available for Windows only'}
            </TooltipContent>
          </Tooltip>
        )}

        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs">
              {status.connected ? (
                <>
                  <Wifi className="h-3.5 w-3.5 text-green-500" />
                  <span className="text-muted-foreground">Backend</span>
                </>
              ) : (
                <>
                  <WifiOff className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="text-muted-foreground">Offline</span>
                </>
              )}
            </div>
          </TooltipTrigger>
          <TooltipContent>
            {status.connected
              ? 'Local backend connected — RAW rotation & in-place rename available'
              : 'Local backend not detected — RAW rotation needs the Windows backend exe'}
          </TooltipContent>
        </Tooltip>

        <Button variant="ghost" size="icon" onClick={toggleTheme}>
          {resolved === 'dark' ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </Button>
      </div>
    </header>
  )
}
