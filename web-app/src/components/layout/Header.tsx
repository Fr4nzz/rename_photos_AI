import { Sun, Moon } from 'lucide-react'
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
      <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label="Toggle theme">
        {resolved === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      </Button>
    </header>
  )
}
