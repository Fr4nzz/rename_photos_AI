import { useEffect, useState } from 'react'
import { ThemeContext, type Theme } from '@/components/ThemeContext'
import { isEmbedded } from '@/lib/embed/host'

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // inside another app's page (light, with no theme switch of its own here) the app is light too
  const embedded = isEmbedded()
  const [theme, setTheme] = useState<Theme>(
    () => (embedded ? 'light' : (localStorage.getItem('theme') as Theme) || 'dark')
  )

  const resolved =
    theme === 'system'
      ? window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light'
      : theme

  useEffect(() => {
    const root = document.documentElement
    root.classList.remove('light', 'dark')
    root.classList.add(resolved)
    if (!embedded) localStorage.setItem('theme', theme)
  }, [theme, resolved, embedded])

  return (
    <ThemeContext.Provider value={{ theme, setTheme, resolved }}>
      {children}
    </ThemeContext.Provider>
  )
}
