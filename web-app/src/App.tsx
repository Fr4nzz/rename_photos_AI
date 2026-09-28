import { TooltipProvider } from '@/components/ui/tooltip'
import { Toaster } from '@/components/ui/sonner'
import { ThemeProvider } from '@/components/ThemeProvider'
import { Header } from '@/components/layout/Header'
import { PhotosView } from '@/components/photos/PhotosView'

function App() {
  return (
    <ThemeProvider>
      <TooltipProvider>
        <div className="flex h-screen flex-col bg-background text-foreground">
          <Header />
          <main className="min-h-0 flex-1 overflow-hidden">
            <PhotosView />
          </main>
          <Toaster />
        </div>
      </TooltipProvider>
    </ThemeProvider>
  )
}

export default App
