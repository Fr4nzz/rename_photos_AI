import { TooltipProvider } from '@/components/ui/tooltip'
import { Toaster } from '@/components/ui/sonner'
import { toast } from 'sonner'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ThemeProvider } from '@/components/ThemeProvider'
import { Header } from '@/components/layout/Header'
import { ProcessTab } from '@/components/process/ProcessTab'
import { ReviewTab } from '@/components/review/ReviewTab'
import { ApiKeysTab } from '@/components/api-keys/ApiKeysTab'
import { useProcessingStore } from '@/stores/processingStore'
import { Camera, ClipboardCheck, Key } from 'lucide-react'

function App() {
  const handleTabChange = (value: string) => {
    if (value === 'review' && useProcessingStore.getState().isProcessing) {
      toast.warning('AI is still processing — results may be incomplete')
    }
  }

  return (
    <ThemeProvider>
      <TooltipProvider>
        <div className="flex h-screen flex-col bg-background text-foreground">
          <Header />

          <Tabs defaultValue="process" className="flex min-h-0 flex-1 flex-col" onValueChange={handleTabChange}>
            <div className="border-b px-4">
              <TabsList className="h-10">
                <TabsTrigger value="process" className="gap-1.5">
                  <Camera className="h-3.5 w-3.5" />
                  Process Images
                </TabsTrigger>
                <TabsTrigger value="review" className="gap-1.5">
                  <ClipboardCheck className="h-3.5 w-3.5" />
                  Review Results
                </TabsTrigger>
                <TabsTrigger value="api-keys" className="gap-1.5">
                  <Key className="h-3.5 w-3.5" />
                  API Keys
                </TabsTrigger>
              </TabsList>
            </div>

            <TabsContent forceMount value="process" className="mt-0 flex-1 overflow-hidden data-[state=inactive]:hidden">
              <ProcessTab />
            </TabsContent>
            <TabsContent forceMount value="review" className="mt-0 flex-1 overflow-hidden data-[state=inactive]:hidden">
              <ReviewTab />
            </TabsContent>
            <TabsContent forceMount value="api-keys" className="mt-0 flex-1 overflow-auto p-4 data-[state=inactive]:hidden">
              <ApiKeysTab />
            </TabsContent>
          </Tabs>

          <Toaster />
        </div>
      </TooltipProvider>
    </ThemeProvider>
  )
}

export default App
