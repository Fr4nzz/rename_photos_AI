import { useState, useEffect } from 'react'
import { toast } from 'sonner'
import { useApiKeysStore } from '@/stores/apiKeysStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { fetchAvailableModels } from '@/lib/aiClient'
import { getErrorMessage } from '@/lib/errors'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Save, ExternalLink, Loader2 } from 'lucide-react'
import { logger } from '@/lib/logger'

export function ApiKeysTab() {
  const { apiKeys, availableModels, setApiKeys, setAvailableModels } = useApiKeysStore()
  const { modelName, updateSetting } = useSettingsStore()
  const [keysText, setKeysText] = useState(() => apiKeys.join('\n'))
  const [loading, setLoading] = useState(false)

  // Sync textarea when store changes externally
  useEffect(() => {
    setKeysText(apiKeys.join('\n'))
  }, [apiKeys])

  // Auto-refresh model list on mount when keys already exist
  useEffect(() => {
    if (apiKeys.length === 0) return
    let cancelled = false
    fetchAvailableModels(apiKeys[0]).then((models) => {
      if (!cancelled && models.length > 0) {
        setAvailableModels(models)
      }
    })
    return () => { cancelled = true }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSave = async () => {
    const keys = keysText
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)

    setApiKeys(keys)

    if (keys.length === 0) {
      toast.info('API keys cleared')
      setAvailableModels([])
      return
    }

    setLoading(true)
    try {
      const models = await fetchAvailableModels(keys[0])
      setAvailableModels(models)

      // Auto-select first Flash model if none selected
      if (!modelName && models.length > 0) {
        updateSetting('modelName', models[0])
      }

      toast.success(`Saved ${keys.length} key(s). Found ${models.length} models.`)
      logger.info(`Fetched ${models.length} models: ${models.slice(0, 5).join(', ')}...`)
    } catch (e: unknown) {
      toast.error(`Failed to fetch models: ${getErrorMessage(e)}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Google AI API Keys</CardTitle>
          <CardDescription>
            Enter one API key per line. Keys are stored in your browser's localStorage.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <a
            href="https://aistudio.google.com/apikey"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
          >
            Get your API key from Google AI Studio
            <ExternalLink className="h-3 w-3" />
          </a>

          <Textarea
            value={keysText}
            onChange={(e) => setKeysText(e.target.value)}
            placeholder="AIzaSy..."
            className="min-h-[120px] resize-y font-mono text-sm"
          />

          <Button onClick={handleSave} disabled={loading} className="gap-1.5">
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            Save API Keys
          </Button>
        </CardContent>
      </Card>

      {availableModels.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Available Models</CardTitle>
            <CardDescription>
              {availableModels.length} Gemini models found (Flash models prioritized)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-1.5">
              {availableModels.map((m) => (
                <Badge
                  key={m}
                  variant={m === modelName ? 'default' : 'outline'}
                  className="cursor-pointer text-xs"
                  onClick={() => updateSetting('modelName', m)}
                >
                  {m}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
