import { createGoogleGenerativeAI } from '@ai-sdk/google'
import { generateText } from 'ai'
import { logger } from './logger'
import { getErrorMessage } from './errors'

interface ApiError {
  status?: number
  statusCode?: number
  data?: { error?: { code?: number } }
}

interface GeminiModelsResponse {
  models?: Array<{ name?: string }>
}

function getGeminiVersion(modelName: string): number {
  const versionMatch = modelName.match(/gemini-(\d+(?:\.\d+)?)/)
  return versionMatch ? Number(versionMatch[1]) : 0
}

function getGeminiFamilyRank(modelName: string): number {
  if (modelName.includes('flash')) return 0
  if (modelName.includes('pro')) return 1
  return 2
}

export class AIClient {
  private apiKeys: string[]
  private currentKeyIndex = 0
  private modelName: string
  private maxRetries: number

  constructor(apiKeys: string[], modelName: string) {
    if (apiKeys.length === 0) throw new Error('API keys list cannot be empty.')
    this.apiKeys = apiKeys
    this.modelName = modelName
    this.maxRetries = apiKeys.length * 2
  }

  private createProvider() {
    return createGoogleGenerativeAI({
      apiKey: this.apiKeys[this.currentKeyIndex],
    })
  }

  private switchKey() {
    this.currentKeyIndex = (this.currentKeyIndex + 1) % this.apiKeys.length
    logger.info(`Switched to API key #${this.currentKeyIndex + 1}`)
  }

  async sendRequest(
    promptText: string,
    imageBase64List: string[]
  ): Promise<{ text: string; success: boolean }> {
    let lastError: Error | null = null

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const google = this.createProvider()

        const content: Array<
          | { type: 'text'; text: string }
          | { type: 'image'; image: string; mimeType?: string }
        > = [
          { type: 'text', text: promptText },
          ...imageBase64List.map((b64) => ({
            type: 'image' as const,
            image: b64,
            mimeType: 'image/jpeg' as const,
          })),
        ]

        logger.info(
          `Sending API request with ${imageBase64List.length} images (Attempt ${attempt + 1}/${this.maxRetries}, Key #${this.currentKeyIndex + 1})`
        )

        const result = await generateText({
          model: google(this.modelName),
          messages: [{ role: 'user', content }],
        })

        const responsePreview =
          result.text.length > 300
            ? result.text.slice(0, 300) + '...'
            : result.text
        logger.info(`AI response: ${responsePreview}`)

        return { text: result.text, success: true }
      } catch (error: unknown) {
        lastError = error instanceof Error ? error : new Error(String(error))
        const apiError = error as ApiError
        const status =
          apiError.status ?? apiError.statusCode ?? apiError.data?.error?.code

        if (status === 429 || status === 500 || status === 503) {
          logger.warn(`API Error (${status}). Rotating key...`)
          this.switchKey()
          await sleep(2000)
          continue
        }

        logger.error(`Non-retryable API Error: ${getErrorMessage(error)}`)
        break
      }
    }

    return {
      text: `Failed after ${this.maxRetries} attempts: ${lastError?.message}`,
      success: false,
    }
  }
}

/**
 * Parse JSON block from AI response text.
 * Looks for ```json ... ``` fenced blocks.
 */
export function parseJsonResponse(
  responseText: string
): Record<string, Record<string, string>> | null {
  const match = responseText.match(/```json\s*([\s\S]+?)\s*```/i)
  if (!match) return null
  try {
    return JSON.parse(match[1])
  } catch {
    return null
  }
}

/**
 * Check if response contains meaningful data.
 */
export function responseHasData(
  responseText: string,
  mainColumn: string
): boolean {
  const data = parseJsonResponse(responseText)
  if (!data) return false
  for (const item of Object.values(data)) {
    if (item[mainColumn]?.trim()) return true
  }
  return false
}

/**
 * Fetch available Gemini models from Google's API.
 * Uses the raw REST API since the AI SDK doesn't expose model listing.
 */
export async function fetchAvailableModels(apiKey: string): Promise<string[]> {
  try {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`
    )
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json() as GeminiModelsResponse

    const excluded = ['tts', 'image', 'audio', 'embedding', 'aqa', 'bison']
    const models: string[] = []

    for (const m of data.models ?? []) {
      const name: string = m.name?.replace('models/', '') ?? ''
      if (!name.startsWith('gemini-')) continue
      if (excluded.some((ex) => name.includes(ex))) continue

      // Require version 2.5+
      const versionMatch = name.match(/gemini-(\d+(?:\.\d+)?)/)
      if (versionMatch) {
        const version = parseFloat(versionMatch[1])
        if (version < 2.5) continue
      }

      models.push(name)
    }

    // Sort: newest Gemini generation first, Flash before Pro within each generation.
    models.sort((a, b) => {
      const versionDiff = getGeminiVersion(b) - getGeminiVersion(a)
      if (versionDiff !== 0) return versionDiff

      const familyDiff = getGeminiFamilyRank(a) - getGeminiFamilyRank(b)
      if (familyDiff !== 0) return familyDiff

      return a.localeCompare(b)
    })

    return models
  } catch (e: unknown) {
    logger.error(`Failed to fetch models: ${getErrorMessage(e)}`)
    return []
  }
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}
