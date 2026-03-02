import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface ApiKeysState {
  apiKeys: string[]
  availableModels: string[]
  setApiKeys: (keys: string[]) => void
  setAvailableModels: (models: string[]) => void
}

export const useApiKeysStore = create<ApiKeysState>()(
  persist(
    (set) => ({
      apiKeys: [],
      availableModels: [],

      setApiKeys: (keys) => set({ apiKeys: keys }),
      setAvailableModels: (models) => set({ availableModels: models }),
    }),
    {
      name: 'ai-photo-processor-api-keys',
    }
  )
)
