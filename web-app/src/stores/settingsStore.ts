import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { AppSettings, CropSettings, SuffixMode } from '@/types'
import { DEFAULT_SETTINGS, DEFAULT_PROMPT } from '@/lib/constants'

interface SettingsState extends AppSettings {
  updateSetting: <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => void
  updateCropSetting: <K extends keyof CropSettings>(key: K, value: CropSettings[K]) => void
  resetPrompt: () => void
  resetAll: () => void
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...DEFAULT_SETTINGS,

      updateSetting: (key, value) =>
        set({ [key]: value }),

      updateCropSetting: (key, value) =>
        set((state) => ({
          cropSettings: { ...state.cropSettings, [key]: value },
        })),

      resetPrompt: () =>
        set({ promptText: DEFAULT_PROMPT }),

      resetAll: () =>
        set({ ...DEFAULT_SETTINGS }),
    }),
    {
      name: 'ai-photo-processor-settings',
    }
  )
)
