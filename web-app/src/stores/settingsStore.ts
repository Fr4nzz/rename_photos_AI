import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { AppSettings, CropSettings } from '@/types'
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
      // 1: RAW files shown among the photos by default (people looked for them); 2: larger photos in the review
      version: 2,
      migrate: (state, version) => {
        const s = { ...(state as SettingsState) }
        if (version < 1) s.previewRaw = true
        if (version < 2 && s.reviewThumbSize === 180) s.reviewThumbSize = 260
        return s
      },
    }
  )
)
