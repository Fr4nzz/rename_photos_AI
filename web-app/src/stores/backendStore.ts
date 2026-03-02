import { create } from 'zustand'
import type { BackendStatus } from '@/types'
import { BACKEND_URL } from '@/lib/constants'

interface BackendState {
  status: BackendStatus
  /** Number of consecutive failed checks (used for backoff) */
  failCount: number
  checkHealth: () => Promise<void>
}

export const useBackendStore = create<BackendState>()((set) => ({
  status: { connected: false, url: BACKEND_URL },
  failCount: 0,

  checkHealth: async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/health`, {
        signal: AbortSignal.timeout(2000),
      })
      if (res.ok) {
        set({ status: { connected: true, url: BACKEND_URL }, failCount: 0 })
      } else {
        set((s) => ({ status: { connected: false, url: BACKEND_URL }, failCount: s.failCount + 1 }))
      }
    } catch {
      set((s) => ({ status: { connected: false, url: BACKEND_URL }, failCount: s.failCount + 1 }))
    }
  },
}))
