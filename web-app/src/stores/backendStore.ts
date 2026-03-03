import { create } from 'zustand'
import type { BackendStatus } from '@/types'
import { BACKEND_URL } from '@/lib/constants'
import { logger } from '@/lib/logger'

interface BackendState {
  status: BackendStatus
  /** Number of consecutive failed checks (used for backoff) */
  failCount: number
  checkHealth: () => Promise<void>
}

export const useBackendStore = create<BackendState>()((set, get) => ({
  status: { connected: false, url: BACKEND_URL },
  failCount: 0,

  checkHealth: async () => {
    const wasPreviouslyConnected = get().status.connected
    try {
      const res = await fetch(`${BACKEND_URL}/health`, {
        signal: AbortSignal.timeout(2000),
      })
      if (res.ok) {
        if (!wasPreviouslyConnected && get().failCount > 0) {
          logger.info(`Backend connected at ${BACKEND_URL}`)
        }
        set({ status: { connected: true, url: BACKEND_URL }, failCount: 0 })
      } else {
        const newCount = get().failCount + 1
        if (newCount === 1) {
          logger.warn(`Backend not reachable at ${BACKEND_URL} (HTTP ${res.status})`)
        }
        set({ status: { connected: false, url: BACKEND_URL }, failCount: newCount })
      }
    } catch {
      const newCount = get().failCount + 1
      if (newCount === 1) {
        logger.warn(`Backend not reachable at ${BACKEND_URL} — retrying silently`)
      }
      set({ status: { connected: false, url: BACKEND_URL }, failCount: newCount })
    }
  },
}))
