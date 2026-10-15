/** The batch's results saved where it was loaded from: the browser, and its folder's rename_files/ with what the CSV does not hold. */
import { saveCsvToFolder, saveCsvToStorage, toCsvString } from './csvHandler'
import { saveSidecar } from './reviewSidecar'
import { useProcessingStore } from '@/stores/processingStore'
import { useSettingsStore } from '@/stores/settingsStore'
import type { PhotoRow } from '@/types'

export async function saveBatch(rows: PhotoRow[]): Promise<void> {
  const { currentCsvName, dirHandle } = useProcessingStore.getState()
  if (!currentCsvName) return
  const text = toCsvString(rows, useSettingsStore.getState().mainColumn)
  await saveCsvToStorage(currentCsvName, text)
  if (dirHandle) {
    await saveCsvToFolder(dirHandle, currentCsvName, text)
    await saveSidecar(dirHandle, currentCsvName)
  }
}
