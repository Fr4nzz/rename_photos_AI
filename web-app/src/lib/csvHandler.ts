import type { PhotoRow } from '@/types'

const DB_NAME = 'ai-photo-processor'
const STORE_NAME = 'csvFiles'
const META_STORE = 'meta'
const DB_VERSION = 2

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME)
      }
      if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE)
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

export async function saveCsvToStorage(name: string, csvText: string): Promise<void> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.objectStore(STORE_NAME).put(csvText, name)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

export async function loadCsvFromStorage(name: string): Promise<string | null> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const req = tx.objectStore(STORE_NAME).get(name)
    req.onsuccess = () => resolve(req.result ?? null)
    req.onerror = () => reject(req.error)
  })
}

export async function listStoredCsvs(): Promise<string[]> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const req = tx.objectStore(STORE_NAME).getAllKeys()
    req.onsuccess = () => resolve((req.result as string[]).sort())
    req.onerror = () => reject(req.error)
  })
}

export async function deleteCsvFromStorage(name: string): Promise<void> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.objectStore(STORE_NAME).delete(name)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

/**
 * Parse CSV text into PhotoRow[].
 */
export function parseCsv(csvText: string, mainColumn: string): PhotoRow[] {
  const lines = csvText.trim().split('\n')
  if (lines.length < 2) return []

  const headers = lines[0].split(',').map((h) => h.trim())
  const rows: PhotoRow[] = []

  for (let i = 1; i < lines.length; i++) {
    const values = parseCsvLine(lines[i])
    if (values.length < headers.length) continue

    const obj: Record<string, string> = {}
    headers.forEach((h, idx) => {
      obj[h] = values[idx] ?? ''
    })

    rows.push({
      from: obj['from'] ?? '',
      currentPath: obj['current_path'] ?? obj['from'] ?? '',
      photoId: parseInt(obj['photo_ID'] ?? '0', 10),
      mainValue: obj[mainColumn] ?? '',
      co: obj['co'] ?? '',
      n: obj['n'] ?? '',
      skip: obj['skip'] ?? '',
      to: obj['to'] ?? '',
      suffix: obj['suffix'] ?? '',
      batchNumber: parseInt(obj['batch_number'] ?? '0', 10),
      captureDate: obj['capture_date'] || null,
      status: (obj['status'] as PhotoRow['status']) || 'Original',
    })
  }

  return rows
}

/**
 * Convert PhotoRow[] to CSV text.
 */
export function toCsvString(rows: PhotoRow[], mainColumn: string): string {
  const headers = [
    'from', 'photo_ID', mainColumn, 'co', 'n', 'skip', 'to', 'suffix',
    'batch_number', 'status', 'current_path',
  ]
  const lines = [headers.join(',')]

  for (const row of rows) {
    const values = [
      escapeCsv(row.from),
      String(row.photoId),
      escapeCsv(row.mainValue),
      escapeCsv(row.co),
      escapeCsv(row.n),
      escapeCsv(row.skip),
      escapeCsv(row.to),
      escapeCsv(row.suffix),
      String(row.batchNumber),
      row.status,
      escapeCsv(row.currentPath),
    ]
    lines.push(values.join(','))
  }

  return lines.join('\n')
}

function escapeCsv(val: string): string {
  if (val.includes(',') || val.includes('"') || val.includes('\n')) {
    return `"${val.replace(/"/g, '""')}"`
  }
  return val
}

function parseCsvLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < line.length && line[i + 1] === '"') {
          current += '"'
          i++
        } else {
          inQuotes = false
        }
      } else {
        current += ch
      }
    } else if (ch === '"') {
      inQuotes = true
    } else if (ch === ',') {
      result.push(current)
      current = ''
    } else {
      current += ch
    }
  }
  result.push(current)
  return result
}

// --- Meta storage (last CSV name, directory handle) ---

async function getMeta(key: string): Promise<any> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(META_STORE, 'readonly')
    const req = tx.objectStore(META_STORE).get(key)
    req.onsuccess = () => resolve(req.result ?? null)
    req.onerror = () => reject(req.error)
  })
}

async function setMeta(key: string, value: any): Promise<void> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(META_STORE, 'readwrite')
    tx.objectStore(META_STORE).put(value, key)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

export async function saveLastCsvName(name: string): Promise<void> {
  await setMeta('lastCsvName', name)
}

export async function getLastCsvName(): Promise<string | null> {
  return getMeta('lastCsvName')
}

export async function saveDirHandle(handle: FileSystemDirectoryHandle): Promise<void> {
  await setMeta('lastDirHandle', handle)
}

export async function getSavedDirHandle(): Promise<FileSystemDirectoryHandle | null> {
  return getMeta('lastDirHandle')
}

/**
 * Trigger download of a CSV string as a file.
 */
export function downloadCsv(csvText: string, filename: string) {
  const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
