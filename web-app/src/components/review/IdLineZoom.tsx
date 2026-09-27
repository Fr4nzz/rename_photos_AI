import { useEffect, useState } from 'react'
import { useProcessingStore } from '@/stores/processingStore'
import { loadImagePreview } from '@/lib/imageProcessing'
import { camidLikeness } from '@/lib/ocr/decide'
import { normalizeReading } from '@/lib/ocr/pipeline'

/** Enlarged crop of the line the reader took the CAMID from (or the most CAMID-like line). */
export function IdLineZoom({ name, file, camid }: { name: string; file: File; camid: string }) {
  const reading = useProcessingStore((s) => s.ocrReadings.get(name))
  const [url, setUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!reading || reading.lines.length === 0) return
    const line = reading.lines.find((l) => normalizeReading(l.text) === camid)
      ?? [...reading.lines].sort((a, b) => camidLikeness(b.text) - camidLikeness(a.text))[0]
    let cancelled = false
    let made: string | null = null
    ;(async () => {
      const canvas = await loadImagePreview(file, 1600)
      const [W, H] = reading.size
      const s = canvas.width / W
      const pts = line.box.map(([x, y]) => (reading.turned ? [W - x, H - y] : [x, y]))
      const xs = pts.map((p) => p[0] * s), ys = pts.map((p) => p[1] * s)
      const pad = 6
      const x0 = Math.max(0, Math.min(...xs) - pad), y0 = Math.max(0, Math.min(...ys) - pad)
      const w = Math.min(canvas.width, Math.max(...xs) + pad) - x0, h = Math.min(canvas.height, Math.max(...ys) + pad) - y0
      if (w <= 0 || h <= 0) return
      const out = document.createElement('canvas')
      const scale = 56 / h
      out.width = Math.round(w * scale)
      out.height = 56
      const ctx = out.getContext('2d')!
      if (reading.turned) {
        ctx.translate(out.width, out.height)
        ctx.rotate(Math.PI)
      }
      ctx.drawImage(canvas, x0, y0, w, h, 0, 0, out.width, out.height)
      out.toBlob((blob) => {
        if (cancelled || !blob) return
        made = URL.createObjectURL(blob)
        setUrl(made)
      }, 'image/jpeg', 0.9)
    })().catch(() => setUrl(null))
    return () => {
      cancelled = true
      if (made) URL.revokeObjectURL(made)
    }
  }, [reading, file, camid])

  if (!url) return null
  return <img src={url} alt="ID line" className="h-14 max-w-full rounded-sm border bg-white object-contain object-left" />
}
