import { useEffect, useState } from 'react'
import { useProcessingStore } from '@/stores/processingStore'
import { loadImagePreview, rotateCanvas } from '@/lib/imageProcessing'
import { camidLikeness } from '@/lib/ocr/decide'
import { normalizeReading } from '@/lib/ocr/pipeline'

/** Enlarged crop of the line the reader took the CAMID from (or the most CAMID-like line). */
export function IdLineZoom({ name, file, camid, applied = 0 }: { name: string; file: File; camid: string; applied?: number }) {
  const reading = useProcessingStore((s) => s.ocrReadings.get(name))
  const [url, setUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!reading || reading.lines.length === 0) return
    const line = reading.lines.find((l) => normalizeReading(l.text) === camid)
      ?? [...reading.lines].sort((a, b) => camidLikeness(b.text) - camidLikeness(a.text))[0]
    let cancelled = false
    let made: string | null = null
    ;(async () => {
      // line boxes refer to the photo as it was read; undo rotations written to the file since then
      const preview = await loadImagePreview(file, 1600)
      const canvas = applied % 360 ? rotateCanvas(preview, 360 - (applied % 360)) : preview
      const [W, H] = reading.size
      const s = canvas.width / W
      const pts = line.box.map(([x, y]) => (reading.turned ? [W - x, H - y] : [x, y]))
      const xs = pts.map((p) => p[0] * s), ys = pts.map((p) => p[1] * s)
      const pad = 6
      const x0 = Math.max(0, Math.min(...xs) - pad), y0 = Math.max(0, Math.min(...ys) - pad)
      const w = Math.min(canvas.width, Math.max(...xs) + pad) - x0, h = Math.min(canvas.height, Math.max(...ys) + pad) - y0
      if (w <= 0 || h <= 0) return
      // show the line the way the reader read it: upright, even when written sideways
      const turn = (reading.turned ? 180 : 0) + (line.turn ?? 0) // counter-clockwise degrees
      const sideways = turn % 180 !== 0
      const scale = 56 / (sideways ? w : h)
      const dw = Math.round(w * scale), dh = Math.round(h * scale)
      const out = document.createElement('canvas')
      out.width = sideways ? dh : dw
      out.height = sideways ? dw : dh
      const ctx = out.getContext('2d')!
      ctx.translate(out.width / 2, out.height / 2)
      ctx.rotate((-turn * Math.PI) / 180)
      ctx.drawImage(canvas, x0, y0, w, h, -dw / 2, -dh / 2, dw, dh)
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
  }, [reading, file, camid, applied])

  if (!url) return null
  return <img src={url} alt="ID line" className="h-14 max-w-full rounded-sm border bg-white object-contain object-left" />
}
