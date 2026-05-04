/**
 * Web Worker for off-main-thread image decoding.
 * Receives a Blob, decodes with createImageBitmap at reduced resolution,
 * and transfers the resulting ImageBitmap back (zero-copy).
 */

export interface DecodeRequest {
  id: number
  blob: Blob
  maxSize: number
  applyExif: boolean
}

export interface DecodeResponse {
  id: number
  bitmap?: ImageBitmap
  error?: string
}

// Worker entry point
const ctx = self as unknown as Worker

ctx.onmessage = async (e: MessageEvent<DecodeRequest>) => {
  const { id, blob, maxSize, applyExif } = e.data
  try {
    const bmp = await createImageBitmap(blob, {
      resizeWidth: maxSize,
      resizeQuality: 'medium',
      imageOrientation: applyExif ? 'from-image' : 'none',
    })
    ctx.postMessage({ id, bitmap: bmp } satisfies DecodeResponse, [bmp])
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'decode failed'
    ctx.postMessage({ id, error: message } satisfies DecodeResponse)
  }
}
