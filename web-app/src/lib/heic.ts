/**
 * HEIC/HEIF decoding (iPhone photos), which Chrome cannot do natively: libheif compiled to
 * WebAssembly. The image's rotation/mirroring (irot/imir) is applied by libheif, so the result is
 * upright like the photo appears on the phone.
 */
import libheif from 'libheif-js/wasm-bundle'

export const HEIC_EXTENSIONS = new Set(['.heic', '.heif'])

export interface RGBA { width: number; height: number; data: Uint8ClampedArray<ArrayBuffer> }

export async function decodeHeic(blob: Blob): Promise<RGBA> {
  const decoder = new libheif.HeifDecoder()
  const images = decoder.decode(new Uint8Array(await blob.arrayBuffer()))
  if (!images.length) throw new Error('not a HEIC image')
  const image = images[0] // the primary image
  const width = image.get_width(), height = image.get_height()
  const data = new Uint8ClampedArray(width * height * 4)
  await new Promise<void>((resolve, reject) => {
    image.display({ data, width, height }, (out: unknown) => (out ? resolve() : reject(new Error('HEIC decoding failed'))))
  })
  for (const img of images) img.free?.()
  return { width, height, data }
}
