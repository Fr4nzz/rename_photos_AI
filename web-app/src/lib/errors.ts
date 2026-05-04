export function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function getErrorName(error: unknown): string | undefined {
  return error instanceof DOMException || error instanceof Error
    ? error.name
    : undefined
}
