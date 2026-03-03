type LogLevel = 'INFO' | 'WARN' | 'ERROR' | 'PERF'

interface LogEntry {
  timestamp: string
  level: LogLevel
  message: string
}

const logs: LogEntry[] = []
const timers = new Map<string, number>()

function log(level: LogLevel, message: string) {
  const timestamp = new Date().toISOString()
  logs.push({ timestamp, level, message })
  const prefix = `[${level.padEnd(5)}]`
  if (level === 'ERROR') {
    console.error(prefix, message)
  } else if (level === 'WARN') {
    console.warn(prefix, message)
  } else if (level === 'PERF') {
    console.log(`%c${prefix} ${message}`, 'color: #8b5cf6')
  } else {
    console.log(prefix, message)
  }
}

export const logger = {
  info: (msg: string) => log('INFO', msg),
  warn: (msg: string) => log('WARN', msg),
  error: (msg: string) => log('ERROR', msg),
  /** Start a named timer. Call timeEnd() with the same label to log the elapsed time. */
  time: (label: string) => {
    timers.set(label, performance.now())
  },
  /** Stop a named timer and log the elapsed time. Returns elapsed ms. Silently ignores missing timers (e.g. React Strict Mode double-mount). */
  timeEnd: (label: string): number => {
    const start = timers.get(label)
    if (start === undefined) return 0
    timers.delete(label)
    const elapsed = performance.now() - start
    log('PERF', `${label}: ${formatMs(elapsed)}`)
    return elapsed
  },
  getLogs: () => [...logs],
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}
