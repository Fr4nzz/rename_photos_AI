type LogLevel = 'INFO' | 'WARN' | 'ERROR'

interface LogEntry {
  timestamp: string
  level: LogLevel
  message: string
}

const logs: LogEntry[] = []

function log(level: LogLevel, message: string) {
  const timestamp = new Date().toISOString()
  logs.push({ timestamp, level, message })
  const prefix = `[${level.padEnd(5)}]`
  if (level === 'ERROR') {
    console.error(prefix, message)
  } else if (level === 'WARN') {
    console.warn(prefix, message)
  } else {
    console.log(prefix, message)
  }
}

export const logger = {
  info: (msg: string) => log('INFO', msg),
  warn: (msg: string) => log('WARN', msg),
  error: (msg: string) => log('ERROR', msg),
  getLogs: () => [...logs],
}
