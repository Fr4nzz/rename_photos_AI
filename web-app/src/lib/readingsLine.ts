/**
 * The models' readings of a photo in one line under it on the review card: the side its wings show,
 * how many, which way up, and how sure the CAMID reading is. A part is marked when it disagrees with
 * what the card will do (the name's d/v, the turn that will be written) or the reading is not sure.
 */
import type { PhotoRow, SuffixMode } from '@/types'
import { AUTO_SURE, type PhotoReading } from './ocr/pipeline'
import { LETTER, PLACE, TURN_SURE, TURN_WINGS, wingTurn, type WingCheck } from './wings/checks'
import type { Wing } from './wings/model'

export interface ReadingPart { text: string; warn: boolean }

const norm = (a: number) => ((a % 360) + 360) % 360
/** how sure, as a whole percentage or two decimals, rounded down (a 0.949 is not shown as sure as 0.95) */
const pct = (p: number) => `${Math.floor(p * 100 + 1e-9)} %`
const two = (p: number) => (Math.floor(p * 100 + 1e-9) / 100).toFixed(2)
const ORDER = { forewing: 0, hindwing: 1, extra: 2 }
const HALF = { left: 0, right: 1, unknown: 2 }

const leaning = (p: number) => (p >= 0.5 ? `dorsal ${pct(p)}` : `ventral ${pct(1 - p)}`)
const wingName = (w: Wing) => (w.kind === 'extra' ? 'extra wing' : w.half === 'unknown' ? `${w.kind} (side unknown)` : `${w.half} ${w.kind}`)
const TURNED: Record<number, [string, string]> = {
  0: ['upright', 'upright'],
  90: ['turned 90° ccw', 'turned 90° counter-clockwise (forewings on top after 90° clockwise)'],
  180: ['upside down', 'upside down (forewings at the bottom)'],
  270: ['turned 90° cw', 'turned 90° clockwise (forewings on top after 90° counter-clockwise)'],
}

/**
 * The parts of a photo's line and their detail (the hover title); null when the photo has not been read.
 * The turn is told as the card shows the photo (turned as chosen), so «upright» agrees with the rotation.
 */
export function readingsLine(row: PhotoRow, check: WingCheck | undefined, reading: PhotoReading | undefined, suffixMode: SuffixMode): { parts: ReadingPart[]; title: string } | null {
  if (!check && !reading) return null
  const parts: ReadingPart[] = [], title: string[] = []
  const live = row.skip !== 'x'
  if (check) {
    const n = check.wings.length
    if (!n) {
      parts.push({ text: 'no wings', warn: false })
      title.push('No wing found')
    } else {
      const named = suffixMode === 'Wing Clips' ? undefined : LETTER[row.suffix[0]]
      const sideText = check.side === 'dorsal' ? `dorsal ${pct(check.p!)}`
        : check.side === 'ventral' ? `ventral ${pct(1 - check.p!)}`
          : check.side === 'mixed' ? 'mixed sides' : `unsure (${leaning(check.p!)})`
      const against = !!named && (check.side === 'mixed' || ((check.side === 'dorsal' || check.side === 'ventral') && check.side !== named))
      parts.push({ text: sideText, warn: live && against })
      parts.push({ text: check.missing.length ? `${4 - check.missing.length} of 4 wings` : `${n} wing${n === 1 ? '' : 's'}`, warn: false })
      const r = norm(90 * check.turnK + check.applied - Number(row.rotChosen || 0))
      const trusted = wingTurn(check, row) !== null
      parts.push({ text: `${TURNED[r][0]} ${pct(check.pTurn)}`, warn: live && trusted && r !== 0 })

      title.push(`Side: P(dorsal) ${check.p!.toFixed(2)}${check.fore.p !== null ? ` · forewings ${check.fore.p.toFixed(2)}` : ''}${check.hind.p !== null ? ` · hindwings ${check.hind.p.toFixed(2)}` : ''}`)
      const wings = [...check.wings].sort((a, b) => ORDER[a.kind] - ORDER[b.kind] || HALF[a.half] - HALF[b.half])
      for (const w of wings) title.push(`  ${wingName(w)}: P(dorsal) ${w.pDorsal.toFixed(2)}`)
      if (check.missing.length) title.push(`Missing: ${check.missing.map((p) => PLACE[p]).join(', ')}`)
      title.push(`Turn: ${TURNED[r][1]} as shown, ${pct(check.pTurn)} sure${trusted ? '' : ` (not used: the wings' turn counts with ${TURN_WINGS} wings or more, ${pct(TURN_SURE)} sure)`}`)
    }
  }
  // the reader's confidence is of the ID it read: not shown once the photo has another one
  if (reading?.camid && reading.camid === row.mainValue.trim()) {
    parts.push({ text: `ID ${two(reading.conf)}`, warn: live && reading.conf < AUTO_SURE })
    title.push(`CAMID read with confidence ${two(reading.conf)} (sure from ${two(AUTO_SURE)})`)
  }
  return parts.length ? { parts, title: title.join('\n') } : null
}
