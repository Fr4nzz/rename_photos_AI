import { describe, expect, it } from 'vitest'
import { chooseTurn, meanLogp, readingOrder, shapeOf, type LineRead, type Turn } from '../envelopeText'
import { ctcDecode } from '../rec'
import type { Pt } from '../lineDet'

/** A reading of `text` whose characters each have probability p. */
const r = (text: string, p: number): LineRead => ({ text, conf: p, logp: [...text].length * Math.log(p) })

/** Readings by line and turn; counts how many were asked for. */
function reader(table: Partial<Record<Turn, LineRead>>[]) {
  const asked: string[] = []
  return {
    asked,
    read: async (i: number, t: Turn) => { asked.push(`${i}:${t}`); return table[i][t] ?? r('', 1) },
  }
}

describe('shapes and scores', () => {
  it('tells tall, wide and square crops apart as the CAM ID reader does', () => {
    expect(shapeOf(40, 200)).toBe('tall')
    expect(shapeOf(200, 40)).toBe('wide')
    expect(shapeOf(50, 70)).toBe('square')
  })
  it('weighs lines by their length', () => {
    expect(meanLogp([r('Mechanitis', 0.99), r('♂', 0.5)])).toBeCloseTo((10 * Math.log(0.99) + Math.log(0.5)) / 11)
    expect(meanLogp([r('', 1)])).toBe(-Infinity)
  })
  it('sums the log-probabilities of the decoded characters', () => {
    // 3 steps over [blank, a, b]: a (0.9), blank, b (0.8)
    const probs = new Float32Array([0.05, 0.9, 0.05, 0.9, 0.05, 0.05, 0.1, 0.1, 0.8])
    const d = ctcDecode(probs, 3, 3, ['blank', 'a', 'b'])
    expect(d.text).toBe('ab')
    expect(d.logp).toBeCloseTo(Math.log(0.9) + Math.log(0.8), 5)
  })
})

describe('one turn per envelope', () => {
  it('keeps an upright envelope upright without reading it turned over', async () => {
    const t = reader([{ 0: r('CAM071234', 0.99) }, { 0: r('Mechanitis', 0.97) }, { 0: r('3/12/22', 0.95) }])
    const c = await chooseTurn({ sizes: [[300, 50], [320, 60], [200, 55]], read: t.read })
    expect(c.turn).toBe(0)
    expect(c.lineTurns).toEqual([0, 0, 0])
    expect(t.asked.some((k) => k.endsWith(':180'))).toBe(false)
  })
  it('turns a poorly read envelope over when that reads better', async () => {
    const t = reader([
      { 0: r('C2770', 0.6), 180: r('CAM071234', 0.98) },
      { 0: r('sihuea', 0.55), 180: r('Mechanitis', 0.97) },
    ])
    const c = await chooseTurn({ sizes: [[300, 50], [320, 60]], read: t.read })
    expect(c.turn).toBe(180)
  })
  it('follows a sure CAM ID read upside down without comparing', async () => {
    const t = reader([{ 180: r('CAM071234', 0.99) }, { 0: r('Mech', 0.9), 180: r('Mechanitis', 0.8) }])
    const c = await chooseTurn({ sizes: [[300, 50], [320, 60]], read: t.read, camid: { line: 0, turn: 180 } })
    expect(c.turn).toBe(180)
    expect(c.lineTurns).toEqual([180, 180])
  })
  it('reads all the sideways lines in the one turn that reads best together', async () => {
    // line 1 alone would prefer 270 (a short confident misread); together the lines read best at 90
    const t = reader([
      { 90: r('Mechanitis', 0.98), 270: r('sinpeuaW', 0.7) },
      { 90: r('♀', 0.9), 270: r('♂', 0.99) },
      { 90: r('Finca Narupa', 0.97), 270: r('edn', 0.6) },
    ])
    const c = await chooseTurn({ sizes: [[50, 300], [40, 90], [55, 330]], read: t.read })
    expect(c.turn).toBe(90)
    expect(c.lineTurns).toEqual([90, 90, 90])
  })
  it('reads a lone sign of the other shape with the main lines', async () => {
    // a ♀ on an upright envelope is taller than wide; read turned 270 it would be a ♂
    const t = reader([{ 0: r('Mechanitis', 0.99) }, { 0: r('polymnia', 0.99) }, { 0: r('♀', 0.99), 90: r('♀', 0.99), 270: r('♂', 0.99) }])
    const c = await chooseTurn({ sizes: [[300, 60], [280, 64], [45, 85]], read: t.read })
    expect(c.lineTurns).toEqual([0, 0, 0])
    expect(c.crossTurn).toBeUndefined()
  })
  it('gives lines written across the rest their own turn', async () => {
    // a sideways envelope with a CAM ID written along the photo's x axis
    const t = reader([
      { 90: r('Mechanitis', 0.98), 270: r('xx', 0.5) },
      { 90: r('Finca Narupa', 0.97), 270: r('edn', 0.6) },
      { 0: r('CAM071234', 0.99) },
    ])
    const c = await chooseTurn({ sizes: [[50, 300], [55, 330], [260, 50]], read: t.read, camid: { line: 2, turn: 0 } })
    expect(c.turn).toBe(90)
    expect(c.crossTurn).toBe(0)
    expect(c.lineTurns).toEqual([90, 90, 0])
  })
  it('lets a sure CAM ID settle a close call between the two sideways turns', async () => {
    const t = reader([{ 90: r('CAM071234', 0.97), 270: r('CAM071234', 0.96) }, { 90: r('abc', 0.9), 270: r('abc', 0.92) }])
    const c = await chooseTurn({ sizes: [[50, 300], [50, 200]], read: t.read, camid: { line: 0, turn: 90 } })
    expect(c.turn).toBe(90)
  })
})

describe('reading order', () => {
  const box = (x0: number, y0: number, x1: number, y1: number): Pt[] => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
  it('goes top to bottom, then left to right, in the turned envelope', () => {
    const boxes = [box(10, 100, 100, 120), box(10, 10, 100, 30), box(120, 12, 200, 32)]
    expect(readingOrder(boxes, 0, 300, 200)).toEqual([1, 2, 0])
    // turned over, the bottom line comes first and the row reads right to left in the photo
    expect(readingOrder(boxes, 180, 300, 200)).toEqual([0, 2, 1])
  })
  it('reads a sideways envelope along its turned lines', () => {
    // two tall lines; turned 90 counter-clockwise, the right-hand one is on top
    const boxes = [box(10, 10, 30, 200), box(60, 10, 80, 200)]
    expect(readingOrder(boxes, 90, 100, 220)).toEqual([1, 0])
    expect(readingOrder(boxes, 270, 100, 220)).toEqual([0, 1])
  })
})
