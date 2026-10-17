import { describe, expect, it } from 'vitest'
import { cpuPlan, planFromTimes } from '../devicePlan'

// a Radeon 760M and its Ryzen 5 7640U, timed by probeDevices (17 Oct 2026)
const card = { envelope: 29, line: 35, rec: 16, wingseg: 96, wingcls: 17 }
const ryzen = { envelope: 154, rec: 50 }

describe('device plan', () => {
  it('runs every model on a card as fast as an integrated Radeon', () => {
    const plan = planFromTimes({ gpu: card, cpu: ryzen }, 12, 8)
    expect(plan.workers).toBe(2)
    expect(Object.keys(plan.runtime.devices).sort()).toEqual(['envelope', 'line', 'rec', 'wingcls', 'wingseg'])
  })

  it('keeps the processor when the card is far slower', () => {
    const slow = Object.fromEntries(Object.entries(card).map(([k, v]) => [k, v * 40]))
    expect(planFromTimes({ gpu: slow, cpu: ryzen }, 12, 8).runtime.devices).toEqual({})
  })

  it('moves to the card what is faster there on a slow processor', () => {
    const plan = planFromTimes({ gpu: { wingseg: 300 }, cpu: { envelope: 400, rec: 130 } }, 4, 8)
    expect(plan.runtime.devices).toEqual({ wingseg: 'gpu' })
  })

  it('without a card time, the processor', () => {
    expect(planFromTimes({ gpu: {}, cpu: ryzen }, 4, 8)).toEqual(cpuPlan(4, 8))
  })

  it('one worker per logical processor, as the memory allows', () => {
    expect(cpuPlan(12, 8).workers).toBe(6)
    expect(cpuPlan(8, 8).workers).toBe(4)
    expect(cpuPlan(4, 8).workers).toBe(4)
    expect(cpuPlan(4, 4).workers).toBe(2)
    expect(cpuPlan(2, 2).workers).toBe(2)
  })
})
