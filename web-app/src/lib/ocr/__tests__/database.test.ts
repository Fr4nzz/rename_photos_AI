import { describe, expect, it } from 'vitest'
import { camidsFromCsv, parseCsv } from '../database'

describe('database CSV', () => {
  it('parses quoted fields with commas and newlines', () => {
    expect(parseCsv('a,b\n"x, y","line1\nline2"\n')).toEqual([['a', 'b'], ['x, y', 'line1\nline2']])
  })
  it('reads only CAM_ID columns', () => {
    const csv = 'Notes,CAM_ID,CAM_ID_insectary\n"old CAM070001",CAM070002,CAM070003\nx,NA,\n'
    expect(camidsFromCsv(csv)).toEqual(['CAM070002', 'CAM070003'])
  })
})

describe.skipIf(process.env.LIVE_DB !== '1')('live database', () => {
  it('fetches the three tabs', async () => {
    const { camidsFromCsv: parse } = await import('../database')
    const res = await fetch('https://docs.google.com/spreadsheets/d/1QZj6YgHAJ9NmFXFPCtu-i-1NDuDmAdMF2Wogts7S2_4/export?format=csv&gid=900206579')
    expect(parse(await res.text()).length).toBeGreaterThan(5000)
  }, 60_000)
})
