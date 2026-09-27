import { describe, expect, it } from 'vitest'
import { companionsOf, indexByStem, planRenames, type PlannedStep } from '../renamePlan'

/** Apply steps to a simulated case-insensitive folder; throws if a step would overwrite. */
function simulate(files: string[], steps: PlannedStep[]): string[] {
  const dir = new Map(files.map((f) => [f.toLowerCase(), f]))
  for (const s of steps) {
    if (!dir.has(s.from.toLowerCase())) throw new Error(`missing ${s.from}`)
    if (dir.has(s.to.toLowerCase()) && s.to.toLowerCase() !== s.from.toLowerCase()) throw new Error(`overwrite ${s.to}`)
    dir.delete(s.from.toLowerCase())
    dir.set(s.to.toLowerCase(), s.to)
  }
  return [...dir.values()].sort()
}

describe('planRenames', () => {
  it('renames plain files directly', () => {
    const files = ['IMG_1.JPG', 'IMG_2.JPG']
    const plan = planRenames([{ src: 'IMG_1.JPG', dst: 'CAM070001d.JPG' }, { src: 'IMG_2.JPG', dst: 'CAM070001v.JPG' }], files)
    expect(plan.refused).toEqual([])
    expect(plan.steps.every((s) => !s.temp)).toBe(true)
    expect(simulate(files, plan.steps)).toEqual(['CAM070001d.JPG', 'CAM070001v.JPG'])
  })

  it('handles a swap through temporary names without overwriting', () => {
    const files = ['a.jpg', 'b.jpg']
    const plan = planRenames([{ src: 'a.jpg', dst: 'b.jpg' }, { src: 'b.jpg', dst: 'a.jpg' }], files)
    expect(simulate(files, plan.steps)).toEqual(['a.jpg', 'b.jpg'])
    expect(plan.steps.filter((s) => s.temp)).toHaveLength(2)
  })

  it('refuses a target held by an unrelated file instead of moving that file', () => {
    const files = ['a.jpg', 'CAM070001d.jpg']
    const plan = planRenames([{ src: 'a.jpg', dst: 'CAM070001d.jpg' }], files)
    expect(plan.steps).toEqual([])
    expect(plan.refused[0].reason).toBe('target-exists')
  })

  it('refuses two files wanting the same name', () => {
    const plan = planRenames([{ src: 'a.jpg', dst: 'x.jpg' }, { src: 'b.jpg', dst: 'X.JPG' }], ['a.jpg', 'b.jpg'])
    expect(plan.steps).toEqual([])
    expect(plan.refused.map((r) => r.reason)).toEqual(['duplicate-target', 'duplicate-target'])
  })

  it('handles case-only renames on case-insensitive folders', () => {
    const files = ['cam070001d.jpg']
    const plan = planRenames([{ src: 'cam070001d.jpg', dst: 'CAM070001d.jpg' }], files)
    expect(simulate(files, plan.steps)).toEqual(['CAM070001d.jpg'])
  })

  it('refuses an op whose target stays because its own op was refused', () => {
    // b -> c is refused (c is unrelated); a -> b must then be refused too
    const files = ['a.jpg', 'b.jpg', 'c.jpg']
    const plan = planRenames([{ src: 'a.jpg', dst: 'b.jpg' }, { src: 'b.jpg', dst: 'c.jpg' }], files)
    expect(plan.steps).toEqual([])
    expect(plan.refused).toHaveLength(2)
  })
})

describe('chains', () => {
  it('propagates a refusal along a chain', () => {
    const files = ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg']
    const plan = planRenames([{ src: 'a.jpg', dst: 'b.jpg' }, { src: 'b.jpg', dst: 'c.jpg' }, { src: 'c.jpg', dst: 'd.jpg' }], files)
    expect(plan.steps).toEqual([])
    expect(plan.refused).toHaveLength(3)
  })
})

describe('companions', () => {
  it('finds RAW partners with any extension case', () => {
    const index = indexByStem(['IMG_1.JPG', 'IMG_1.CR2', 'img_1.nef', 'IMG_2.JPG', 'IMG_1.xmp'])
    expect(companionsOf('IMG_1.JPG', index, new Set(['.cr2', '.nef'])).sort()).toEqual(['IMG_1.CR2', 'img_1.nef'])
  })
})
