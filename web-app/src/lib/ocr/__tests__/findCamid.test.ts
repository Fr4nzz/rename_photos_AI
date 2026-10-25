import { describe, expect, it } from 'vitest'
import { chooseCamid, findCamid, joinPieces, type LineReading, type PhotoReading } from '../pipeline'

describe('findCamid', () => {
  it('takes a whole line', () => {
    expect(findCamid('CAM 075192')).toEqual({ id: 'CAM075192', how: 'line' })
    expect(findCamid('Cam071295')).toEqual({ id: 'CAM071295', how: 'line' })
  })
  it('takes an ID followed only by a sex sign or punctuation', () => {
    expect(findCamid('CAM075192♀')).toEqual({ id: 'CAM075192', how: 'token' })
    expect(findCamid('CAM075192 ♂.')).toEqual({ id: 'CAM075192', how: 'token' })
  })
  it('takes an ID ending a longer line', () => {
    expect(findCamid('Ela-meme-1-CAM079134')).toEqual({ id: 'CAM079134', how: 'token' })
    expect(findCamid('Eva-mneme.8CAm079141')).toEqual({ id: 'CAM079141', how: 'token' })
  })
  it('takes an ID whose first letters are cut off, at the start or after a separator', () => {
    expect(findCamid('AM074594')).toEqual({ id: 'CAM074594', how: 'cut' })
    expect(findCamid('(AM074586')).toEqual({ id: 'CAM074586', how: 'cut' })
    expect(findCamid('CM077250')).toEqual({ id: 'CAM077250', how: 'cut' })
    expect(findCamid('M077392')).toEqual({ id: 'CAM077392', how: 'cut' })
  })
  it('refuses anything else', () => {
    for (const t of ['CAM0790710', 'CAM07907', 'CAM079071 Oleria', 'CAM07907l', 'XAM074594', 'AM0745941', 'ME22-335', 'XM074594', '077352', '642022', 'CAN074594', '074594', 'CAM075192♀ 2'])
      expect(findCamid(t), t).toBeNull()
  })
})

const reading = (cands: PhotoReading['candidates']): PhotoReading =>
  ({ camid: null, conf: 0, candidates: cands, source: 'envelope', envelope: null, turned: 0, lines: [], size: [1600, 1067] })

describe('chooseCamid', () => {
  const known = new Set(['CAM074594', 'CAM079134', 'CAM071000'])
  it('prefers a whole line in the database over an ID inside a line', () => {
    const r = reading([{ id: 'CAM071000', conf: 0.9 }, { id: 'CAM079134', conf: 0.99, how: 'token' }])
    expect(chooseCamid(r, known)?.id).toBe('CAM071000')
  })
  it('takes an ID inside a line when the database knows it', () => {
    expect(chooseCamid(reading([{ id: 'CAM079134', conf: 0.9, how: 'token' }]), known)?.id).toBe('CAM079134')
    expect(chooseCamid(reading([{ id: 'CAM079135', conf: 0.99, how: 'token' }]), known)).toBeNull()
  })
  it('takes an ID with letters of CAM missing only when the database knows it, or very sure without the database', () => {
    expect(chooseCamid(reading([{ id: 'CAM074594', conf: 0.9, how: 'cut' }]), known)?.id).toBe('CAM074594')
    expect(chooseCamid(reading([{ id: 'CAM074595', conf: 0.999, how: 'cut' }]), known)).toBeNull()
    expect(chooseCamid(reading([{ id: 'CAM074594', conf: 0.95, how: 'cut' }]), null)).toBeNull()
    expect(chooseCamid(reading([{ id: 'CAM074594', conf: 0.995, how: 'cut' }]), null)?.id).toBe('CAM074594')
  })
})

describe('joinPieces', () => {
  const line = (text: string, x0: number, x1: number, y = 100): LineReading =>
    ({ text, conf: 0.97, box: [[x0, y], [x1, y], [x1, y + 30], [x0, y + 30]] })
  it('joins CAM and 6 digits, and a CAM piece with the digits it lacks', () => {
    expect(joinPieces([line('CAM', 10, 70), line('076987', 80, 200)]).map((l) => l.text)).toEqual(['CAM076987'])
    expect(joinPieces([line('CAM078', 10, 130), line('715', 140, 200)]).map((l) => l.text)).toEqual(['CAM078715'])
  })
  it('does not join pieces that do not add up to 6 digits, are on another row or in the wrong order', () => {
    expect(joinPieces([line('CAM07', 10, 110), line('715', 120, 180)])).toEqual([])
    expect(joinPieces([line('CAM078', 10, 130), line('715', 140, 200, 300)])).toEqual([])
    expect(joinPieces([line('076816', 10, 130), line('CAM07', 140, 240)])).toEqual([])
  })
})
