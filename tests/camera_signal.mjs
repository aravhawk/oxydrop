// Live acquisition regressions; published offline estimator remains parity checked separately.
import assert from 'node:assert/strict';
import { estimateHr, GATE } from '../app/js/hr.js';
import { scoreSignal, ringState, chooseMode, SCORE_MIN, FS, PROBE_SEC, SETTLE_SEC } from '../app/js/lensprobe.js';
const guarded = { ...GATE, harmonicGuard: true };
const wave = (hz, a = 1, harmonic = 0) => Array.from({ length: 900 }, (_, i) => a * Math.sin(2 * Math.PI * hz * i / FS) + harmonic * Math.sin(4 * Math.PI * hz * i / FS + .3));
const doubled = wave(1.2, .2, 1);
assert.ok(estimateHr(doubled).hr > 140, 'fixture reproduces secondary-peak doubling');
const fixed = estimateHr(doubled, FS, guarded);
assert.ok(fixed.valid && Math.abs(fixed.hr - 72) < 2, JSON.stringify(fixed));
for (const hz of [1, 1.2, 1.8, 2.4, 2.8]) {
  const e = estimateHr(wave(hz), FS, guarded);
  assert.ok(e.valid && Math.abs(e.hr - 60 * hz) < 2, `pure sine ${hz}: ${JSON.stringify(e)}`);
}
let seed = 1;
const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
const series = (pulse, secs = PROBE_SEC + SETTLE_SEC) => Array.from({ length: Math.round(secs * FS) }, (_, i) => {
  const x = 2 * Math.PI * 1.2 * i / FS;
  return [150, 110, 80].map((v, c) => v * (1 + pulse * [1, 1.6, 1.3][c] * (Math.sin(x) + .5 * Math.sin(2 * x + 1))) + (rnd() - .5) * (pulse ? .4 : 6));
});
for (let trial = 1; trial <= 20; trial++) {
  seed = trial;
  const noise = scoreSignal(series(0)), pulse = scoreSignal(series(.015));
  assert.ok(pulse.score > noise.score && pulse.score >= SCORE_MIN, `pulse seed ${trial}`);
  assert.ok(noise.score < SCORE_MIN, `noise seed ${trial}: ${noise.score}`);
}
const clippedRed = scoreSignal(series(.015, 8).map((p) => [255, p[1], p[2]]));
assert.ok(clippedRed.score >= SCORE_MIN && clippedRed.channel !== 0);
assert.equal(clippedRed.channelSat[0], 1);
assert.equal(scoreSignal(series(.015).map(() => [255, 255, 255])).reason, 'saturated');
const dim = scoreSignal(series(.015, 8).map((p) => p.map((v) => v / 10)));
assert.notEqual(dim.reason, 'dark', 'dim red signal is not near-black');
assert.ok(dim.score >= SCORE_MIN);
const result = (label, score) => ({ mode: 'torch', ok: true, torchOn: true, label, deviceId: label, score });
assert.equal(chooseMode([result('Back Triple Camera', 95), result('Back Camera', 65)]).label, 'Back Camera');
assert.equal(chooseMode([result('Back Triple Camera', 95), result('Back Camera', 0)]).label, 'Back Triple Camera');
console.log('CAMERA SIGNAL PASS: harmonic replay, pure high HR, 20 noise seeds, red clipping, dim signal, physical preference');

// Replay the actual live Chain through its 60 s calibration, not just its estimator.
const { Chain } = await import('../app/js/engine.js');
const { readFileSync } = await import('node:fs');
const weights = JSON.parse(readFileSync(new URL('../app/data/weights.json', import.meta.url)));
for (const scenario of [
  { name: 'dominant second harmonic', hz: 1.2, fundamental: .2, second: 1, redClipped: false },
  { name: 'clipped red, healthy green', hz: 1.2, fundamental: 1, second: .3, redClipped: true },
  { name: 'genuine fast pulse', hz: 2.4, fundamental: 1, second: 0, redClipped: false },
]) {
  const chain = new Chain(weights, { hrGate: guarded }), rows = [];
  for (let i = 0; i < 120 * FS; i++) {
    const x = 2 * Math.PI * scenario.hz * i / FS;
    const w = scenario.fundamental * Math.sin(x) + scenario.second * Math.sin(2 * x + .3);
    const mod = 1 + .002 * Math.sin(2 * Math.PI * .2 * i / FS);
    const rgb = [150, 110, 80].map((v, c) => v * mod * (1 + [.006, .01, .008][c] * w));
    if (scenario.redClipped) rgb[0] = 255;
    const d = chain.push(...rgb);
    if (d && !d.calibrating) rows.push(d);
  }
  const accepted = rows.filter((d) => d.valid && Number.isFinite(d.hr));
  assert.ok(accepted.length >= 10, `${scenario.name}: only ${accepted.length}/${rows.length} valid post-calibration windows`);
  assert.ok(accepted.every((d) => Math.abs(d.hr - scenario.hz * 60) < 3), `${scenario.name}: ${accepted.map((d) => d.hr).join(',')}`);
  if (scenario.redClipped) assert.ok(accepted.every((d) => d.ch !== 0), 'clipped red channel never selected');
  console.log(`LIVE CHAIN REPLAY PASS: ${scenario.name}, ${accepted.length}/${rows.length} valid, HR ${accepted[0].hr.toFixed(1)}`);
}

assert.equal(ringState([20, 2, 1]).covered, true, 'dim red tissue ring agrees with placement');
assert.equal(scoreSignal(series(.015).map(() => [255, 1, 0])).reason, 'saturated', 'clipped red with no usable other channel prompts pressure/exposure change');
