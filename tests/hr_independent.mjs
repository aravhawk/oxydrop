import assert from 'node:assert/strict';
import test from 'node:test';
import { estimateHR } from '../src/hr_independent.js';

function frames(redHz, greenHz = redHz, fundamental = 1, secondHarmonic = 0) {
  return Array.from({ length: 250 }, (_, i) => {
    const timestampMs = i * 40 + (i % 5 - 2) * 1.7;
    const seconds = timestampMs / 1000;
    const pulse = (hz) => fundamental * Math.sin(2 * Math.PI * hz * seconds) +
      secondHarmonic * Math.sin(4 * Math.PI * hz * seconds);
    return {
      timestampMs,
      red: 160 + 0.7 * seconds + 5 * pulse(redHz),
      green: 60 + 0.5 * seconds + 3 * pulse(greenHz),
    };
  });
}

function assertRejected(input) {
  const result = estimateHR(input);
  assert.equal(result.bpm, null);
  assert.equal(result.method, 're-measure');
  assert.ok(result.sqi >= 0 && result.sqi < 0.6);
}

test('uses jittered frame timestamps at 25 Hz and tracks both band edges', () => {
  for (const bpm of [42, 72, 144, 210]) {
    const result = estimateHR(frames(bpm / 60));
    assert.equal(result.method, 'autocorrelation+spectral');
    assert.ok(Math.abs(result.bpm - bpm) <= 1);
    assert.ok(result.sqi >= 0.6 && result.sqi <= 1);
  }
});

test('reports one beat per cycle with a normal harmonic and withholds an ambiguous rate', () => {
  assert.equal(estimateHR(frames(1.2, 1.2, 1, 0.6)).bpm, 72);
  assertRejected(frames(1.2, 1.2, 0.3, 1));
});

test('withholds an estimate when strong channels report different rates', () => {
  assertRejected(frames(1.2, 2.4));
});

test('withholds an estimate for linear drift and large timestamp gaps', () => {
  assertRejected(frames(1.2, 1.2, 0));
  assertRejected(frames(1.2).filter((_, i) => i < 40 || i > 80));
});

test('withholds an estimate for duplicate or malformed timestamps', () => {
  const input = frames(1.2);
  input[10] = input[9];
  assertRejected(input);
  input[10] = { ...input[11], timestampMs: Infinity };
  assertRejected(input);
});
