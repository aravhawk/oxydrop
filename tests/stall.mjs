// Frame-stall accounting and gating. Usage: node tests/stall.mjs
// 1. Resampler: a stall is recorded with its duration (not just one "gap" event).
// 2. End to end with the shipped chain and decision rule: a clean synthetic pulse with a stall of 0 / 0.5 / 3 / 6 s must make every
//    decision window that overlaps the stall UNSURE (frame_gaps), and a clean window after it must recover to NORMAL.
// 3. Staleness (camera_stalled), calibration stall rule inputs, and decision gating on the new context fields.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Resampler } from '../app/js/resample.js';
import { Chain, FS, WIN } from '../app/js/engine.js';
import { decide, hrToShow, gapContext, stalledDecision, qualityReason, STALL_MS, GAP_MAX_S, GAP_TOTAL_S } from '../app/js/decision.js';
const W = JSON.parse(fs.readFileSync(new URL('../app/data/weights.json', import.meta.url)));
let n = 0; const t = (name, fn) => { fn(); n++; console.log('ok  ', name); };

let seed = 7; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296), gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
/** Frames [t, r, g, b] at 30 fps for dur seconds, 72 bpm pulse with a second harmonic; frames with stallAt <= t < stallAt + stall are removed. */
function frames(dur, stallAt = 0, stall = 0, fps = 30) {
  const out = [], f = 1.2;
  for (let i = 0; i < dur * fps; i++) {
    const tt = i / fps; if (stall > 0 && tt >= stallAt && tt < stallAt + stall) continue;
    const x = 2 * Math.PI * f * tt, w = Math.sin(x) + 0.5 * Math.sin(2 * x + 1);
    out.push([tt + gauss() * 0.002, 150 * (1 + 0.006 * w) + gauss() * 0.15, 110 * (1 + 0.010 * w) + gauss() * 0.15, 80 * (1 + 0.008 * w) + gauss() * 0.15]);
  }
  return out;
}

// ---- 1. resampler accounting
t('no stall: no gap spans', () => { const rs = new Resampler(() => {}); for (const fr of frames(40)) rs.push(...fr); assert.equal(rs.gapSpans.length, 0); assert.equal(rs.gapStats(0).max, 0); });
for (const stall of [0.5, 3, 6]) {
  t(`stall ${stall} s is recorded with its duration`, () => {
    const rs = new Resampler(() => {}); for (const fr of frames(60, 30, stall)) rs.push(...fr);
    assert.equal(rs.gaps, 1, 'one gap event');
    const g = rs.gapStats(0); assert.equal(g.count, 1);
    assert.ok(Math.abs(g.max - stall) < 0.1, `max ${g.max}`); assert.ok(Math.abs(g.total - stall) < 0.15, `total ${g.total}`);
    // output samples keep arriving at 30 Hz across the stall (interpolated), so the span must cover ~stall*30 samples
    const s = rs.gapSpans[0]; assert.ok(Math.abs((s.end - s.start) - stall * FS) <= 3);
  });
}
t('gapStats only counts gaps overlapping the requested tail', () => {
  const rs = new Resampler(() => {}); for (const fr of frames(100, 20, 3)) rs.push(...fr);
  assert.equal(rs.gapStats(rs.n - 10 * FS).max, 0);          // stall ended ~77 s ago
  assert.ok(rs.gapStats(0).max > 2.9);
});
t('isolated single dropped frames stay below the gating thresholds', () => {
  const rs = new Resampler(() => {}); const fr = frames(60).filter((_, i) => i % 40 !== 7);   // 1 frame in 40 dropped
  for (const f of fr) rs.push(...f);
  const g = gapContext(rs); assert.ok(g.gapMax <= GAP_MAX_S && g.gapTotal <= GAP_TOTAL_S, JSON.stringify(g));
});

// ---- 2. end to end: shipped engine + decision
function run(stall, stallAt = 90, dur = 140) {
  const chain = new Chain(W), res = [], rs = new Resampler((r, g, b) => {
    const d = chain.push(r, g, b);
    if (d && !d.calibrating) { const ctx = { fps: 30, finger: true, moving: false, calOk: true, recentGaps: 0, ...gapContext(rs) }; res.push({ d, dec: decide(d, ctx), end: chain.n / FS }); }
  });
  for (const fr of frames(dur, stallAt, stall)) rs.push(...fr);
  return { res, rs };
}
const clean = run(0);
t('clean pulse: windows are answered NORMAL (the test signal is usable)', () => {
  const late = clean.res.filter((x) => x.end > 80), ok = late.filter((x) => x.dec.state === 'NORMAL').length;
  assert.ok(late.length > 20 && ok / late.length >= 0.9, `${ok}/${late.length}`);
  assert.equal(clean.res.filter((x) => x.dec.reason === 'frame_gaps').length, 0);
});
for (const stall of [3, 6]) {
  t(`stall ${stall} s from t = 90 s: every window overlapping it is UNSURE (frame_gaps); the window after it recovers`, () => {
    const { res } = run(stall);
    const over = res.filter((x) => x.end > 90 && x.end - WIN < 90 + stall + 1e-6);      // 10 s window [end-10, end] overlaps [90, 90+stall]
    assert.ok(over.length >= 3, `${over.length} overlapping windows`);
    for (const x of over) { assert.equal(x.dec.state, 'UNSURE', `window ending ${x.end}`); assert.equal(x.dec.reason, 'frame_gaps'); assert.equal(x.dec.hr, null, 'no heart-rate number from a stalled window'); }
    // gap leaves the 10 s lookback, then the 30 s heart-rate span; after both the answer must be NORMAL again (never stuck UNSURE)
    const last = res.filter((x) => x.end >= 90 + stall + 34);
    assert.ok(last.length >= 2 && last.every((x) => x.dec.reason !== 'frame_gaps'), 'recovered');
  });
}
t('a 0.5 s hiccup is below the 1 s single-gap limit and does not flag frame_gaps', () => {
  const { res } = run(0.5); assert.equal(res.filter((x) => x.dec.reason === 'frame_gaps').length, 0);
});

// ---- 3. gating on the new context fields
const good = (o = {}) => ({ calibrating: false, valid: true, sat: 0, hr: 72, label: 'normal', p: 0.1, ...o });
const ok = (o = {}) => ({ fps: 30, recentGaps: 0, gapMax: 0, gapTotal: 0, stalledMs: 0, finger: true, moving: false, calOk: true, ...o });
t('gapMax above 1 s -> frame_gaps, exactly 1 s passes', () => { assert.equal(decide(good(), ok({ gapMax: 1.01 })).reason, 'frame_gaps'); assert.equal(decide(good(), ok({ gapMax: 1.0 })).state, 'NORMAL'); });
t('gapTotal above 3 s -> frame_gaps, exactly 3 s passes', () => { assert.equal(decide(good(), ok({ gapTotal: 3.01 })).reason, 'frame_gaps'); assert.equal(decide(good(), ok({ gapTotal: 3.0 })).state, 'NORMAL'); });
t('a LOW label never survives a stall', () => { for (const c of [ok({ gapMax: 5 }), ok({ gapTotal: 5 }), ok({ stalledMs: STALL_MS + 1 })]) assert.equal(decide(good({ label: 'HYPOXIC' }), c).state, 'UNSURE'); });
t('staleness: no frame for > 1.5 s -> camera_stalled and no HR tile; exactly 1.5 s passes', () => {
  const r = decide(good(), ok({ stalledMs: 1600 })); assert.equal(r.state, 'UNSURE'); assert.equal(r.reason, 'camera_stalled'); assert.equal(r.hr, null);
  assert.equal(hrToShow(good(), ok({ stalledMs: 1e9 })), null);
  assert.equal(decide(good(), ok({ stalledMs: STALL_MS })).state, 'NORMAL');
});
t('stalledDecision() is UNSURE with the stall reason and no HR', () => { const r = stalledDecision(); assert.equal(r.state, 'UNSURE'); assert.equal(r.reason, 'camera_stalled'); assert.equal(r.hr, null); assert.equal(r.text, 'Unsure, re-measure'); assert.ok(r.detail.length > 20); });
t('stall outranks the other reasons (it is the most basic fault)', () => { assert.equal(qualityReason(good({ valid: false }), ok({ stalledMs: 5000, finger: false })), 'camera_stalled'); });

console.log(`STALL PASS (${n} checks)`);
