// Capture-path simulation: synthetic fingertip PPG (strong dicrotic notch) -> jittery/60 fps/dropped frames -> Resampler -> chain.
// Compares the v1 beat-interval HR feature with the v2 robust estimate. Usage: node tests/capture_sim.mjs
import { Resampler } from '../app/js/resample.js';
import { Chain as ChainV2 } from '../app/js/engine.js';
import { Chain as ChainV1 } from '../app/js/engine_v1.js';
import fs from 'node:fs';
const W2 = JSON.parse(fs.readFileSync(new URL('../app/data/weights.json', import.meta.url))), W1 = JSON.parse(fs.readFileSync(new URL('../python/tiny_chain_v1_weights.json', import.meta.url)));
let seed = 12345; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296), gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
function ppg(t, f, a2, ph) { const x = 2 * Math.PI * f * t; return Math.sin(x) + a2 * Math.sin(2 * x + ph); }
function trial({ hr, a2, fps, jitterMs, dropP, expStep, noise }) {
  const f = hr / 60, dur = 130, frames = [], ph = rnd() * 6;
  for (let i = 0; i < dur * fps; i++) {
    if (rnd() < dropP) continue;
    const t = i / fps, ts = t + gauss() * jitterMs / 1000;
    let g = 1 + (expStep && t > 80 ? 0.08 : 0);                 // auto-exposure step at 80 s
    const w = ppg(t, f, a2, ph), dcr = 150 * g, dcg = 110 * g, dcb = 80 * g;
    frames.push([ts, dcr * (1 + 0.006 * w) + gauss() * noise, dcg * (1 + 0.010 * w) + gauss() * noise, dcb * (1 + 0.008 * w) + gauss() * noise]);
  }
  const res = {};
  for (const [name, C, Wt] of [['v1', ChainV1, W1], ['v2', ChainV2, W2]]) {
    const chain = new C(Wt), rs = new Resampler((r, g, b) => { const d = chain.push(r, g, b); if (d && d.s >= 40) (res[name] ||= []).push(d); });
    for (const fr of frames) rs.push(...fr);
    res.rs = rs;
  }
  return res;
}
const cases = [];
for (const hr of [55, 70, 85, 100]) for (const a2 of [0.4, 0.8, 1.2]) for (const [fps, jit, drop] of [[30, 3, 0], [60, 4, 0], [60, 6, 0.03], [24, 3, 0]]) cases.push({ hr, a2, fps, jitterMs: jit, dropP: drop, expStep: true, noise: 0.25 });
const S = { v1: { n: 0, dbl: 0, ok5: 0, err: [] }, v2: { n: 0, dbl: 0, ok5: 0, err: [], valid: 0 } }; let gapsSum = 0;
for (const c of cases) {
  const r = trial(c); gapsSum += r.rs.gaps;
  for (const d of r.v1 || []) { const h = d.hr; if (!Number.isFinite(h)) continue; S.v1.n++; const e = Math.abs(h - c.hr); S.v1.err.push(e); if (e <= 5) S.v1.ok5++; if (Math.abs(h / c.hr - 2) < 0.12) S.v1.dbl++; }
  for (const d of r.v2 || []) { S.v2.n++; if (!d.valid) continue; S.v2.valid++; const e = Math.abs(d.hrEst - c.hr); S.v2.err.push(e); if (e <= 5) S.v2.ok5++; if (Math.abs(d.hrEst / c.hr - 2) < 0.12) S.v2.dbl++; }
}
const med = (a) => a.sort((x, y) => x - y)[a.length >> 1];
console.log(`trials=${cases.length} (HR 55-100 bpm, dicrotic 2nd-harmonic 0.4-1.2x, 24/30/60 fps, jitter, drops, exposure step)`);
console.log(`v1 beat-interval HR : windows ${S.v1.n}  median err ${med(S.v1.err).toFixed(1)} bpm  within5 ${(S.v1.ok5 / S.v1.n * 100).toFixed(1)}%  DOUBLED ${(S.v1.dbl / S.v1.n * 100).toFixed(1)}%`);
console.log(`v2 robust HR (gated): windows ${S.v2.n}  accepted ${(S.v2.valid / S.v2.n * 100).toFixed(1)}%  median err ${med(S.v2.err).toFixed(1)} bpm  within5 ${(S.v2.ok5 / S.v2.valid * 100).toFixed(1)}% of accepted  DOUBLED ${(S.v2.dbl / S.v2.valid * 100).toFixed(2)}% of accepted`);
