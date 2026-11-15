// OxyNet parity: JS forward pass (app/js/cnn.js) vs PyTorch outputs, and streaming input features (app/js/cnn_features.js) vs the numpy reference.
// Weights = member 0 of the shipped hybrid (app/data/hybrid_R_s0.json); the fixture is written by python/oxydrop_ml/export_js.py --ckpt R_s0.pt. Usage: node tests/cnn_parity.mjs [weights.json] [cnn_ref.json]
import fs from 'node:fs';
import { loadModel, forward, decodeHr } from '../app/js/cnn.js';
import { windowsFromRgb, N_CH, N_SIDE, WIN } from '../app/js/cnn_features.js';

const wpath = process.argv[2] ?? new URL('../app/data/hybrid_R_s0.json', import.meta.url);
const rpath = process.argv[3] ?? new URL('./fixtures/cnn_ref.json', import.meta.url);
const model = loadModel(JSON.parse(fs.readFileSync(wpath)));
const ref = JSON.parse(fs.readFileSync(rpath)), TOL = ref.tolerance ?? 1e-4;
const f32 = (b64) => new Float32Array(Uint8Array.from(Buffer.from(b64, 'base64')).buffer);
let fail = false, worst = 0;
const check0 = (cond, msg) => { if (!cond) { console.log('FAIL', msg); process.exitCode = 1; } };
if (process.argv[2] === undefined && process.argv[3] === undefined) {   // default files: the fixture must come from the checkpoint of shipped member R_s0
  const hw = JSON.parse(fs.readFileSync(new URL('../app/data/hybrid_weights.json', import.meta.url)));
  const m0 = hw.members.find((m) => m.family === 'R' && m.seed === 0);
  check0(ref.ckpt_sha256 === m0.ckpt_sha256, `cnn_ref.json ckpt_sha256 ${ref.ckpt_sha256} must equal the shipped member R_s0 (${m0.ckpt_sha256})`);
}
const check = (cond, msg) => { if (!cond) { console.log('FAIL', msg); fail = true; } };

// ---- 1. every head, every fixture case
const heads = ['hr_logits', 'spo2', 'spo2_logvar', 'lowox_logit', 'sqi_logit', 'emb'];
const perHead = Object.fromEntries(heads.map((h) => [h, 0]));
for (const c of ref.cases) {
  const o = forward(model, f32(c.x), { domain: c.domain, side: c.side, embBase: c.emb_base });
  let mx = 0;
  for (const h of heads) {
    const a = typeof o[h] === 'number' ? [o[h]] : o[h], b = typeof c.out[h] === 'number' ? [c.out[h]] : c.out[h];
    check(a.length === b.length, `${c.name}.${h} length`);
    let d = 0; for (let i = 0; i < b.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
    perHead[h] = Math.max(perHead[h], d); mx = Math.max(mx, d);
  }
  worst = Math.max(worst, mx);
  console.log(`  ${c.name.padEnd(18)} domain ${c.domain}  max|JS-torch| = ${mx.toExponential(2)}  HR(argmax)=${decodeHr(o.hr_logits, 40, 'argmax')}`);
  check(mx < TOL, `${c.name} exceeds ${TOL}`);
}
console.log('per head worst:', heads.map((h) => `${h} ${perHead[h].toExponential(1)}`).join('  '));

// ---- 2. the output must depend on the domain, the side vector and the baseline embedding (guards against ignored inputs)
{
  const c = ref.cases[0], x = f32(c.x);
  const a = forward(model, x, { domain: 0, side: c.side, embBase: c.emb_base }), lo0 = a.lowox_logit, sp0 = a.spo2, h0 = a.hr_logits[0];
  const b = forward(model, x, { domain: 1, side: c.side, embBase: c.emb_base });
  if (model.cfg.stem === 'shared') check(Math.abs(b.hr_logits[0] - h0) > 1e-6, 'domain has no effect');   // the joint stem has no domain input
  const e = forward(model, x, { domain: 0, side: c.side, embBase: c.emb_base.map((v) => v + 1) });
  check(Math.abs(e.lowox_logit - lo0) > 1e-6 && e.spo2 === sp0, 'baseline embedding must change only the low-oxygen head');
  const s = forward(model, x, { domain: 0, side: c.side.map((v) => v + 1), embBase: c.emb_base });
  check(Math.abs(s.spo2 - sp0) > 1e-6, 'side vector has no effect');
}

// ---- 3. latency per 10 s window (single thread, includes nothing but forward())
{
  const c = ref.cases[0], x = f32(c.x), opt = { domain: 0, side: c.side, embBase: c.emb_base };
  for (let i = 0; i < 20; i++) forward(model, x, opt);
  const N = 100, lat = [];
  for (let i = 0; i < N; i++) { const t0 = performance.now(); forward(model, x, opt); lat.push(performance.now() - t0); }
  lat.sort((a, b) => a - b);
  console.log(`LATENCY forward() per 10 s window: median ${lat[N >> 1].toFixed(1)} ms, p95 ${lat[Math.floor(N * 0.95)].toFixed(1)} ms, params ${model.params} (target < 30 ms, laptop, Node ${process.version})`);
}

// ---- 4. streaming features vs numpy reference on a recorded Hoffman file
const feat = ref.features;
if (feat) {
  const rows = fs.readFileSync(new URL(`./fixtures/ppg-csv/${feat.record}`, import.meta.url), 'utf8').trim().split('\n').slice(1, 1 + feat.n_samples).map((l) => l.split(',').map(Number));
  const dtPat = (i) => (1 / 30) * (1 + 0.5 * (i % 11 === 0) + 3 * (i % 401 === 0));
  for (const [vn, V] of Object.entries(feat.variants)) {
    const dt = vn === 'irregular_dt' ? rows.map((_, i) => dtPat(i)) : null;
    const mask = V.mask ? V.mask.map(Number) : undefined;
    let src = rows;
    if (V.rgb_f32) { const f = f32(V.rgb_f32); src = Array.from({ length: f.length / 3 }, (_, i) => [f[3 * i], f[3 * i + 1], f[3 * i + 2]]); }
    if (vn === 'contact_mask') src = rows.map((r) => [r[0], r[1], 0]);
    const t0 = performance.now(), W = windowsFromRgb(src, dt, mask), ms = performance.now() - t0;
    check(W.length === V.te.length, `${vn}: ${W.length} windows vs ${V.te.length}`);
    let dm = 0, ds = 0, da = 0, dfull = 0, flags = 0;
    const nn = (u) => (u === null ? NaN : u), dif = (p, q) => (Number.isNaN(p) && Number.isNaN(q) ? 0 : Math.abs(p - q));
    const nBase = V.base.some((u) => u !== null);
    if (nBase !== !!W[W.length - 1].base) flags++;
    if (nBase) V.base.forEach((u, i) => { ds = Math.max(ds, dif(nn(u), W[W.length - 1].base[i])); });
    W.forEach((w, k) => {
      if (w.te !== V.te[k] || w.calibrating !== V.calibrating[k]) flags++;
      for (let c = 0; c < N_CH; c++) {
        let m = 0, a = 0; for (let t = 0; t < WIN; t++) { const v = w.x[c * WIN + t]; m += v; a += Math.abs(v); }
        dm = Math.max(dm, Math.abs(m / WIN - V.ch_mean[k][c]), Math.abs(a / WIN - V.ch_absmean[k][c]));
      }
      for (let i = 0; i < N_SIDE; i++) ds = Math.max(ds, dif(w.side[i], nn(V.side[k][i])));
      for (let i = 0; i < w.abs.length; i++) da = Math.max(da, dif(w.abs[i], nn(V.abs[k][i])));
    });
    for (const fw of V.full_windows) { const r = f32(fw.x), x = W[fw.k].x; for (let i = 0; i < r.length; i++) dfull = Math.max(dfull, Math.abs(x[i] - r[i])); }
    const nBeats = W.reduce((a, w) => a + w.abs[9], 0), okFrac = W.reduce((a, w) => a + w.abs[8], 0) / W.length;
    console.log(`  features ${vn.padEnd(12)} ${W.length} windows (${nBeats} beats, beat_ok ${(100 * okFrac).toFixed(0)} %)  max|channel stat diff|=${dm.toExponential(1)}  side/base=${ds.toExponential(1)}  abs=${da.toExponential(1)}  full-window max|diff|=${dfull.toExponential(1)}  flag mismatches=${flags}  (${(ms / W.length).toFixed(2)} ms/window)`);
    check(dm < 1e-4 && ds < 1e-4 && da < 1e-3 && dfull < 1e-4 && flags === 0 && (nBeats > 100 || vn === 'noise'), `${vn}: feature mismatch`);
    // end-to-end: JS features -> JS network == torch on the recorded-window model cases
    if (V.model_cases) for (const c of V.model_cases) {
      const k = +c.name.slice(c.name.lastIndexOf('_k') + 2), o = forward(model, W[k].x, { domain: c.domain, side: W[k].side, embBase: c.emb_base });
      const d = Math.max(Math.abs(o.spo2 - c.out.spo2), Math.abs(o.lowox_logit - c.out.lowox_logit), Math.abs(o.sqi_logit - c.out.sqi_logit));
      check(d < 1e-3, `end-to-end ${c.name} ${d}`);
    }
  }
}
console.log(`TOTAL worst max|JS-torch| over all heads and cases = ${worst.toExponential(2)} (limit ${TOL})`);
if (fail || worst >= TOL) { console.log('CNN PARITY FAIL'); process.exit(1); } else console.log('CNN PARITY PASS');
