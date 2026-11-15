// Hybrid engine tests (app/js/hybrid_engine.js, the app's main inference path):
//  1. combiner: the exported meta-learner reproduces the Python pipeline logit on tests/fixtures/hybrid_ref.json (537 windows)
//  2. conformal: the three-state rule reproduces app/data/conformal.json 'examples' for every alpha, and the shipped thresholds are the documented ones
//  3. streaming engine on the two Hoffman fixture recordings vs python/oxydrop_ml/make_engine_ref.py (PyTorch checkpoints of the exported members,
//     numpy features, the v1 chain of python/oxydrop): chain logit c, CNN correction, CNN HR, baseline embedding, decision logit z, every window
//  4. decision parity: decide() (decision.js, unchanged quality gates) in front of the conformal state; HR rule 'agree'
//  5. latency per window in Node and the asset sizes
// Usage: node tests/hybrid_engine.mjs
import fs from 'node:fs';
import { HybridEngine, OOD_LAC_MIN, loadHybridAssets, metaLogit, conformalState, hrAgree, chainLogit } from '../app/js/hybrid_engine.js';
import { decide, STATE } from '../app/js/decision.js';

const root = new URL('../', import.meta.url);
const rd = (p) => JSON.parse(fs.readFileSync(new URL(p, root)));
let fail = false;
const check = (cond, msg) => { if (!cond) { console.log('FAIL', msg); fail = true; } };

const assets = await loadHybridAssets(async (rel) => rd('app/' + rel));
const { hybrid, conformal } = assets;

// ---- 1. combiner
{
  const ref = rd('tests/fixtures/hybrid_ref.json');
  check(ref.source === 'R' && !ref.calibration && ref.k === 1, 'fixture pipeline must be R / no calibration / k=1');
  let mx = 0;
  ref.rows.forEach((r, i) => { mx = Math.max(mx, Math.abs(metaLogit(hybrid.pipeline, r.c + r.corr) - ref.expected_logit[i])); });
  console.log(`combiner: ${ref.rows.length} windows, max|z_js - z_python| = ${mx.toExponential(2)}`);
  check(mx < 1e-9, 'combiner parity');
  check(Math.abs(chainLogit(0.5)) < 1e-12 && chainLogit(0) === Math.log(1e-6 / (1 - 1e-6)), 'chain logit clip');
}

// ---- 2. conformal
{
  let n = 0;
  for (const ex of conformal.examples) for (const [a, want] of Object.entries(ex.state)) {
    if (conformal.alphas[a].past_crossing) continue;      // documented as not usable; the engine refuses it
    n++; check(conformalState(conformal, ex.z, Number(a)) === want, `conformal z=${ex.z} alpha=${a}: ${conformalState(conformal, ex.z, Number(a))} != ${want}`);
  }
  const t = conformal.alphas['0.10'];
  check(conformal.default_alpha === 0.1 && Math.abs(t.normal_below + 0.9653109532046704) < 1e-12 && Math.abs(t.low_above - 0.389431721484113) < 1e-12, 'default alpha thresholds');
  let threw = false; try { conformalState(conformal, 0, 0.20); } catch (e) { threw = true; }
  check(threw, 'past-crossing alpha must be refused');
  check(conformalState(conformal, NaN) === 'UNSURE', 'NaN logit is UNSURE');
  check(hrAgree(72, 75) === 73.5 && hrAgree(72, 80) === 72 && hrAgree(72, NaN) === 72 && hrAgree(NaN, 70) === 70, 'HR rule agree');
  console.log(`conformal: ${n} example states match; alpha 0.10 thresholds (${t.normal_below.toFixed(3)}, ${t.low_above.toFixed(3)})`);
}

// ---- 3 + 4. streaming engine on the two fixture recordings
const ref = rd('tests/fixtures/hybrid_engine_ref.json');
{   // the fixture was generated from exactly the checkpoints whose hashes are recorded in the shipped weights (closes the link checkpoint -> fixture -> app/data)
  const want = Object.fromEntries(hybrid.members.filter((m) => m.family === 'R').map((m) => [`R_s${m.seed}`, m.ckpt_sha256]));
  check(Object.keys(want).length === 3 && Object.entries(want).every(([k, v]) => ref.ckpt_sha256 && ref.ckpt_sha256[k] === v), 'hybrid_engine_ref.json ckpt_sha256 must equal the ckpt_sha256 of every shipped member');
}
const TOL = { c: 1e-6, corr: 1e-4, hr: 1e-3, z: 5e-4, emb: 1e-4 };
const lat = [];
const summary = {}, syncWins = {}, recRows = {};
for (const [id, side] of [['100001L', 'Left'], ['100004R', 'Right']]) {
  const rows = fs.readFileSync(new URL(`tests/fixtures/ppg-csv/${side}/${id.slice(0, -1)}.csv`, root), 'utf8').trim().split('\n').slice(1).map((l) => l.split(',').map(Number));
  const R = ref.records[id], eng = new HybridEngine(assets, { forceCnn: true }), wins = [];
  for (const [r, g, b] of rows) { const d = eng.push(r, g, b); if (d) wins.push(d); }
  lat.push(...eng.lat); syncWins[id] = wins; recRows[id] = rows;
  const dec = wins.filter((d) => !d.calibrating && d.s >= 60);
  check(dec.length === R.windows.length, `${id}: ${dec.length} decision windows, reference has ${R.windows.length}`);
  const mx = { c: 0, corr: 0, hr: 0, z: 0, emb: 0 };
  R.windows.forEach((w, i) => {
    const d = dec[i]; if (!d || d.s !== w.s) { check(false, `${id} window ${i} start`); return; }
    mx.c = Math.max(mx.c, Math.abs(d.c - w.c)); mx.corr = Math.max(mx.corr, Math.abs(d.corr - w.corr.reduce((a, b) => a + b, 0) / w.corr.length));
    const hs = [...w.hr].sort((a, b) => a - b); mx.hr = Math.max(mx.hr, Math.abs(d.cnnHr - hs[1]));
    mx.z = Math.max(mx.z, Math.abs(d.z - w.z));
  });
  eng.embBase.forEach((e, k) => { for (let j = 0; j < e.length; j++) mx.emb = Math.max(mx.emb, Math.abs(e[j] - R.emb_base[k][j])); });
  console.log(`${id}: ${dec.length} windows  max|c| ${mx.c.toExponential(1)}  |corr| ${mx.corr.toExponential(1)}  |cnn HR| ${mx.hr.toExponential(1)} bpm  |z| ${mx.z.toExponential(1)}  |emb_base| ${mx.emb.toExponential(1)}`);
  for (const k of Object.keys(TOL)) check(mx[k] < TOL[k], `${id} ${k} parity ${mx[k]} >= ${TOL[k]}`);

  // 4. decision parity with the production engine and decide()
  const pw = wins.filter((d) => !d.calibrating);
  let nValid = 0, nNormal = 0, nLow = 0, nUnsure = 0, right = 0, wrong = 0, hrBad = 0;
  const rp = rd(`app/data/replay/${id}.json`).spo2, refAt = (s0) => { const g = rp.slice(s0, s0 + 10).filter((v) => v != null); return g.length >= 5 ? g.reduce((a, b) => a + b, 0) / g.length : null; };   // study reference, as the replay UI uses it
  pw.forEach((d, i) => {
    const q = decide(d, {});
    if (!d.valid) { check(q.state === STATE.UNSURE && q.hr === null && d.label === 'nolock', `${id} invalid window must be UNSURE without HR`); return; }
    nValid++;
    check(q.state === conformalState(conformal, d.z) && d.label === { NORMAL: 'normal', LOW: 'HYPOXIC', UNSURE: 'unsure' }[q.state], `${id} state/label mismatch at s=${d.s}`);
    check(Math.abs(d.p - 1 / (1 + Math.exp(-d.z))) < 1e-12 && Number.isFinite(d.hr), `${id} p/hr at s=${d.s}`);
    check(d.hr === hrAgree(d.cnnHr, d.specHr), `${id} HR rule at s=${d.s}`);
    if (q.state === STATE.NORMAL) nNormal++; else if (q.state === STATE.LOW) nLow++; else nUnsure++;
    const rf = refAt(d.s); if (q.state !== STATE.UNSURE && rf != null) ((q.state === STATE.LOW) === (rf < 90)) ? right++ : wrong++;
  });
  summary[id] = { windows: pw.length, valid: nValid, normal: nNormal, low: nLow, unsure: nUnsure, answeredCorrect: `${right}/${right + wrong}` };
  console.log(`${id}: engine ${pw.length} windows, ${nValid} locked, ${nNormal} normal / ${nLow} possible low oxygen / ${nUnsure} unsure (in-sample for the exported model: not a test), answered correct vs the study reference ${right}/${right + wrong}`);
}
check(summary['100001L'].valid > 100, 'person 1 must lock on many windows');

// ---- 5b. generic low-amplitude guard (a far-from-the-lens flash): attenuate the pulse of person 1 and check what the engine does. Not tuned on any phone:
//          the cut is the training range (scripts/ood_range.py)
{
  const rows = recRows['100001L'].slice(0, 150 * 30), N = rows.length;
  const dc = rows.map((_, i) => { const a = Math.max(0, i - 90), b = Math.min(N, i + 90), s = [0, 0, 0]; for (let j = a; j < b; j++) for (let c = 0; c < 3; c++) s[c] += rows[j][c]; return s.map((v) => v / (b - a)); });
  const out = {};
  for (const k of [1, 0.03]) {
    const eng = new HybridEngine(assets), ws = [];
    rows.forEach((r, i) => { const d = eng.push(...r.map((v, c) => dc[i][c] + k * (v - dc[i][c]))); if (d && !d.calibrating) ws.push(d); });
    out[k] = ws;
  }
  const locked = out[1].filter((d) => d.valid);
  check(locked.length > 30 && locked.every((d) => d.label !== 'low_ac' && Number.isFinite(d.lacMax) && d.lacMax > OOD_LAC_MIN), 'unattenuated windows are inside the trained amplitude range');
  const weak = out[0.03].filter((d) => d.valid);
  const refused = weak.filter((d) => d.label === 'low_ac');
  check(weak.length === 0 || (refused.length === weak.length && refused.every((d) => decide(d, {}).state === STATE.UNSURE && decide(d, {}).reason === 'low_ac' && Number.isFinite(decide(d, {}).hr))), 'weak-pulse windows that still lock must be refused as low_ac with the spectral HR only');
  console.log(`low-AC guard: unattenuated ${locked.length} locked windows, lac_max ${Math.min(...locked.map((d) => d.lacMax)).toFixed(2)} to ${Math.max(...locked.map((d) => d.lacMax)).toFixed(2)} (cut ${OOD_LAC_MIN}); pulse x0.03: ${weak.length} still locked, ${refused.length} refused as low_ac, ${out[0.03].length - weak.length} no lock`);
}

// ---- 6. the live path: main-thread quality chain + Web Worker (app/js/hybrid_worker.js, hybrid_async.js) through a loopback Worker, first 130 s of person 1
{
  const { HybridWorker, HybridWorkerEngine, loadHybridLite } = await import('../app/js/hybrid_async.js');
  const loop = { w: null };
  globalThis.self = { postMessage: (m) => setTimeout(() => loop.w.onmessage({ data: m }), 0) };
  globalThis.fetch = async (u) => ({ ok: true, json: async () => JSON.parse(fs.readFileSync(u)) });
  globalThis.Worker = class { constructor() { loop.w = this; } postMessage(m) { setTimeout(() => globalThis.self.onmessage({ data: m }), 0); } terminate() {} };
  await import('../app/js/hybrid_worker.js');
  const lite = await loadHybridLite(async (rel) => rd('app/' + rel));
  const worker = new HybridWorker('loopback', lite.conformal.default_alpha); await worker.ready;
  const got = [];
  const eng = new HybridWorkerEngine(lite, worker, lite.conformal.default_alpha, (d) => got.push(d));
  const rows = recRows['100001L'].slice(0, 130 * 30);
  for (let i = 0; i < rows.length; i++) { const d = eng.push(...rows[i]); if (d) got.push(d); if (i % 60 === 59) await new Promise((r) => setTimeout(r, 0)); }
  for (let k = 0; k < 600 && eng.inFlight > 0; k++) await new Promise((r) => setTimeout(r, 10));
  check(eng.inFlight === 0, 'worker results all delivered');
  const want = syncWins['100001L'].slice(0, got.length);
  check(got.length === 61, `worker path delivered ${got.length} windows (expected 61)`);
  let mx = 0, lab = 0;
  got.sort((a, b) => a.s - b.s).forEach((d, i) => { const w = want[i]; if (!w || w.s !== d.s) { check(false, `worker window ${i} start`); return; } if (!d.calibrating) { mx = Math.max(mx, Math.abs(d.z - w.z), Math.abs(d.c - w.c), Math.abs(d.cnnHr - w.cnnHr), Math.abs(d.hr - w.hr)); if (d.label !== w.label) lab++; } });
  console.log(`worker path: ${got.length} windows, max|worker - in-thread| = ${mx.toExponential(1)}, label mismatches ${lab}, model ${eng.latency().coreMedian.toFixed(1)} ms in the worker`);
  check(mx < 1e-9 && lab === 0, 'worker engine must equal the in-thread engine');
  // a stale generation (new recording) must not deliver old results
  eng.close();
}

// ---- 5. latency
lat.sort((a, b) => a - b);
const med = lat[lat.length >> 1], p95 = lat[Math.floor(lat.length * 0.95)];
const kb = (f) => (fs.statSync(new URL('app/data/' + f, root)).size / 1024).toFixed(0);
console.log(`LATENCY hybrid engine per 2 s window (v2 chain + v1 chain + features + ${assets.models.length} CNN forwards, with forceCnn so every window runs the CNN): median ${med.toFixed(1)} ms, p95 ${p95.toFixed(1)} ms over ${lat.length} windows (Node ${process.version})`);
console.log('ASSETS KB:', ['hybrid_weights.json', 'conformal.json', 'weights.json', ...assets.members.map((m) => m.weights)].map((f) => `${f} ${kb(f)}`).join(', '));
console.log(`MODEL ${assets.models[0].params} parameters per member x ${assets.models.length} seeds`);
if (fail) { console.log('HYBRID ENGINE FAIL'); process.exit(1); } else console.log('HYBRID ENGINE PASS');
