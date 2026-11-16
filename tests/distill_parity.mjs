// PhotonDistill parity: app/js/distill_features.js + distill_engine.js (and the engine slot of photon_engine.js) against the Python reference
// tests/fixtures/ref_distill.json (python/oxydrop_ml/photon/distill_app_export.py: flashgap.rec_features + the app weight file in numpy).
// Recordings: tests/fixtures/ppg-csv/Left/100001.csv, Right/100004.csv and tests/fixtures/photon_L4_100001L.csv (stress level 4, simulated).
// Tolerances: every feature 1e-6, probability 1e-5, SpO2 estimate 1e-4, abstain flag exact. Also: causality, three-state mapping, latency.
import fs from 'node:fs';
import { DistillNet, DistillCore } from '../app/js/distill_engine.js';
import { COLS } from '../app/js/distill_features.js';
import { PhotonEngine, finishPhoton, DEFAULT_PHOTON_MODEL } from '../app/js/photon_engine.js';
import { decide, STATE } from '../app/js/decision.js';

const root = new URL('../', import.meta.url);
const ref = JSON.parse(fs.readFileSync(new URL('tests/fixtures/ref_distill.json', root)));
const model = JSON.parse(fs.readFileSync(new URL('app/data/distill.json', root)));
const v2Weights = JSON.parse(fs.readFileSync(new URL('app/data/weights.json', root)));
const TOL_X = 1e-6, TOL_P = 1e-5, TOL_S = 1e-4;
let fail = false;
const check = (cond, msg) => { if (!cond) { console.log('FAIL', msg); fail = true; } };
const loadCsv = (rel) => fs.readFileSync(new URL('tests/fixtures/' + rel, root), 'utf8').trim().split('\n').slice(1).map((l) => l.split(',').map(Number));

check(DEFAULT_PHOTON_MODEL === 'distill', `default model of the engine slot is ${DEFAULT_PHOTON_MODEL}, expected distill`);
check(ref.cols.length === COLS.length && ref.cols.every((c, i) => c === COLS[i]), 'feature list differs from the reference');

let worstX = 0, worstP = 0, worstS = 0, nWin = 0, nAbsMismatch = 0;
const lat = [];
for (const [name, R] of Object.entries(ref.recordings)) {
  const rgb = loadCsv(R.csv);
  check(rgb.length === R.n_samples, `${name}: ${rgb.length} samples, reference ${R.n_samples}`);
  const core = new DistillCore(model), out = [];
  for (const [r, g, b] of rgb) { const d = core.push(r, g, b); if (d && !d.calibrating) { out.push(d); lat.push(d.ms); } }
  const want = R.t.map((t, i) => i).filter((i) => R.eval[i]);
  check(out.length === want.length, `${name}: ${out.length} model windows, reference ${want.length}`);
  let mx = 0, mp = 0, ms = 0, worstCol = '';
  for (let k = 0; k < Math.min(out.length, want.length); k++) {
    const i = want[k], d = out[k];
    check(d.s === R.t[i], `${name}: window ${k} starts at ${d.s}, reference ${R.t[i]}`);
    for (let j = 0; j < COLS.length; j++) {
      const a = d.x[j], b = R.X[i][j];
      const e = b === null ? (Number.isFinite(a) ? Infinity : 0) : Math.abs(a - b);
      if (!(e <= mx)) { mx = e; worstCol = `${COLS[j]} at s=${d.s} (js ${a}, py ${b})`; }
    }
    mp = Math.max(mp, Math.abs(d.p - R.p[i])); ms = Math.max(ms, Math.abs(d.spo2 - R.spo2[i]));
    if (d.abstain !== !R.answer[i]) { nAbsMismatch++; check(false, `${name}: abstain flag differs at s=${d.s} (conf js ${d.conf}, py ${R.conf[i]}, tau ${model.tau})`); }
    nWin++;
  }
  check(mx < TOL_X, `${name}: worst feature error ${mx.toExponential(2)} >= ${TOL_X}: ${worstCol}`);
  check(mp < TOL_P, `${name}: worst |dp| ${mp.toExponential(2)} >= ${TOL_P}`);
  check(ms < TOL_S, `${name}: worst |dSpO2| ${ms.toExponential(2)} >= ${TOL_S}`);
  worstX = Math.max(worstX, mx); worstP = Math.max(worstP, mp); worstS = Math.max(worstS, ms);
  const ans = out.filter((d) => !d.abstain).length;
  console.log(`  ${name.padEnd(11)} ${out.length} windows  answered ${ans}  max|dx| ${mx.toExponential(2)}  max|dp| ${mp.toExponential(2)}  max|dSpO2| ${ms.toExponential(2)}`);
}
check(nWin > 1000, `only ${nWin} windows compared`);
console.log(`features ${worstX.toExponential(2)} (tol ${TOL_X}), probability ${worstP.toExponential(2)} (tol ${TOL_P}), SpO2 ${worstS.toExponential(2)}, abstain mismatches ${nAbsMismatch}, ${nWin} windows`);

// trained export's own 2-window reference (distill_export.py) through the app file
{
  const src = JSON.parse(fs.readFileSync(new URL(model.source.split(' ')[0], root)));
  const net = new DistillNet(model);
  src.reference.X.forEach((x, i) => {
    const o = net.predict(x);
    check(Math.abs(o.p - src.reference.score[i]) < 1e-6, `weights-file reference window ${i}: p ${o.p} vs ${src.reference.score[i]}`);
    check(o.answer === src.reference.answer[i], `weights-file reference window ${i}: answer differs`);
  });
}

// causality: the first 300 s of a recording give the same windows as the full recording
{
  const R = ref.recordings['100001L'], rgb = loadCsv(R.csv);
  const run = (n) => { const c = new DistillCore(model), o = []; for (const [r, g, b] of rgb.slice(0, n)) { const d = c.push(r, g, b); if (d && !d.calibrating) o.push(d); } return o; };
  const a = run(300 * 30), b = run(rgb.length);
  check(a.length > 100 && a.every((d, k) => d.s === b[k].s && d.p === b[k].p && d.x.every((v, j) => Object.is(v, b[k].x[j]))), 'truncating the recording changed earlier windows (not causal)');
}

// engine slot: three states exactly from the model output
{
  const d2 = { s: 80, calibrating: false, valid: false, sat: 0.5, sqi: 0.1, hr: 70, label: 'nolock' };
  const m = (p, abstain) => ({ p, abstain, pred: p > 0.5, spo2: 88 });
  const st = (h) => decide(finishPhoton(d2, h, 'x')).state;
  check(st(m(0.9, true)) === STATE.UNSURE && st(m(0.1, true)) === STATE.UNSURE, 'abstaining window is not Unsure');
  check(st(m(0.9, false)) === STATE.LOW, 'answered p > 0.5 is not Possible low oxygen (no pulse-lock or saturation mask may be added)');
  check(st(m(0.2, false)) === STATE.NORMAL, 'answered p <= 0.5 is not Normal');
  check(decide(finishPhoton(d2, m(0.9, false), 'x'), { finger: false }).state === STATE.UNSURE, 'no finger must still give Unsure (measurement check)');
  check(decide(finishPhoton(d2, m(0.9, false), 'x')).hr === null, 'no heart rate is shown without the v2 pulse lock');
  check(decide(finishPhoton({ ...d2, valid: true, sat: 0 }, m(0.2, false), 'x')).hr === 70, 'the heart rate is shown when the v2 chain locked');
  // the slot streams the recording and its windows carry the model output unchanged
  const R = ref.recordings['100004R'], rgb = loadCsv(R.csv), e = new PhotonEngine({ name: 'distill', model, v2Weights }), got = [];
  for (const [r, g, b] of rgb) { const d = e.push(r, g, b); if (d && !d.calibrating) got.push(d); }
  const want = R.t.map((t, i) => i).filter((i) => R.eval[i]);
  check(got.length === want.length && got.every((d, k) => Math.abs(d.modelP - R.p[want[k]]) < TOL_P && d.modelAbstain === !R.answer[want[k]]), 'engine slot windows differ from the reference');
  check(got.every((d) => d.label === (d.modelAbstain ? 'unsure' : d.modelP > 0.5 ? 'HYPOXIC' : 'normal')), 'engine slot label is not the model output');
  check(e.info.id === 'oxydrop-photon-distill2-v1', `engine id ${e.info.id}`);
  const L = e.latency(); console.log(`engine slot (v2 quality chain + distill) per 2 s window: median ${L.median.toFixed(2)} ms, p95 ${L.p95.toFixed(2)} ms`);
}

lat.sort((a, b) => a - b);
console.log(`distill core (features + 3 students) per 2 s window in Node: median ${lat[lat.length >> 1].toFixed(2)} ms, p95 ${lat[Math.floor(lat.length * 0.95)].toFixed(2)} ms, max ${lat[lat.length - 1].toFixed(2)} ms`);
const size = fs.statSync(new URL('app/data/distill.json', root)).size;
check(size < 400000, `app/data/distill.json is ${size} bytes`);
console.log(fail ? 'distill parity: FAIL' : `distill parity: OK (app/data/distill.json ${(size / 1024).toFixed(0)} kB)`);
process.exit(fail ? 1 : 0);
