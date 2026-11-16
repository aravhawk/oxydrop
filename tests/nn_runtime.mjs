// nn_runtime parity: app/js/nn_runtime.js (float64 forward for Linear, Conv1d, activations, LayerNorm, BatchNorm, GRU, pooling) vs PyTorch float64 outputs.
// Fixture: tests/fixtures/nn_runtime_ref.json, written by python/oxydrop_ml/photon/make_nn_fixture.py. Usage: node tests/nn_runtime.mjs [fixture.json]
import fs from 'node:fs';
import { loadNN, erf } from '../app/js/nn_runtime.js';

const rpath = process.argv[2] ?? new URL('./fixtures/nn_runtime_ref.json', import.meta.url);
const ref = JSON.parse(fs.readFileSync(rpath)), TOL = ref.tolerance ?? 1e-6;
let fail = false;
const check = (cond, msg) => { if (!cond) { console.log('FAIL', msg); fail = true; } };

// ---- 0. erf against reference values (Abramowitz-Stegun tables / mpmath, 1e-14)
for (const [x, v] of [[0, 0], [0.1, 0.1124629160182849], [0.5, 0.5204998778130465], [1, 0.8427007929497149], [2, 0.9953222650189527], [2.5, 0.9995930479825550], [3, 0.9999779095030014], [-1.3, -0.9340079449406524]]) {
  check(Math.abs(erf(x) - v) < 1e-14, `erf(${x}) = ${erf(x)} expected ${v}`);
}

// ---- 1. every model, every case, every output
let worst = 0;
const models = {};
for (const m of ref.models) {
  const net = loadNN(m.graph);
  models[m.name] = net;
  let mx = 0;
  for (const [ci, c] of m.cases.entries()) {
    const out = net.forward(c.inputs);
    for (const [name, want] of Object.entries(c.outputs)) {
      const got = out[name];
      check(got && got.length === want.length, `${m.name}[${ci}].${name}: length ${got?.length} vs ${want.length}`);
      let d = 0;
      for (let i = 0; i < want.length; i++) d = Math.max(d, Math.abs(got[i] - want[i]));
      mx = Math.max(mx, d);
      check(d < TOL, `${m.name}[${ci}].${name}: max |JS - torch| = ${d.toExponential(2)} >= ${TOL}`);
      check(want.some((v) => Math.abs(v) > 1e-3), `${m.name}.${name}: reference output is degenerate`);
    }
  }
  worst = Math.max(worst, mx);
  console.log(`  ${m.name.padEnd(14)} ${String(m.cases.length)} case(s)  ops ${m.graph.ops.map((o) => o.op).join(',')}  max|JS-torch| = ${mx.toExponential(2)}`);
}
console.log(`worst over all models: ${worst.toExponential(2)} (tolerance ${TOL})`);

// ---- 2. causality of the causal conv stack: output at step k depends only on inputs <= k and within the receptive field
{
  const m = ref.models.find((q) => q.name === 'tcn'), net = loadNN(m.graph), x = Float64Array.from(m.cases[0].inputs.x), T = 40, C = 3;
  const seqName = m.causal.output, base = Float64Array.from(net.forward({ x })[seqName]);
  const k = 20, x2 = Float64Array.from(x);
  for (let c = 0; c < C; c++) for (let t = k + 1; t < T; t++) x2[c * T + t] += 5;                 // change the future only
  const o2 = net.forward({ x: x2 })[seqName], rows = base.length / T;
  let dPast = 0, dFuture = 0;
  for (let r = 0; r < rows; r++) for (let t = 0; t < T; t++) { const d = Math.abs(o2[r * T + t] - base[r * T + t]); if (t <= k) dPast = Math.max(dPast, d); else dFuture = Math.max(dFuture, d); }
  check(dPast === 0, `causal stack: changing inputs after step ${k} changed outputs at or before it (${dPast})`);
  check(dFuture > 1e-6, 'causal stack: the future perturbation did not reach later outputs (test is vacuous)');
  const x3 = Float64Array.from(x), far = k - m.causal.lookback - 1;                                   // outside the receptive field of step k
  for (let c = 0; c < C; c++) for (let t = 0; t <= far; t++) x3[c * T + t] += 5;
  const o3 = net.forward({ x: x3 })[seqName];
  let dFar = 0; for (let r = 0; r < rows; r++) dFar = Math.max(dFar, Math.abs(o3[r * T + k] - base[r * T + k]));
  check(dFar === 0, `causal stack: step ${k} depends on inputs outside its receptive field (${dFar})`);
  console.log(`  causality: past change ${dPast}, far-past change ${dFar}, future effect ${dFuture.toExponential(2)}`);
}

// ---- 3. streaming GRU: the hidden state of call 1 fed to call 2 reproduces the one-pass torch output (checked in section 1); a wrong state must change the output
{
  const m = ref.models.find((q) => q.name === 'gru_state'), net = loadNN(m.graph);
  const c2 = m.cases[1], ok = Float64Array.from(net.forward(c2.inputs).h_out);
  const bad = Float64Array.from(net.forward({ x: c2.inputs.x, h0: new Array(5).fill(0) }).h_out);
  let d = 0; for (let i = 0; i < ok.length; i++) d = Math.max(d, Math.abs(ok[i] - bad[i]));
  check(d > 1e-3, 'gru: the carried hidden state has no effect (test is vacuous)');
}

// ---- 4. loader errors are explicit
{
  const bad = (g, re) => { try { loadNN(g); check(false, `loadNN accepted ${re}`); } catch (e) { check(re.test(e.message), `unexpected error "${e.message}" for ${re}`); } };
  const m = ref.models.find((q) => q.name === 'mlp').graph;
  bad({ ...m, format: 'nope' }, /unsupported model format/);
  bad({ ...m, ops: [{ op: 'frobnicate', in: 'x', out: 'y' }], outputs: ['y'] }, /unsupported op/);
  bad({ ...m, ops: [{ op: 'act', fn: 'relu', in: 'zz', out: 'y' }], outputs: ['y'] }, /used before/);
  const net = loadNN(m);
  try { net.forward({ x: [1, 2] }); check(false, 'forward accepted a wrong input length'); } catch (e) { check(/expected 8 values/.test(e.message), `unexpected error "${e.message}"`); }
}

if (fail) { console.log('nn_runtime parity FAILED'); process.exit(1); }
console.log('nn_runtime parity OK');
