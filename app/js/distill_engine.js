// PhotonDistill in the browser: the round-2 trained student network (python/oxydrop_ml/photon/distill_student.py), app file app/data/distill.json
// (format 'oxydrop-photon-distill-app-v1', written by python/oxydrop_ml/photon/distill_app_export.py from results/photon/distill_weights.json).
// Pure ES module (no DOM), shared by the app (through the engine slot of photon_engine.js) and the Node tests.
//   30 Hz R,G,B -> distill_features.js (39 causal features relative to the label-free first minute)
//               -> per student: median-impute, clip, standardise (the student's own training constants) -> nn_runtime.js MLP 39-32-32-3
//               -> 3-student ensemble: p = mean sigmoid(hyp), spo2 = mean (90 + 5 spo2 head), conf = mean sigmoid(abstain), answer = conf >= tau
// The model's own abstain head decides Unsure; nothing else is added. Parity: tests/distill_parity.mjs against tests/fixtures/ref_distill.json.

import { loadNN } from './nn_runtime.js';
import { DistillFeatures, COLS } from './distill_features.js';

export const DISTILL_FORMAT = 'oxydrop-photon-distill-app-v1';
const sig = (z) => { const v = Math.min(Math.max(z, -30), 30); return 1 / (1 + Math.exp(-v)); };

function check(M) {
  if (!M || M.format !== DISTILL_FORMAT) throw new Error('distill model: wrong format');
  if (!Array.isArray(M.cols) || M.cols.length !== COLS.length || M.cols.some((c, i) => c !== COLS[i])) throw new Error('distill model: feature list does not match distill_features.js');
  if (!Number.isFinite(M.tau) || !Array.isArray(M.students) || !M.students.length) throw new Error('distill model: missing tau or students');
  for (const s of M.students) for (const k of ['med', 'lo', 'hi', 'mu', 'sd']) if (!s.prep || !Array.isArray(s.prep[k]) || s.prep[k].length !== COLS.length) throw new Error(`distill model: prep.${k} has the wrong length`);
}

/** The network part: 39 features -> {p, spo2, conf, answer, abstain}. */
export class DistillNet {
  constructor(model) {
    check(model);
    this.M = model; this.tau = model.tau;
    this.nets = model.students.map((s) => loadNN(s.graph)); this.preps = model.students.map((s) => s.prep);
    this.z = new Float64Array(COLS.length);
    this.params = model.students.map((s) => Object.values(s.graph.tensors).reduce((a, t) => a + t.shape.reduce((x, y) => x * y, 1), 0));
  }
  predict(x) {
    let p = 0, sp = 0, cf = 0; const n = this.nets.length;
    for (let k = 0; k < n; k++) {
      const P = this.preps[k], z = this.z;
      for (let j = 0; j < z.length; j++) { const v = Number.isFinite(x[j]) ? x[j] : P.med[j]; z[j] = (Math.min(Math.max(v, P.lo[j]), P.hi[j]) - P.mu[j]) / P.sd[j]; }
      const o = this.nets[k].forward({ z }).o;
      p += sig(o[0]); sp += 90 + 5 * o[1]; cf += sig(o[2]);
    }
    p /= n; sp /= n; cf /= n;
    const answer = cf >= this.tau;
    return { p, spo2: sp, conf: cf, answer, abstain: !answer, pred: p > 0.5 };
  }
}

/** Runtime of the engine slot (photon_engine.js contract): push(r, g, b) -> null | {s, calibrating} | {s, calibrating: false, p, abstain, pred, spo2, conf, x, ms}. */
export class DistillCore {
  constructor(model) { this.net = new DistillNet(model); this.fe = new DistillFeatures(); this.M = model; }
  get info() {
    const pp = this.net.params;
    return { id: this.M.id, model: `PhotonDistill student MLP 39-32-32-3, ${pp.length} seeds (abstain head, tau ${this.net.tau.toFixed(3)})`, params: pp[0], paramsTotal: pp.reduce((a, b) => a + b, 0), seeds: pp.length };
  }
  push(r, g, b) {
    const t0 = performance.now(), f = this.fe.push(r, g, b);
    if (!f || f.calibrating) return f;
    const o = this.net.predict(f.x);
    return { s: f.s, calibrating: false, ...o, x: f.x, ms: performance.now() - t0 };
  }
}
