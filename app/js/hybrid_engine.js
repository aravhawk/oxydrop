// OxyDrop hybrid engine: the final model, running on the device. Pure ES module (no DOM), shared by the app and the Node tests.
//
//   30 Hz R,G,B  ->  v1 physics chain (ratio-of-ratios and pulse statistics, per-person baseline)         -> c  = logit(p_chain)
//                ->  OxyNet CNN, 3 seeds, residual head fed with the baseline-relative physics channels   -> corr = mean lowox_logit
//                ->  r = c + corr  ->  logistic meta-learner  ->  decision logit z (z > 0: possible low oxygen)
//                ->  Mondrian conformal thresholds (conformal.json, default alpha 0.10): z < normal_below NORMAL, z > low_above LOW, else UNSURE
//   heart rate:     CNN HR head (median of the 3 seeds) and the v2 spectral estimate, rule 'agree' (mean if within 5 bpm, else the CNN)
//   quality:        the v2 chain's window validity (pulse lock, beat agreement, saturation, baseline) decides 'nolock'; decision.js adds its own gates.
//
// The output object has the shape of engine.js Chain windows ({s, calibrating, valid, sqi, sat, hrEst, hr, p, label, dHR, ...}) so app/js/decision.js
// decide() and the UI are unchanged. label: 'normal' | 'HYPOXIC' | 'unsure' | 'nolock'.
// Definitions and the evidence: results/ml/HYBRID.md, results/ml/CONFORMAL.md, python/oxydrop_ml/MODEL_SPEC.md. Nothing here is fitted to a phone.

import { Chain as V2Chain, FS, CAL, WIN } from './engine.js';
import { Chain as V1Chain } from './engine_v1.js';
import { CnnInput } from './cnn_features.js';
import { loadModel, forward, decodeHr, meanEmbedding, sigmoid } from './cnn.js';

export { FS, CAL, WIN };
export const ENGINE_ID = 'oxydrop-hybrid-v1';
export const HR_AGREE_BPM = 5;      // rule 'agree' (python/oxydrop_ml/hybrid_hr.py): mean of CNN and spectral if within 5 bpm, else the CNN
const CHAIN_CLIP = 1e-6;
// Generic low-amplitude (out-of-range) guard. lac = log(std of the band-passed log-intensity of the 10 s window) is about the AC/DC ratio of the pulse; the guard uses the
// largest of the three channels. The cut is the 0.1th percentile of that number over all windows of the six Hoffman people, the data the low-oxygen part was trained on
// (scripts/ood_range.py, results/ml/ood_ac_range.csv: -6.02, an AC/DC ratio of 0.24 %). Below it the network has never seen a pulse that weak, so the low-oxygen state is
// refused (UNSURE) whatever the score says. Derived from training windows only; no phone model, no label.
export const OOD_LAC_MIN = -6.02;

const median = (a) => { const b = Array.from(a).sort((x, y) => x - y), n = b.length; return n % 2 ? b[(n - 1) / 2] : 0.5 * (b[n / 2 - 1] + b[n / 2]); };
const mean = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };

/** c = logit of the mean probability of the two v1 chain models, clipped like the Python pipeline. */
export function chainLogit(p) { const q = Math.min(Math.max(p, CHAIN_CLIP), 1 - CHAIN_CLIP); return Math.log(q / (1 - q)); }

/** Meta-learner of the exported pipeline (source R, features [r], no per-user calibration, k = 1): z = intercept + coef * (r - mean) / scale. */
export function metaLogit(pipeline, r) { const m = pipeline.meta; return m.intercept + m.coef[0] * (r - m.mean[0]) / m.scale[0]; }

/** Conformal three-state rule (format oxydrop-conformal-v1): 'NORMAL' if z < normal_below, 'LOW' if z > low_above, otherwise 'UNSURE'. */
export function conformalState(conf, z, alpha = conf.default_alpha) {
  const a = conf.alphas[alpha.toFixed(2)];
  if (!a) throw new Error(`no conformal thresholds for alpha ${alpha}`);
  if (a.past_crossing) throw new Error(`alpha ${alpha} is past the crossing point: not usable`);
  if (!Number.isFinite(z)) return 'UNSURE';
  return z < a.normal_below ? 'NORMAL' : z > a.low_above ? 'LOW' : 'UNSURE';
}
const LABEL = { NORMAL: 'normal', LOW: 'HYPOXIC', UNSURE: 'unsure' };

/** Final HR rule: mean of CNN and spectral estimates if within 5 bpm, else the CNN estimate. */
export function hrAgree(cnn, spec) {
  if (!Number.isFinite(cnn)) return Number.isFinite(spec) ? spec : NaN;
  return Number.isFinite(spec) && Math.abs(cnn - spec) <= HR_AGREE_BPM ? 0.5 * (cnn + spec) : cnn;
}

/** Fetch and build everything the engine needs. fetchJson(relativePath) -> parsed JSON (the app passes fetch(), the tests read the disk). */
export async function loadHybridAssets(fetchJson) {
  const [hybrid, conformal, v2Weights] = await Promise.all([fetchJson('data/hybrid_weights.json'), fetchJson('data/conformal.json'), fetchJson('data/weights.json')]);
  if (hybrid.format !== 'oxydrop-hybrid-v1') throw new Error('unsupported hybrid format ' + hybrid.format);
  if (conformal.format !== 'oxydrop-conformal-v1') throw new Error('unsupported conformal format ' + conformal.format);
  const p = hybrid.pipeline;
  if (p.source !== 'R' || p.features.length !== 1 || p.features[0] !== 'r' || p.calibration || p.aggregate_windows !== 1) throw new Error('this engine implements the pipeline source R, features [r], no calibration, k = 1');
  const members = hybrid.members.filter((m) => m.role === 'corr');
  if (!members.length || members.some((m) => !m.weights)) throw new Error('missing CNN member weights');
  const models = [];
  for (const m of members) models.push(loadModel(await fetchJson('data/' + m.weights)));
  return { hybrid, conformal, v2Weights, models, members };
}

/** The heavy half of the engine: v1 physics chain + CNN features + the 3 CNN forward passes + meta-learner + conformal rule. No v2 chain, no quality gate.
 *  Runs in a Web Worker in the live app (app/js/hybrid_worker.js) so that a CNN pass on a slow phone never delays camera frames, and in-thread in the tests. */
export class HybridCore {
  constructor(assets, opts = {}) {
    this.A = assets; this.alpha = opts.alpha ?? assets.conformal.default_alpha;
    conformalState(assets.conformal, 0, this.alpha);            // throws early for an unsupported alpha
    this.v1 = new V1Chain(assets.hybrid.chain.engine_v1_weights);
    this.cin = new CnnInput([1, 1, 1]); this.models = assets.models;
    this.embBase = null; this.baseTried = false;
  }
  get info() {
    const m0 = this.models[0];
    return { id: ENGINE_ID, model: `OxyNet residual CNN, ${this.models.length} seeds`, params: m0.params, paramsTotal: m0.params * this.models.length, alpha: this.alpha, thresholds: this.A.conformal.alphas[this.alpha.toFixed(2)] };
  }
  _buildBaseline() {
    this.baseTried = true;
    const wins = this.cin.baselineWindows(8);
    if (!wins.length) return;
    this.embBase = this.models.map((m) => meanEmbedding(wins.map((w) => Float64Array.from(forward(m, w.x, { domain: 0, side: w.side }).emb))));
  }
  /** One 30 Hz sample. wantCnn: whether this window needs the CNN (the pulse-lock gate passed); only read on the sample that completes a window.
   *  Returns null between windows, else {calibrating} or {noBase} or {skipped} or {c, corr, cnnHr, spo2Exp, z, ms}. */
  push(r, g, b, dt = 1 / FS, wantCnn = true) {
    const d1 = this.v1.push(r, g, b), w = this.cin.push(r, g, b, dt);
    if (!d1) return null;
    const t0 = performance.now();
    if (!this.baseTried && w && !w.calibrating) this._buildBaseline();          // first window after the first minute: the baseline exists
    if (d1.calibrating || !w || w.calibrating) return { calibrating: true };
    if (!this.embBase) return { noBase: true };
    if (!wantCnn) return { skipped: true };
    const lo = [], hr = [], sp = [];
    this.models.forEach((m, i) => {
      const o = forward(m, w.x, { domain: 0, side: w.side, embBase: this.embBase[i] });
      lo.push(o.lowox_logit); hr.push(decodeHr(o.hr_logits, m.hr.lo, 'expect', 7)); sp.push(o.spo2);
    });
    const c = chainLogit(d1.p), corr = mean(lo);
    const lacMax = Math.max(...[3, 4, 5].map((i) => w.abs[i]).filter(Number.isFinite));
    return { c, corr, cnnHr: median(hr), spo2Exp: mean(sp), lacMax, z: metaLogit(this.A.hybrid.pipeline, c + corr), ms: performance.now() - t0 };
  }
}

/** Merge the v2 quality window d2 (engine.js Chain) with the core result h into the window object the UI and decision.js use. */
export function finishWindow(d2, h, A, alpha) {
  if (d2.calibrating) return { ...d2, engine: ENGINE_ID };
  const out = { ...d2, engine: ENGINE_ID, label: 'nolock', p: NaN, hr: NaN, z: NaN, c: NaN, corr: NaN, cnnHr: NaN, specHr: d2.hrEst, spo2Exp: NaN };
  if (!h || h.calibrating || h.noBase) { if (h && h.noBase) out.valid = false; return out; }   // no usable personal baseline: nothing to decide on
  if (h.skipped) return out;
  Object.assign(out, { c: h.c, corr: h.corr, cnnHr: h.cnnHr, spo2Exp: h.spo2Exp, z: h.z, p: sigmoid(h.z), state: conformalState(A.conformal, h.z, alpha) });
  out.lacMax = h.lacMax;
  if (d2.valid) {   // no pulse lock: stays 'nolock', no HR
    if (h.lacMax < OOD_LAC_MIN) { out.label = 'low_ac'; out.hr = d2.hrEst; out.lowAc = true; }   // weaker than anything the model was trained on: no low-oxygen state, spectral HR only
    else { out.label = LABEL[out.state]; out.hr = hrAgree(h.cnnHr, d2.hrEst); }
  }
  return out;
}

/** The engine in one thread: v2 quality chain + HybridCore. Same push() contract as engine.js Chain. Used by the tests, the replay demo and as the fallback when
 *  module workers are unavailable. opts: {alpha, forceCnn (tests: also run the CNN on windows that fail the pulse-lock gate; the label stays 'nolock')}. */
export class HybridEngine {
  constructor(assets, opts = {}) {
    this.core = new HybridCore(assets, opts); this.A = assets; this.alpha = this.core.alpha; this.forceCnn = !!opts.forceCnn;
    this.v2 = new V2Chain(assets.v2Weights);
    this.ac = this.v2.ac; this.lg = this.v2.lg; this.n = 0;      // live views for the UI (waveform); same arrays the v2 chain appends to
    this.lat = []; this.lastMs = NaN;
  }
  get info() { return this.core.info; }
  get embBase() { return this.core.embBase; }
  /** Latency of the engine per 2 s window (ms): {last, median, p95, n}. Includes the v2 chain, v1 chain, features and all CNN forward passes. */
  latency() { return latencyStats(this.lat, this.lastMs); }
  push(r, g, b, dt = 1 / FS) {
    const t0 = performance.now(), d2 = this.v2.push(r, g, b);
    const h = this.core.push(r, g, b, dt, !!(d2 && (d2.valid || this.forceCnn)));
    this.n++;
    if (!d2) return null;
    const d = finishWindow(d2, h, this.A, this.alpha);
    this.lastMs = performance.now() - t0; this.lat.push(this.lastMs); if (this.lat.length > 200) this.lat.shift();
    return d;
  }
  calibrationSummary() { return this.v2.calibrationSummary(); }
}

export function latencyStats(lat, last = NaN) {
  const n = lat.length; if (!n) return { last, median: NaN, p95: NaN, n: 0 };
  const s = [...lat].sort((a, b) => a - b); return { last, median: s[n >> 1], p95: s[Math.min(n - 1, Math.floor(n * 0.95))], n };
}

/** Batch helper (tests): rgb = array of [r,g,b] at 30 Hz. Returns every window (calibration windows included). */
export function runBatchHybrid(rgb, assets, opts) {
  const e = new HybridEngine(assets, opts), out = [];
  for (const [r, g, b] of rgb) { const d = e.push(r, g, b); if (d) out.push(d); }
  return out;
}
