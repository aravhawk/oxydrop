// OxyDrop engine slot: runs one model file, app/data/<name>.json, in the page thread. This is the DEFAULT engine of the app (no ?engine= or ?engine=distill);
// ?model=<name> picks another file of a registered format (debug). The hybrid stays behind ?engine=hybrid and the v2 logistic chain behind ?engine=logistic.
// Pure ES module (no DOM), shared by the app and the Node tests.
//
//   30 Hz R,G,B -> v2 chain (engine.js): heart rate, pulse lock, saturation and the calibration summary for the UI (it does NOT decide anything here)
//               -> the model runtime (chosen by the model file's `format`, see RUNTIMES) -> per 10 s window {p, abstain, pred?, spo2?}
//               -> three states, exactly the model's output: abstain -> 'unsure' ("Unsure, re-measure"), else pred (default p > 0.5) -> 'HYPOXIC', else 'normal'.
//   No extra masking: no pulse-lock ('nolock'), low-AC, saturation or SNR gate is added on top of the model (`ownQuality` tells decision.js to skip its
//   signal gates). The v2 pulse lock is kept as `pulseLock` for the signal tile and the debug view. decision.js still applies the app's measurement checks
//   that are not about the signal the model reads (camera stalled, calibration failed, no finger, movement, frame rate, frame gaps).
//
// Runtime contract (add a class and registerRuntime(format, factory) for a new model family):
//   new Runtime(model)              model = the parsed JSON; throw on anything malformed
//   runtime.push(r, g, b)           one 30 Hz sample -> null between windows | {s, calibrating: true} | {s, calibrating: false, p, abstain, pred?, spo2?, ...}
//                                   windows complete every 60 samples once 300 are in (engine.js FS, WIN, STEP); first 60 s = calibration (CAL)
//   runtime.info                    {id, model, params, paramsTotal}
// Registered: 'oxydrop-photon-distill-app-v1' = PhotonDistill (app/js/distill_engine.js, app/data/distill.json; parity tests/distill_parity.mjs).

import { Chain as V2Chain, FS, CAL, WIN } from './engine.js';
import { DistillCore, DISTILL_FORMAT } from './distill_engine.js';

export { FS, CAL, WIN };
export const DEFAULT_PHOTON_MODEL = 'distill';
const MODEL_NAME = /^[a-z0-9_-]{1,64}$/;

// ---------- runtime registry ----------
const RUNTIMES = new Map();
export function registerRuntime(format, factory) { RUNTIMES.set(format, factory); }
export function makeRuntime(model) {
  const f = RUNTIMES.get(model && model.format);
  if (!f) throw new Error(`no in-browser runtime for model format ${model && model.format}`);
  return f(model);
}
export const runtimeFormats = () => [...RUNTIMES.keys()];
registerRuntime(DISTILL_FORMAT, (m) => new DistillCore(m));

// ---------- the slot ----------
const STATE_OF = { normal: 'NORMAL', HYPOXIC: 'LOW', unsure: 'UNSURE' };
const logit = (p) => { const q = Math.min(Math.max(p, 1e-6), 1 - 1e-6); return Math.log(q / (1 - q)); };

/** Three-state mapping of one model window h onto the v2 window d2 (heart rate, pulse lock, saturation): exactly the model's output, no extra masking. */
export function finishPhoton(d2, h, id) {
  if (d2.calibrating) return { ...d2, engine: id };
  const out = { ...d2, engine: id, pulseLock: !!d2.valid, valid: true, ownQuality: true, label: 'unsure', p: NaN, z: NaN, c: NaN, corr: NaN, cnnHr: NaN, specHr: d2.hrEst, spo2Exp: NaN,
    modelP: NaN, modelSpo2: NaN, modelAbstain: true, model: null };
  if (!h || h.calibrating || !Number.isFinite(h.p)) return out;          // no model output for this window: nothing to show but Unsure
  const pred = h.pred ?? h.p > 0.5, label = h.abstain ? 'unsure' : pred ? 'HYPOXIC' : 'normal';
  Object.assign(out, { label, state: STATE_OF[label], p: h.p, z: logit(h.p), modelP: h.p, modelSpo2: Number.isFinite(h.spo2) ? h.spo2 : NaN, modelAbstain: !!h.abstain, model: h });
  return out;
}

export function latencyStats(lat, last = NaN) {
  const n = lat.length; if (!n) return { last, median: NaN, p95: NaN, n: 0 };
  const s = [...lat].sort((a, b) => a - b); return { last, median: s[n >> 1], p95: s[Math.min(n - 1, Math.floor(n * 0.95))], n };
}

/** Fetch and check everything the slot needs: data/<name>.json (the model) and data/weights.json (the v2 chain for HR / pulse lock / calibration). */
export async function loadPhotonAssets(fetchJson, name = DEFAULT_PHOTON_MODEL) {
  if (!MODEL_NAME.test(name)) throw new Error(`bad model name ${JSON.stringify(name)}`);
  const [model, v2Weights] = await Promise.all([fetchJson(`data/${name}.json`), fetchJson('data/weights.json')]);
  makeRuntime(model);                        // throws early for an unknown format or a malformed file
  return { name, model, v2Weights };
}

/** The slot in the page thread: v2 chain + the model runtime. Same push() contract as engine.js Chain. If the model throws, the engine marks itself
 *  `degraded` and from then on returns the v2 logistic chain's own windows; the app shows a visible fallback notice (app.js photonFailed). */
export class PhotonEngine {
  constructor(assets, opts = {}) {
    this.A = assets; this.core = makeRuntime(assets.model); this.id = this.core.info.id;
    this.v2 = new V2Chain(assets.v2Weights, opts.hrGate ? { hrGate: opts.hrGate } : {});
    this.ac = this.v2.ac; this.lg = this.v2.lg; this.n = 0;      // live views for the UI (waveform); same arrays the v2 chain appends to
    this.lat = []; this.lastMs = NaN; this.degraded = false; this.failMsg = '';
  }
  get info() { return this.core.info; }
  /** true while the model runs: its windows carry their own quality judgement (decision.js and app.js add no signal gate on top). */
  get ownQuality() { return !this.degraded; }
  /** Latency of the engine per 2 s window (ms): {last, median, p95, n}, v2 chain and model together. */
  latency() { return latencyStats(this.lat, this.lastMs); }
  degrade(msg = 'model error') { if (!this.degraded) { this.degraded = true; this.failMsg = msg; } }
  push(r, g, b) {
    const t0 = performance.now(), d2 = this.v2.push(r, g, b);
    let h = null;
    if (!this.degraded) { try { h = this.core.push(r, g, b); } catch (e) { this.degrade(e && e.message ? e.message : String(e)); } }
    this.n++;
    if (!d2) return null;
    if (this.degraded) return d2;
    const d = finishPhoton(d2, h, this.id);
    this.lastMs = performance.now() - t0; this.lat.push(this.lastMs); if (this.lat.length > 200) this.lat.shift();
    return d;
  }
  calibrationSummary() { return this.v2.calibrationSummary(); }
}

/** Batch helper (tests): rgb = array of [r,g,b] at 30 Hz. Returns every window (calibration windows included). */
export function runBatchPhoton(rgb, assets, opts) {
  const e = new PhotonEngine(assets, opts), out = [];
  for (const [r, g, b] of rgb) { const d = e.push(r, g, b); if (d) out.push(d); }
  return out;
}
