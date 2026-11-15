// Main-thread side of the hybrid engine in the live app: the v2 quality chain stays on the main thread (cheap, it decides the pulse-lock gate and drives the
// waveform), the CNN half runs in app/js/hybrid_worker.js. A window completes asynchronously (typically 50 to 300 ms after its last sample); calibration windows
// complete at once. Same push() contract as engine.js Chain except that CNN windows are delivered through `onWindow(d)`.
import { Chain as V2Chain, FS } from './engine.js';
import { finishWindow, latencyStats, ENGINE_ID } from './hybrid_engine.js';

// A 2 s hop must finish before the next one can build an unbounded backlog.
export const WORKER_WINDOW_TIMEOUT_MS = 2000, MAX_WORKER_PENDING = 2;

/** The small files the main thread needs (the CNN member files are only read by the worker or by an in-thread fallback). */
export async function loadHybridLite(fetchJson) {
  const [hybrid, conformal, v2Weights] = await Promise.all([fetchJson('data/hybrid_weights.json'), fetchJson('data/conformal.json'), fetchJson('data/weights.json')]);
  if (hybrid.format !== 'oxydrop-hybrid-v1' || conformal.format !== 'oxydrop-conformal-v1') throw new Error('unsupported hybrid or conformal format');
  return { hybrid, conformal, v2Weights };
}

/** Owner of the Worker. ready resolves with the model info once the worker has loaded the weights; rejects on a worker error or after timeoutMs. */
export class HybridWorker {
  constructor(url, alpha, timeoutMs = 12000) {
    this.w = new Worker(url, { type: 'module' }); this.gen = 0; this.handler = null; this.error = null; this.info = null;
    this.ready = new Promise((res, rej) => {
      const t = setTimeout(() => { this.error = 'the model worker did not start'; this.terminate(); rej(new Error(this.error)); }, timeoutMs);
      this.w.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') { clearTimeout(t); this.info = m.info; res(m.info); }
        else if (m.type === 'error') { this.error = m.message; clearTimeout(t); this.terminate(); rej(new Error(m.message)); if (this.onError) this.onError(m.message); }
        else if (this.handler) this.handler(m);
      };
      this.w.onerror = (e) => { const msg = (e && e.message) || 'worker error'; this.error = msg; clearTimeout(t); this.terminate(); rej(new Error(msg)); if (this.onError) this.onError(msg); };
    });
    this.ready.catch(() => {});
    this.w.postMessage({ type: 'init', alpha });
  }
  reset() { this.gen++; this.w.postMessage({ type: 'reset', gen: this.gen }); return this.gen; }
  terminate() { try { this.w.terminate(); } catch (e) { /* ignore */ } }
}

export class HybridWorkerEngine {
  /** lite: loadHybridLite() result; worker: HybridWorker; onWindow(d): called for every CNN window (post calibration). */
  constructor(lite, worker, alpha, onWindow, opts = {}) {
    this.A = { conformal: lite.conformal }; this.alpha = alpha; this.worker = worker; this.onWindow = onWindow;
    this.v2 = new V2Chain(lite.v2Weights, { hrGate: opts.hrGate }); this.ac = this.v2.ac; this.lg = this.v2.lg; this.n = 0;
    this.buf = []; this.id = 0; this.pending = new Map(); this.lat = []; this.latV2 = []; this.latCore = []; this.lastMs = NaN; this.dead = false; this.degraded = false;
    this.timeoutMs = opts.timeoutMs ?? WORKER_WINDOW_TIMEOUT_MS; this.maxPending = opts.maxPending ?? MAX_WORKER_PENDING; this.degradeReason = '';
    this.gen = worker.reset(); this._h = (m) => this._msg(m); worker.handler = this._h;
  }
  get info() { return this.worker.info; }
  get embBase() { return null; }
  latency() { const l = latencyStats(this.lat, this.lastMs); l.v2Median = latencyStats(this.latV2).median; l.coreMedian = latencyStats(this.latCore).median; l.async = true; return l; }
  push(r, g, b, dt = 1 / FS) {
    if (!this.degraded && this.oldestPendingMs() >= this.timeoutMs) this._fallback('model worker exceeded the 2 s capture budget');
    if (this.degraded) { this.n++; return this.v2.push(r, g, b); }   // the worker is gone: the quality chain's own decision is the answer (the logistic chain, same weights)
    const t0 = performance.now(), d2 = this.v2.push(r, g, b);
    this.buf.push(r, g, b, dt); this.n++;
    if (!d2) return null;
    this.latV2.push(performance.now() - t0); if (this.latV2.length > 200) this.latV2.shift();
    const samples = Float64Array.from(this.buf); this.buf.length = 0;
    if (this.pending.size >= this.maxPending) { this._fallback('model worker backlog exceeded the capture budget'); return d2; }
    const id = ++this.id, p = { d2, n: this.n, t0: performance.now(), timer: null };
    // Track calibration messages too: baseline construction can be the slowest CNN operation.
    this.pending.set(id, p);
    p.timer = setTimeout(() => this._fallback('model worker exceeded the 2 s capture budget'), this.timeoutMs);
    try { this.worker.w.postMessage({ type: 'win', gen: this.gen, id, want: !!d2.valid, samples: samples.buffer }, [samples.buffer]); }
    catch (e) { this.pending.delete(id); clearTimeout(p.timer); this._fallback('model worker could not receive a window'); return d2; }
    if (d2.calibrating) return finishWindow(d2, null, this.A, this.alpha);
    return null;
  }
  _msg(m) {
    if (m.type !== 'res' || m.gen !== this.gen || this.dead) return;
    const p = this.pending.get(m.id); if (!p) return; this.pending.delete(m.id); clearTimeout(p.timer);
    if (p.d2.calibrating) return; // already delivered synchronously for the calibration UI
    this.lastMs = performance.now() - p.t0; this.lat.push(this.lastMs); if (this.lat.length > 200) this.lat.shift();
    if (m.h && Number.isFinite(m.h.ms)) { this.latCore.push(m.h.ms); if (this.latCore.length > 200) this.latCore.shift(); }
    const d = finishWindow(p.d2, m.h, this.A, this.alpha); d.n = p.n;   // d.n: samples pushed when the window ended (the result arrives later)
    if (this.onWindow) this.onWindow(d);
  }
  /** Windows sent to the worker whose result has not arrived yet. */
  get inFlight() { return this.pending.size; }
  /** Age in ms of the oldest window the worker has not answered (0 if none). A healthy worker answers in 50 to 300 ms; a dead or hung one never does. */
  oldestPendingMs() { let m = 0; const now = performance.now(); for (const p of this.pending.values()) m = Math.max(m, now - p.t0); return m; }
  _fallback(reason) {
    if (this.dead || this.degraded) return;
    this.degradeReason = reason; this.degrade();
    // Stop expensive work as well as waiting for it. The page's handler displays the fallback notice.
    if (this.worker.terminate) this.worker.terminate();
    if (this.worker.onError) this.worker.onError(reason);
  }
  /** Keep the calibrated quality chain, flush queued decisions as logistic results, and never accept a late CNN reply. */
  degrade() {
    if (this.degraded) return; this.degraded = true; this.buf.length = 0;
    const queued = [...this.pending.values()]; this.pending.clear();
    if (this.worker.handler === this._h) this.worker.handler = null;
    for (const p of queued) { clearTimeout(p.timer); if (!p.d2.calibrating && this.onWindow) this.onWindow({ ...p.d2, n: p.n }); }
  }
  close() { this.dead = true; if (this.worker.handler === this._h) this.worker.handler = null; for (const p of this.pending.values()) clearTimeout(p.timer); this.pending.clear(); }
  calibrationSummary() { return this.v2.calibrationSummary(); }
}
export { ENGINE_ID };
