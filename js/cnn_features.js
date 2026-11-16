// Causal construction of the OxyNet input tensor from the resampler's 30 Hz R,G,B samples (0..255 scale) and frame-time deltas.
// Port of python/oxydrop_ml/features.windows_to_tensor (FEAT-003), step for step; see that docstring and python/oxydrop_ml/MODEL_SPEC.md.
// Pure ES module (no DOM). The band-pass is the v2 chain's SOS cascade, imported from engine.js (same coefficients as scipy butter(3,[0.7,4])).
//
// Every 2 s the last 15 s (CTX 5 s of filter warm-up + the 10 s window) are re-filtered from a warm start, so the result does not depend on
// anything older than 15 s. Windows that end before 15 s of recording have the missing warm-up odd-reflected about the first sample.
// Windows ending inside the first 60 s only collect baseline statistics (built with no baseline); later windows use that personal baseline.

import { SOS } from './engine.js';

export const FS = 30, WIN = 300, HOP = 60, CTX = 150, L_BUF = CTX + WIN, BASE_SEC = 60;
export const AC_SCALE = 100, DC_TAU = 3, V_TAU = 2, MIN_GAP = 10, PROM_K = 0.8, TIMEOUT = 3 * FS, AMP_EPS = 1e-4, ROR_CLIP = 3, MIN_BASE_WIN = 3;
export const CHANNELS = ['ac_R', 'ac_G', 'ac_B', 'dt', 'ror_RG_rel', 'ror_RB_rel', 'beat_ok', 'dc_R_rel', 'dc_G_rel', 'dc_B_rel'];
export const SIDE = ['d_ror_RG', 'd_ror_RB', 'd_dc_R', 'd_dc_G', 'd_dc_B', 'd_lac_R', 'd_lac_G', 'd_lac_B', 'beat_frac'];
export const ABS = ['dc_R', 'dc_G', 'dc_B', 'lac_R', 'lac_G', 'lac_B', 'ror_RG', 'ror_RB', 'beat_frac', 'n_beats', 'hr_beat'];
export const N_CH = CHANNELS.length, N_SIDE = SIDE.length, N_ABS = ABS.length, N_BASE = 9;

// scipy.signal.sosfilt_zi(SOS): steady state of the cascade for a unit step (each section sees the previous sections' DC gain)
const ZI = (() => {
  let scale = 1; const zi = [];
  for (const [b0, b1, b2, , a1, a2] of SOS) {
    const h = (b0 + b1 + b2) / (1 + a1 + a2), z1 = b2 - a2 * h, z0 = b1 - a1 * h + z1;
    zi.push([scale * z0, scale * z1]); scale *= h;
  }
  return zi;
})();

const nanmedian = (a) => {
  const b = Array.from(a).filter((v) => Number.isFinite(v)).sort((p, q) => p - q), n = b.length;
  return n ? (n % 2 ? b[(n - 1) / 2] : 0.5 * (b[n / 2 - 1] + b[n / 2])) : NaN;
};

/** STEP 2: forward-only band-pass with warm start (state = zi * first sample). */
function causalAc(x) {
  const L = x.length, y = new Float64Array(L), z = ZI.map((zz) => [zz[0] * x[0], zz[1] * x[0]]);
  for (let n = 0; n < L; n++) {
    let v = x[n];
    for (let s = 0; s < SOS.length; s++) {
      const [b0, b1, b2, , a1, a2] = SOS[s], zs = z[s], o = b0 * v + zs[0];
      zs[0] = b1 * v - a1 * o + zs[1]; zs[1] = b2 * v - a2 * o; v = o;
    }
    y[n] = v;
  }
  return y;
}

/** STEP 3: first-order IIR, initial state = first sample. */
function ema(x, tau) {
  const a = 1 - Math.exp(-1 / (tau * FS)), y = new Float64Array(x.length); let p = x[0];
  for (let n = 0; n < x.length; n++) { p = a * x[n] + (1 - a) * p; y[n] = p; }
  return y;
}

/** STEP 4: per-beat log ratio-of-ratios, held between beats. ac: 3 arrays (already zero for absent wavelengths). */
function beatTrace(ac, mask) {
  const L = ac[0].length, x = new Float64Array(L);
  for (let t = 0; t < L; t++) x[t] = -((ac[0][t] + ac[1][t]) + ac[2][t]);
  const aV = 1 - Math.exp(-1 / (V_TAU * FS)), ror0 = new Float64Array(L), ror1 = new Float64Array(L), ok = new Uint8Array(L), peak = new Uint8Array(L);
  let v = x[0] * x[0], lastPeak = -1e6, trough = Infinity, troughI = 0, seen = false, r0 = 0, r1 = 0;
  const clip = (u) => Math.min(Math.max(u, -ROR_CLIP), ROR_CLIP);
  for (let t = 0; t < L; t++) {
    const xt = x[t];
    if (t >= 2) {
      const j = t - 1;
      if (x[j - 1] < x[j] && x[j] >= xt && (j - lastPeak) >= MIN_GAP && (x[j] - trough) >= PROM_K * Math.sqrt(v)) {
        const a0 = Math.abs(ac[0][j] - ac[0][troughI]), a1 = Math.abs(ac[1][j] - ac[1][troughI]), a2 = Math.abs(ac[2][j] - ac[2][troughI]);
        r0 = clip(Math.log((a0 + AMP_EPS) / (a1 + AMP_EPS)));
        r1 = mask[2] ? clip(Math.log((a0 + AMP_EPS) / (a2 + AMP_EPS))) : 0;
        seen = true; lastPeak = j; peak[j] = 1; trough = x[j]; troughI = j;
      }
    }
    ror0[t] = r0; ror1[t] = r1; ok[t] = seen && (t - lastPeak) <= TIMEOUT ? 1 : 0;
    if (xt < trough) { trough = xt; troughI = t; }
    v += aV * (xt * xt - v);
  }
  return { ror0, ror1, ok, peak };
}

/** Beat HR (bpm) of a window from its peak flags: 60 FS / median inter-peak interval, NaN with fewer than 2 valid intervals. */
function windowHr(peak, from) {
  const pos = []; for (let i = from; i < peak.length; i++) if (peak[i]) pos.push(i - from);
  const ibi = []; for (let i = 1; i < pos.length; i++) { const d = pos[i] - pos[i - 1]; if (!(d < 0.3 * FS || d > 1.6 * FS)) ibi.push(d); }
  return ibi.length >= 2 ? 60 * FS / nanmedian(ibi) : NaN;
}

/**
 * Stateless reference port of windows_to_tensor for ONE window.
 * lr: 3 arrays of length L >= 300 (ln(rgb+1)), dt: array of L seconds, base: Float64Array(9) from baselineFromAbs() or null, mask: [1,1,1] (wavelength present).
 * Returns {x: Float32Array(10*300) channel-major, side: Float32Array(9), abs: Float64Array(11)}.
 */
export function windowToTensor(lr, dt, base = null, mask = [1, 1, 1]) {
  const L = lr[0].length, f = 3, s0 = L - WIN;
  const m = mask.map(Boolean), mf = m.map(Number);
  const b = base ? Array.from(base) : new Array(N_BASE).fill(NaN), bnan = b.map((u) => !Number.isFinite(u)), b0 = b.map((u, i) => (bnan[i] ? 0 : u));
  const ac = lr.map((c, i) => { const y = causalAc(c); for (let n = 0; n < L; n++) y[n] *= mf[i]; return y; });
  const dc = lr.map((c) => ema(c, DC_TAU));
  const bt = beatTrace(ac, m);
  const x = new Float32Array(N_CH * WIN), side = new Float32Array(N_SIDE), ab = new Float64Array(N_ABS);
  let cnt = 0, rs0 = 0, rs1 = 0;
  for (let n = 0; n < WIN; n++) {
    const t = s0 + n, ok = bt.ok[t];
    for (let c = 0; c < f; c++) x[c * WIN + n] = AC_SCALE * ac[c][t];
    x[3 * WIN + n] = Math.min(Math.max(dt[t] * FS - 1, -1), 8);
    x[4 * WIN + n] = bnan[6] || !(m[0] && m[1]) ? 0 : (bt.ror0[t] - b0[6]) * ok;
    x[5 * WIN + n] = bnan[7] || !(m[0] && m[2]) ? 0 : (bt.ror1[t] - b0[7]) * ok;
    x[6 * WIN + n] = ok;
    for (let c = 0; c < f; c++) x[(7 + c) * WIN + n] = bnan[c] ? 0 : (dc[c][t] - b0[c]) * mf[c];
    if (ok) { cnt++; rs0 += bt.ror0[t]; rs1 += bt.ror1[t]; }
  }
  const dcm = [0, 0, 0], lac = [0, 0, 0];
  for (let c = 0; c < f; c++) {
    let sm = 0, sa = 0; for (let n = 0; n < WIN; n++) { sm += lr[c][s0 + n]; sa += ac[c][s0 + n]; }
    dcm[c] = sm / WIN; const ma = sa / WIN; let v = 0; for (let n = 0; n < WIN; n++) { const d = ac[c][s0 + n] - ma; v += d * d; }
    lac[c] = Math.log(Math.sqrt(v / WIN) + 1e-6);
  }
  const rm = cnt ? [rs0 / cnt, rs1 / cnt] : [NaN, NaN], bf = cnt / WIN;
  let nPk = 0; for (let n = 0; n < WIN; n++) nPk += bt.peak[s0 + n];
  for (let c = 0; c < f; c++) { ab[c] = m[c] ? dcm[c] : NaN; ab[3 + c] = m[c] ? lac[c] : NaN; }
  ab[6] = rm[0]; ab[7] = m[2] ? rm[1] : NaN; ab[8] = bf; ab[9] = nPk; ab[10] = windowHr(bt.peak, s0);
  for (let k = 0; k < 2; k++) { const d = rm[k] - b[6 + k]; side[k] = Number.isFinite(d) ? d : 0; }
  for (let c = 0; c < f; c++) {
    let d = dcm[c] - b[c]; side[2 + c] = Number.isFinite(d) && m[c] ? d : 0;
    d = lac[c] - b[3 + c]; side[5 + c] = Number.isFinite(d) && m[c] ? d : 0;
  }
  side[8] = bf;
  return { x, side, abs: ab };
}

/** Label-free personal baseline: median over windows with beats (beat_frac > 0.3) of [dc R,G,B, lac R,G,B, ror RG, RB, hr]; all NaN with < 3 such windows. */
export function baselineFromAbs(absList, minWindows = MIN_BASE_WIN) {
  const out = new Float64Array(N_BASE).fill(NaN), good = absList.filter((a) => a[8] > 0.3);
  if (good.length < minWindows) return out;
  [0, 1, 2, 3, 4, 5, 6, 7, 10].forEach((c, i) => { out[i] = nanmedian(good.map((a) => a[c])); });
  return out;
}

export class CnnInput {
  /** mask: wavelengths present, [1,1,1] for the phone camera, [1,1,0] for contact red/IR data stored as [red, IR, 0]. */
  constructor(mask = [1, 1, 1]) { this.mask = mask; this.reset(); }
  reset() {
    this.lr = [0, 1, 2].map(() => new Float64Array(L_BUF)); this.dt = new Float64Array(L_BUF);
    this.n = 0; this.calAbs = []; this.base = null;
    this.hist = [0, 1, 2].map(() => new Float64Array(BASE_SEC * FS)); this.histDt = new Float64Array(BASE_SEC * FS);   // the first minute, kept to rebuild the baseline windows once the baseline exists
  }
  /** One 30 Hz sample (0..255 scale); dt = source frame interval in seconds (default 1/30). Returns a window object every 2 s once 10 s exist, else null.
   *  Window: {x: Float32Array(10*300), side: Float32Array(9), te, s (start, seconds), calibrating (still collecting the baseline: no baseline channels),
   *  warm (baseline fixed but the window starts inside the first minute), abs, base (Float64Array(9) or null)}. */
  push(r, g, b, dt = 1 / FS) {
    const v = [r, g, b], i = this.n % L_BUF;
    for (let c = 0; c < 3; c++) this.lr[c][i] = Math.log(Math.max(v[c], 1e-3) + 1.0);
    this.dt[i] = dt;
    if (this.n < BASE_SEC * FS) { for (let c = 0; c < 3; c++) this.hist[c][this.n] = this.lr[c][i]; this.histDt[this.n] = dt; }
    this.n++;
    if (this.n < WIN || (this.n - WIN) % HOP !== 0) return null;
    return this._window();
  }
  _window() {
    const te = this.n, buf = [0, 1, 2].map(() => new Float64Array(L_BUF)), dt = new Float64Array(L_BUF);
    for (let k = 0; k < L_BUF; k++) {
      const ii = te - L_BUF + k;
      if (ii >= 0) { const p = ii % L_BUF; for (let c = 0; c < 3; c++) buf[c][k] = this.lr[c][p]; dt[k] = this.dt[p]; }
      else {   // odd reflection about the first sample; only reached while te <= 450, when the ring has not wrapped yet
        for (let c = 0; c < 3; c++) buf[c][k] = 2 * this.lr[c][0] - this.lr[c][-ii];
        dt[k] = this.dt[0];
      }
    }
    const calibrating = te <= BASE_SEC * FS;
    if (!calibrating && !this.base) this.base = baselineFromAbs(this.calAbs);
    const w = windowToTensor(buf, dt, calibrating ? null : this.base, this.mask);
    if (calibrating) this.calAbs.push(w.abs);
    return { ...w, te, s: (te - WIN) / FS, calibrating, warm: !calibrating && te - WIN < BASE_SEC * FS, base: this.base };
  }
}

// numpy.round (half to even) on a non-negative number
const roundHalfEven = (v) => { const f = Math.floor(v), d = v - f; return d < 0.5 ? f : d > 0.5 ? f + 1 : (f % 2 === 0 ? f : f + 1); };

/** The windows that define the baseline embedding of the low-oxygen head: up to maxRows evenly spaced windows (numpy linspace + round, unique) among those
 *  that end inside the first minute, REBUILT with the finished personal baseline (training convention, python/oxydrop_ml/train.group_emb).
 *  Call after the baseline exists (a window ending after 60 s has been pushed). Returns [] when there is no usable baseline. Allocates; once per person. */
CnnInput.prototype.baselineWindows = function baselineWindows(maxRows = 8) {
  if (!this.base || !this.base.every(Number.isFinite) || this.n < BASE_SEC * FS) return [];
  const tes = []; for (let te = WIN; te <= BASE_SEC * FS; te += HOP) tes.push(te);
  const K = tes.length, m = Math.min(maxRows, K), pick = new Set();
  for (let i = 0; i < m; i++) pick.add(roundHalfEven(m === 1 ? 0 : (i * (K - 1)) / (m - 1)));
  const out = [];
  for (const k of [...pick].sort((a, b) => a - b)) {
    const te = tes[k], buf = [0, 1, 2].map(() => new Float64Array(L_BUF)), dt = new Float64Array(L_BUF);
    for (let j = 0; j < L_BUF; j++) {
      const ii = te - L_BUF + j;
      if (ii >= 0) { for (let c = 0; c < 3; c++) buf[c][j] = this.hist[c][ii]; dt[j] = this.histDt[ii]; }
      else { for (let c = 0; c < 3; c++) buf[c][j] = 2 * this.hist[c][0] - this.hist[c][-ii]; dt[j] = this.histDt[0]; }
    }
    const w = windowToTensor(buf, dt, this.base, this.mask);
    out.push({ x: w.x, side: w.side, te });
  }
  return out;
};

/** Offline helper: all windows of an array of [r,g,b] rows (optionally with a dt array). */
export function windowsFromRgb(rgb, dt, mask) {
  const f = new CnnInput(mask), out = [];
  for (let i = 0; i < rgb.length; i++) { const w = f.push(rgb[i][0], rgb[i][1], rgb[i][2], dt ? dt[i] : 1 / FS); if (w) out.push(w); }
  return out;
}
