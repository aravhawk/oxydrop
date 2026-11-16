// PhotonDistill front end: the 39 causal per-window features of python/oxydrop_ml/photon/flashgap.py rec_features, in float64 JavaScript.
// Pure ES module (no DOM, no Node APIs), runs in the browser and in the Node tests. Streaming: push one 30 Hz R,G,B sample at a time.
//   0-18   FUSION  chain v1 deltas (python/oxydrop/chain.py deltas): per-window features of engine_v1.js winFeats, causal forward fill, minus the
//                  median of the first-minute windows (t + 10 <= 60 s), NaN -> that median
//   19-33  QUAL    absolute signal quality (flashgap.quality_feats): spectral SNR per channel, log AC sd, log DC, clipped fraction, quantisation
//                  (zero-difference) fraction per channel, autocorrelation peak of the best channel, beat fraction (trailing 30 s)
//   34-38  RESP    respiratory / autonomic group over the trailing 60 s (flashgap.resp_feats), forward filled and made relative to the first minute
//                  (flashgap._ffill_base; a first-minute median that is NaN becomes 0)
// FFTs of numpy are replaced by direct DFTs of the bins that are used (same values to rounding). Parity: tests/distill_parity.mjs against
// tests/fixtures/ref_distill.json (python/oxydrop_ml/photon/distill_app_export.py), tolerance 1e-6 on every feature.

import { Chain as V1Chain, winFeats, findPeaks, FS, WIN, STEP, CAL, FUSION } from './engine_v1.js';

export { FS, WIN, STEP, CAL, FUSION };
export const QUAL = ['snr_R', 'snr_G', 'snr_B', 'lpi_R', 'lpi_G', 'lpi_B', 'ldc_R', 'ldc_G', 'ldc_B', 'clip', 'qz_R', 'qz_G', 'qz_B', 'acf', 'beat_frac'];
export const RESP = ['rr_ibi60', 'rsa60', 'rmssd60', 'rr_dc60', 'resp_dc60'];
export const COLS = [...FUSION, ...QUAL, ...RESP];
const L = WIN * FS;                    // 300 samples per window

// ---------- numpy helpers ----------
const mean = (a, lo = 0, hi = a.length) => { let s = 0; for (let i = lo; i < hi; i++) s += a[i]; return s / (hi - lo); };
const stdp = (a) => { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; } return Math.sqrt(s / a.length); };
function median(a) { const b = Array.from(a).sort((x, y) => x - y), n = b.length; return n % 2 ? b[(n - 1) / 2] : 0.5 * (b[n / 2 - 1] + b[n / 2]); }
const nanmedian = (a) => { const b = a.filter((v) => !Number.isNaN(v)); return b.length ? median(b) : NaN; };
/** np.hanning(M): 0.5 + 0.5 cos(pi n / (M - 1)), n = 1 - M, 3 - M, ..., M - 1 */
function hanning(M) { const w = new Float64Array(M); if (M === 1) { w[0] = 1; return w; } for (let i = 0; i < M; i++) w[i] = 0.5 + 0.5 * Math.cos(Math.PI * (1 - M + 2 * i) / (M - 1)); return w; }
/** np.fft.rfftfreq(n, d)[k] */
const rfftfreq = (k, n, d) => k * (1.0 / (n * d));
/** np.interp for increasing xp */
function interp(xq, xp, fp) {
  const out = new Float64Array(xq.length), n = xp.length;
  for (let q = 0; q < xq.length; q++) {
    const v = xq[q];
    if (v <= xp[0]) { out[q] = fp[0]; continue; }
    if (v >= xp[n - 1]) { out[q] = fp[n - 1]; continue; }
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= v) lo = m; else hi = m; }
    out[q] = fp[lo] + (fp[hi] - fp[lo]) * (v - xp[lo]) / (xp[hi] - xp[lo]);
  }
  return out;
}
/** y - polyval(polyfit(x, y, 1), x) (least-squares line) */
function detrend(x, y) {
  const n = x.length, mx = mean(x), my = mean(y); let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  const a = sxy / sxx, b = my - a * mx, out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = y[i] - (a * x[i] + b);
  return out;
}
/** |rfft(x * hanning(len), n=nfft)| argmax over the bins with lo <= freq <= hi -> that frequency (first maximum, as np.argmax) */
function peakFreq(x, nfft, d, lo, hi) {
  const M = x.length, w = hanning(M); let best = -1, bf = NaN;
  for (let k = 0; k <= nfft / 2; k++) {
    const f = rfftfreq(k, nfft, d); if (!(f >= lo && f <= hi)) continue;
    let re = 0, im = 0;
    for (let n = 0; n < M; n++) { const a = 2 * Math.PI * ((k * n) % nfft) / nfft, v = x[n] * w[n]; re += v * Math.cos(a); im -= v * Math.sin(a); }
    const mag = Math.hypot(re, im); if (mag > best) { best = mag; bf = f; }
  }
  return bf;
}
/** std of irfft(G, n) where G = rfft(x) with the bins outside [lo, hi] (frequency, spacing d) set to 0 */
function bandStd(x, d, lo, hi) {
  const n = x.length, keep = [];
  for (let k = 0; k <= Math.floor(n / 2); k++) { const f = rfftfreq(k, n, d); if (f >= lo && f <= hi) keep.push(k); }
  const G = keep.map((k) => { let re = 0, im = 0; for (let j = 0; j < n; j++) { const a = 2 * Math.PI * ((k * j) % n) / n; re += x[j] * Math.cos(a); im -= x[j] * Math.sin(a); } return [k, re, im]; });
  const y = new Float64Array(n);
  for (let m = 0; m < n; m++) {
    let s = 0;
    for (const [k, re, im] of G) {
      const c = (k === 0 || (n % 2 === 0 && k === n / 2)) ? 1 : 2, a = 2 * Math.PI * ((k * m) % n) / n;
      s += c * (re * Math.cos(a) - (k === 0 || (n % 2 === 0 && k === n / 2) ? 0 : im * Math.sin(a)));
    }
    y[m] = s / n;
  }
  return stdp(y);
}
/** chosen beat channel of a trailing AC span (green unless its sd is < 0.3 x red), negated, as flashgap / chain do */
function beatSignal(ac, lo, e) {
  const n = e - lo, s = [0, 1].map((c) => stdp(ac[c].slice(lo, e)));
  const g = s[1] > s[0] * 0.3 ? ac[1] : ac[0], x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = -g[lo + i];
  return x;
}

// ---------- spectral constants of quality_feats (nf = 1024) ----------
const NF = 1024;
const FR = (k) => rfftfreq(k, NF, 1.0 / FS);
const BAND_K = [], KS = [];
for (let k = 0; k <= NF / 2; k++) { const f = FR(k); if (f >= 0.65 && f <= 4.0) BAND_K.push(k); if (f >= 0.85 && f <= 3.0) KS.push(k); }
const W_NEAR = Math.round(0.10 / (FS / NF));
const K_MAX = Math.min(2 * KS[KS.length - 1] + W_NEAR + 1, NF / 2 + 1);   // bins 0 .. K_MAX - 1 are all that quality_feats reads
const HANN_L = hanning(L);
const COSN = new Float64Array(NF), SINN = new Float64Array(NF);
for (let j = 0; j < NF; j++) { COSN[j] = Math.cos(2 * Math.PI * j / NF); SINN[j] = Math.sin(2 * Math.PI * j / NF); }

/** flashgap.quality_feats for the window ending at sample te (exclusive). rgb: raw samples [3][n]; ac: band-passed log [3][n]. */
export function qualityFeats(rgb, ac, te) {
  const s0 = te - L, A = [0, 1, 2].map((c) => { const m = mean(ac[c], s0, te), a = new Float64Array(L); for (let i = 0; i < L; i++) a[i] = ac[c][s0 + i] - m; return a; });
  // power spectrum (zero-padded to 1024, Hann window), prefix sums (np.cumsum)
  const P = [0, 1, 2].map(() => new Float64Array(K_MAX)), Cs = [0, 1, 2].map(() => new Float64Array(K_MAX + 1));
  for (let c = 0; c < 3; c++) {
    const xw = new Float64Array(L); for (let n = 0; n < L; n++) xw[n] = A[c][n] * HANN_L[n];
    for (let k = 0; k < K_MAX; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < L; n++) { const j = (k * n) % NF; re += xw[n] * COSN[j]; im -= xw[n] * SINN[j]; }
      P[c][k] = re * re + im * im; Cs[c][k + 1] = Cs[c][k] + P[c][k];
    }
  }
  const tot = [0, 1, 2].map((c) => { let s = 0; for (const k of BAND_K) s += P[c][k]; return s; });
  const Ssum = (k) => P[0][k] / (tot[0] + 1e-30) + P[1][k] / (tot[1] + 1e-30) + P[2][k] / (tot[2] + 1e-30);
  let kb = KS[0], bestS = -Infinity;
  for (const k of KS) { const v = Ssum(k) + 0.7 * Ssum(2 * k); if (v > bestS) { bestS = v; kb = k; } }
  const rs = (c, lo, hi) => Cs[c][Math.min(hi, NF / 2 + 1)] - Cs[c][Math.max(lo, 0)];
  const snr = [], lpi = [], ldc = [], qz = [];
  for (let c = 0; c < 3; c++) {
    const near = rs(c, kb - W_NEAR, kb + W_NEAR + 1) + rs(c, 2 * kb - W_NEAR, 2 * kb + W_NEAR + 1), sd = stdp(A[c]);
    const v = Math.log((near + 1e-12) / (Math.max(tot[c] - near, 0) + 1e-12));
    snr.push(sd < 1e-6 ? -6.0 : Math.min(Math.max(v, -6.0), 6.0));
    lpi.push(Math.log(sd + 1e-6));
    ldc.push(Math.log(mean(rgb[c], s0, te) + 1.0));
    let z = 0; for (let i = s0 + 1; i < te; i++) if (rgb[c][i] - rgb[c][i - 1] === 0) z++;
    qz.push(z / (L - 1));
  }
  let nclip = 0;
  for (let i = s0; i < te; i++) if ([0, 1, 2].some((c) => rgb[c][i] >= 254.5 || rgb[c][i] <= 0.5)) nclip++;
  const clip = nclip / L;
  let best = 0; for (let c = 1; c < 3; c++) if (snr[c] > snr[best]) best = c;
  const xb = A[best], r0 = (() => { let s = 0; for (let n = 0; n < L; n++) s += xb[n] * xb[n]; return s; })();
  let acf = -Infinity;
  for (let lag = Math.trunc(FS / 3.0); lag < Math.trunc(FS / 0.85) + 1; lag++) {
    let s = 0; for (let n = 0; n + lag < L; n++) s += xb[n] * xb[n + lag];
    acf = Math.max(acf, s / (r0 + 1e-30));
  }
  const f0 = FR(kb), lo30 = Math.max(0, te - 30 * FS), x = beatSignal(ac, lo30, te);
  const pk = findPeaks(x, Math.trunc(FS * 0.33), stdp(x) * 0.3 + 1e-12);
  let nv = 0; for (let i = 1; i < pk.length; i++) { const v = (pk[i] - pk[i - 1]) / FS; if (v > 0.3 && v < 1.6) nv++; }
  const beat = Math.min(nv / Math.max(f0 * (te - lo30) / FS - 1.0, 1.0), 2.0);
  return [...snr, ...lpi, ...ldc, clip, ...qz, acf, beat];
}

/** flashgap.resp_feats for the window ending at te (trailing 60 s). lg, ac: [3][n]. NaN where undefined. */
export function respFeats(lg, ac, te) {
  const out = [NaN, NaN, NaN, NaN, NaN], lo = Math.max(0, te - 60 * FS), x = beatSignal(ac, lo, te);
  const pk = findPeaks(x, Math.trunc(FS * 0.33), stdp(x) * 0.3 + 1e-12);
  let ibi = [], tt = [];
  for (let i = 1; i < pk.length; i++) { const v = (pk[i] - pk[i - 1]) / FS; if (v > 0.3 && v < 1.6) { ibi.push(v); tt.push(pk[i] / FS); } }
  if (ibi.length >= 8) {
    const d = []; for (let i = 1; i < ibi.length; i++) { const v = ibi[i] - ibi[i - 1]; if (Math.abs(v) < 0.3) d.push(v); }
    if (d.length >= 4) { let s = 0; for (const v of d) s += v * v; out[2] = Math.log(Math.sqrt(s / d.length) + 1e-3); }
    const nTi = Math.max(0, Math.ceil((tt[tt.length - 1] - tt[0]) / 0.25)), ti = new Float64Array(nTi);
    for (let i = 0; i < nTi; i++) ti[i] = tt[0] + i * 0.25;
    if (nTi >= 32) {
      const sr = detrend(ti, interp(ti, tt, ibi));
      out[0] = peakFreq(sr, 512, 0.25, 0.1, 0.6) * 60;
      out[1] = Math.log(bandStd(sr, 0.25, 0.1, 0.6) + 1e-4);
    }
  }
  const n3 = Math.floor((te - lo) / 10);
  if (n3 >= 48) {
    const d3 = new Float64Array(n3), tx = new Float64Array(n3);
    for (let i = 0; i < n3; i++) { d3[i] = mean(lg[0], lo + 10 * i, lo + 10 * i + 10); tx[i] = i / 3.0; }
    const dd = detrend(tx, d3);
    out[3] = peakFreq(dd, 512, 1 / 3.0, 0.1, 0.6) * 60;
    out[4] = Math.log(bandStd(dd, 1 / 3.0, 0.1, 0.6) + 1e-6);
  }
  return out;
}

/** Streaming feature extractor. push(r, g, b) -> null between windows | {s, calibrating: true} | {s, calibrating: false, x: Float64Array(39), ...}.
 *  Windows every 2 s once 10 s are in; the first minute (windows with s + 10 <= 60) is the label-free per-recording baseline; s >= 60 are model windows. */
export class DistillFeatures {
  constructor() {
    // the v1 chain (engine_v1.js) supplies the lg / ac streams and the chain deltas; its own logistic output is never computed here
    this.chain = new V1Chain(null); this.chain._window = (te) => this._chainWindow(te);
    this.rgb = [[], [], []]; this.respLast = new Array(RESP.length).fill(NaN); this.respCal = []; this.respBase = null; this._out = null;
  }
  get n() { return this.chain.n; }
  push(r, g, b) {
    this.rgb[0].push(r); this.rgb[1].push(g); this.rgb[2].push(b);
    return this.chain.push(r, g, b);
  }
  _chainWindow(te) {
    const ch = this.chain, s = te / FS - WIN, f = winFeats(...ch.lg, ...ch.ac, te);
    const row = FUSION.map((k, j) => { const v = f[k] === undefined ? NaN : f[k]; return Number.isNaN(v) ? ch.last[j] : v; });
    ch.last = row.slice();
    const rr = respFeats(ch.lg, ch.ac, te).map((v, j) => (Number.isNaN(v) ? this.respLast[j] : v));
    this.respLast = rr.slice();
    const hr = row[0];
    if (s + WIN <= CAL) { ch.calRows.push(row); this.respCal.push(rr); return { s, calibrating: true, hr }; }
    if (!ch.base) ch.base = FUSION.map((_, j) => nanmedian(ch.calRows.map((r) => r[j])));
    if (!this.respBase) this.respBase = RESP.map((_, j) => { const m = nanmedian(this.respCal.map((r) => r[j])); return Number.isFinite(m) ? m : 0.0; });
    if (s < CAL) return { s, calibrating: true, warm: true, hr };
    const D = row.map((v, j) => (Number.isNaN(v) ? ch.base[j] : v) - ch.base[j]);
    const Q = qualityFeats(this.rgb, ch.ac, te);
    const R = rr.map((v, j) => (Number.isNaN(v) ? this.respBase[j] : v) - this.respBase[j]);
    return { s, calibrating: false, x: Float64Array.from([...D, ...Q, ...R]), hr, dHR: D[0], baseHR: ch.base[0], sqi: f._sqi };
  }
  calibrationSummary() { return this.chain.calibrationSummary(); }
}

/** Batch helper (tests): rgb = array of [r, g, b] at 30 Hz -> every model window {s, x}. */
export function featuresBatch(rgb) {
  const fe = new DistillFeatures(), out = [];
  for (const [r, g, b] of rgb) { const d = fe.push(r, g, b); if (d && !d.calibrating) out.push(d); }
  return out;
}
