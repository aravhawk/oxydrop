// OxyDrop TINY chain, JavaScript port of python/tiny_chain.py + python/stages.py.
// Pure ES module with no DOM/Node dependencies, so the same file runs in the browser and in the Node parity test.
// Input: per-frame mean R,G,B (0..255 scale) resampled to a fixed 30 Hz grid. Fully causal, streaming.
// Parity target: Python reference (scipy sosfilt / find_peaks, numpy) to ~1e-9 on the probability.

export const FS = 30, WIN = 10, STEP = 2, CAL = 60, BAND = 0.10;
export const FUSION = ['hr', 'rmssd', 'sdnn', 'rr_amp', 'rr_ibi', 'pi_R', 'pi_G', 'pi_B', 'beat_cv',
  'ror_RG', 'ror_RB', 'ror_GB', 'sl_RG', 'sl_RB', 'sl30_RG', 'ror30_RG', 'dc_R', 'dc_G', 'dc_B'];

// scipy.signal.butter(3, [0.7, 4.0], 'band', fs=30, output='sos')
const SOS = [
  [0.023033552196294126, 0.04606710439258825, 0.023033552196294126, 1.0, -1.3774469840172525, 0.4705642812122516],
  [1.0, 0.0, -1.0, 1.0, -1.0980118950671502, 0.5789751164600165],
  [1.0, -2.0, 1.0, 1.0, -1.8699353828611232, 0.8923205348852282],
];

// ---------- small numeric helpers (numpy semantics) ----------
const mean = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
const stdp = (a) => { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; } return Math.sqrt(s / a.length); };
function median(a) {
  const b = Array.from(a).sort((x, y) => x - y), n = b.length;
  return n % 2 ? b[(n - 1) / 2] : 0.5 * (b[n / 2 - 1] + b[n / 2]);
}
const nanmedian = (a) => { const b = Array.from(a).filter((v) => !Number.isNaN(v)); return b.length ? median(b) : NaN; };
function cov3(rows) { // rows: array of [r,g,b]; ddof = 1 like np.cov
  const n = rows.length, m = [0, 0, 0];
  for (const r of rows) { m[0] += r[0]; m[1] += r[1]; m[2] += r[2]; }
  m[0] /= n; m[1] /= n; m[2] /= n;
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const r of rows) for (let i = 0; i < 3; i++) for (let j = i; j < 3; j++) C[i][j] += (r[i] - m[i]) * (r[j] - m[j]);
  for (let i = 0; i < 3; i++) for (let j = i; j < 3; j++) { C[i][j] /= (n - 1); C[j][i] = C[i][j]; }
  return C;
}

// ---------- scipy.signal.find_peaks(x, distance, prominence) port ----------
function localMaxima(x) {
  const out = []; let i = 1; const iMax = x.length - 1;
  while (i < iMax) {
    if (x[i - 1] < x[i]) {
      let ahead = i + 1;
      while (ahead < iMax && x[ahead] === x[i]) ahead++;
      if (x[ahead] < x[i]) { out.push(((i + ahead - 1) / 2) | 0); i = ahead; }
    }
    i++;
  }
  return out;
}
function selectByDistance(peaks, x, distance) {
  const n = peaks.length, d = Math.ceil(distance), keep = new Uint8Array(n).fill(1);
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (x[peaks[a]] - x[peaks[b]]) || (a - b));
  for (let i = n - 1; i >= 0; i--) {
    const j = order[i]; if (!keep[j]) continue;
    for (let k = j - 1; k >= 0 && peaks[j] - peaks[k] < d; k--) keep[k] = 0;
    for (let k = j + 1; k < n && peaks[k] - peaks[j] < d; k++) keep[k] = 0;
  }
  return peaks.filter((_, i) => keep[i]);
}
function prominence(x, p) {
  let leftMin = x[p]; for (let i = p; i >= 0 && x[i] <= x[p]; i--) if (x[i] < leftMin) leftMin = x[i];
  let rightMin = x[p]; for (let i = p; i < x.length && x[i] <= x[p]; i++) if (x[i] < rightMin) rightMin = x[i];
  return x[p] - Math.max(leftMin, rightMin);
}
export function findPeaks(x, distance, minProm) {
  let pk = localMaxima(x);
  pk = selectByDistance(pk, x, distance);
  return pk.filter((p) => prominence(x, p) >= minProm);
}

// np.interp with clamping
function interp(xq, xp, fp) {
  return xq.map((v) => {
    if (v <= xp[0]) return fp[0];
    if (v >= xp[xp.length - 1]) return fp[fp.length - 1];
    let lo = 0, hi = xp.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= v) lo = m; else hi = m; }
    return fp[lo] + (fp[hi] - fp[lo]) * (v - xp[lo]) / (xp[hi] - xp[lo]);
  });
}

// dominant respiratory frequency (bpm) of a 4 Hz series: |rfft(x*hann, n=1024)| argmax over 0.1-0.6 Hz
function respRate(x) {
  const M = x.length; if (M < 8) return NaN;
  const m = mean(x), xc = x.map((v) => v - m); if (stdp(xc) === 0) return NaN;
  const w = xc.map((v, n) => v * (0.5 - 0.5 * Math.cos(2 * Math.PI * n / (M - 1))));
  let best = -1, bestK = -1;
  for (let k = 0; k <= 512; k++) {
    const fr = k * 0.00390625; if (fr < 0.1 || fr > 0.6) continue;
    let re = 0, im = 0; const ang = 2 * Math.PI * k / 1024;
    for (let n = 0; n < M; n++) { re += w[n] * Math.cos(ang * n); im -= w[n] * Math.sin(ang * n); }
    const mag = Math.hypot(re, im); if (mag > best) { best = mag; bestK = k; }
  }
  return bestK * 0.00390625 * 60;
}

// spectral-concentration signal-quality index (cardiac band) of a 10 s green AC window (app quality gate; same definition as stages.py 'sqi')
export function sqiOf(a10g) {
  const M = a10g.length, N = 2048, w = a10g.map((v, n) => v * (0.5 - 0.5 * Math.cos(2 * Math.PI * n / (M - 1))));
  const mags = []; let kMax = -1;
  for (let k = 0; k <= N / 2; k++) {
    const fr = k * FS / N; if (fr < 0.7 || fr > 3.5) continue;
    let re = 0, im = 0; const ang = 2 * Math.PI * k / N;
    for (let n = 0; n < M; n++) { re += w[n] * Math.cos(ang * n); im -= w[n] * Math.sin(ang * n); }
    mags.push([k, Math.hypot(re, im)]);
  }
  let tot = 0, bi = 0; mags.forEach(([, m], i) => { tot += m; if (m > mags[bi][1]) bi = i; });
  let near = 0; for (let i = Math.max(bi - 4, 0); i < Math.min(bi + 5, mags.length); i++) near += mags[i][1];
  return near / (tot + 1e-9);
}

// ---------- per-window features ----------
export function winFeats(lgR, lgG, lgB, acR, acG, acB, te) {
  const f = {}, s10 = te - 10 * FS, s30 = Math.max(0, te - 30 * FS);
  const a10 = [], a30 = [];
  for (let i = s10; i < te; i++) a10.push([acR[i], acG[i], acB[i]]);
  for (let i = s30; i < te; i++) a30.push([acR[i], acG[i], acB[i]]);
  const sd = [0, 1, 2].map((c) => stdp(a10.map((r) => r[c])) + 1e-9);
  const dc = [mean(lgR.slice(s10, te)), mean(lgG.slice(s10, te)), mean(lgB.slice(s10, te))];
  f.ror_RG = Math.log(sd[0] / sd[1]); f.ror_RB = Math.log(sd[0] / sd[2]); f.ror_GB = Math.log(sd[1] / sd[2]);
  const C = cov3(a10); f.sl_RG = C[0][1] / (C[1][1] + 1e-12); f.sl_RB = C[0][2] / (C[2][2] + 1e-12);
  const C3 = cov3(a30); f.sl30_RG = C3[0][1] / (C3[1][1] + 1e-12); f.ror30_RG = 0.5 * Math.log(C3[0][0] / (C3[1][1] + 1e-12));
  f.dc_R = dc[0]; f.dc_G = dc[1]; f.dc_B = dc[2];
  f.pi_R = Math.log(sd[0]); f.pi_G = Math.log(sd[1]); f.pi_B = Math.log(sd[2]);
  const x = a30.map((r) => -(sd[1] > sd[0] * 0.3 ? r[1] : r[0]));
  const pk = findPeaks(x, Math.trunc(FS * 0.33), stdp(x) * 0.3);
  let ibi = [];
  if (pk.length >= 6) { for (let i = 1; i < pk.length; i++) { const v = (pk[i] - pk[i - 1]) / FS; if (v > 0.3 && v < 1.6) ibi.push(v); } }
  if (ibi.length >= 5) {
    f.hr = 60 / median(ibi);
    let ss = 0; for (let i = 1; i < ibi.length; i++) ss += (ibi[i] - ibi[i - 1]) ** 2;
    f.rmssd = Math.log(Math.sqrt(ss / (ibi.length - 1)) + 1e-3);
    f.sdnn = Math.log(stdp(ibi) + 1e-3);
    const amp = pk.map((p) => x[p]), tt = pk.map((p) => p / FS);
    const nTi = Math.ceil((tt[tt.length - 1] - tt[0]) / 0.25), ti = [];
    for (let i = 0; i < nTi; i++) ti.push(tt[0] + i * 0.25);
    if (ti.length > 16) {
      f.rr_amp = respRate(interp(ti, tt, amp));
      const dpk = []; for (let i = 1; i < pk.length; i++) dpk.push((pk[i] - pk[i - 1]) / FS);
      f.rr_ibi = respRate(interp(ti, tt.slice(1), dpk));
    }
    f.beat_cv = stdp(amp) / (mean(amp.map(Math.abs)) + 1e-9);
  }
  // quality (not a model input)
  f._sqi = sqiOf(a10.map((r) => r[1]));
  return f;
}

const sigm = (z) => 1 / (1 + Math.exp(-z));
function lr(m, x) { let z = m.b; for (let i = 0; i < x.length; i++) z += ((x[i] - m.mu[i]) / m.sd[i]) * m.coef[i]; return sigm(z); }

// ---------- streaming chain ----------
export class Chain {
  constructor(weights, opts = {}) {
    this.W = weights; this.band = opts.band ?? BAND;
    this.z = [0, 1, 2].map(() => SOS.map(() => [0, 0]));  // sosfilt state per channel/section
    this.lg = [[], [], []]; this.ac = [[], [], []]; this.n = 0;
    this.last = new Array(FUSION.length).fill(NaN); this.calRows = []; this.base = null;
  }
  /** push one 30 Hz sample (mean R,G,B on 0..255 scale). Returns a decision object when a new 2 s window completes, else null. */
  push(r, g, b) {
    const v = [r, g, b];
    for (let c = 0; c < 3; c++) {
      const l = Math.log(Math.max(v[c], 1e-3) + 1.0); this.lg[c].push(l);
      let x = l;
      for (let s = 0; s < SOS.length; s++) {
        const [b0, b1, b2, , a1, a2] = SOS[s], z = this.z[c][s];
        const y = b0 * x + z[0]; z[0] = b1 * x - a1 * y + z[1]; z[1] = b2 * x - a2 * y; x = y;
      }
      this.ac[c].push(x);
    }
    this.n++;
    if (this.n < WIN * FS || (this.n - WIN * FS) % (STEP * FS) !== 0) return null;
    return this._window(this.n);
  }
  _window(te) {
    const s = te / FS - WIN, f = winFeats(...this.lg, ...this.ac, te);
    const row = FUSION.map((k, j) => { const val = f[k] === undefined ? NaN : f[k]; return Number.isNaN(val) ? this.last[j] : val; });
    this.last = row.slice();                       // causal forward fill
    if (s + WIN <= CAL) { this.calRows.push(row); return { s, calibrating: true, sqi: f._sqi, hr: row[0] }; }
    if (!this.base) this.base = FUSION.map((_, j) => nanmedian(this.calRows.map((r) => r[j])));
    const X = row.map((v, j) => (Number.isNaN(v) ? this.base[j] : v) - this.base[j]);
    if (s < CAL) return { s, calibrating: true, warm: true, sqi: f._sqi, hr: row[0] };
    const p = 0.5 * (lr(this.W.hr, [X[0]]) + lr(this.W.fusion, X));
    const label = Math.abs(p - 0.5) < this.band ? 'unsure' : (p > 0.5 ? 'HYPOXIC' : 'normal');
    return { s, calibrating: false, p, label, hr: row[0], dHR: X[0], baseHR: this.base[0], sqi: f._sqi, X };
  }
  calibrationSummary() {
    const hrs = this.calRows.map((r) => r[0]).filter((v) => !Number.isNaN(v));
    return { hr: hrs.length ? median(hrs) : NaN, hrSpread: hrs.length ? Math.max(...hrs) - Math.min(...hrs) : NaN, windows: this.calRows.length };
  }
}

/** Batch helper (parity test): rgb = array of [r,g,b] at 30 Hz. Returns decisions with s >= CAL. */
export function runBatch(rgb, weights, opts) {
  const ch = new Chain(weights, opts), out = [];
  for (const [r, g, b] of rgb) { const d = ch.push(r, g, b); if (d && !d.calibrating) out.push(d); }
  return out;
}
